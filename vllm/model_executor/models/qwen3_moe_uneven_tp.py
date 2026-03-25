"""
Experimental Qwen3-MoE implementation with *uneven* tensor parallel sharding.

This file is intentionally self‑contained and does NOT integrate with
vLLM's full weight‑loading / quantization pipeline. It is meant as a
reference for how one could implement 1/3–2/3 style sharding at the
layer level, not as a drop‑in replacement for `qwen3_moe.py`.

Design assumptions:
- TP size == 2.
- Rank 0 holds 1/3 of certain tensor‑parallel dimensions.
- Rank 1 holds 2/3 of those dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Any, Iterable, List, Tuple

import torch
import torch.nn as nn

from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import set_default_rope_theta

from .interfaces import MixtureOfExperts, SupportsEagle3, SupportsLoRA, SupportsPP
from .qwen3_moe import Qwen3MoeModel, Qwen3MoeSparseMoeBlock
from .utils import (
    PPMissingLayer,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


def _compute_uneven_splits(
    total: int, ratios: List[int]
) -> Tuple[List[int], List[int]]:
    """Given a total size and per‑rank ratios, compute integer splits.

    Returns (sizes, offsets), where:
    - sizes[i]  is the local size for rank i
    - offsets[i] is the start offset for rank i in the global tensor
    """
    n = len(ratios)
    assert n > 0
    assert total > 0

    s = sum(ratios)
    # Initial floor allocation.
    raw_sizes = [total * r // s for r in ratios]
    allocated = sum(raw_sizes)
    # Distribute remainder starting from rank 0.
    remainder = total - allocated
    i = 0
    while remainder > 0:
        raw_sizes[i] += 1
        remainder -= 1
        i = (i + 1) % n

    sizes = raw_sizes
    offsets = []
    cur = 0
    for sz in sizes:
        offsets.append(cur)
        cur += sz
    assert cur == total, (cur, total)
    return sizes, offsets


def _compute_uneven_gqa_splits(
    num_heads: int,
    num_kv_heads: int,
    ratios: List[int],
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Compute uneven splits for GQA that preserve num_heads % num_kv_heads == 0 per rank.

    The Attention layer requires local_num_heads % local_num_kv_heads == 0.
    We achieve this by splitting num_kv_heads first, then assigning
    local_num_heads = local_num_kv_heads * (num_heads // num_kv_heads).

    Returns (q_sizes, q_offsets, kv_sizes, kv_offsets).
    """
    assert num_heads % num_kv_heads == 0
    num_heads_per_kv = num_heads // num_kv_heads

    kv_sizes, kv_offsets = _compute_uneven_splits(num_kv_heads, ratios)
    q_sizes = [kv_sizes[i] * num_heads_per_kv for i in range(len(ratios))]
    q_offsets = []
    cur = 0
    for sz in q_sizes:
        q_offsets.append(cur)
        cur += sz
    assert cur == num_heads, (q_sizes, num_heads)
    return q_sizes, q_offsets, kv_sizes, kv_offsets


@dataclass
class UnevenTPConfig:
    """Config for uneven TP sharding.

    For now we only support:
      - tp_world_size == 2 by default
      - ratios == [1, 2]  (rank0: 1/3, rank1: 2/3) if not overridden
    """

    ratios: List[int]

    @classmethod
    def qwen3_default(cls) -> "UnevenTPConfig":
        tp = get_tensor_model_parallel_world_size()
        assert tp == 2, "This experimental implementation assumes TP=2."
        return cls(ratios=[1, 2])

    @classmethod
    def from_model_config(cls, hf_config: Any) -> "UnevenTPConfig":
        """Create UnevenTPConfig from the HF text config if provided.

        Users can set `uneven_tp_ratios` in the model config JSON, e.g.:
          "uneven_tp_ratios": [1, 2]

        If not set, fall back to the default 1:2 split.
        """
        tp = get_tensor_model_parallel_world_size()
        ratios = getattr(hf_config, "uneven_tp_ratios", None)
        if ratios is None:
            return cls.qwen3_default()

        # Allow list/tuple/other iterable of ints.
        ratios_list = list(ratios)
        assert len(ratios_list) == tp, (
            f"len(uneven_tp_ratios) ({len(ratios_list)}) "
            f"must equal tensor_parallel_size ({tp})."
        )
        assert all(int(r) > 0 for r in ratios_list), "uneven_tp_ratios must be > 0."
        return cls(ratios=[int(r) for r in ratios_list])

    @property
    def world_size(self) -> int:
        return len(self.ratios)


class UnevenColumnLinear(nn.Module):
    """Column‑parallel linear with uneven sharding of the output dimension.

    Each rank i stores a different number of output channels, decided by
    `ratios`. Forward is purely local; the full concatenated output can be
    reconstructed via all‑gather (with padding) if needed.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        uneven_cfg: UnevenTPConfig | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if uneven_cfg is None:
            uneven_cfg = UnevenTPConfig.qwen3_default()

        self.in_features = in_features
        self.out_features = out_features
        self.uneven_cfg = uneven_cfg

        tp_rank = get_tensor_model_parallel_rank()
        assert tp_rank < uneven_cfg.world_size

        sizes, offsets = _compute_uneven_splits(out_features, uneven_cfg.ratios)
        self.local_out_features = sizes[tp_rank]
        self.local_offset = offsets[tp_rank]

        self.weight = nn.Parameter(
            torch.empty(self.local_out_features, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.local_out_features))
        else:
            self.register_parameter("bias", None)

        # Simple initialization; in a real system you'd want to mimic Qwen init.
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = in_features
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_features]
        y = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            y = y + self.bias
        # Shape: [..., local_out_features]
        return y


class UnevenRowLinear(nn.Module):
    """Row‑parallel linear with uneven sharding of the input dimension.

    Each rank i stores a different slice of the input dimension. Forward:
      y_local = x_local @ W_local^T
      y = all_reduce_sum(y_local)

    This works with different local input sizes as long as all ranks agree
    on the same output size.

    When `sizes` is provided, use those explicit sizes instead of computing
    from ratios (needed when input dim is split by GQA-aware head allocation).

    When `input_is_local=True`, x is already the local chunk per rank
    (e.g. from column-parallel output without all-gather). No slicing is done.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        uneven_cfg: UnevenTPConfig | None = None,
        bias: bool = False,
        sizes: List[int] | None = None,
        input_is_local: bool = False,
    ) -> None:
        super().__init__()
        if uneven_cfg is None:
            uneven_cfg = UnevenTPConfig.qwen3_default()

        self.in_features = in_features
        self.out_features = out_features
        self.uneven_cfg = uneven_cfg
        self.input_is_local = input_is_local

        tp_rank = get_tensor_model_parallel_rank()
        assert tp_rank < uneven_cfg.world_size

        if sizes is not None:
            assert sum(sizes) == in_features, (sizes, in_features)
            self.local_in_features = sizes[tp_rank]
            self.local_offset = sum(sizes[:tp_rank])
        else:
            sizes_list, offsets = _compute_uneven_splits(
                in_features, uneven_cfg.ratios
            )
            self.local_in_features = sizes_list[tp_rank]
            self.local_offset = offsets[tp_rank]

        self.weight = nn.Parameter(
            torch.empty(out_features, self.local_in_features)
        )
        if bias:
            # Bias is replicated across ranks; we'll add it after all‑reduce.
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = in_features
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_is_local:
            x_local = x
        else:
            x_local = x[
                ..., self.local_offset : self.local_offset + self.local_in_features
            ]
        y_local = torch.matmul(x_local, self.weight.t())

        # All‑reduce partial outputs across TP group.
        if get_tensor_model_parallel_world_size() > 1:
            y_local = tensor_model_parallel_all_reduce(y_local)

        if self.bias is not None:
            y_local = y_local + self.bias
        # Now every rank has full output: [..., out_features]
        return y_local


class UnevenMergedColumnLinear(nn.Module):
    """Merged column-parallel linear with uneven sharding of output dimensions.

    Similar to MergedColumnParallelLinear but each rank holds a different
    number of output channels per part. Used for gate_up_proj where gate
    and up are merged: each part (gate, up) is split by ratios, and rank i
    gets part_0_i and part_1_i concatenated.

    Args:
        in_features: input dimension.
        output_size_per_part: total output size for each merged part
            (e.g., intermediate_size for each of gate and up).
        n_parts: number of merged parts (e.g., 2 for gate_up).
    """

    def __init__(
        self,
        in_features: int,
        output_size_per_part: int,
        n_parts: int = 2,
        uneven_cfg: UnevenTPConfig | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if uneven_cfg is None:
            uneven_cfg = UnevenTPConfig.qwen3_default()

        self.in_features = in_features
        self.output_size_per_part = output_size_per_part
        self.n_parts = n_parts
        self.uneven_cfg = uneven_cfg

        tp_rank = get_tensor_model_parallel_rank()
        assert tp_rank < uneven_cfg.world_size

        sizes, offsets = _compute_uneven_splits(
            output_size_per_part, uneven_cfg.ratios
        )
        self.local_out_per_part = sizes[tp_rank]
        self.local_offset_per_part = offsets[tp_rank]
        self.local_out_features = self.local_out_per_part * n_parts

        self.weight = nn.Parameter(
            torch.empty(self.local_out_features, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.local_out_features))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = in_features
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            y = y + self.bias
        return y


class Qwen3UnevenTPMLP(nn.Module):
    """Qwen3MoE-style MLP with uneven tensor parallel sharding.

    gate_up_proj and down_proj use uneven 1:2 split of intermediate_size.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        uneven_cfg: UnevenTPConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if uneven_cfg is None:
            uneven_cfg = UnevenTPConfig.qwen3_default()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.uneven_cfg = uneven_cfg

        self.gate_up_proj = UnevenMergedColumnLinear(
            in_features=hidden_size,
            output_size_per_part=intermediate_size,
            n_parts=2,
            uneven_cfg=uneven_cfg,
            bias=False,
        )
        self.down_proj = UnevenRowLinear(
            in_features=intermediate_size,
            out_features=hidden_size,
            uneven_cfg=uneven_cfg,
            bias=False,
            input_is_local=True,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        from vllm.model_executor.layers.activation import SiluAndMul

        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3UnevenTPAttention(nn.Module, AttentionLayerBase):
    """Simplified Qwen3‑style attention with uneven TP over heads.

    This shows how to:
      - shard QKV projection outputs unevenly across TP ranks
      - still use the existing `Attention` kernel on each rank
      - use a row‑parallel output projection with uneven input sharding
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int | None = None,
        cache_config: Any | None = None,
        prefix: str = "",
        rms_norm_eps: float = 1e-6,
        rope_parameters: dict | None = None,
        max_position_embeddings: int = 8192,
        dual_chunk_attention_config: dict | None = None,
        uneven_cfg: UnevenTPConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        self.head_dim = head_dim or (hidden_size // num_heads)

        self.uneven_cfg = uneven_cfg or UnevenTPConfig.qwen3_default()
        tp_rank = get_tensor_model_parallel_rank()

        # Compute per‑rank head allocation. Use GQA‑aware split so that
        # local_num_heads % local_num_kv_heads == 0 (required by Attention).
        q_sizes, q_offsets, kv_sizes, kv_offsets = _compute_uneven_gqa_splits(
            self.total_num_heads,
            self.total_num_kv_heads,
            self.uneven_cfg.ratios,
        )

        self.local_num_heads = q_sizes[tp_rank]
        self.local_q_offset = q_offsets[tp_rank]
        self.local_num_kv_heads = kv_sizes[tp_rank]
        self.local_kv_offset = kv_offsets[tp_rank]

        self.local_q_size = self.local_num_heads * self.head_dim
        self.local_kv_size = self.local_num_kv_heads * self.head_dim

        # Local QKV projection: outputs only this rank's heads.
        # Use nn.Linear (not UnevenColumnLinear) since we already have the
        # per-rank local size; UnevenColumnLinear would split again.
        self.qkv_proj = nn.Linear(
            hidden_size,
            self.local_q_size + 2 * self.local_kv_size,
            bias=False,
        )

        # QK norm (required by Qwen3; applied per-head before attention)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # Output projection: row‑parallel, uneven on input dim.
        # Must use GQA-aware head split so boundaries align with full_heads.
        head_dim_sizes = [q_sizes[i] * self.head_dim for i in range(len(q_sizes))]
        self.o_proj = UnevenRowLinear(
            in_features=self.total_num_heads * self.head_dim,
            out_features=hidden_size,
            uneven_cfg=self.uneven_cfg,
            bias=False,
            sizes=head_dim_sizes,
        )

        # RoPE (required by Qwen3; applied before attention)
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters or {"rope_type": "default"},
            dual_chunk_attention_config=dual_chunk_attention_config,
        )

        # Reuse the existing Attention kernel on per‑rank heads.
        self.attn = Attention(
            num_heads=self.local_num_heads,
            head_size=self.head_dim,
            scale=self.head_dim**-0.5,
            num_kv_heads=self.local_num_kv_heads,
            cache_config=cache_config,
            prefix=prefix,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        """Delegate to the underlying Attention layer."""
        return self.attn.get_attn_backend()

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        """Delegate to the underlying Attention layer."""
        return self.attn.get_kv_cache_spec(vllm_config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states: [B, S, H] or [num_tokens, H] when flattened (2D)
        ndim = hidden_states.dim()
        if ndim == 2:
            # [num_tokens, hidden] -> keep 2D, use [B, total_heads * head_dim]
            bsz = hidden_states.size(0)
            seqlen = 1  # logical seqlen for allocation
        else:
            bsz, seqlen = hidden_states.size(0), hidden_states.size(1)

        qkv_local = self.qkv_proj(hidden_states)
        # [B, S, local_q + 2*local_kv]
        q_local, k_local, v_local = torch.split(
            qkv_local,
            [self.local_q_size, self.local_kv_size, self.local_kv_size],
            dim=-1,
        )

        # Apply QK norm (required by Qwen3)
        q_by_head = q_local.view(
            *q_local.shape[:-1], self.local_num_heads, self.head_dim
        )
        q_by_head = self.q_norm(q_by_head)
        q_local = q_by_head.reshape(q_local.shape)

        k_by_head = k_local.view(
            *k_local.shape[:-1], self.local_num_kv_heads, self.head_dim
        )
        k_by_head = self.k_norm(k_by_head)
        k_local = k_by_head.reshape(k_local.shape)

        q_local, k_local = self.rotary_emb(positions, q_local, k_local)
        attn_out_local = self.attn(q_local, k_local, v_local)
        # Shape: [B, S, local_num_heads * head_dim] or [B, local_size] when 2D

        # We now have local heads; concatenate all heads logically by
        # padding to the global head dimension and using UnevenRowLinear.
        start = self.local_q_offset * self.head_dim
        end = start + self.local_q_size
        device = attn_out_local.device
        head_dim_size = self.total_num_heads * self.head_dim

        if ndim == 2:
            # Keep 2D: full_heads [B, head_dim_size], output [B, hidden_size]
            full_heads = torch.zeros(
                bsz,
                head_dim_size,
                device=device,
                dtype=attn_out_local.dtype,
            )
            full_heads[:, start:end] = attn_out_local
        else:
            full_heads = torch.zeros(
                bsz,
                seqlen,
                head_dim_size,
                device=device,
                dtype=attn_out_local.dtype,
            )
            full_heads[..., start:end] = attn_out_local

        output = self.o_proj(full_heads)
        # [B, S, hidden_size] or [B, hidden_size] when 2D
        return output


class Qwen3UnevenTPDecoderLayer(nn.Module):
    """Decoder layer using Qwen3UnevenTPAttention.

    MLP / MoE components are reused from the standard Qwen3Moe implementation.
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        self.uneven_cfg = UnevenTPConfig.from_model_config(config)
        set_default_rope_theta(config, default_theta=1000000)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )

        self.self_attn = Qwen3UnevenTPAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn_uneven_tp",
            rms_norm_eps=config.rms_norm_eps,
            rope_parameters=getattr(config, "rope_parameters", None),
            max_position_embeddings=max_position_embeddings,
            dual_chunk_attention_config=dual_chunk_attention_config,
            uneven_cfg=self.uneven_cfg,
        )

        # `mlp_only_layers` in the config.
        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        num_experts = getattr(config, "num_experts", 0)
        decoder_sparse_step = getattr(config, "decoder_sparse_step", 1)
        if (layer_idx not in mlp_only_layers) and (
            num_experts > 0 and (layer_idx + 1) % decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeSparseMoeBlock(
                vllm_config=vllm_config, prefix=f"{prefix}.mlp"
            )
        else:
            self.mlp = Qwen3UnevenTPMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                 uneven_cfg=self.uneven_cfg,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3UnevenTPModel(nn.Module):
    """Qwen3MoE‑style model whose attention layers use uneven TP sharding."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        # Keep for shadow loading using the reference Qwen3MoeModel.
        self._vllm_config = vllm_config
        self._prefix = prefix

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        self.quant_config = quant_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen3UnevenTPDecoderLayer(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            if layer_idx in self.aux_hidden_state_layers:
                aux_hidden_state = (
                    hidden_states + residual if residual is not None else hidden_states
                )
                aux_hidden_states.append(aux_hidden_state)
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """Reuse the reference Qwen3MoeModel logic to compute expert mapping."""
        shadow = Qwen3MoeModel(
            vllm_config=self._vllm_config, prefix=self._prefix
        )
        return shadow.get_expert_mapping()

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        """Load HF-style Qwen3 weights into the uneven-TP model.

        Strategy:
          1) Construct a reference Qwen3MoeModel and delegate the full
             weight-loading logic to it (including MoE, MLP, norms,
             embeddings, etc.).
          2) Copy all non-attention parameters from the reference model
             into this uneven-TP model (shapes must match).
          3) For attention projections (Q/K/V/O), bypass the reference
             model and instead shard the HF weights explicitly into the
             UnevenColumnLinear / UnevenRowLinear weights.

        Expected attention weight names (per layer):
          - model.layers.{i}.self_attn.q_proj.weight
          - model.layers.{i}.self_attn.k_proj.weight
          - model.layers.{i}.self_attn.v_proj.weight
          - model.layers.{i}.self_attn.o_proj.weight

        """

        # Materialize list so we can iterate multiple times.
        weight_items = list(weights)
        loaded_params: set[str] = set()

        # 1) Delegate to a reference Qwen3MoeModel for all "standard"
        #    loading logic (MLP, MoE, norms, embeddings, etc.).
        #    HF checkpoint uses "model.xxx" but the shadow (standalone) has
        #    params "xxx" (no model prefix). Strip "model." when loading.
        def _name_for_shadow(name: str) -> str:
            return name[6:] if name.startswith("model.") else name

        weight_items_for_shadow = [
            (_name_for_shadow(name), w) for name, w in weight_items
        ]
        shadow = Qwen3MoeModel(
            vllm_config=self._vllm_config, prefix=self._prefix
        )
        shadow.load_weights(weight_items_for_shadow)

        # 2) Copy non-attention parameters from the shadow model into
        #    this uneven-TP model whenever shapes match.
        #    Skip .self_attn. (handled explicitly). MLP layers with
        #    Qwen3UnevenTPMLP will have shape mismatch and be skipped;
        #    we load those explicitly below.
        self_params = dict(self.named_parameters())
        for name, param in shadow.named_parameters():
            if ".self_attn." in name:
                continue
            if name not in self_params:
                continue
            dst = self_params[name]
            if dst.shape != param.shape:
                continue
            dst.data.copy_(param.data)
            full_name = f"{self._prefix}.{name}" if self._prefix else name
            loaded_params.add(full_name)

        # 3) Handle attention Q/K/V/O projections explicitly using the
        #    uneven sharding rules.

        def _parse_layer_idx(name: str) -> int | None:
            # Example: "model.layers.3.self_attn.q_proj.weight"
            marker = "layers."
            try:
                base, rest = name.split(marker, 1)
                idx_str, _ = rest.split(".", 1)
                return int(idx_str)
            except Exception:
                return None

        for name, loaded_weight in weight_items:
            if ".self_attn." not in name:
                continue

            layer_idx = _parse_layer_idx(name)
            if layer_idx is None or layer_idx >= len(self.layers):
                continue

            layer = self.layers[layer_idx]
            if isinstance(layer, PPMissingLayer):
                continue

            # We expect Qwen3UnevenTPDecoderLayer here.
            attn: Qwen3UnevenTPAttention = layer.self_attn  # type: ignore[attr-defined]

            qkv_param_name = name.replace(
                ".q_proj.weight", ".qkv_proj.weight"
            ).replace(".k_proj.weight", ".qkv_proj.weight").replace(
                ".v_proj.weight", ".qkv_proj.weight"
            )

            if name.endswith(".self_attn.qkv_proj.weight"):
                # Merged QKV from packed_modules_mapping: split and load.
                q_size = attn.total_num_heads * attn.head_dim
                k_size = attn.total_num_kv_heads * attn.head_dim
                q_part = loaded_weight[:q_size]
                k_part = loaded_weight[q_size : q_size + k_size]
                v_part = loaded_weight[q_size + k_size : q_size + 2 * k_size]
                _load_uneven_qkv_weight(attn, q_part, kind="q")
                _load_uneven_qkv_weight(attn, k_part, kind="k")
                _load_uneven_qkv_weight(attn, v_part, kind="v")
                loaded_params.add(name)
            elif name.endswith(".self_attn.q_proj.weight"):
                _load_uneven_qkv_weight(attn, loaded_weight, kind="q")
                loaded_params.add(qkv_param_name)
            elif name.endswith(".self_attn.k_proj.weight"):
                _load_uneven_qkv_weight(attn, loaded_weight, kind="k")
                loaded_params.add(qkv_param_name)
            elif name.endswith(".self_attn.v_proj.weight"):
                _load_uneven_qkv_weight(attn, loaded_weight, kind="v")
                loaded_params.add(qkv_param_name)
            elif name.endswith(".self_attn.o_proj.weight"):
                _load_uneven_o_proj_weight(attn, loaded_weight)
                loaded_params.add(name)
            elif name.endswith(".self_attn.q_norm.weight"):
                attn.q_norm.weight.data.copy_(
                    loaded_weight.to(attn.q_norm.weight.device)
                )
                loaded_params.add(name)
            elif name.endswith(".self_attn.k_norm.weight"):
                attn.k_norm.weight.data.copy_(
                    loaded_weight.to(attn.k_norm.weight.device)
                )
                loaded_params.add(name)

        # 4) Handle MLP layers with Qwen3UnevenTPMLP explicitly.
        mlp_weights: dict[int, dict[str, torch.Tensor]] = {}
        for name, loaded_weight in weight_items:
            if ".mlp." not in name or not name.endswith(".weight"):
                continue
            layer_idx = _parse_layer_idx(name)
            if layer_idx is None or layer_idx >= len(self.layers):
                continue
            layer = self.layers[layer_idx]
            if isinstance(layer, PPMissingLayer):
                continue
            if not isinstance(layer.mlp, Qwen3UnevenTPMLP):
                continue
            if layer_idx not in mlp_weights:
                mlp_weights[layer_idx] = {}
            if "gate_proj" in name:
                mlp_weights[layer_idx]["gate_proj"] = loaded_weight
            elif "up_proj" in name:
                mlp_weights[layer_idx]["up_proj"] = loaded_weight
            elif "gate_up_proj" in name:
                mlp_weights[layer_idx]["gate_up_proj"] = loaded_weight
            elif "down_proj" in name:
                mlp_weights[layer_idx]["down_proj"] = loaded_weight

        for layer_idx, wdict in mlp_weights.items():
            if "down_proj" not in wdict:
                continue
            layer = self.layers[layer_idx]
            assert isinstance(layer.mlp, Qwen3UnevenTPMLP)
            mlp = layer.mlp
            if "gate_up_proj" in wdict:
                gate_up = wdict["gate_up_proj"]
            elif "gate_proj" in wdict and "up_proj" in wdict:
                gate_up = torch.cat(
                    [wdict["gate_proj"], wdict["up_proj"]], dim=0
                )
            else:
                continue
            _load_uneven_gate_up_proj_weight(mlp, gate_up)
            _load_uneven_down_proj_weight(mlp, wdict["down_proj"])
            loaded_params.add(
                f"model.layers.{layer_idx}.mlp.gate_up_proj.weight"
            )
            if "gate_proj" in wdict:
                loaded_params.add(
                    f"model.layers.{layer_idx}.mlp.gate_proj.weight"
                )
            if "up_proj" in wdict:
                loaded_params.add(
                    f"model.layers.{layer_idx}.mlp.up_proj.weight"
                )
            loaded_params.add(
                f"model.layers.{layer_idx}.mlp.down_proj.weight"
            )

        return loaded_params


def _load_uneven_qkv_weight(
    attn: Qwen3UnevenTPAttention,
    loaded_weight: torch.Tensor,
    *,
    kind: str,
) -> None:
    """Shard HF Q/K/V projection weights into an UnevenColumnLinear.

    HF weights are expected to have shapes:
      - q_proj.weight: [total_num_heads * head_dim, hidden_size]
      - k_proj.weight: [total_num_kv_heads * head_dim, hidden_size]
      - v_proj.weight: [total_num_kv_heads * head_dim, hidden_size]

    For each rank r (0 or 1), we compute the local head range using
    the same uneven ratios as the runtime attention, then slice the
    corresponding rows from the global weight and copy them into the
    appropriate block in attn.qkv_proj.weight:

      [ Q_rows | K_rows | V_rows ]
      [  0:q   | q:q+k  | q+k:q+2k ]
    """
    assert kind in ("q", "k", "v")

    device = attn.qkv_proj.weight.device
    loaded_weight = loaded_weight.to(device=device, dtype=attn.qkv_proj.weight.dtype)
    head_dim = attn.head_dim

    tp_rank = get_tensor_model_parallel_rank()

    # Use the same GQA-aware split as the attention layer for consistency.
    q_sizes, q_offsets, kv_sizes, kv_offsets = _compute_uneven_gqa_splits(
        attn.total_num_heads,
        attn.total_num_kv_heads,
        attn.uneven_cfg.ratios,
    )

    if kind == "q":
        local_heads = q_sizes[tp_rank]
        local_offset = q_offsets[tp_rank]
        total_heads = attn.total_num_heads

        src_start = local_offset * head_dim
        src_end = src_start + local_heads * head_dim

        dst_start = 0
        dst_end = dst_start + local_heads * head_dim
    else:
        local_heads = kv_sizes[tp_rank]
        local_offset = kv_offsets[tp_rank]
        total_heads = attn.total_num_kv_heads

        src_start = local_offset * head_dim
        src_end = src_start + local_heads * head_dim

        if kind == "k":
            dst_start = attn.local_q_size
        else:  # "v"
            dst_start = attn.local_q_size + attn.local_kv_size
        dst_end = dst_start + local_heads * head_dim

    # Sanity checks.
    assert loaded_weight.shape[0] == total_heads * head_dim, (
        loaded_weight.shape,
        total_heads,
        head_dim,
    )
    assert loaded_weight.shape[1] == attn.hidden_size

    qkv_weight = attn.qkv_proj.weight
    if qkv_weight.shape[0] < dst_end:
        raise ValueError(
            f"qkv_proj weight shape mismatch: need {dst_end} rows for {kind} "
            f"but have {qkv_weight.shape[0]}. "
            f"tp_rank={tp_rank}, dst_start={dst_start}, local_heads={local_heads}, "
            f"head_dim={head_dim}, loaded_weight.shape={loaded_weight.shape}"
        )
    if qkv_weight.shape[1] != loaded_weight.shape[1]:
        raise ValueError(
            f"qkv_proj hidden_size mismatch: {qkv_weight.shape[1]} vs "
            f"{loaded_weight.shape[1]}"
        )

    qkv_weight.data[dst_start:dst_end, :] = loaded_weight[src_start:src_end, :]


def _load_uneven_o_proj_weight(
    attn: Qwen3UnevenTPAttention,
    loaded_weight: torch.Tensor,
) -> None:
    """Shard HF O projection weight into an UnevenRowLinear.

    HF weight shape:
      - o_proj.weight: [hidden_size, hidden_size]

    We shard along the input (column) dimension using the same GQA-aware
    head split as the o_proj forward, so boundaries align with full_heads.
    """
    device = attn.o_proj.weight.device
    loaded_weight = loaded_weight.to(device=device, dtype=attn.o_proj.weight.dtype)

    hidden_size = attn.hidden_size

    q_sizes, q_offsets, _, _ = _compute_uneven_gqa_splits(
        attn.total_num_heads,
        attn.total_num_kv_heads,
        attn.uneven_cfg.ratios,
    )
    head_dim_sizes = [q_sizes[i] * attn.head_dim for i in range(len(q_sizes))]
    tp_rank = get_tensor_model_parallel_rank()
    local_in = head_dim_sizes[tp_rank]
    local_offset = sum(head_dim_sizes[:tp_rank])

    assert loaded_weight.shape == (hidden_size, hidden_size)

    src = loaded_weight[:, local_offset : local_offset + local_in]
    dst = attn.o_proj.weight
    assert dst.shape == src.shape
    dst.data.copy_(src)


def _load_uneven_gate_up_proj_weight(
    mlp: "Qwen3UnevenTPMLP",
    loaded_weight: torch.Tensor,
) -> None:
    """Shard HF gate_up_proj weight into UnevenMergedColumnLinear.

    HF weight shape: [2*intermediate_size, hidden_size]
    Layout: gate rows [0:intermediate_size], up rows [intermediate_size:2*intermediate_size]
    Each part is split by ratios; rank i gets gate_i and up_i concatenated.
    """
    device = mlp.gate_up_proj.weight.device
    loaded_weight = loaded_weight.to(
        device=device, dtype=mlp.gate_up_proj.weight.dtype
    )
    intermediate_size = mlp.intermediate_size
    hidden_size = mlp.hidden_size
    assert loaded_weight.shape == (2 * intermediate_size, hidden_size)

    sizes, offsets = _compute_uneven_splits(
        intermediate_size, mlp.uneven_cfg.ratios
    )
    tp_rank = get_tensor_model_parallel_rank()
    local_sz = sizes[tp_rank]
    local_off = offsets[tp_rank]

    gate_part = loaded_weight[local_off : local_off + local_sz]
    up_part = loaded_weight[
        intermediate_size + local_off : intermediate_size + local_off + local_sz
    ]
    local_weight = torch.cat([gate_part, up_part], dim=0)
    mlp.gate_up_proj.weight.data.copy_(local_weight)


def _load_uneven_down_proj_weight(
    mlp: "Qwen3UnevenTPMLP",
    loaded_weight: torch.Tensor,
) -> None:
    """Shard HF down_proj weight into UnevenRowLinear.

    HF weight shape: [hidden_size, intermediate_size]
    We shard along the input (column) dimension.
    """
    device = mlp.down_proj.weight.device
    loaded_weight = loaded_weight.to(
        device=device, dtype=mlp.down_proj.weight.dtype
    )
    intermediate_size = mlp.intermediate_size
    hidden_size = mlp.hidden_size
    assert loaded_weight.shape == (hidden_size, intermediate_size)

    sizes, offsets = _compute_uneven_splits(
        intermediate_size, mlp.uneven_cfg.ratios
    )
    tp_rank = get_tensor_model_parallel_rank()
    local_sz = sizes[tp_rank]
    local_off = offsets[tp_rank]

    src = loaded_weight[:, local_off : local_off + local_sz]
    dst = mlp.down_proj.weight
    assert dst.shape == src.shape
    dst.data.copy_(src)


class Qwen3UnevenTPForCausalLM(
    nn.Module, SupportsPP, SupportsLoRA, SupportsEagle3, MixtureOfExperts
):
    """Causal LM wrapper around Qwen3UnevenTPModel."""

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ]
    }

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        # Only perform the following mapping when Qwen3MoeMLP exists
        if getattr(config, "mlp_only_layers", []):
            self.packed_modules_mapping["gate_up_proj"] = ["gate_proj", "up_proj"]

        self.model = Qwen3UnevenTPModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters (mirrors Qwen3MoeForCausalLM).
        self.expert_weights = []

        self.moe_layers = []
        example_layer = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, Qwen3UnevenTPDecoderLayer)
            if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_layer is None:
            # Dense model (no MoE layers): use dummy MoE metadata.
            self.num_moe_layers = 0
            self.num_expert_groups = 1
            self.num_shared_experts = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_redundant_experts = 0
        else:
            self.num_moe_layers = len(self.moe_layers)
            self.num_expert_groups = 1
            self.num_shared_experts = 0
            self.num_logical_experts = example_layer.n_logical_experts
            self.num_physical_experts = example_layer.n_physical_experts
            self.num_local_physical_experts = example_layer.n_local_physical_experts
            self.num_routed_experts = example_layer.n_routed_experts
            self.num_redundant_experts = example_layer.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        if self.num_moe_layers == 0:
            return
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights into both the backbone and lm_head."""
        weight_items = list(weights)
        loaded = self.model.load_weights(weight_items)

        # Load lm_head if it is not tied to embeddings or if a separate
        # lm_head weight is provided. Use weight_loader for TP sharding.
        for name, w in weight_items:
            if not name.endswith("lm_head.weight"):
                continue
            if self.config.tie_word_embeddings:
                # Already loaded via embed_tokens
                loaded.add("lm_head.weight")
                break
            param = self.lm_head.weight
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, w)
            loaded.add("lm_head.weight")
            break
        else:
            # When tie_word_embeddings, lm_head shares embed_tokens; already loaded.
            if self.config.tie_word_embeddings:
                loaded.add("lm_head.weight")

        return loaded

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

