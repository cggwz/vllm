#!/bin/bash
# 1. 设置只编译当前显卡架构 (这是最关键的一步，请根据你的显卡修改数值)
export TORCH_CUDA_ARCH_LIST="8.6" 

# 2. 设置最大编译线程数
export MAX_JOBS=$(nproc)

# 3. 确保使用 Ninja
# pip install ninja

# 4. 执行增量安装 (不要带 --no-build-isolation，除非依赖有问题)
# -v 可以让你看到详细的编译进度，确定增量编译是否生效
pip install -e  . --no-build-isolation