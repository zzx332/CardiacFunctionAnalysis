#!/bin/bash
# setup_env.sh — 一键创建 CardiacAI conda 环境
# 用法：bash setup_env.sh [env_name]
# 默认环境名：cardiacai
set -e

ENV_NAME="${1:-cardiacai}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 清华镜像加速 ────────────────────────────────────────────
# pip：让 environment.yml 中的 pip 依赖走清华源
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# conda：临时使用清华 channel（不修改全局 .condarc）
export CONDA_CHANNEL_PRIORITY=flexible

echo "=========================================="
echo " CardiacAI 环境安装"
echo " 环境名：$ENV_NAME"
echo "=========================================="

# 1. 检测 CUDA 版本，自动选择 pytorch-cuda
if command -v nvidia-smi &> /dev/null; then
    CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "检测到 GPU，驱动版本：$CUDA_VER"
    # 根据驱动版本选择 pytorch-cuda
    DRIVER_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    if [ "$DRIVER_MAJOR" -ge 525 ]; then
        PYTORCH_CUDA="12.1"
    else
        PYTORCH_CUDA="11.8"
    fi
    echo "将使用 pytorch-cuda=$PYTORCH_CUDA"
else
    echo "未检测到 GPU，将安装 CPU 版本"
    PYTORCH_CUDA="cpu"
fi

# 2. 创建或更新 conda 环境
if conda env list | grep -q "^$ENV_NAME "; then
    echo "环境 $ENV_NAME 已存在，执行更新..."
    conda env update -n "$ENV_NAME" -f "$SCRIPT_DIR/environment.yml" --prune
else
    echo "创建新环境 $ENV_NAME ..."
    # 动态替换 pytorch-cuda 版本
    sed "s/pytorch-cuda=11\.8/pytorch-cuda=$PYTORCH_CUDA/g" \
        "$SCRIPT_DIR/environment.yml" > /tmp/cardiacai_env.yml
    conda env create -n "$ENV_NAME" -f /tmp/cardiacai_env.yml
fi

echo ""
echo "=========================================="
echo " 安装完成！激活环境："
echo "   conda activate $ENV_NAME"
echo " 运行项目（在 CardiacAI/ 目录下）："
echo "   python main.py --task landmark ..."
echo "=========================================="
