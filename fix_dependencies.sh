#!/bin/bash

# vLLM 의존성 충돌 해결 스크립트
echo "🔧 vLLM 의존성 충돌 해결"
echo "========================"

cd ~/vllm_server

# 기존 패키지 정리
echo "🧹 기존 패키지 정리..."
pip uninstall -y torch torchvision vllm transformers 2>/dev/null || true

# 캐시 정리
echo "🗑️ pip 캐시 정리..."
pip cache purge

# 단계별 재설치
echo "📦 단계별 재설치..."

echo "  1️⃣ PyTorch 설치 (CUDA 118)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "  2️⃣ Transformers 설치..."
pip install transformers

echo "  3️⃣ 기본 의존성 설치..."
pip install fastapi uvicorn Pillow python-dotenv requests

echo "  4️⃣ Ray 설치..."
pip install "ray[default]"

echo "  5️⃣ vLLM 설치..."
pip install vllm

echo "✅ 설치 완료!"

# 확인
echo "🔍 설치 확인..."
python -c "
import torch
import vllm
import transformers
print(f'PyTorch: {torch.__version__}')
print(f'vLLM: {vllm.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
