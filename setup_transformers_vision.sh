#!/bin/bash

# Transformers 기반 Vision 서버 설정
# vLLM 호환성 문제 시 대안

set -e

echo "🔄 Transformers 기반 Vision 서버 설정..."

# 가상환경 활성화
source venv/bin/activate

# Transformers 및 Vision 관련 패키지 설치
echo "📦 Vision 처리 패키지 설치..."
pip install --upgrade torch torchvision transformers
pip install --upgrade Pillow opencv-python qwen-vl-utils
pip install --upgrade accelerate bitsandbytes

echo "✅ Vision 패키지 설치 완료"

echo "💡 사용법:"
echo "1. Qwen2.5-VL을 Transformers로 직접 로드"
echo "2. 이미지 처리 최적화"
echo "3. FastAPI 엔드포인트에서 직접 호출"
