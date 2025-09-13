#!/bin/bash

# vLLM 안정화 스크립트
# Qwen2.5-VL 모델 호환성을 위한 안정적인 버전 설치

set -e

echo "🔄 vLLM 안정화 작업 시작..."

# 가상환경 활성화
source venv/bin/activate

# 현재 vLLM 버전 확인
echo "📋 현재 vLLM 버전:"
pip show vllm | grep Version || echo "vLLM 미설치"

# Qwen2.5-VL 호환 버전으로 설치
echo "⬇️ vLLM 안정 버전 설치 (Qwen2.5-VL 호환)..."

# 기존 vLLM 제거
pip uninstall -y vllm || true

# 안정적인 버전 설치 (0.6.0은 Qwen2.5-VL과 호환성이 좋음)
pip install vllm==0.6.0

# 의존성 확인
pip install --upgrade transformers accelerate

echo "✅ vLLM 0.6.0 설치 완료"
pip show vllm | grep Version

echo "🔄 서버 재시작이 필요합니다."
echo "GPU 서버에서 다음 명령 실행:"
echo "  ./start_vllm.sh stop"
echo "  ./start_vllm.sh start-bg"
