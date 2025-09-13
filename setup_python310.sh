#!/bin/bash

# GPU 서버 Python 3.10 설치 스크립트
echo "🐍 GPU 서버 Python 3.10 환경 설치"
echo "=================================="

# 시스템 업데이트
echo "📦 시스템 패키지 업데이트..."
sudo apt update

# Python 3.10 및 관련 패키지 설치
echo "🐍 Python 3.10 설치..."
sudo apt install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    python3.10-distutils

# pip 설치 확인
echo "📋 pip 설치 확인..."
if ! python3.10 -m pip --version >/dev/null 2>&1; then
    echo "📥 pip 설치 중..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
fi

# 기본 Python 심볼릭 링크 (선택적)
echo "🔗 Python 심볼릭 링크 설정..."
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# 시스템 개발 도구 설치
echo "🔧 개발 도구 설치..."
sudo apt install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    unzip \
    htop \
    tmux \
    screen

# CUDA 도구 확인
echo "🔥 CUDA 환경 확인..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "⚠️ nvidia-smi를 찾을 수 없습니다. CUDA 드라이버를 확인하세요."
fi

# /data 디렉토리 권한 설정
echo "📁 /data 디렉토리 권한 설정..."
sudo mkdir -p /data/huggingface_models
sudo chown -R $USER:$USER /data/huggingface_models
sudo chmod -R 755 /data/huggingface_models

echo "✅ Python 3.10 환경 설치 완료!"
echo ""
echo "다음 단계:"
echo "1. cd ~/vllm_server"
echo "2. ./start_vllm.sh"
