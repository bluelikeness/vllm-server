#!/bin/bash

set -euo pipefail

# vLLM Server 시작 스크립트 (foreground/background 지원)
ACTION=${1:-start} # start | start-bg | stop | status | logs | restart
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$BASE_DIR")"
LOG_DIR="$BASE_DIR/logs"
PID_FILE="$BASE_DIR/vllm_server.pid"
TS="$(date +%Y%m%d-%H%M%S)"
TS_LOG="$LOG_DIR/server-$TS.log"
CUR_LOG="$LOG_DIR/server.log"
CUDA_HOME=/usr/local/cuda
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
PATH=/usr/local/cuda/bin:$PATH

echo "🚀 CloudLLM vLLM 서버 관리: $ACTION"

# start/start-bg/restart 에서만 무거운 준비 작업 수행
DO_SETUP=false
if [[ "$ACTION" == "start" || "$ACTION" == "start-bg" || "$ACTION" == "restart" ]]; then
    DO_SETUP=true
fi

if [ "$DO_SETUP" = true ]; then
    # 가상환경 활성화 (필요시)
    echo "🐍 Python 가상환경 확인..."

# Python 버전 확인
PYTHON_CMD=""
for py_cmd in python3.10 python3.11 python3.9 python3 python; do
    if command -v $py_cmd >/dev/null 2>&1; then
        PY_VERSION=$($py_cmd --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
        if [[ "$PY_VERSION" =~ ^3\.(8|9|10|11)$ ]]; then
            PYTHON_CMD=$py_cmd
            echo "  ✅ 호환 가능한 Python 발견: $py_cmd (버전 $PY_VERSION)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "  ❌ 호환 가능한 Python 버전(3.8-3.11)을 찾을 수 없습니다."
    echo "  📥 Python 3.10 설치 명령:"
    echo "    sudo apt update && sudo apt install python3.10 python3.10-venv python3.10-dev"
    exit 1
fi

# 가상환경 생성 또는 활성화
if [ ! -d "venv" ]; then
    echo "  📦 가상환경 생성 중... (Python $PY_VERSION 사용)"
    $PYTHON_CMD -m venv venv
    if [ $? -ne 0 ]; then
        echo "  ❌ 가상환경 생성 실패"
        echo "  🔧 venv 모듈 설치: sudo apt install python3.10-venv"
        exit 1
    fi
    echo "  ✅ 가상환경 생성 완료"
fi

echo "  🔄 가상환경 활성화 중..."
source venv/bin/activate

# 가상환경 내 Python 버전 확인
VENV_PY_VERSION=$(python --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
echo "  📍 가상환경 Python 버전: $VENV_PY_VERSION"

# pip 업그레이드
echo "  🔧 pip 업그레이드 중..."
python -m pip install --upgrade pip setuptools wheel

# 모델 캐시 디렉토리 확인 및 설정
echo "📁 모델 캐시 디렉토리 확인..."

# Hugging Face 캐시 디렉토리 설정 - 데이터 디스크 사용
export HF_HOME="/data/huggingface_models"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"

# LoRA 어댑터 중앙 저장소 설정
export LORA_ADAPTERS_HOME="$HF_HOME/lora_adapters"
export LORA_CONFIG_DIR="$LORA_ADAPTERS_HOME"

# 캐시 디렉토리 생성 및 확인
mkdir -p "$HF_HOME"
mkdir -p "$TRANSFORMERS_CACHE"
mkdir -p "$LORA_ADAPTERS_HOME"  # LoRA 어댑터 디렉토리 생성

# 디렉토리 권한 확인
if [ ! -w "$HF_HOME" ]; then
    echo "  ⚠️ $HF_HOME 디렉토리에 쓰기 권한이 없습니다."
    echo "  🔧 권한 수정을 시도합니다..."
    sudo mkdir -p "$HF_HOME" "$LORA_ADAPTERS_HOME" 2>/dev/null || true
    sudo chown -R $USER:$USER "$HF_HOME" 2>/dev/null || true
    sudo chmod -R 755 "$HF_HOME" 2>/dev/null || true
fi

echo "  📍 HF_HOME: $HF_HOME"
echo "  📍 TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "  📍 LORA_ADAPTERS_HOME: $LORA_ADAPTERS_HOME"
echo "  🔑 디렉토리 권한: $(ls -ld $HF_HOME 2>/dev/null | awk '{print $1}' || echo 'N/A')"

# 캐시 디렉토리 용량 확인
echo "💾 캐시 디렉토리 용량 확인..."
if [ -d "$HF_HOME" ]; then
    CACHE_SIZE=$(du -sh "$HF_HOME" 2>/dev/null | cut -f1 || echo "N/A")
    echo "  📊 현재 캐시 크기: $CACHE_SIZE"
fi

# 디스크 공간 확인 - /data 파티션
echo "💽 디스크 공간 확인..."
df -h /data 2>/dev/null | tail -1 | awk '{print "  📊 /data 사용 가능 공간: " $4 " (사용률: " $5 ")"}' || \
df -h . | tail -1 | awk '{print "  📊 현재 위치 사용 가능 공간: " $4 " (사용률: " $5 ")"}'

# 환경변수 로드 확인
if [ ! -f ".env" ]; then
    echo "❌ .env 파일이 없습니다. 생성해주세요."
    exit 1
fi

# .env 파일 로드
echo "📋 환경변수 로드 중..."
set -a  # 자동으로 변수를 export
source .env
set +a
echo "  ✅ .env 파일 로드 완료"

# 선택적 모드 인자 처리 (예: ./start_vllm.sh start int4)
MODE_ARG="${2:-}"
if [[ -n "$MODE_ARG" ]]; then
    MODE_LC="${MODE_ARG,,}"
    case "$MODE_LC" in
        fp16)
            export RUNTIME_VLLM_LOAD_MODE=fp16
            if [[ -n "${FP16_MODEL_NAME:-}" ]]; then
                export RUNTIME_MODEL_NAME="$FP16_MODEL_NAME"
                echo "  🔧 모드 인자 적용: FP16 (MODEL_NAME→$RUNTIME_MODEL_NAME)"
            else
                echo "  🔧 모드 인자 적용: FP16 (MODEL_NAME 유지)"
            fi
            ;;
        int4|4bit)
            export RUNTIME_VLLM_LOAD_MODE=4bit
            if [[ -n "${INT4_MODEL_NAME:-}" ]]; then
                export RUNTIME_MODEL_NAME="$INT4_MODEL_NAME"
                echo "  🔧 모드 인자 적용: INT4 (MODEL_NAME→$RUNTIME_MODEL_NAME)"
            else
                echo "  ⚠️ INT4 모델명이 .env에 비어 있습니다(INT4_MODEL_NAME). 기본 MODEL_NAME 사용"
            fi
            ;;
        int8|8bit)
            export RUNTIME_VLLM_LOAD_MODE=8bit
            if [[ -n "${INT8_MODEL_NAME:-}" ]]; then
                export RUNTIME_MODEL_NAME="$INT8_MODEL_NAME"
                echo "  🔧 모드 인자 적용: INT8 (MODEL_NAME→$RUNTIME_MODEL_NAME)"
            else
                echo "  🔧 모드 인자 적용: INT8 (MODEL_NAME 유지)"
            fi
            ;;
        bf16|bfloat16)
            export RUNTIME_VLLM_LOAD_MODE=bf16
            if [[ -n "${BF16_MODEL_NAME:-}" ]]; then
                export RUNTIME_MODEL_NAME="$BF16_MODEL_NAME"
                echo "  🔧 모드 인자 적용: BF16 (MODEL_NAME→$RUNTIME_MODEL_NAME)"
            else
                echo "  🔧 모드 인자 적용: BF16 (MODEL_NAME 유지)"
            fi
            ;;
        *)
            echo "  ⚠️ 알 수 없는 모드 인자: $MODE_ARG (무시)"
            ;;
    esac
fi

# 런타임 오버라이드 적용 (start_vllm_with_options.sh 등에서 전달)
if [[ -n "${RUNTIME_VLLM_LOAD_MODE:-}" ]]; then
    export VLLM_LOAD_MODE="$RUNTIME_VLLM_LOAD_MODE"
    echo "  🔧 런타임 모드 오버라이드 적용: VLLM_LOAD_MODE=$VLLM_LOAD_MODE"
fi
if [[ -n "${RUNTIME_MODEL_NAME:-}" ]]; then
    export MODEL_NAME="$RUNTIME_MODEL_NAME"
    echo "  🔧 런타임 모델 오버라이드 적용: MODEL_NAME=$MODEL_NAME"
fi

# 모델명 확인 (환경변수에서)
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-32B-Instruct}"
echo "🤖 로드할 모델: $MODEL_NAME"

# GPU 메모리 확인
echo "🔍 GPU 상태 확인..."
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  📊 GPU 메모리 상태:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
    while IFS=, read -r idx name mem_used mem_total util; do
        mem_used=$(echo $mem_used | xargs)
        mem_total=$(echo $mem_total | xargs)
        util=$(echo $util | xargs)
        mem_percent=$((mem_used * 100 / mem_total))
        echo "    GPU$idx ($name): ${mem_used}MB/${mem_total}MB (${mem_percent}%) - 사용률: ${util}%"

        # 메모리 부족 경고
        if [ $mem_percent -gt 90 ]; then
            echo "    ⚠️ GPU$idx 메모리 사용률이 높습니다!"
        fi
    done

    # GPU 프로세스 확인
    echo "  🔄 실행 중인 GPU 프로세스:"
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader,nounits 2>/dev/null | \
    while IFS=, read -r pid name mem; do
        if [ -n "$pid" ]; then
            echo "    PID: $pid, 프로세스: $name, 메모리: ${mem}MB"
        fi
    done || echo "    (실행 중인 GPU 프로세스 없음)"
else
    echo "  ❌ nvidia-smi를 찾을 수 없습니다. CUDA가 설치되어 있는지 확인하세요."
    exit 1
fi

# vLLM 의존성 설치 확인
echo "📚 vLLM 설치 확인..."

# 필수 패키지 설치 확인 및 설치
echo "  �� 설치된 패키지 확인 중..."
NEED_INSTALL=false

# 기본 패키지들 확인
for pkg in torch transformers fastapi uvicorn; do
    if ! python -c "import ${pkg//-/_}" 2>/dev/null; then
        echo "  ❌ $pkg 누락"
        NEED_INSTALL=true
    else
        echo "  ✅ $pkg 설치됨"
    fi
done

# Pillow는 PIL로 import
if ! python -c "import PIL" 2>/dev/null; then
    echo "  ❌ Pillow 누락"
    NEED_INSTALL=true
else
    echo "  ✅ Pillow 설치됨"
fi

# vLLM 확인
if ! python -c "import vllm" 2>/dev/null; then
    echo "  ❌ vLLM 누락"
    NEED_INSTALL=true
else
    echo "  ✅ vLLM 설치됨"
fi

if [ "$NEED_INSTALL" = true ]; then
    echo "  📥 누락된 패키지 설치 중..."

    # 단계별 설치로 의존성 충돌 방지
    echo "    1️⃣ PyTorch 설치..."
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    echo "    2️⃣ Transformers 및 기본 라이브러리 설치..."
    python -m pip install transformers accelerate

    echo "    3️⃣ 웹 프레임워크 설치..."
    python -m pip install fastapi uvicorn[standard] python-multipart aiofiles

    echo "    4️⃣ 유틸리티 라이브러리 설치..."
    python -m pip install Pillow opencv-python numpy scipy
    python -m pip install python-dotenv PyPDF2 python-docx openpyxl requests

    echo "    5️⃣ Ray 설치..."
    python -m pip install ray[default]

    echo "    6️⃣ vLLM 설치... (시간이 걸릴 수 있습니다)"
    python -m pip install vllm

    echo "  ✅ 패키지 설치 완료"
else
    echo "  ✅ 모든 필수 패키지가 이미 설치되어 있습니다"
fi

# 4bit 모드일 경우 bitsandbytes 확인 및 설치
if [ "${VLLM_LOAD_MODE,,}" = "4bit" ] || [ "${VLLM_LOAD_MODE,,}" = "int4" ]; then
    echo "🔍 4bit 모드 감지: bitsandbytes 확인 중..."
    if ! python -c "import bitsandbytes" 2>/dev/null; then
        echo "  ❌ bitsandbytes 미설치 -> 설치 진행(b>=0.46.1)"
    if ! python -m pip install --upgrade 'bitsandbytes>=0.46.1'; then
            echo "  ❌ bitsandbytes 설치 실패. FP16 모드로 폴백합니다."
            export VLLM_LOAD_MODE=fp16
            export VLLM_QUANTIZATION_METHOD=
        else
            echo "  ✅ bitsandbytes 설치 완료"
    fi
    else
        echo "  ✅ bitsandbytes 설치됨"
    fi
fi

# vLLM 설치 최종 확인
python -c "import vllm; print(f'✅ vLLM 버전: {vllm.__version__}')" 2>/dev/null || {
    echo "❌ vLLM 설치 실패"
    echo "수동 설치 명령:"
    echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    echo "  pip install vllm transformers"
    exit 1
}

# CUDA 설치 확인
echo "🔥 CUDA 호환성 확인..."
python -c "
import torch
print(f'  📍 PyTorch 버전: {torch.__version__}')
print(f'  🔥 CUDA 사용 가능: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  🖥️ GPU 개수: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('  ⚠️ CUDA를 사용할 수 없습니다.')
"

# 모델 파일 존재 확인
echo "🔍 모델 파일 확인..."

# MODEL_NAME이 절대 경로인지 확인
if [[ "$MODEL_NAME" == /* ]]; then
    # 절대 경로인 경우 그대로 사용
    MODEL_PATH="$MODEL_NAME"
    echo "  📍 모델 경로 (직접 지정): $MODEL_PATH"
else
    # 모델명인 경우 캐시 경로로 변환
    # Hugging Face 캐시 경로 형식: models--org--model-name (슬래시는 --로 변환)
    CACHE_MODEL_NAME=$(echo $MODEL_NAME | sed 's#/#--#g')
    MODEL_PATH="$HF_HOME/models--$CACHE_MODEL_NAME"
    echo "  📍 모델 경로 (캐시 계산): $MODEL_PATH"
    echo "  🔍 변환 과정: $MODEL_NAME → models--$CACHE_MODEL_NAME"
fi

if [ -d "$MODEL_PATH" ]; then
    echo "  ✅ 모델이 존재합니다: $MODEL_PATH"
    MODEL_SIZE=$(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1 || echo "N/A")
    echo "  📊 모델 크기: $MODEL_SIZE"

    # 모델 파일들 확인
    echo "  📂 모델 파일 목록:"
    ls -la "$MODEL_PATH/snapshots" 2>/dev/null | head -5 || \
    ls -la "$MODEL_PATH" 2>/dev/null | head -5 || echo "    (모델 파일을 찾을 수 없음)"
else
    echo "  ⚠️ 모델이 없습니다. 첫 실행 시 다운로드됩니다."
    echo "  📥 예상 다운로드 크기: ~64GB (Qwen2.5-VL-32B)"
    echo "  ⏱️ 예상 다운로드 시간: 10-30분 (네트워크 속도에 따라)"
    echo "  📍 다운로드 위치: $MODEL_PATH"

    # /data 디스크 공간 충분한지 확인
    AVAILABLE_GB=$(df /data 2>/dev/null | tail -1 | awk '{print int($4/1024/1024)}' || echo "0")
    if [ "$AVAILABLE_GB" -lt 70 ]; then
        echo "  ❌ /data 디스크 공간이 부족합니다 (필요: 70GB, 사용가능: ${AVAILABLE_GB}GB)"
        exit 1
    fi

    # 사용자 확인
    echo -n "  ❓ 계속 진행하시겠습니까? (y/N): "
    read -r confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "❌ 사용자가 취소했습니다."
    exit 1
    fi
fi

fi # DO_SETUP

# 실행/제어 함수들
start_foreground() {
    echo "🔥 vLLM 서버(포그라운드) 시작 중..."
    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}${PARENT_DIR}"
    python -m vllm_server.server
}

start_background() {
    echo "🔥 vLLM 서버(백그라운드) 시작 중..."
    mkdir -p "$LOG_DIR"
    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}${PARENT_DIR}"
    # 기존 PID 처리
    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE" || true)
        if [ -n "${OLD_PID}" ] && ps -p "$OLD_PID" > /dev/null 2>&1; then
            echo "  ⚠️ 기존 서버(PID $OLD_PID)가 실행 중입니다. 먼저 stop 하세요."
            exit 1
        else
            rm -f "$PID_FILE"
        fi
    fi
    # nohup으로 데몬화
    nohup python -m vllm_server.server >> "$TS_LOG" 2>&1 &
    NEW_PID=$!
    echo "$NEW_PID" > "$PID_FILE"
    ln -sf "$TS_LOG" "$CUR_LOG"
    echo "  ✅ 백그라운드 시작: PID=$NEW_PID"
    echo "  📜 로그: $TS_LOG (심볼릭: $CUR_LOG)"
}

stop_server() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE" || true)
        if [ -n "${PID}" ] && ps -p "$PID" > /dev/null 2>&1; then
            echo "🛑 서버 중지 중... (PID $PID)"
            kill "$PID" || true
            # 최대 10초 대기
            for i in {1..20}; do
                if ps -p "$PID" > /dev/null 2>&1; then
                    sleep 0.5
                else
                    break
                fi
            done
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "  ⛔ 강제 종료(SIGKILL)"
                kill -9 "$PID" || true
            fi
            rm -f "$PID_FILE"
            echo "  ✅ 서버 중지 완료"
        else
            echo "  ℹ️ 실행 중인 서버를 찾지 못했습니다. PID 파일 정리"
            rm -f "$PID_FILE"
        fi
    else
        echo "  ℹ️ PID 파일이 없습니다. 서버가 실행 중이 아닐 수 있습니다."
    fi
}

status_server() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE" || true)
    if [ -n "${PID}" ] && ps -p "$PID" > /dev/null 2>&1; then
            echo "✅ 서버 실행 중 (PID $PID)"
        else
            echo "❌ PID 파일은 있으나 프로세스가 없습니다."
        fi
    else
        echo "❌ 서버가 실행 중이 아닙니다."
    fi
    echo "📜 로그 파일: $CUR_LOG"
    [ -f "$CUR_LOG" ] && tail -n 30 "$CUR_LOG" || echo "(로그 없음)"
}

case "$ACTION" in
    start)
        echo "⚙️  포그라운드 모드로 실행합니다 (터미널 종료 시 서버가 함께 종료)."
        echo "  👉 백그라운드는: $0 start-bg"
        echo
        echo "🔥 vLLM 서버 시작 중..."
        start_foreground
        ;;
    start-bg)
        echo "⚙️  백그라운드(데몬) 모드로 실행합니다 (터미널 종료와 무관)."
        echo
        start_background
        ;;
    stop)
        stop_server
        ;;
    restart)
        stop_server
        echo "🔄 서버 재시작..."
        start_background
        ;;
    status)
        status_server
        ;;
    logs)
        echo "📜 실시간 로그 보기: tail -f $CUR_LOG"
        [ -f "$CUR_LOG" ] && tail -n 100 "$CUR_LOG" || echo "(로그 없음)"
        ;;
    *)
    echo "사용법: $0 [start|start-bg|stop|status|logs|restart] [fp16|int4|int8|bf16]"
        exit 1
        ;;
esac

echo "✅ 작업 완료: $ACTION"
