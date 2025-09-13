#!/bin/bash

# vLLM Server 시작 스크립트 - 모델/모드 선택 및 LoRA 어댑터 지원
# 사용법: ./start_vllm_with_options.sh [start|start-bg|status] [FP16|INT4|INT8|BF16] [--with-lora]

set -euo pipefail

ACTION=${1:-start}          # start | start-bg | status
MODE=${2:-FP16}             # FP16 | INT4 | INT8 | BF16 (대소문자 무관)
WITH_LORA=${3:-}            # --with-lora

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$BASE_DIR/.env"

echo "🚀 CloudLLM vLLM 서버 시작: $ACTION (모드: ${MODE})"

# LoRA 어댑터 확인 및 설정
check_and_setup_lora() {
    if [[ "$WITH_LORA" == "--with-lora" ]]; then
        echo "🎯 LoRA 어댑터 설정 확인 중..."
        
        # LoRA 관리자로 어댑터 스캔
        if [[ -f "$BASE_DIR/lora_manager.py" ]]; then
            python "$BASE_DIR/lora_manager.py" scan
            python "$BASE_DIR/lora_manager.py" update-env
            echo "✅ LoRA 설정 완료"
        else
            echo "⚠️ LoRA 관리자를 찾을 수 없습니다. 수동 설정이 필요합니다."
        fi
    else
        echo "🎯 LoRA 어댑터: 비활성화 (베이스 모델만 사용)"
        # LoRA 설정 초기화
        sed -i '/^LORA_/d' "$ENV_FILE" || true
        cat >> "$ENV_FILE" << EOF

# LoRA 어댑터 설정 (비활성화)
LORA_ADAPTERS=
LORA_ADAPTER_NAMES=
DEFAULT_LORA_ADAPTER=
EOF
    fi
}

to_lower() { echo "$1" | tr '[:upper:]' '[:lower:]'; }

# 모드별 런타임 오버라이드(export) 구성: .env 파일을 직접 수정하지 않고 주입
configure_mode_overrides() {
    local mode=$(to_lower "$1")
    echo "🔧 모드별 런타임 설정 적용 중... ($mode)"
    case "$mode" in
        fp16)
            export RUNTIME_VLLM_LOAD_MODE=fp16
            ;;
        int4|4bit)
            export RUNTIME_VLLM_LOAD_MODE=4bit
            ;;
        int8|8bit)
            export RUNTIME_VLLM_LOAD_MODE=8bit
            ;;
        bf16|bfloat16)
            export RUNTIME_VLLM_LOAD_MODE=bf16
            ;;
        *)
            echo "❌ 지원하지 않는 모드: $1"
            echo "📝 사용 가능한 모드: FP16, INT4(4bit), INT8(8bit), BF16"
            exit 1
            ;;
    esac

    # 모드별 MODEL_NAME 스위칭 (설정돼 있으면 대체)
    set -a; source "$ENV_FILE"; set +a
    case "$mode" in
        fp16)
            if [[ -n "${FP16_MODEL_NAME:-}" ]]; then export RUNTIME_MODEL_NAME="$FP16_MODEL_NAME"; fi
            ;;
        int4|4bit)
            if [[ -n "${INT4_MODEL_NAME:-}" ]]; then export RUNTIME_MODEL_NAME="$INT4_MODEL_NAME"; fi
            ;;
        int8|8bit)
            if [[ -n "${INT8_MODEL_NAME:-}" ]]; then export RUNTIME_MODEL_NAME="$INT8_MODEL_NAME"; fi
            ;;
        bf16|bfloat16)
            if [[ -n "${BF16_MODEL_NAME:-}" ]]; then export RUNTIME_MODEL_NAME="$BF16_MODEL_NAME"; fi
            ;;
    esac
}

# 현재 설정 표시
show_current_config() {
    echo "📊 현재 모델 설정:"
    echo "  - 로드 모드(실효값): ${VLLM_LOAD_MODE:-$(grep -m1 '^VLLM_LOAD_MODE=' "$ENV_FILE" | cut -d'=' -f2)}"
    echo "  - 모델명(실효값): ${MODEL_NAME:-$(grep -m1 '^MODEL_NAME=' "$ENV_FILE" | cut -d'=' -f2)}"
    echo "  - 양자화(프리셋): $(grep -m1 '^INT4_QUANTIZATION_METHOD=' "$ENV_FILE" | cut -d'=' -f2 || echo '')"
    echo "  - 최대 길이(프리셋): $(grep -m1 "^${MODE^^}_MAX_MODEL_LEN=" "$ENV_FILE" | cut -d'=' -f2 || echo '')"
    echo "  - GPU 사용률(프리셋): $(grep -m1 "^${MODE^^}_GPU_MEMORY_UTILIZATION=" "$ENV_FILE" | cut -d'=' -f2 || echo '')"
    echo "  - 최대 시퀀스(프리셋): $(grep -m1 "^${MODE^^}_MAX_NUM_SEQS=" "$ENV_FILE" | cut -d'=' -f2 || echo '')"
    
    # LoRA 설정 표시
    local lora_adapters=$(grep "^LORA_ADAPTERS=" "$ENV_FILE" | cut -d'=' -f2 || echo "")
    local default_lora=$(grep "^DEFAULT_LORA_ADAPTER=" "$ENV_FILE" | cut -d'=' -f2 || echo "")
    
    if [[ -n "$lora_adapters" ]]; then
        echo "  - LoRA 어댑터: 활성화"
        if [[ -n "$default_lora" ]]; then
            echo "    기본 어댑터: $default_lora"
        fi
    else
        echo "  - LoRA 어댑터: 비활성화"
    fi
}

# 메인 로직
case $ACTION in
    start|start-bg)
        configure_mode_overrides "$MODE"
        check_and_setup_lora
        show_current_config
        echo "🚀 vLLM 서버 시작 중..."
        # 환경변수는 export로 이미 전달되므로 직접 실행
        ./start_vllm.sh "$ACTION"
        ;;
    status)
        # .env만 읽을 경우 실효값이 보이지 않을 수 있으므로, 현재 셸의 override도 반영
        configure_mode_overrides "$MODE"
        show_current_config
        ;;
    logs)
        # 최신 심볼릭 로그를 팔로우
        LOG_DIR="$BASE_DIR/logs"
        CUR_LOG="$LOG_DIR/server.log"
        echo "📜 실시간 로그 보기: tail -f $CUR_LOG"
        [ -f "$CUR_LOG" ] && tail -f "$CUR_LOG" || echo "(로그 없음)"
        ;;
    *)
        echo "사용법: $0 [start|start-bg|status|logs] [FP16|INT4|INT8|BF16] [--with-lora]"
        echo ""
        echo "예시:"
        echo "  $0 start FP16                 # FP16 모드로 포그라운드 시작"
        echo "  $0 start-bg INT4 --with-lora  # INT4(AWQ)+LoRA로 백그라운드 시작"
        echo "  $0 status INT8                # INT8 프리셋 상태 확인"
        echo "  $0 logs FP16                  # 최신 서버 로그 팔로우"
        exit 1
        ;;
esac
