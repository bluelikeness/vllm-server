#!/bin/bash

###############################################################################
# vLLM 서버 배포 스크립트 (로깅 개선 버전)
# 
# 기능:
# - 로깅 시스템 개선 사항 자동 적용
# - 환경변수 설정
# - 서비스 재시작
# - 로그 확인
###############################################################################

set -e  # 오류 발생 시 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 함수 정의
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# 스크립트 디렉토리 확인
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

print_header "vLLM 서버 배포 시작 (로깅 개선 버전)"

# 1. 환경변수 확인
print_info "현재 환경변수 확인..."
if [ -f .env ]; then
    print_success ".env 파일 존재"
    
    # LOG_LEVEL 확인
    if grep -q "LOG_LEVEL" .env; then
        current_log_level=$(grep "LOG_LEVEL" .env | cut -d '=' -f2)
        print_info "현재 로그 레벨: $current_log_level"
    else
        print_warning "LOG_LEVEL 설정이 없습니다"
        echo ""
        echo "로그 레벨을 선택하세요:"
        echo "1) INFO  - 프로덕션 환경 (기본)"
        echo "2) DEBUG - 개발/디버깅 환경"
        read -p "선택 (1 또는 2): " choice
        
        case $choice in
            2)
                echo "LOG_LEVEL=DEBUG" >> .env
                print_success "로그 레벨 DEBUG 설정 완료"
                ;;
            *)
                echo "LOG_LEVEL=INFO" >> .env
                print_success "로그 레벨 INFO 설정 완료"
                ;;
        esac
    fi
else
    print_error ".env 파일이 없습니다"
    print_info ".env 파일을 생성하시겠습니까? (y/n)"
    read -p "> " create_env
    
    if [ "$create_env" = "y" ]; then
        print_info "로그 레벨을 선택하세요:"
        echo "1) INFO  - 프로덕션 환경 (기본)"
        echo "2) DEBUG - 개발/디버깅 환경"
        read -p "선택 (1 또는 2): " choice
        
        case $choice in
            2)
                echo "LOG_LEVEL=DEBUG" > .env
                print_success "로그 레벨 DEBUG 설정 완료"
                ;;
            *)
                echo "LOG_LEVEL=INFO" > .env
                print_success "로그 레벨 INFO 설정 완료"
                ;;
        esac
    else
        print_error "배포를 중단합니다"
        exit 1
    fi
fi

# 2. 로그 디렉토리 확인
print_info "로그 디렉토리 확인..."
LOG_DIR="/tmp"
if [ ! -w "$LOG_DIR" ]; then
    print_warning "$LOG_DIR에 쓰기 권한이 없습니다"
    LOG_DIR="./logs"
    mkdir -p "$LOG_DIR"
    print_success "로컬 로그 디렉토리 생성: $LOG_DIR"
    
    # logger_config.py 수정
    if [ -f "logger_config.py" ]; then
        sed -i "s|/tmp/vllm_app.log|$LOG_DIR/vllm_app.log|g" logger_config.py
        sed -i "s|/tmp/vllm_engine.log|$LOG_DIR/vllm_engine.log|g" logger_config.py
        print_success "로그 경로 업데이트 완료"
    fi
fi

# 3. 기존 프로세스 확인 및 종료
print_info "기존 vLLM 서버 프로세스 확인..."
if pgrep -f "vllm_server" > /dev/null; then
    print_warning "실행 중인 vLLM 서버 발견"
    print_info "기존 서버를 종료하시겠습니까? (y/n)"
    read -p "> " kill_server
    
    if [ "$kill_server" = "y" ]; then
        pkill -f "vllm_server" || true
        sleep 2
        print_success "기존 서버 종료 완료"
    else
        print_error "배포를 중단합니다"
        exit 1
    fi
else
    print_success "실행 중인 프로세스 없음"
fi

# 4. 파일 존재 확인
print_info "필수 파일 확인..."
required_files=("logger_config.py" "app.py" "engine.py" "server.py")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        print_error "필수 파일이 없습니다: $file"
        exit 1
    fi
done
print_success "모든 필수 파일 존재"

# 5. Python 패키지 확인
print_info "Python 패키지 확인..."
python3 -c "import vllm, fastapi, torch" 2>/dev/null || {
    print_error "필수 Python 패키지가 설치되지 않았습니다"
    print_info "requirements.txt를 사용하여 설치하시겠습니까? (y/n)"
    read -p "> " install_packages
    
    if [ "$install_packages" = "y" ]; then
        pip install -r requirements.txt
        print_success "패키지 설치 완료"
    else
        print_error "배포를 중단합니다"
        exit 1
    fi
}
print_success "Python 패키지 확인 완료"

# 6. 서버 시작 방법 선택
print_header "서버 시작 방법 선택"
echo "1) 포그라운드 실행 (로그 실시간 확인)"
echo "2) 백그라운드 실행 (nohup)"
echo "3) systemd 서비스로 실행"
echo "4) 배포만 하고 시작하지 않음"
read -p "선택 (1-4): " start_choice

case $start_choice in
    1)
        print_success "포그라운드 실행 선택"
        print_info "서버를 시작합니다..."
        print_warning "Ctrl+C로 중지할 수 있습니다"
        sleep 2
        
        # .env 로드 및 실행
        export $(cat .env | xargs)
        python3 -m vllm_server.server
        ;;
    
    2)
        print_success "백그라운드 실행 선택"
        
        # 로그 파일 경로
        OUTPUT_LOG="./vllm_server_$(date +%Y%m%d_%H%M%S).log"
        
        print_info "서버를 백그라운드로 시작합니다..."
        print_info "로그 파일: $OUTPUT_LOG"
        
        # .env 로드 및 백그라운드 실행
        nohup bash -c "export \$(cat .env | xargs) && python3 -m vllm_server.server" > "$OUTPUT_LOG" 2>&1 &
        
        SERVER_PID=$!
        print_success "서버 시작 완료 (PID: $SERVER_PID)"
        print_info "로그 확인: tail -f $OUTPUT_LOG"
        
        # PID 파일 저장
        echo $SERVER_PID > vllm_server.pid
        print_success "PID 파일 저장: vllm_server.pid"
        
        # 5초 후 상태 확인
        sleep 5
        if ps -p $SERVER_PID > /dev/null; then
            print_success "서버가 정상적으로 실행 중입니다"
        else
            print_error "서버 시작 실패"
            print_info "로그 확인: cat $OUTPUT_LOG"
            exit 1
        fi
        ;;
    
    3)
        print_success "systemd 서비스 설정 선택"
        
        # systemd 서비스 파일 생성
        SERVICE_FILE="/etc/systemd/system/vllm-server.service"
        
        print_info "systemd 서비스 파일을 생성합니다..."
        
        sudo bash -c "cat > $SERVICE_FILE" <<EOF
[Unit]
Description=vLLM Server with Enhanced Logging
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$SCRIPT_DIR
EnvironmentFile=$SCRIPT_DIR/.env
ExecStart=/usr/bin/python3 -m vllm_server.server
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=vllm-server

[Install]
WantedBy=multi-user.target
EOF
        
        print_success "서비스 파일 생성 완료: $SERVICE_FILE"
        
        # systemd 리로드
        sudo systemctl daemon-reload
        print_success "systemd 데몬 리로드 완료"
        
        # 서비스 시작
        sudo systemctl enable vllm-server
        sudo systemctl start vllm-server
        print_success "서비스 시작 완료"
        
        # 상태 확인
        sleep 3
        sudo systemctl status vllm-server --no-pager
        
        print_info ""
        print_info "서비스 관리 명령어:"
        print_info "  - 상태 확인: sudo systemctl status vllm-server"
        print_info "  - 시작: sudo systemctl start vllm-server"
        print_info "  - 중지: sudo systemctl stop vllm-server"
        print_info "  - 재시작: sudo systemctl restart vllm-server"
        print_info "  - 로그 확인: sudo journalctl -u vllm-server -f"
        ;;
    
    4)
        print_success "배포만 완료되었습니다"
        print_info "수동으로 서버를 시작하려면:"
        print_info "  export \$(cat .env | xargs)"
        print_info "  python3 -m vllm_server.server"
        exit 0
        ;;
    
    *)
        print_error "잘못된 선택입니다"
        exit 1
        ;;
esac

# 7. 배포 완료 메시지
print_header "배포 완료"
print_success "vLLM 서버 로깅 개선 버전 배포 완료!"

# 현재 로그 레벨 출력
current_log_level=$(grep "LOG_LEVEL" .env | cut -d '=' -f2)
print_info "현재 로그 레벨: $current_log_level"

if [ "$current_log_level" = "DEBUG" ]; then
    print_warning "DEBUG 모드는 상세한 로그를 출력합니다"
    print_warning "프로덕션 환경에서는 INFO 레벨 사용을 권장합니다"
fi

print_info ""
print_info "로그 확인 방법:"
if [ "$start_choice" = "2" ]; then
    print_info "  tail -f $OUTPUT_LOG"
elif [ "$start_choice" = "3" ]; then
    print_info "  sudo journalctl -u vllm-server -f"
else
    print_info "  tail -f /tmp/vllm_app.log"
fi

print_info ""
print_info "로그 레벨 변경:"
print_info "  1. .env 파일에서 LOG_LEVEL 수정"
print_info "  2. 서비스 재시작"

print_info ""
print_info "상세한 가이드: LOGGING_GUIDE.md 참고"
