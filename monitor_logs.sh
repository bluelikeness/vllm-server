#!/bin/bash

###############################################################################
# vLLM 서버 로그 모니터링 스크립트
# 
# 기능:
# - 실시간 로그 모니터링
# - 특정 패턴 필터링
# - 로그 분석
###############################################################################

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 로그 파일 경로
APP_LOG="/tmp/vllm_app.log"
ENGINE_LOG="/tmp/vllm_engine.log"

# 로컬 로그 디렉토리도 확인
if [ ! -f "$APP_LOG" ] && [ -f "./logs/vllm_app.log" ]; then
    APP_LOG="./logs/vllm_app.log"
    ENGINE_LOG="./logs/vllm_engine.log"
fi

# 함수 정의
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_menu() {
    print_header "vLLM 서버 로그 모니터링"
    echo "1) 전체 로그 실시간 모니터링 (app)"
    echo "2) 전체 로그 실시간 모니터링 (engine)"
    echo "3) 에러만 모니터링"
    echo "4) 특정 request_id 추적"
    echo "5) 성능 지표만 확인"
    echo "6) GPU 메모리 사용량 추적"
    echo "7) 최근 요청 요약"
    echo "8) 로그 분석 (통계)"
    echo "9) 로그 레벨 변경"
    echo "0) 종료"
    echo ""
}

monitor_all_logs() {
    local log_file=$1
    echo -e "${CYAN}실시간 로그 모니터링 중... (Ctrl+C로 중지)${NC}"
    echo ""
    tail -f "$log_file" | while read line; do
        # 색상 적용
        if [[ "$line" == *"ERROR"* ]] || [[ "$line" == *"❌"* ]]; then
            echo -e "${RED}$line${NC}"
        elif [[ "$line" == *"WARNING"* ]] || [[ "$line" == *"⚠️"* ]]; then
            echo -e "${YELLOW}$line${NC}"
        elif [[ "$line" == *"✅"* ]] || [[ "$line" == *"SUCCESS"* ]]; then
            echo -e "${GREEN}$line${NC}"
        elif [[ "$line" == *"DEBUG"* ]]; then
            echo -e "${CYAN}$line${NC}"
        else
            echo "$line"
        fi
    done
}

monitor_errors() {
    echo -e "${CYAN}에러 모니터링 중... (Ctrl+C로 중지)${NC}"
    echo ""
    tail -f "$APP_LOG" "$ENGINE_LOG" 2>/dev/null | grep --line-buffered -E "ERROR|❌|실패" | while read line; do
        echo -e "${RED}$line${NC}"
    done
}

track_request() {
    echo "추적할 request_id를 입력하세요:"
    read -p "> " request_id
    
    if [ -z "$request_id" ]; then
        echo -e "${RED}request_id가 입력되지 않았습니다${NC}"
        return
    fi
    
    echo -e "${CYAN}Request ID [$request_id] 추적 결과:${NC}"
    echo ""
    
    echo -e "${YELLOW}=== App 로그 ===${NC}"
    grep "$request_id" "$APP_LOG" 2>/dev/null || echo "결과 없음"
    
    echo ""
    echo -e "${YELLOW}=== Engine 로그 ===${NC}"
    grep "$request_id" "$ENGINE_LOG" 2>/dev/null || echo "결과 없음"
}

monitor_performance() {
    echo -e "${CYAN}성능 지표 모니터링 중... (Ctrl+C로 중지)${NC}"
    echo ""
    tail -f "$APP_LOG" "$ENGINE_LOG" 2>/dev/null | grep --line-buffered -E "성능 지표|generation_ms|tokens_per_second|⏱️|🚀" | while read line; do
        echo -e "${GREEN}$line${NC}"
    done
}

monitor_gpu() {
    echo -e "${CYAN}GPU 메모리 모니터링 중... (Ctrl+C로 중지)${NC}"
    echo ""
    tail -f "$ENGINE_LOG" 2>/dev/null | grep --line-buffered "GPU 메모리" | while read line; do
        # 메모리 사용률 추출 및 색상 적용
        if [[ "$line" =~ ([0-9]+\.[0-9]+)% ]]; then
            usage="${BASH_REMATCH[1]}"
            if (( $(echo "$usage > 90" | bc -l) )); then
                echo -e "${RED}$line${NC}"
            elif (( $(echo "$usage > 75" | bc -l) )); then
                echo -e "${YELLOW}$line${NC}"
            else
                echo -e "${GREEN}$line${NC}"
            fi
        else
            echo "$line"
        fi
    done
}

show_recent_requests() {
    echo -e "${CYAN}최근 10개 요청 요약:${NC}"
    echo ""
    
    # 최근 요청 ID 추출
    request_ids=$(grep "===== 요청 시작" "$APP_LOG" 2>/dev/null | tail -10 | grep -oP '\[\K[^\]]+' | sort -u)
    
    if [ -z "$request_ids" ]; then
        echo "요청 기록이 없습니다"
        return
    fi
    
    echo -e "${BLUE}Request ID | 엔드포인트 | 상태 | 소요시간${NC}"
    echo "-------------------------------------------"
    
    for req_id in $request_ids; do
        # 엔드포인트 추출
        endpoint=$(grep "\[$req_id\].*요청 시작" "$APP_LOG" | grep -oP '요청 시작: \K[^ ]+' | head -1)
        
        # 상태 확인
        if grep -q "\[$req_id\].*요청 성공" "$APP_LOG"; then
            status="${GREEN}성공${NC}"
        elif grep -q "\[$req_id\].*요청 실패" "$APP_LOG"; then
            status="${RED}실패${NC}"
        else
            status="${YELLOW}진행중${NC}"
        fi
        
        # 소요시간 추출
        duration=$(grep "\[$req_id\].*총 소요 시간" "$APP_LOG" | grep -oP '시간: \K[0-9.]+' | head -1)
        duration="${duration:-N/A}"
        
        echo -e "$req_id | $endpoint | $status | ${duration}초"
    done
}

analyze_logs() {
    print_header "로그 분석 통계"
    
    if [ ! -f "$APP_LOG" ]; then
        echo "로그 파일이 없습니다: $APP_LOG"
        return
    fi
    
    # 총 요청 수
    total_requests=$(grep -c "===== 요청 시작" "$APP_LOG" 2>/dev/null || echo "0")
    echo -e "${BLUE}총 요청 수:${NC} $total_requests"
    
    # 성공/실패 요청 수
    success_requests=$(grep -c "===== 요청 성공" "$APP_LOG" 2>/dev/null || echo "0")
    failed_requests=$(grep -c "===== 요청 실패" "$APP_LOG" 2>/dev/null || echo "0")
    echo -e "${GREEN}성공 요청:${NC} $success_requests"
    echo -e "${RED}실패 요청:${NC} $failed_requests"
    
    # 엔드포인트별 요청 수
    echo ""
    echo -e "${BLUE}엔드포인트별 요청 수:${NC}"
    grep "요청 시작:" "$APP_LOG" 2>/dev/null | grep -oP '요청 시작: \K[^ ]+' | sort | uniq -c | while read count endpoint; do
        echo "  $endpoint: $count"
    done
    
    # 평균 응답 시간
    echo ""
    echo -e "${BLUE}성능 통계:${NC}"
    avg_time=$(grep "총 소요 시간" "$APP_LOG" 2>/dev/null | grep -oP '시간: \K[0-9.]+' | awk '{s+=$1; n++} END {if(n>0) print s/n; else print "N/A"}')
    echo "  평균 응답 시간: ${avg_time}초"
    
    # 에러 수
    error_count=$(grep -c "ERROR\|❌" "$APP_LOG" 2>/dev/null || echo "0")
    echo "  에러 수: $error_count"
    
    # 경고 수
    warning_count=$(grep -c "WARNING\|⚠️" "$APP_LOG" 2>/dev/null || echo "0")
    echo "  경고 수: $warning_count"
    
    # 이미지 처리 요청 수
    image_requests=$(grep -c "이미지 포함: 예" "$ENGINE_LOG" 2>/dev/null || echo "0")
    echo "  이미지 처리 요청: $image_requests"
    
    # LoRA 어댑터 사용
    echo ""
    echo -e "${BLUE}LoRA 어댑터 사용:${NC}"
    grep "LoRA 어댑터:" "$ENGINE_LOG" 2>/dev/null | grep -oP 'LoRA 어댑터: \K.*' | sort | uniq -c | while read count adapter; do
        echo "  $adapter: $count"
    done
}

change_log_level() {
    print_header "로그 레벨 변경"
    
    if [ ! -f ".env" ]; then
        echo -e "${RED}.env 파일이 없습니다${NC}"
        return
    fi
    
    current_level=$(grep "LOG_LEVEL" .env | cut -d '=' -f2)
    echo "현재 로그 레벨: $current_level"
    echo ""
    echo "새로운 로그 레벨을 선택하세요:"
    echo "1) INFO  - 프로덕션 환경"
    echo "2) DEBUG - 개발/디버깅 환경"
    read -p "선택 (1 또는 2): " choice
    
    case $choice in
        2)
            sed -i 's/LOG_LEVEL=.*/LOG_LEVEL=DEBUG/' .env
            echo -e "${GREEN}로그 레벨을 DEBUG로 변경했습니다${NC}"
            ;;
        1)
            sed -i 's/LOG_LEVEL=.*/LOG_LEVEL=INFO/' .env
            echo -e "${GREEN}로그 레벨을 INFO로 변경했습니다${NC}"
            ;;
        *)
            echo -e "${RED}잘못된 선택입니다${NC}"
            return
            ;;
    esac
    
    echo ""
    echo -e "${YELLOW}변경 사항을 적용하려면 서비스를 재시작해야 합니다${NC}"
    echo "서비스를 재시작하시겠습니까? (y/n)"
    read -p "> " restart
    
    if [ "$restart" = "y" ]; then
        # PID 파일 확인
        if [ -f "vllm_server.pid" ]; then
            pid=$(cat vllm_server.pid)
            if ps -p $pid > /dev/null 2>&1; then
                echo "서버를 재시작합니다..."
                kill $pid
                sleep 2
                nohup bash -c "export \$(cat .env | xargs) && python3 -m vllm_server.server" > vllm_server_$(date +%Y%m%d_%H%M%S).log 2>&1 &
                echo $! > vllm_server.pid
                echo -e "${GREEN}서버 재시작 완료${NC}"
            fi
        else
            echo -e "${YELLOW}PID 파일이 없습니다. 수동으로 재시작하세요${NC}"
        fi
    fi
}

# 메인 루프
while true; do
    print_menu
    read -p "선택: " choice
    echo ""
    
    case $choice in
        1)
            monitor_all_logs "$APP_LOG"
            ;;
        2)
            monitor_all_logs "$ENGINE_LOG"
            ;;
        3)
            monitor_errors
            ;;
        4)
            track_request
            ;;
        5)
            monitor_performance
            ;;
        6)
            monitor_gpu
            ;;
        7)
            show_recent_requests
            ;;
        8)
            analyze_logs
            ;;
        9)
            change_log_level
            ;;
        0)
            echo "종료합니다"
            exit 0
            ;;
        *)
            echo -e "${RED}잘못된 선택입니다${NC}"
            ;;
    esac
    
    echo ""
    read -p "계속하려면 Enter를 누르세요..."
    clear
done
