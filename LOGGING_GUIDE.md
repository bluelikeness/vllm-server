# 로깅 개선 가이드

## 개요
vLLM 서버의 로깅 시스템이 대폭 개선되었습니다. 이제 환경변수를 통해 로그 레벨을 제어하고, DEBUG 모드에서는 입출력 정보를 상세하게 확인할 수 있습니다.

## 주요 기능

### 1. 환경변수 기반 로그 레벨 제어
```bash
# INFO 레벨 (프로덕션 권장)
LOG_LEVEL=INFO

# DEBUG 레벨 (개발/디버깅)
LOG_LEVEL=DEBUG
```

### 2. 로그 레벨별 출력 내용

#### INFO 레벨 (프로덕션)
- 요청/응답 요약 정보
- 성능 지표 (처리 시간, 토큰 속도 등)
- GPU 메모리 사용량
- 이미지 메타 정보 (크기, 모드)
- 짧은 프롬프트/응답 미리보기 (100자)
- 에러 정보

#### DEBUG 레벨 (개발/디버깅)
- INFO 레벨의 모든 정보
- **전체 프롬프트 내용** (최대 500자)
- **전체 응답 내용** (최대 500자)
- 이미지 Base64 데이터 길이
- 이미지 예상 파일 크기
- JSON 파싱 결과 상세 내용
- 함수 호출 스택 정보
- 상세한 타이밍 정보
- 에러 스택 트레이스

### 3. 구조화된 로깅

각 요청은 고유한 request_id를 가지며, 다음과 같은 형식으로 로깅됩니다:

```
2025-01-09 14:30:25 - INFO - 🎯 [a1b2c3d4] ===== 요청 시작: /vision =====
2025-01-09 14:30:25 - INFO - ⏰ [a1b2c3d4] 시작 시각: 2025-01-09 14:30:25.123
2025-01-09 14:30:25 - INFO - 📝 [a1b2c3d4] 프롬프트 길이: 250자
2025-01-09 14:30:25 - INFO - 🖼️ [a1b2c3d4] 이미지 포함: 예
2025-01-09 14:30:25 - INFO - 📐 [a1b2c3d4] 이미지 크기: 1920x1080
2025-01-09 14:30:25 - INFO - 🎨 [a1b2c3d4] 이미지 모드: RGB
2025-01-09 14:30:25 - INFO - ⚙️ [a1b2c3d4] 생성 파라미터:
   - max_tokens: 512
   - temperature: 0.7
2025-01-09 14:30:25 - INFO - 🖥️ [a1b2c3d4] GPU 메모리 (생성 전): 15.23GB / 80.00GB (19.0%)
2025-01-09 14:30:27 - INFO - 📤 [a1b2c3d4] 응답 길이: 342자
2025-01-09 14:30:27 - INFO - 💡 [a1b2c3d4] 응답 미리보기: 이미지에는 사무실 환경이...
2025-01-09 14:30:27 - INFO - ⏱️ [a1b2c3d4] 성능 지표:
   - generation_ms: 1850ms
   - tokens_generated: 145
   - tokens_per_second: 78.4
2025-01-09 14:30:27 - INFO - ✅ [a1b2c3d4] ===== 요청 성공 =====
2025-01-09 14:30:27 - INFO - ⏱️ [a1b2c3d4] 총 소요 시간: 2.156초
```

### 4. 컬러 로그 출력
터미널에서 로그 레벨별로 색상이 구분되어 표시됩니다:
- DEBUG: 청록색
- INFO: 녹색
- WARNING: 노란색
- ERROR: 빨간색
- CRITICAL: 자홍색

### 5. 로그 파일
로그는 다음 위치에 저장됩니다:
- `/tmp/vllm_app.log` - FastAPI 앱 로그
- `/tmp/vllm_engine.log` - vLLM 엔진 로그

## 사용 방법

### 1. .env 파일 설정

```bash
# 프로덕션 환경
LOG_LEVEL=INFO

# 개발/디버깅 환경
LOG_LEVEL=DEBUG
```

### 2. 환경변수로 직접 설정

```bash
# Linux/WSL
export LOG_LEVEL=DEBUG
python -m vllm_server.server

# PowerShell
$env:LOG_LEVEL="DEBUG"
python -m vllm_server.server
```

### 3. 실행 시 인라인 설정

```bash
LOG_LEVEL=DEBUG python -m vllm_server.server
```

## 로그 분석 예제

### 1. 특정 요청 추적
```bash
# request_id로 필터링
grep "a1b2c3d4" /tmp/vllm_app.log
```

### 2. 에러만 확인
```bash
grep "ERROR" /tmp/vllm_app.log
grep "❌" /tmp/vllm_app.log
```

### 3. 성능 지표만 추출
```bash
grep "성능 지표" /tmp/vllm_app.log
```

### 4. GPU 메모리 사용량 추적
```bash
grep "GPU 메모리" /tmp/vllm_engine.log
```

### 5. 실시간 로그 모니터링
```bash
tail -f /tmp/vllm_app.log
```

## 디버깅 시나리오

### 시나리오 1: 이미지 분석 문제
```bash
# DEBUG 모드로 전환
export LOG_LEVEL=DEBUG

# 서비스 재시작
# 로그에서 다음 정보 확인:
# - 이미지 크기 및 모드
# - Base64 데이터 길이
# - 프롬프트에 비전 태그 포함 여부
# - 멀티모달 처리 성공/실패
```

### 시나리오 2: 응답 품질 문제
```bash
# DEBUG 모드에서 전체 프롬프트와 응답 확인
export LOG_LEVEL=DEBUG

# 로그에서 다음 정보 확인:
# - 전체 프롬프트 내용 (최대 500자)
# - 전체 응답 내용 (최대 500자)
# - JSON 파싱 결과
```

### 시나리오 3: 성능 문제
```bash
# INFO 모드로도 충분
export LOG_LEVEL=INFO

# 로그에서 다음 정보 확인:
# - generation_ms (생성 시간)
# - tokens_per_second (처리 속도)
# - GPU 메모리 사용량
# - endpoint_total_ms (전체 처리 시간)
```

### 시나리오 4: LoRA 어댑터 문제
```bash
# DEBUG 모드로 전환
export LOG_LEVEL=DEBUG

# 로그에서 다음 정보 확인:
# - 요청된 LoRA 어댑터 이름
# - 기본 어댑터 사용 여부
# - LoRA 설정 성공/실패 메시지
```

## 프로덕션 배포 권장사항

### 1. 로그 레벨
```bash
# 프로덕션에서는 INFO 사용
LOG_LEVEL=INFO
```

### 2. 로그 로테이션
로그 파일이 너무 커지지 않도록 로그 로테이션 설정:

```bash
# logrotate 설정 예제 (/etc/logrotate.d/vllm-server)
/tmp/vllm_*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 user user
}
```

### 3. 모니터링
```bash
# 에러 알림 스크립트
#!/bin/bash
tail -f /tmp/vllm_app.log | grep --line-buffered "ERROR" | \
while read line; do
    echo "$line" | mail -s "vLLM Server Error" admin@example.com
done
```

### 4. 로그 수집
ELK Stack, Grafana Loki 등을 사용하여 중앙 집중식 로그 관리:

```yaml
# Docker Compose 예제
services:
  vllm-server:
    ...
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## 개선 사항 요약

1. ✅ 환경변수 기반 로그 레벨 제어 (LOG_LEVEL)
2. ✅ DEBUG 모드에서 전체 프롬프트/응답 로깅
3. ✅ 요청별 고유 ID 추적 (request_id)
4. ✅ 구조화된 로그 포맷
5. ✅ 컬러 터미널 출력
6. ✅ 이미지 메타 정보 상세 로깅
7. ✅ GPU 메모리 사용량 추적
8. ✅ LoRA 어댑터 정보 로깅
9. ✅ 성능 지표 자동 계산
10. ✅ 에러 스택 트레이스 (DEBUG 모드)
11. ✅ 파일 기반 로그 저장

## 문의 및 지원

문제가 발생하거나 추가 기능이 필요한 경우:
1. DEBUG 모드로 전환하여 상세 로그 확인
2. `/tmp/vllm_app.log` 및 `/tmp/vllm_engine.log` 파일 확인
3. 특정 request_id로 요청 추적
