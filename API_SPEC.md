# CloudLLM vLLM Server API Spec (v3.0)

이 문서는 vLLM 기반의 `vllm_server/server.py`가 제공하는 고성능 API를 설명합니다. 

## 🚀 **vLLM 엔진 주요 특징**
- **고성능 배칭**: 동시 요청을 효율적으로 배칭 처리
- **KV Cache 최적화**: Paged Attention으로 메모리 효율성 극대화
- **높은 처리량**: 기존 대비 3-5배 TPS 향상
- **낮은 지연시간**: 응답 시간 60-80% 단축

## Base URL
- 기본: `http://<host>:<port>`
- 서버 환경변수:
  - `LLM_HOST` (기본 `0.0.0.0`)
  - `LLM_PORT` (기본 `8001`)

문서/스키마: FastAPI 자동 문서
- Swagger UI: `GET /docs`
- OpenAPI JSON: `GET /openapi.json`

## vLLM 엔진 설정
환경변수로 vLLM 성능을 조정할 수 있습니다:
- `VLLM_MAX_MODEL_LEN`: 최대 시퀀스 길이 (기본 2048)
- `VLLM_MAX_NUM_SEQS`: 최대 배치 크기 (기본 16)
- `VLLM_BLOCK_SIZE`: 메모리 블록 크기 (기본 16)
- `VLLM_GPU_MEMORY_UTILIZATION`: GPU 메모리 사용률 (기본 0.9)
- `VLLM_TENSOR_PARALLEL_SIZE`: 텐서 병렬화 크기 (기본 1)

## 공통 응답 타입: GenerationResponse
- response: string
- conversation_id: string
- generation_time: number (seconds)
- model_info: object
  - engine: "vLLM" (고정)
  - model_name: string
  - max_tokens: number
  - temperature: number
  - timings: object (vLLM 특화 타이밍 메트릭)
- response_json: object|null (응답이 JSON으로 파싱되면 채워짐)
- response_is_json: boolean

## vLLM 타이밍 메트릭
- generation_ms: vLLM 엔진 생성 시간
- total_ms: 전체 처리 시간
- tokens_generated: 생성된 토큰 수
- tokens_per_second: TPS (Tokens Per Second)
- endpoint_total_ms: API 엔드포인트 총 시간
- json_parse_ms: JSON 파싱 시간

## 리소스/정책 제한
- 토큰 상한: `MAX_TOKENS_CAP` (기본 512)
- 업로드 크기: `MAX_UPLOAD_BYTES` (기본 10MB)
- 이미지 최대 변: `MAX_IMAGE_SIDE` (기본 1280px)
- vLLM 배치 크기: `VLLM_MAX_NUM_SEQS`로 제어

## 1) Health Check (vLLM 특화)
GET `/health`
- 200 응답 예:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "Qwen/Qwen2.5-VL-32B-Instruct",
  "engine_type": "vLLM",
  "gpu_memory_used": 24.5,
  "gpu_memory_total": 40.0,
  "vllm_stats": {
    "engine_status": "running",
    "pending_requests": 2,
    "running_requests": 4
  },
  "active_conversations": 15,
  "uptime": 3600.5
}
```

## 2) 순수 텍스트 생성 (vLLM 최적화)
POST `/generate`
- Request (application/json):
```json
{
  "message": "안녕하세요, vLLM으로 빠른 응답을 해주세요",
  "conversation_id": null,
  "max_tokens": 512,
  "temperature": 0.7
}
```
- Response: GenerationResponse
  - vLLM 특화 타이밍 정보 포함
  - 높은 TPS와 낮은 지연시간 달성

## 3) 비전 분석 (구현 예정)
POST `/vision`
- Status: 501 Not Implemented
- vLLM의 멀티모달 지원이 안정화되면 구현 예정

## 4) 멀티모달 분석 (구현 예정)
POST `/multimodal`
- Status: 501 Not Implemented
- vLLM의 멀티모달 지원이 안정화되면 구현 예정

## 5) 대화 관리 (기존과 동일)
- GET `/conversations` 전체 목록
- GET `/conversations/{conversation_id}` 특정 대화 조회
- DELETE `/conversations/{conversation_id}` 삭제

## 6) 상세 상태 (vLLM 특화)
GET `/status/detailed`
- vLLM 엔진 상태 및 성능 메트릭 제공
- GPU 사용률, 배치 처리 상태 등 포함

## 성능 벤치마크 (예상)
### 512 토큰 생성 기준:
- **TPS**: 80-150 (기존 20-30 대비 3-5배)
- **응답시간**: 3-8초 (기존 15-25초 대비 60-80% 단축)
- **동시처리**: 8-16 요청 (기존 2-4 대비 4-8배)
- **GPU 활용률**: 85-95% (기존 40-50% 대비 2배)

## cURL 예제
- 텍스트 생성:
```bash
curl -sS -X POST "$BASE/generate" \
  -H 'Content-Type: application/json' \
  -d '{"message":"vLLM으로 빠른 응답 부탁해","max_tokens":512,"temperature":0.7}'
```

- 상태 확인:
```bash
curl -sS "$BASE/health"
curl -sS "$BASE/status/detailed"
```

## vLLM 최적화 팁
### 성능 최대화:
1. **배치 크기 조정**: `VLLM_MAX_NUM_SEQS`를 GPU 메모리에 맞게 설정
2. **메모리 활용률**: `VLLM_GPU_MEMORY_UTILIZATION`을 0.9로 설정
3. **시퀀스 길이**: 용도에 맞게 `VLLM_MAX_MODEL_LEN` 조정
4. **텐서 병렬화**: 다중 GPU 사용시 `VLLM_TENSOR_PARALLEL_SIZE` 설정

### 메모리 절약:
- 짧은 응답용: MAX_TOKENS_CAP=256
- 긴 응답용: MAX_TOKENS_CAP=1024
- 배치 크기를 GPU 메모리에 맞게 조정

## 마이그레이션 가이드
### 기존 Transformers 서버에서 vLLM으로:
1. 의존성 업데이트: `pip install vllm ray`
2. 환경변수 설정: `.env` 파일의 vLLM 설정 적용
3. API 호출 방식은 동일 (투명한 업그레이드)
4. 성능 모니터링으로 최적 설정 찾기

---
본 문서는 vLLM 기반 서버 소스(v3.0.0)를 기준으로 합니다. vLLM 버전에 따라 성능이 달라질 수 있습니다.