# vLLM 서버 모델 모드 가이드

## 📋 모델 모드 비교

| 항목 | FP16 모드 | 4bit 모드 |
|------|-----------|-----------|
| **메모리 사용량** | ~60GB | ~30GB |
| **추론 속도** | 빠름 | 중간 (5-15% 느림) |
| **동시 처리** | 24 시퀀스 | 48 시퀀스 |
| **최대 컨텍스트** | 8192 토큰 | 12288 토큰 |
| **Fine-tuning** | ✅ 가능 | ✅ QLoRA 가능 |
| **메모리 효율성** | 보통 | 높음 |
| **처리량** | 높음 | 매우 높음 |

## 🚀 사용법

### 1. FP16 모드 (고성능)
```bash
# Foreground 실행
./start_vllm_with_options.sh start fp16

# Background 실행  
./start_vllm_with_options.sh start-bg fp16
```

### 2. 4bit 모드 (메모리 효율)
```bash
# Foreground 실행
./start_vllm_with_options.sh start 4bit

# Background 실행
./start_vllm_with_options.sh start-bg 4bit
```

### 3. LoRA 어댑터와 함께 실행
```bash
# FP16 + LoRA 어댑터
./start_vllm_with_options.sh start fp16 --with-lora

# 4bit + LoRA 어댑터
./start_vllm_with_options.sh start-bg 4bit --with-lora
```

### 4. 현재 설정 확인
```bash
./start_vllm_with_options.sh status
```

## 🔧 추론 요청 시 LoRA 어댑터 선택

### 요청별 LoRA 어댑터 지정
```bash
# 채팅 요청 with specific LoRA
curl -X POST "http://localhost:8001/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "안녕하세요!",
    "lora_adapter": "vision_v1",
    "max_tokens": 512
  }'

# 이미지 분석 with specific LoRA  
curl -X POST "http://localhost:8001/vision" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "이 이미지를 분석해주세요",
    "image_data": "data:image/jpeg;base64,/9j/4AAQ...",
    "lora_adapter": "document_v2"
  }'
```

### LoRA 어댑터 관리 API
```bash
# 등록된 LoRA 어댑터 목록
curl http://localhost:8001/lora/adapters

# 기본 LoRA 어댑터 설정
curl -X POST "http://localhost:8001/lora/set-default?adapter_name=vision_v1"

# 현재 모델 정보 확인
curl http://localhost:8001/models
```

## 🎯 LoRA 어댑터 관리

### LoRA 어댑터 자동 스캔
```bash
cd vllm_server/
python lora_manager.py scan
```

### LoRA 어댑터 수동 추가
```bash
python lora_manager.py add --name my_adapter --path /path/to/adapter --default
```

### LoRA 어댑터 목록 확인
```bash
python lora_manager.py list
```

### 기본 LoRA 어댑터 설정
```bash
python lora_manager.py set-default --name my_adapter
```

### .env 파일 자동 업데이트
```bash
python lora_manager.py update-env
```

## 🎯 권장 사용 시나리오

### FP16 모드 권장:
- **실시간 채팅**: 빠른 응답이 중요한 경우
- **단일 사용자**: 동시 처리가 많지 않은 경우  
- **고품질 추론**: 최고 품질이 필요한 경우

### 4bit 모드 권장:
- **다중 사용자**: 동시 접속자가 많은 경우
- **배치 처리**: 대량 작업 처리시
- **메모리 제약**: GPU 메모리가 부족한 경우
- **비용 절약**: 인프라 비용 절감이 중요한 경우

## 🔧 Fine-tuning 가이드

### FP16 모델 Fine-tuning
```bash
cd training/
python train_vision_lora.py --config config_vision.json
```

### 4bit 모델 Fine-tuning (QLoRA)
```bash
cd training/
# config_vision.json에서 qlora 설정 활성화됨
python train_vision_lora.py --config config_vision.json --use_4bit
```

## 📊 성능 모니터링

### GPU 메모리 확인
```bash
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits
```

### vLLM 서버 상태 확인
```bash
curl http://localhost:8001/health
```

### 처리량 테스트
```bash
cd vllm_server/
python test_vllm_performance.sh
```

## ⚠️ 주의사항

### 모드 전환시:
1. 기존 서버 완전 종료 필요
2. GPU 메모리 완전 해제 확인
3. 새로운 모드로 재시작

### 메모리 관리:
- **FP16**: GPU 메모리 92% 사용
- **4bit**: GPU 메모리 85% 사용 (안정성 고려)

### 호환성:
- vLLM 최신 버전 권장
- CUDA 11.8+ 필요
- PyTorch 2.0+ 필요

## 🚨 문제 해결

### 메모리 부족 오류:
```bash
# GPU 메모리 강제 해제
python -c "import torch; torch.cuda.empty_cache()"

# 프로세스 확인 및 종료
ps aux | grep vllm
kill -9 <PID>
```

### 양자화 오류:
```bash
# bitsandbytes 재설치
pip uninstall bitsandbytes -y
pip install bitsandbytes
```

### 모델 로드 오류:
```bash
# 모델 캐시 삭제
rm -rf ~/.cache/huggingface/transformers/
```
