# vLLM Server 설치 및 실행 가이드

## 🚀 **설치 순서**

### 1. 의존성 설치
```bash
# 기본 패키지 설치
pip install -r requirements.txt

# vLLM 설치 (CUDA 11.8 기준)
pip install vllm

# Ray 설치 (vLLM 의존성)
pip install ray
```

### 2. 환경 설정
```bash
# .env 파일이 이미 생성되어 있습니다
# 필요시 수정하세요:
vi .env
```

### 3. GPU 확인
```bash
# NVIDIA GPU 확인
nvidia-smi

# CUDA 버전 확인
nvcc --version

# GPU 메모리 최소 24GB 권장 (32B 모델)
```

## ⚙️ **설정 최적화**

### GPU 메모리에 따른 설정:
```bash
# 24GB GPU (RTX 3090, 4090)
VLLM_MAX_NUM_SEQS=8
VLLM_GPU_MEMORY_UTILIZATION=0.85

# 40GB GPU (A100)
VLLM_MAX_NUM_SEQS=16
VLLM_GPU_MEMORY_UTILIZATION=0.9

# 80GB GPU (A100 80GB)
VLLM_MAX_NUM_SEQS=32
VLLM_GPU_MEMORY_UTILIZATION=0.95
```

## 🔥 **실행 방법**

### 방법 1: 스크립트 실행
```bash
chmod +x start_vllm.sh
./start_vllm.sh
```

### 방법 2: 직접 실행
```bash
python server.py
```

## 🧪 **성능 테스트**

```bash
# 테스트 스크립트 실행
chmod +x test_vllm_performance.sh
./test_vllm_performance.sh
```

## 📊 **모니터링**

### 실시간 GPU 모니터링
```bash
# GPU 사용률 모니터링
watch -n 1 nvidia-smi

# 서버 로그 모니터링
tail -f /var/log/vllm_server.log
```

### API로 상태 확인
```bash
# 간단한 상태
curl http://localhost:8001/health

# 상세한 상태
curl http://localhost:8001/status/detailed
```

## 🐛 **트러블슈팅**

### 1. vLLM 설치 오류
```bash
# CUDA 버전 확인 후 재설치
pip uninstall vllm
pip install vllm --no-cache-dir
```

### 2. 메모리 부족 오류
```bash
# 배치 크기 줄이기
export VLLM_MAX_NUM_SEQS=4

# GPU 메모리 사용률 줄이기
export VLLM_GPU_MEMORY_UTILIZATION=0.8
```

### 3. 모델 로딩 실패
```bash
# 디스크 공간 확인
df -h

# 모델 캐시 정리
rm -rf ~/.cache/huggingface/transformers/
```

### 4. 성능이 예상보다 낮은 경우
```bash
# CUDA Graph 비활성화 시도
export VLLM_ENFORCE_EAGER=true

# Flash Attention 확인
python -c "import flash_attn; print('Flash Attention 사용 가능')"
```

## 🔧 **고급 설정**

### 다중 GPU 사용
```bash
# .env 파일에서 설정
VLLM_TENSOR_PARALLEL_SIZE=2  # GPU 개수에 맞게
```

### 메모리 최적화
```bash
# 스왑 공간 설정
VLLM_SWAP_SPACE=8  # GB

# 블록 크기 조정
VLLM_BLOCK_SIZE=32  # 더 큰 블록으로 메모리 효율 향상
```

## 📈 **예상 성능**

### 하드웨어별 예상 TPS (512 토큰 기준):
- **RTX 4090**: 60-80 TPS
- **A100 40GB**: 100-120 TPS  
- **A100 80GB**: 120-150 TPS
- **H100**: 200+ TPS

### 동시 처리 능력:
- **24GB GPU**: 4-8 동시 요청
- **40GB GPU**: 8-16 동시 요청
- **80GB GPU**: 16-32 동시 요청

## ✅ **설치 완료 체크리스트**

- [ ] Python 3.10+ 설치
- [ ] CUDA 11.8+ 설치  
- [ ] vLLM 패키지 설치
- [ ] .env 파일 설정
- [ ] GPU 메모리 확인 (24GB+ 권장)
- [ ] 서버 정상 시작 확인
- [ ] API 응답 테스트 완료
- [ ] 성능 벤치마크 실행

설치 완료 후 `curl http://localhost:8001/health`로 서버 상태를 확인하세요!