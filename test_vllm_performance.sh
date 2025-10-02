#!/bin/bash

# vLLM 서버 성능 테스트 스크립트
BASE_URL="http://localhost:8001"

echo "🧪 vLLM 서버 성능 테스트 시작..."

# 1. 서버 상태 확인
echo "📊 서버 상태 확인..."
curl -s "$BASE_URL/health" | jq '.'

echo ""
echo "📈 상세 상태 확인..."
curl -s "$BASE_URL/status/detailed" | jq '.'

# 2. 단일 요청 테스트
echo ""
echo "⚡ 단일 요청 성능 테스트..."
time curl -s -X POST "$BASE_URL/generate" \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "안녕하세요! vLLM의 성능을 테스트하고 있습니다. 512토큰 정도의 긴 응답을 해주세요.",
    "max_tokens": 512,
    "temperature": 0.7
  }' | jq '.generation_time, .model_info.timings'

# 3. 동시 요청 테스트
echo ""
echo "🔥 동시 요청 테스트 (5개 동시)..."
for i in {1..5}; do
  {
    echo "요청 $i 시작..."
    time curl -s -X POST "$BASE_URL/generate" \
      -H 'Content-Type: application/json' \
      -d "{
        \"message\": \"동시 요청 테스트 $i번째입니다. 빠른 응답 부탁드립니다.\",
        \"max_tokens\": 256,
        \"temperature\": 0.7
      }" > /tmp/vllm_test_$i.json
    echo "요청 $i 완료"
  } &
done

wait
echo "✅ 모든 동시 요청 완료"

# 4. 결과 분석
echo ""
echo "📊 결과 분석..."
for i in {1..5}; do
  if [ -f "/tmp/vllm_test_$i.json" ]; then
    echo "요청 $i: $(cat /tmp/vllm_test_$i.json | jq -r '.generation_time')초, TPS: $(cat /tmp/vllm_test_$i.json | jq -r '.model_info.timings.tokens_per_second // "N/A"')"
    rm /tmp/vllm_test_$i.json
  fi
done

echo ""
echo "🎯 테스트 완료!"