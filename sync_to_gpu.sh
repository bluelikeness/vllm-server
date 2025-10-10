#!/bin/bash
# vLLM 서버 코드를 원격 GPU 서버와 동기화하는 스크립트

set -euo pipefail

LOCAL_DIR="/home/kckang/work/0.CloudLLM-vllm-server/"
REMOTE_USER="ubuntu"
REMOTE_HOST="210.109.80.82"
REMOTE_DIR="/home/ubuntu/work/vllm_server/"
SSH_KEY="/home/kckang/.ssh/aiocr-poc.pem"

rsync -avz --progress \
  -e "ssh -i ${SSH_KEY}" \
  --exclude 'venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.git/' \
  --exclude '*.log' \
  "${LOCAL_DIR}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
