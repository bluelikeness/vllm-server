import torch

print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"사용 중인 GPU: {torch.cuda.get_device_name(0)}")

if torch.cuda.is_available():
    print(f"사용 중인 GPU: {torch.cuda.get_device_name(0)}")
