import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version PyTorch built with: {torch.version.cuda}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is NOT available. Problem persists.")
exit()