import torch
import time

# Use large matrix size to exaggerate the difference
size = 40960

# Set device
cpu = torch.device("cpu")
gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate two large random matrices
a_cpu = torch.randn(size, size, device=cpu)
b_cpu = torch.randn(size, size, device=cpu)

a_gpu = torch.randn(size, size, device=gpu)
b_gpu = torch.randn(size, size, device=gpu)

# Warm-up (important for fair GPU timing)
_ = torch.mm(a_gpu, b_gpu)

# Time CPU
start = time.time()
_ = torch.mm(a_cpu, b_cpu)
cpu_time = time.time() - start

# Time GPU (sync ensures accurate timing)
torch.cuda.synchronize()
start = time.time()
_ = torch.mm(a_gpu, b_gpu)
torch.cuda.synchronize()
gpu_time = time.time() - start

print(f"CPU time: {cpu_time:.4f} seconds")
print(f"GPU time: {gpu_time:.4f} seconds")
