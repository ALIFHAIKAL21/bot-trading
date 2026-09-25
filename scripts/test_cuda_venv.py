import torch

print("Device Name:", torch.cuda.get_device_name(0))
vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"Total VRAM: {vram:.2f} GB")

# Test tensor operations on GPU
x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
y = x @ x
torch.cuda.synchronize()
print("GPU Float16 Matmul (4096 x 4096): Success!")
print(f"Allocated VRAM: {torch.cuda.memory_allocated() / (1024*1024):.1f} MB")
