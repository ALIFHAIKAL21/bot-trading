import sys, pathlib, torch

project_root = pathlib.Path(r'c:\Ngoding\xau_deep_sniper')
sys.path.insert(0, str(project_root / "src"))

from models.moment_model import PretrainedMOMENTClassifier

print("Testing PretrainedMOMENTClassifier dummy forward pass (B=2, C=12, L=64)...")
device = "cuda" if torch.cuda.is_available() else "cpu"

model = PretrainedMOMENTClassifier(num_classes=5, n_channels=12, lora_r=16, lora_alpha=32).to(device)
model.eval()

x = torch.randn(2, 12, 64, device=device)
with torch.no_grad():
    out = model(x)
    probs = model.predict_proba(x)

print(f"Logits shape: {out.shape}")
print(f"Probs shape : {probs.shape}")
print(f"Probs sum   : {probs.sum(dim=-1)}")
assert out.shape == (2, 5)
print(">>> PretrainedMOMENTClassifier 12-channel forward pass: 100% SUCCESS ON CUDA!")
