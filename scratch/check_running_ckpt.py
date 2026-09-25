import torch

ckpt_path = r'c:\Ngoding\xau_deep_sniper\checkpoints\best_moment_pretrained_lora.pt'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

print(f"Epoch: {ckpt.get('epoch')}")
print(f"Composite Score: {ckpt.get('composite_score'):.4f}")
vm = ckpt.get('val_metrics', {})
print(f"Accuracy: {vm.get('accuracy'):.4f}")
print(f"Macro F1: {vm.get('macro_f1'):.4f}")
print(f"Non-HOLD Signals: {vm.get('non_hold_signals')}")
print(f"Non-HOLD Precision: {vm.get('non_hold_precision'):.4f}")
print(f"Directional Accuracy: {vm.get('directional_accuracy'):.4f}")
print(f"Val Loss: {vm.get('loss'):.4f}")
print("Per Class Breakdown:")
for c, s in vm.get('per_class', {}).items():
    print(f"  Class {c}: TP={s.get('true_positive')} FP={s.get('false_positive')} Prec={s.get('precision'):.3f} Rec={s.get('recall'):.3f} F1={s.get('f1'):.3f}")
