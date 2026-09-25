import json

with open(r'c:\Ngoding\xau_deep_sniper\reports\model_training_report.json', 'r') as f:
    d = json.load(f)

th = d.get('training_history', {})
epochs = th.get('epoch', [])
print(f"Total epochs trained: {len(epochs)}")
print(f"Epoch list: {epochs}")
print(f"Train loss: {th.get('train_loss')}")
print(f"Val loss: {th.get('val_loss')}")
print(f"Val Dir Acc: {th.get('val_dir_acc')}")
print(f"Val Non-Hold Prec: {th.get('val_non_hold_prec')}")
