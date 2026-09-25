with open(r'c:\Ngoding\xau_deep_sniper\src\models\moment_model.py', 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()

idx = 0
for i, line in enumerate(lines):
    if "class PretrainedMOMENTClassifier" in line:
        idx = i
        break

for i in range(idx, min(idx + 70, len(lines))):
    print(f"{i+1:3d}: {lines[i]}", end='')
