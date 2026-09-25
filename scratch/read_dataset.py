with open(r'c:\Ngoding\xau_deep_sniper\src\models\dataset.py', 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()

for i in range(80, min(140, len(lines))):
    print(f"{i+1:3d}: {lines[i]}", end='')
