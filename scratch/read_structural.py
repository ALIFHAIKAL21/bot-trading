with open(r'c:\Ngoding\xau_deep_sniper\src\pipeline\features_structural.py', 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    print(f"{i+1:3d}: {line}", end='')
