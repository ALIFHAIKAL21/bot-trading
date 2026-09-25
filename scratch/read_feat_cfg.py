with open(r'c:\Ngoding\xau_deep_sniper\src\pipeline\feature_config.py', 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()

for i in range(50, min(120, len(lines))):
    print(f"{i+1:3d}: {lines[i]}", end='')
