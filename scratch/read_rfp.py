with open(r'c:\Ngoding\xau_deep_sniper\scripts\run_feature_pipeline.py', 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()

for i in range(85, min(125, len(lines))):
    print(f"{i+1:3d}: {lines[i]}", end='')
