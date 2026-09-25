with open(r'c:\Ngoding\xau_deep_sniper\src\pipeline\feature_pipeline.py', 'r', encoding='utf-8-sig') as f:
    text = f.read()

idx = text.find('def validate_all_features')
print(text[idx:idx+1500])
