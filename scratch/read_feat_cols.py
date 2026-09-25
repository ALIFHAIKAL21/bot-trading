with open(r'c:\Ngoding\xau_deep_sniper\src\pipeline\feature_pipeline.py', 'r', encoding='utf-8-sig') as f:
    text = f.read()

idx = text.find('def get_feature_columns')
if idx != -1:
    print(text[idx:idx+800])
else:
    print("Not found")
