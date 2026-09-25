import pathlib

p = pathlib.Path(r'c:\Ngoding\xau_deep_sniper\scripts\run_feature_pipeline.py')
text = p.read_text(encoding='utf-8')

old_assert = 'assert X_train_sample.shape == (200 - L + 1, 9, L), "Shape mismatch!"'
new_assert = 'assert X_train_sample.shape == (200 - L + 1, fcfg.NUM_CHANNELS, L), "Shape mismatch!"'

if old_assert in text:
    text = text.replace(old_assert, new_assert)
    p.write_text(text, encoding='utf-8')
    print("Fixed line 99 assertion in run_feature_pipeline.py!")
else:
    print("Warning: old_assert not found!")
