import pathlib

p = pathlib.Path(r'c:\Ngoding\xau_deep_sniper\scripts\train_moment_classifier.py')
text = p.read_text(encoding='utf-8')

# Update default args
text = text.replace('default=40, help="Number of fine-tuning epochs (default: 40)"', 'default=30, help="Number of fine-tuning epochs (default: 30)"')
text = text.replace('default=1.0, help="Minimum alpha weight for class 0 HOLD (default: 1.0)"', 'default=0.43, help="Minimum alpha weight for class 0 HOLD (default: 0.43 to prevent HOLD collapse)"')

# Update standalone config n_channels=9 -> 12
text = text.replace('n_channels=9,', 'n_channels=12,')

# Explicitly pass n_channels=12 to PretrainedMOMENTClassifier
old_model_call = '''        model = PretrainedMOMENTClassifier(
            num_classes=5,
            dropout=0.2,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
        )'''

new_model_call = '''        model = PretrainedMOMENTClassifier(
            num_classes=5,
            n_channels=12,
            dropout=0.2,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
        )'''

if old_model_call in text:
    text = text.replace(old_model_call, new_model_call)

p.write_text(text, encoding='utf-8')
print("Updated train_moment_classifier.py successfully!")
