import pathlib

# 1. Update config.py to define CHECKPOINTS_DIR
p_cfg = pathlib.Path(r'c:\Ngoding\xau_deep_sniper\src\pipeline\config.py')
cfg_text = p_cfg.read_text(encoding='utf-8')
if "CHECKPOINTS_DIR" not in cfg_text:
    cfg_text += '\nCHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"\n'
    p_cfg.write_text(cfg_text, encoding='utf-8')
    print("Added CHECKPOINTS_DIR to config.py!")

# 2. Update train_moment_classifier.py
p_train = pathlib.Path(r'c:\Ngoding\xau_deep_sniper\scripts\train_moment_classifier.py')
text = p_train.read_text(encoding='utf-8')

old_block = '''    # Setup Cosine Annealing with Warmup Scheduler
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = (len(train_loader) // args.grad_accum) * args.warmup_epochs

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(1e-6 / args.lr, 0.5 * (1.0 + np.cos(np.pi * progress)))

    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lr_lambda)

    # 6. Execute Training
    checkpoint_file = config.CHECKPOINTS_DIR / chk_name
    print(f"\\n[6/6] Launching Training ({args.epochs} epochs, warmup {args.warmup_epochs})...")
    print(f"      Target Checkpoint: {checkpoint_file}")
    t_train_start = time.time()

    train_results = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=args.epochs,
        best_checkpoint_file=checkpoint_file,
        verbose=True,
    )'''

new_block = '''    # 6. Execute Training via Institutional Trainer
    chk_dir = project_root / "checkpoints"
    chk_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = chk_dir / chk_name

    print(f"\\n[6/6] Launching Training ({args.epochs} epochs, warmup {args.warmup_epochs})...")
    print(f"      Target Checkpoint: {checkpoint_file}")
    t_train_start = time.time()

    train_results = trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        checkpoint_dir=str(chk_dir),
        checkpoint_name=chk_name,
        verbose=True,
    )'''

if old_block in text:
    text = text.replace(old_block, new_block)
    p_train.write_text(text, encoding='utf-8')
    print("Fixed train_moment_classifier.py to call trainer.fit correctly!")
else:
    print("Warning: old_block not found in train_moment_classifier.py!")
