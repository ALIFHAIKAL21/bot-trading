import pathlib

p = pathlib.Path(r'c:\Ngoding\xau_deep_sniper\src\models\trainer.py')
text = p.read_text(encoding='utf-8')

old_score = '''            # Simpan model terbaik berdasarkan Institutional Composite Score (Directional Accuracy 60% + Macro F1 40%)
            dir_acc = val_metrics["directional_accuracy"]
            f1 = val_metrics["macro_f1"]
            composite_score = (dir_acc * 0.6 + f1 * 0.4) if dir_acc > 0 else f1 * 0.4

            is_best = (composite_score > best_composite_score) if best_composite_score >= 0 else True'''

new_score = '''            # Simpan model terbaik berdasarkan Institutional Composite Score
            # Activity Threshold Gate: Wajib minimal 500 sinyal non-HOLD di data validasi (11k bar)
            # untuk mencegah jebakan HOLD Collapse (model cuma trade belasan kali)
            total_signals = val_metrics["non_hold_signals"]
            min_signals = 500
            dir_acc = val_metrics["directional_accuracy"]
            signal_prec = val_metrics["non_hold_precision"]
            f1 = val_metrics["macro_f1"]

            if total_signals >= min_signals:
                composite_score = (signal_prec * 0.4 + dir_acc * 0.4 + f1 * 0.2)
            else:
                composite_score = 0.0  # Diskualifikasi model yang malas / pasif (< 500 sinyal)

            is_best = (composite_score > best_composite_score and composite_score > 0.0)'''

if old_score in text:
    text = text.replace(old_score, new_score)
    p.write_text(text, encoding='utf-8')
    print("Updated trainer.py with Activity Threshold Gate!")
else:
    print("Warning: old_score not found in trainer.py!")
