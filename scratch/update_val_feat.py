import pathlib

p_fp = pathlib.Path(r'c:\Ngoding\xau_deep_sniper\src\pipeline\feature_pipeline.py')
text = p_fp.read_text(encoding='utf-8')

old_code = '''    # Validate technical features
    warmup = get_warmup_period()
    for ch in ["smi", "ma_ribbon_slope"]:
        if ch not in df.columns:
            report[ch] = {"status": "MISSING"}
            continue
        data = df[ch].iloc[warmup:].dropna()
        report[ch] = {
            "count": len(data),
            "mean": round(float(data.mean()), 4),
            "std": round(float(data.std()), 4),
            "min": round(float(data.min()), 4),
            "max": round(float(data.max()), 4),
            "inf_count": int(np.isinf(data).sum()),
            "status": "PASS" if not np.isinf(data).any() else "FAIL",
            "issues": ["INF_VALUES"] if np.isinf(data).any() else [],
        }
    
    # Validate structural features
    for ch in ["liquidity_distance", "fvg_status"]:
        if ch not in df.columns:
            report[ch] = {"status": "MISSING"}
            continue
        data = df[ch].iloc[warmup:].dropna()
        report[ch] = {
            "count": len(data),
            "mean": round(float(data.mean()), 4),
            "std": round(float(data.std()), 4),
            "min": round(float(data.min()), 4),
            "max": round(float(data.max()), 4),
            "inf_count": int(np.isinf(data).sum()),
            "status": "PASS" if not np.isinf(data).any() else "FAIL",
            "issues": ["INF_VALUES"] if np.isinf(data).any() else [],
        }'''

new_code = '''    # Validate all 12 feature channels
    warmup = get_warmup_period()
    for ch_idx, ch in fcfg.FEATURE_CHANNELS.items():
        if ch in report:
            continue
        if ch not in df.columns:
            report[ch] = {"status": "MISSING"}
            continue
        data = df[ch].iloc[warmup:].dropna()
        report[ch] = {
            "count": len(data),
            "mean": round(float(data.mean()), 4),
            "std": round(float(data.std()), 4),
            "min": round(float(data.min()), 4),
            "max": round(float(data.max()), 4),
            "inf_count": int(np.isinf(data).sum()),
            "status": "PASS" if not np.isinf(data).any() else "FAIL",
            "issues": ["INF_VALUES"] if np.isinf(data).any() else [],
        }'''

if old_code in text:
    text = text.replace(old_code, new_code)
    p_fp.write_text(text, encoding='utf-8')
    print("Updated validate_all_features to validate all 12 channels!")
else:
    print("Warning: old_code not found in feature_pipeline.py!")
