import matplotlib.pyplot as plt
import numpy as np

periods = ['2021', '2022', '2023', '2024', '2025', '2026 (Q1)']
baseline = [6.19, 38.01, -9.23, -11.55, 23.29, 10.12]
model_15ch_tau38 = [24.70, 113.59, 57.56, 67.39, 61.30, 5.79]
model_15ch_tau40 = [18.58, 86.19, 81.81, 40.34, 36.74, 10.78]

x = np.arange(len(periods))
width = 0.28

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(13, 6.5), dpi=150)

rects1 = ax.bar(x - width, baseline, width, label='1. Baseline (12 Channel M30)', color='#4a5568')
rects2 = ax.bar(x, model_15ch_tau38, width, label='2. Model Baru 15-Channel (Tau = 0.380)', color='#3182ce')
rects3 = ax.bar(x + width, model_15ch_tau40, width, label='3. Model Baru 15-Channel (Tau = 0.400 Sniper)', color='#38a169')

ax.set_ylabel('Annual Return (%)', fontsize=12, fontweight='bold', color='#e2e8f0')
ax.set_title('AUDIT INSTITUSIONAL MENDALAM: Baseline 12-Ch vs Model Baru 15-Ch (M30 + H4 Macro)\nTransformasi Performa 2021 - 2026 Tanpa Kompromi', fontsize=13, fontweight='bold', color='#ffffff', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(periods, fontsize=11, fontweight='bold')
ax.axhline(0, color='#718096', linewidth=1, linestyle='--')
ax.legend(frameon=True, facecolor='#1a202c', edgecolor='#4a5568', fontsize=10)
ax.grid(axis='y', linestyle=':', alpha=0.3)

for rects in [rects1, rects2, rects3]:
    for rect in rects:
        h = rect.get_height()
        va = 'bottom' if h >= 0 else 'top'
        ax.annotate(f'{h:+.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3 if h >= 0 else -8),
                    textcoords="offset points",
                    ha='center', va=va, fontsize=8, fontweight='bold')

plt.tight_layout()
out_path = r'c:\Ngoding\xau_deep_sniper\reports\img\audit_15ch_vs_baseline.png'
plt.savefig(out_path, dpi=150)
print(f"Chart saved to {out_path}")
