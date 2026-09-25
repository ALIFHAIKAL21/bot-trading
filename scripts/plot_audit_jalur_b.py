import matplotlib.pyplot as plt
import numpy as np

years = ['2021', '2022', '2023', '2024', '2025']
baseline = [5.05, 38.36, -7.98, -11.18, 21.14]
jalur_b_naive = [7.48, 25.30, -2.01, 2.83, 17.38]
jalur_b_proper = [6.19, 40.70, -4.43, -7.84, 23.60]
jalur_b_master = [5.38, 41.36, -1.43, -7.87, 26.44]

x = np.arange(len(years))
width = 0.20

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

rects1 = ax.bar(x - 1.5*width, baseline, width, label='1. Baseline (M30 Asli)', color='#4a5568')
rects2 = ax.bar(x - 0.5*width, jalur_b_naive, width, label='2. Jalur B Naif (Memotong Trade - Merusak 2022/2025)', color='#e53e3e')
rects3 = ax.bar(x + 0.5*width, jalur_b_proper, width, label='3. Jalur B Benar (Asymmetric Risk - 0 Trade Dibuang)', color='#3182ce')
rects4 = ax.bar(x + 1.5*width, jalur_b_master, width, label='4. Jalur B Master (Surgical Gate + Runner)', color='#38a169')

ax.set_ylabel('Annual Return (%)', fontsize=12, fontweight='bold', color='#e2e8f0')
ax.set_title('AUDIT REALISTIS: Perbandingan Hasil Pengujian Jalur B (2021 - 2025)\nMembuktikan Batas Matematis Rule-Based vs Kebutuhan Retraining', fontsize=13, fontweight='bold', color='#ffffff', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(years, fontsize=11, fontweight='bold')
ax.axhline(0, color='#718096', linewidth=1, linestyle='--')
ax.legend(frameon=True, facecolor='#1a202c', edgecolor='#4a5568', fontsize=9)
ax.grid(axis='y', linestyle=':', alpha=0.3)

for rects in [rects1, rects2, rects3, rects4]:
    for rect in rects:
        h = rect.get_height()
        va = 'bottom' if h >= 0 else 'top'
        ax.annotate(f'{h:+.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3 if h >= 0 else -8),
                    textcoords="offset points",
                    ha='center', va=va, fontsize=7.5, fontweight='bold')

plt.tight_layout()
out_path = r'c:\Ngoding\xau_deep_sniper\reports\img\audit_jalur_b_comparison.png'
plt.savefig(out_path, dpi=150)
print(f"Chart saved to {out_path}")
