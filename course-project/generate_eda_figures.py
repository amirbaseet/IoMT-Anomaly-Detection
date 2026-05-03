"""
Course Project — EDA figürlerini üret (Bölüm 3 için).

Bu script eda_output/train_cleaned.csv dosyasını kullanır.
Avantajı: orijinal değerlerle çalışır (standardize edilmemiş),
            histogramlar ve Cohen's d daha okunabilir olur.

Üretilenler:
- fig01_class_distribution.png    : Sınıf dağılımı log scale bar chart
- fig02_feature_histograms.png    : Top 5 özellik histogramları
- fig03_correlation_heatmap.png   : 44x44 korelasyon heatmap
- fig04_cohens_d.png              : Cohen's d top 10
- ../cohens_d_values.csv          : Tüm özellikler için Cohen's d (rapor için)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Görsel ayarları
# ============================================================
plt.rcParams.update({
    'figure.dpi': 100,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'font.size': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# ============================================================
# Yollar
# ============================================================
PROJECT_ROOT = os.path.expanduser('~/IoMT-Project')
DATA_PATH = f'{PROJECT_ROOT}/eda_output/train_cleaned.csv'
OUT_DIR = f'{PROJECT_ROOT}/course-project/figures'
PARENT_DIR = f'{PROJECT_ROOT}/course-project'

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Veriyi yükle (subsample — bellek için)
# ============================================================
print('Veri yükleniyor (subsample 500K satır — bellek dostu)...')

# 4.5M satır çok fazla; rastgele subsample yap (görsel için yeterli)
SAMPLE_SIZE = 500_000

# pandas'ın skiprows trick'i ile rastgele subsample
total_rows = 4_515_081
np.random.seed(42)
skip = np.random.choice(
    np.arange(1, total_rows + 1),
    size=total_rows - SAMPLE_SIZE,
    replace=False
)
df = pd.read_csv(DATA_PATH, skiprows=skip)

print(f'  Yüklenen örnek sayısı: {len(df):,}')
print(f'  Sütun sayısı: {len(df.columns)}')

# 44 feature + label + category + split (sondaki 3 sütun feature değil)
NON_FEATURE_COLS = ['label', 'category', 'split']
feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
print(f'  Feature sayısı: {len(feature_cols)}')
print(f'  Sınıf sayısı: {df["label"].nunique()}')

# Drate sıfır varyans — kontrol
if 'Drate' in df.columns:
    drate_var = df['Drate'].var()
    print(f'  Drate varyans: {drate_var:.6f}')
    if drate_var == 0:
        print('  → Drate sıfır varyans, EDA için çıkarıyorum.')
        df = df.drop(columns=['Drate'])
        feature_cols = [c for c in feature_cols if c != 'Drate']

print(f'  Final feature sayısı: {len(feature_cols)}')

X = df[feature_cols]
y = df['label']

# ============================================================
# Figür 1: Sınıf Dağılımı
# ============================================================
print('\n[1/4] Figür 1: Sınıf dağılımı oluşturuluyor...')

class_counts = y.value_counts().sort_values(ascending=True)

# Renk fonksiyonu
def get_color(name):
    if name == 'Benign':
        return '#1D9E75'  # teal
    if name.startswith('DDoS'):
        return '#A32D2D'  # red
    if name.startswith('DoS'):
        return '#E24B4A'  # light red
    if name.startswith('Recon'):
        return '#EF9F27'  # amber
    if name.startswith('MQTT'):
        return '#534AB7'  # purple
    if name.startswith('ARP'):
        return '#D85A30'  # coral
    return '#888780'

colors = [get_color(name) for name in class_counts.index]

fig, ax = plt.subplots(figsize=(11, 9))
bars = ax.barh(
    range(len(class_counts)),
    class_counts.values,
    color=colors,
    edgecolor='black',
    linewidth=0.4,
)
ax.set_yticks(range(len(class_counts)))
ax.set_yticklabels(class_counts.index, fontsize=10)
ax.set_xscale('log')
ax.set_xlabel('Örnek Sayısı (logaritmik ölçek)', fontsize=11)
ax.set_title(
    f'CICIoMT2024 — 19 Sınıfın Dağılımı (n={SAMPLE_SIZE:,} subsample)',
    fontsize=13,
    pad=15,
)

# Bar değerleri
for i, (bar, val) in enumerate(zip(bars, class_counts.values)):
    ax.text(val * 1.05, i, f'{val:,}', va='center', fontsize=9)

# Lejant
from matplotlib.patches import Patch

legend = [
    Patch(facecolor='#1D9E75', label='Benign'),
    Patch(facecolor='#A32D2D', label='DDoS'),
    Patch(facecolor='#E24B4A', label='DoS'),
    Patch(facecolor='#EF9F27', label='Recon'),
    Patch(facecolor='#534AB7', label='MQTT'),
    Patch(facecolor='#D85A30', label='ARP Spoofing'),
]
ax.legend(handles=legend, loc='lower right', framealpha=0.95)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig01_class_distribution.png')
plt.close()
print(f'  ✓ Kaydedildi: fig01_class_distribution.png')

# ============================================================
# Figür 2: Top 5 Özellik Histogramları
# ============================================================
print('\n[2/4] Figür 2: Özellik histogramları oluşturuluyor...')

# Top 5 ayırıcı özellik (Cohen's d hesaplanmadı henüz, ama bilinen güçlü olanlar)
top_features = ['Rate', 'IAT', 'Header_Length', 'Tot size', 'Duration']
top_features = [f for f in top_features if f in X.columns]

if len(top_features) < 5:
    print(f'  Sadece {len(top_features)} özellik bulundu, hepsini kullanıyorum')

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

benign_mask = (y == 'Benign')

for idx, feat in enumerate(top_features):
    ax = axes[idx]

    benign_vals = X.loc[benign_mask, feat].values
    attack_vals = X.loc[~benign_mask, feat].values

    # Outlier'ları kırp — log scale yerine
    p99 = np.percentile(np.concatenate([benign_vals, attack_vals]), 99.5)
    p1 = np.percentile(np.concatenate([benign_vals, attack_vals]), 0.5)
    bins = np.linspace(p1, p99, 60)

    ax.hist(
        benign_vals,
        bins=bins,
        alpha=0.6,
        label='Benign',
        color='#1D9E75',
        edgecolor='black',
        linewidth=0.3,
        density=True,
    )
    ax.hist(
        attack_vals,
        bins=bins,
        alpha=0.6,
        label='Saldırı',
        color='#A32D2D',
        edgecolor='black',
        linewidth=0.3,
        density=True,
    )

    ax.set_title(feat, fontsize=12)
    ax.set_xlabel('Değer (orijinal)')
    ax.set_ylabel('Yoğunluk')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

# Boş subplot
if len(top_features) < 6:
    axes[5].axis('off')

plt.suptitle('Top 5 Özelliğin Saldırı vs Benign Dağılımları', fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig02_feature_histograms.png')
plt.close()
print(f'  ✓ Kaydedildi: fig02_feature_histograms.png')

# ============================================================
# Figür 3: Korelasyon Heatmap
# ============================================================
print('\n[3/4] Figür 3: Korelasyon heatmap oluşturuluyor...')

# Subsample içinden de daha küçük bir örnek (heatmap için 100K yeterli)
heatmap_sample = X.sample(min(100_000, len(X)), random_state=42)
corr = heatmap_sample.corr()

n_features = len(feature_cols)

fig, ax = plt.subplots(figsize=(15, 13))
sns.heatmap(
    corr,
    cmap='RdBu_r',
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.2,
    cbar_kws={'shrink': 0.7, 'label': 'Pearson Korelasyon Katsayısı'},
    ax=ax,
    xticklabels=True,
    yticklabels=True,
)

ax.set_title(
    f'{n_features} Özellik Arasındaki Pearson Korelasyon Matrisi\n'
    f'(n={len(heatmap_sample):,} örnek)',
    fontsize=13,
    pad=15,
)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig03_correlation_heatmap.png')
plt.close()
print(f'  ✓ Kaydedildi: fig03_correlation_heatmap.png')

# ============================================================
# Figür 4: Cohen's d Top 10
# ============================================================
print('\n[4/4] Figür 4: Cohen\'s d hesaplanıyor...')

benign_data = X[benign_mask]
attack_data = X[~benign_mask]

cohens_d = {}
for col in X.columns:
    m1, m2 = attack_data[col].mean(), benign_data[col].mean()
    s1, s2 = attack_data[col].std(), benign_data[col].std()
    n1, n2 = len(attack_data), len(benign_data)

    if n1 + n2 - 2 > 0:
        pooled_var = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
        pooled = np.sqrt(pooled_var) if pooled_var > 0 else 0
        d = abs(m1 - m2) / pooled if pooled > 0 else 0
    else:
        d = 0

    cohens_d[col] = d

# Top 10
top10 = sorted(cohens_d.items(), key=lambda x: x[1], reverse=True)[:10]
features_top10 = [t[0] for t in top10][::-1]
values_top10 = [t[1] for t in top10][::-1]

fig, ax = plt.subplots(figsize=(11, 7))

# Renk: d büyüklüğüne göre
colors_d = []
for d in values_top10:
    if d > 2.0:
        colors_d.append('#04342C')  # Olağanüstü
    elif d > 1.0:
        colors_d.append('#0F6E56')  # Çok büyük
    elif d > 0.8:
        colors_d.append('#1D9E75')  # Büyük
    else:
        colors_d.append('#5DCAA5')  # Orta

bars = ax.barh(
    features_top10,
    values_top10,
    color=colors_d,
    edgecolor='black',
    linewidth=0.4,
)

# Eşik çizgileri
ax.axvline(x=0.5, linestyle=':', color='gray', alpha=0.5, label='Orta etki (d=0.5)')
ax.axvline(x=0.8, linestyle='--', color='gray', alpha=0.7, label='Büyük etki (d=0.8)')
ax.axvline(x=2.0, linestyle='-.', color='red', alpha=0.7, label='Olağanüstü (d=2.0)')

# Bar değerleri
for bar, val in zip(bars, values_top10):
    ax.text(
        val + max(values_top10) * 0.02,
        bar.get_y() + bar.get_height() / 2,
        f'{val:.2f}',
        va='center',
        fontsize=10,
        fontweight='bold',
    )

ax.set_xlabel("Cohen's d (mutlak değer)", fontsize=11)
ax.set_title(
    "Saldırı vs Benign — Top 10 Ayırıcı Özellik (Cohen's d Etki Büyüklüğü)",
    fontsize=13,
    pad=15,
)
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig04_cohens_d.png')
plt.close()
print(f'  ✓ Kaydedildi: fig04_cohens_d.png')

# Cohen's d değerlerini CSV'ye kaydet
cohens_df = pd.DataFrame(
    sorted(cohens_d.items(), key=lambda x: x[1], reverse=True),
    columns=['Özellik', 'Cohens_d'],
)
cohens_df.to_csv(f'{PARENT_DIR}/cohens_d_values.csv', index=False)
print(f'  ✓ Cohen\'s d değerleri: {PARENT_DIR}/cohens_d_values.csv')

# ============================================================
# Özet
# ============================================================
print('\n' + '=' * 60)
print('TÜM EDA FIGÜRLERI ÜRETILDI!')
print(f'Kayıt yeri: {OUT_DIR}')
print('=' * 60)
print('\nÜretilen figürler:')
print('  ✓ fig01_class_distribution.png')
print('  ✓ fig02_feature_histograms.png')
print('  ✓ fig03_correlation_heatmap.png')
print('  ✓ fig04_cohens_d.png')
print('\nBonus dosya:')
print('  ✓ cohens_d_values.csv (rapora değerler için)')
print('\n🎯 Sonraki adım: Markdown rapora figürleri göm.')