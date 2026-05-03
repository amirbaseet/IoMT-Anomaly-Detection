"""
Course Project — Mevcut markdown raporda 'Görsel X' notlarını
gerçek figür satırları ile değiştir.

Kullanım:
    python embed_figures_in_report.py

Bu script Course_Project_FULL_REPORT.md dosyasını okur,
yer-tutucu satırları gerçek ![](figures/...) satırları ile değiştirir,
ve Course_Project_FULL_REPORT_with_figures.md olarak kaydeder.
"""

import os
import re

# ============================================================
# Yollar
# ============================================================
PROJECT_ROOT = os.path.expanduser('~/IoMT-Project')
COURSE_DIR = f'{PROJECT_ROOT}/course-project'

INPUT_MD = f'{COURSE_DIR}/Course_Project_FULL_REPORT.md'
OUTPUT_MD = f'{COURSE_DIR}/Course_Project_FULL_REPORT_with_figures.md'

# Eğer ana dosya yoksa hata
if not os.path.exists(INPUT_MD):
    print(f'HATA: {INPUT_MD} bulunamadı.')
    print('Önce Course_Project_FULL_REPORT.md dosyasını course-project/ klasörüne koy.')
    exit(1)

# ============================================================
# Figür değiştirme planı
# ============================================================
# Her tuple: (yer_tutucu_pattern, figür_satırı_değiştirilecek)
# Pattern'ler regex; en uzun-eşleşen-önce sıralı

REPLACEMENTS = [
    # Bölüm 3.1
    (
        r'> \*\*Görsel 1 \(zorunlu\):\*\*[^\n]*(\n>[^\n]*)*',
        '''![Şekil 1 — Sınıf Dağılımı](figures/fig01_class_distribution.png)

*Şekil 1. CICIoMT2024 eğitim setinde 19 sınıfın örnek sayıları (logaritmik ölçek). En çok temsil edilen sınıf Benign, en az temsil edilen sınıf ARP_Spoofing — yaklaşık 75:1 dengesizlik oranı görülmektedir.*'''
    ),

    # Bölüm 3.2
    (
        r'> \*\*Görsel 2 \(zorunlu\):\*\*[^\n]*(\n>[^\n]*)*',
        '''![Şekil 2 — Top 5 Özellik Histogramları](figures/fig02_feature_histograms.png)

*Şekil 2. En güçlü beş ayırıcı özelliğin (Rate, IAT, Header_Length, Tot size, Duration) saldırı ve benign sınıfları için yoğunluk dağılımları. Saldırı dağılımı (kırmızı) ile benign dağılımı (yeşil) arasında belirgin ayrışma görülmektedir; özellikle Rate ve IAT'da örtüşme oldukça sınırlıdır.*'''
    ),

    # Bölüm 3.3
    (
        r'> \*\*Görsel 3 \(zorunlu\):\*\*[^\n]*(\n>[^\n]*)*',
        '''![Şekil 3 — Korelasyon Heatmap](figures/fig03_correlation_heatmap.png)

*Şekil 3. 44 özellik arasındaki Pearson korelasyon matrisi. Dört yüksek-korelasyon kümesi belirgindir: paket boyutu istatistikleri (Min, Max, AVG, Std, Tot size, Tot sum, Magnitue, Radius), TCP flag sayaçları, hız özellikleri (Rate ↔ Srate, r ≈ 0.99) ve akış istatistikleri.*'''
    ),

    # Bölüm 3.4 — opsiyonel
    (
        r'> \*\*Görsel 4 \(opsiyonel\)[^\n]*(\n>[^\n]*)*',
        ''  # Opsiyonel görsel — atlanacak
    ),

    # Bölüm 3.5 — Cohen's d (Görsel 5 olarak yazılı)
    (
        r'> \*\*Görsel 5 \(zorunlu\):\*\*[^\n]*(\n>[^\n]*)*',
        '''![Şekil 4 — Cohen's d Top 10](figures/fig04_cohens_d.png)

*Şekil 4. Saldırı ve benign sınıfları arasında en güçlü ayırıcı 10 özellik (Cohen's d etki büyüklüğü). Tüm top 10 özellik d > 0.8 (büyük etki) eşiğini geçmektedir; Rate, IAT ve Srate gibi özellikler d > 2.0 ile olağanüstü güçlü ayırıcılar olarak öne çıkmaktadır.*'''
    ),

    # Bölüm 5.4 — Confusion matrix (Görsel 6)
    (
        r'> \*\*Görsel 6 \(zorunlu\):\*\*[^\n]*(\n>[^\n]*)*',
        '''![Şekil 5 — XGBoost Confusion Matrix](figures/fig05_xgb_confusion_matrix.png)

*Şekil 5. XGBoost (E7) modelinin 19-sınıf test setindeki normalized confusion matrix'i. Diagonal yüksek doğruluk (>0.95 çoğu sınıfta) görülmektedir; en belirgin karışıklıklar DDoS ile DoS aile saldırıları arasındadır.*'''
    ),

    # Bölüm 5.4.2 — Per-class F1 (Görsel 7)
    (
        r'> \*\*Görsel 7 \(önerilen\):\*\*[^\n]*(\n>[^\n]*)*',
        '''![Şekil 6 — E1-E8 Konfigürasyon Karşılaştırması](figures/fig06_e1_e8_comparison.png)

*Şekil 6. Sekiz deneysel konfigürasyonun (E1-E8) karşılaştırmalı performans grafiği. E7 (XGBoost + entropy + SMOTETomek + class_weight=balanced) en yüksek macro F1 skorunu elde etmiştir.*

![Şekil 7 — SMOTETomek Etkisi](figures/fig07_smote_effect.png)

*Şekil 7. SMOTETomek dengesizlik yönetimi tekniğinin az temsil edilen sınıflar üzerindeki etkisi. ARP_Spoofing, Recon alt türleri gibi nadir sınıflar için anlamlı performans iyileşmesi sağlamıştır.*'''
    ),

    # Bölüm 5.6 — SHAP/Feature importance (Görsel 8)
    (
        r'> \*\*Görsel 8 \(önerilen\):\*\*[^\n]*(\n>[^\n]*)*',
        '''![Şekil 8 — Random Forest Feature Importance](figures/fig08_feature_importance.png)

*Şekil 8. Random Forest modelinin en önemli 15 özelliği (feature importance gain skorları). EDA'daki Cohen's d sıralaması ile büyük ölçüde tutarlıdır; IAT ve Rate en üstte yer almaktadır.*

![Şekil 14 — SHAP Global Önemi](figures/fig14_shap_importance.png)

*Şekil 14. SHAP (mean absolute) global özellik önemi. Modelin karar sürecinde IAT ve Rate özellikleri en yüksek katkıyı sağlamaktadır.*'''
    ),

    # Bölüm 6.2 — AE mimarisi (Görsel 9 — bu mimari diyagramı, atla)
    (
        r'> \*\*Görsel 9 \(zorunlu\):\*\*[^\n]*(\n>[^\n]*)*',
        ''  # AE mimarisi text-art zaten var, ekstra görsel gerekmez
    ),

    # Bölüm 6.2.4 — AE Eğitim eğrisi (Görsel 10)
    (
        r'> \*\*Görsel 10 \(zorunlu\):\*\*[^\n]*(\n>[^\n]*)*',
        '''![Şekil 9 — Otoenkoder Eğitim Eğrisi](figures/fig09_ae_loss_curve.png)

*Şekil 9. Otoenkoder eğitim ve validation loss değerlerinin epoch'lara göre değişimi. Erken durdurma (early stopping) ile yaklaşık 50. epoch civarında en iyi model elde edilmiş; train ile validation arası farkın küçük olması overfitting bulunmadığını göstermektedir.*'''
    ),

    # Bölüm 6.3 — Recon error histogram (Görsel 11)
    (
        r'> \*\*Görsel 11 \(zorunlu\):\*\*[^\n]*(\n>[^\n]*)*',
        '''![Şekil 10 — AE Reconstruction Error Histogram](figures/fig10_ae_recon_error.png)

*Şekil 10. Test setinde Otoenkoder reconstruction error dağılımları. Benign akışlar düşük error bölgesinde, saldırılar p90 eşiğinin (0.2013) sağında yoğunlaşmaktadır. İki dağılım net ayrım göstermektedir.*'''
    ),

    # Bölüm 6.4.2 — Per-class detection (Görsel 12)
    (
        r'> \*\*Görsel 12 \(önerilen\):\*\*[^\n]*(\n>[^\n]*)*',
        '''![Şekil 11 — AE Per-Class Reconstruction Error Boxplot](figures/fig11_ae_per_class_boxplot.png)

*Şekil 11. 19 sınıf için Otoenkoder reconstruction error dağılımları (boxplot). DDoS aile saldırıları yüksek error medyanlarına sahipken, ARP_Spoofing ve MQTT_Malformed_Data benign'e daha yakın değerlerde kalmaktadır — bu zor sınıfları işaret eder.*

![Şekil 12 — Detection Rate Heatmap](figures/fig12_detection_rate_heatmap.png)

*Şekil 12. 18 saldırı türü için Otoenkoder detection rate heatmap'i. DDoS ve MQTT-DDoS aile saldırıları %95 üzerinde tespit edilirken, ARP_Spoofing ve MQTT_Malformed_Data %55-62 aralığında kalmaktadır.*'''
    ),

    # Bölüm 6.5 — ROC curves (Görsel 13)
    (
        r'> \*\*Görsel 13 \(önerilen\):\*\*[^\n]*(\n>[^\n]*)*',
        '''![Şekil 13 — AE vs Isolation Forest ROC Eğrileri](figures/fig13_ae_vs_if_roc.png)

*Şekil 13. Otoenkoder ve Isolation Forest modellerinin ROC eğrileri. Otoenkoder AUC = 0.9892, Isolation Forest AUC = 0.9543 — yaklaşık 3.5 puanlık fark gözlemlenmektedir. Otoenkoder özellikle yüksek hacimli flood saldırılarında üstünlük sağlamaktadır.*'''
    ),

    # Bölüm 6.6 — AE vs IF (Görsel 14 — bu da AE vs IF için ama farklı ifade ile yazılmış)
    (
        r'> \*\*Görsel 14 \(önerilen\):\*\*[^\n]*(\n>[^\n]*)*',
        ''  # Şekil 13 ile zaten kapsanıyor
    ),
]

# ============================================================
# Replace işlemi
# ============================================================
print(f'Okunuyor: {INPUT_MD}')

with open(INPUT_MD, 'r', encoding='utf-8') as f:
    content = f.read()

original_size = len(content)

# Her replacement'ı uygula
for pattern, replacement in REPLACEMENTS:
    matches = re.findall(pattern, content, flags=re.MULTILINE)
    if matches:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        print(f'  ✓ Değiştirildi: {pattern[:60]}...')
    else:
        print(f'  ⚠ Bulunamadı: {pattern[:60]}... (atlanıyor)')

# Kaydet
with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
    f.write(content)

new_size = len(content)
diff = new_size - original_size

print(f'\nKaydedildi: {OUTPUT_MD}')
print(f'Boyut: {original_size:,} byte → {new_size:,} byte (fark: {diff:+,})')
print('\n' + '='*60)
print('TAMAM!')
print('='*60)
print(f'\nYeni dosya: {OUTPUT_MD}')
print('VS Code\'da bu dosyayı aç ve "Markdown PDF: Export PDF" yap.')
print('\nNot: Orijinal dosya değişmedi (Course_Project_FULL_REPORT.md).')
