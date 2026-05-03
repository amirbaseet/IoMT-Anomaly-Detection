# IoMT Ağlarında Hibrit Saldırı Tespit Sistemi

Bu proje, **CICIoMT2024** veri seti üzerinde Internet of Medical Things (IoMT) ağlarında siber saldırıları tespit etmek için **denetimli (XGBoost)** ve **denetimsiz (Otoenkoder)** makine öğrenmesi katmanlarını ayrı ayrı analiz eder. Veri seti **5.4 milyon ağ akışı**, **19 saldırı/normal sınıfı** ve **44 özellik** içerir. Tıbbi cihaz ağlarında saldırı tespiti, hasta güvenliği ve cihaz uyumluluğu açısından kritik bir savunma katmanıdır; bu çalışma iki yaklaşımın **tamamlayıcılığını** ampirik olarak doğrular.

## Yazarlar ve Bağlam

| | |
|---|---|
| **Üniversite** | Sakarya Üniversitesi |
| **Ders** | Siber Güvenlik Analitiği |
| **Dönem** | 2025-2026 Bahar |
| **Sunum Tarihi** | 5 Mayıs 2026 |
| **Yazarlar** | AMRO MOUSA ISMAIL BASEET (Y255012028) · MOTAZ ARMASH (Y255012163) |

## Teslim Edilen Çıktılar

```
course-project/
├── Course_Project_FULL_REPORT_FINAL.md     (~50 sayfa, 10 bölüm)
├── Course_Project_FULL_REPORT_FINAL.pdf    (raporun PDF sürümü)
├── notebooks/
│   ├── course_project_demo.ipynb            (54 hücre, ~7 dakika çalıştırma)
│   ├── PRESENTATION_GUIDE.md                (15 dakikalık sunum akışı önerisi)
│   ├── README.md                            (bug-report retrospektifi)
│   └── test_log.txt                         (son nbconvert verifikasyonu)
└── figures/                                  (14 yayın-kalitesinde figür)
```

GitHub: <https://github.com/amirbaseet/IoMT-Anomaly-Detection/tree/main/course-project>

## Ana Bulgular

Tüm sayısal değerler `results/` ve `eda_output/` altındaki kanonik dosyalardan yüklenir.

### Birincil — Otoenkoder vs Isolation Forest = +12.8 pp AUC

| Metrik | Otoenkoder | Isolation Forest | Fark |
|---|---:|---:|---:|
| AUC (ROC, test) | **0.9892** | 0.8612 | **+12.8 pp** |
| Per-class avg detection | **%80.0** | %16.3 | **+63.7 pp** |

Kaynak: `results/unsupervised/metrics/model_comparison.csv` ve canlı hesaplama. Bu fark hibrit yaklaşımda Otoenkoder seçiminin **ampirik gerekçesidir**.

### İkincil — SMOTETomek 4/4 konfigürasyonda macro F1'i düşürdü

`results/supervised/metrics/overall_comparison.csv` üzerinden:

| Konfigürasyon çifti | Original → SMOTE | Fark (pp) |
|---|---|---:|
| RF · reduced (E1→E2) | 0.847 → 0.836 | -1.1 |
| RF · full (E5→E6) | 0.855 → 0.838 | -1.7 |
| XGB · reduced (E3→E4) | 0.899 → 0.854 | -4.5 |
| XGB · full (E7→E8) | 0.908 → 0.871 | -3.7 |

"Sentetik azınlık örnekleme her zaman yardım eder" varsayımına aykırı bir gözlem; IDS literatüründe yeterince vurgulanmamıştır.

### Üçüncül — Model swap (RF→XGB) ARP_Spoofing'te +25 pp

ARP_Spoofing F1: Random Forest seviyesinde ≈ 0.50 (E5: 0.502, E5G: 0.495); XGBoost (E7) ile **0.758** — **+25 pp** kazanım. Bu, çalışmadaki tek-sınıf bazlı en büyük model-kaynaklı iyileşmedir.

Kaynak: `results/supervised/metrics/E5_classification_report_test.json`, `E7_classification_report_test.json`.

### Final XGBoost (E7) Performansı — Bilinen 19 Sınıf

XGBoost (E7) test seti üzerinde:

- **%99.27 accuracy** · **0.9076 macro F1** · **0.9906 MCC**

Kaynak: `results/supervised/metrics/E7_multiclass.json`.

### Null Sonuç — Entropy vs Gini Criterion

Random Forest hiperparametrelerinde tek farkı `criterion` parametresi olan iki versiyon eğitildi (E5 entropy, E5G gini). Macro F1 farkı yalnızca **+0.47 pp** — istatistiksel gürültü mertebesinde. scikit-learn dokümantasyonunun "iki criterion benzer ağaçlar üretir" ifadesi bu veri setinde de doğrulandı. Negatif sonuç, raporun §5.5'inde *metodolojik disiplin örneği* olarak şeffafça raporlandı.

Kaynak: `results/supervised/metrics/E5_vs_E5G_comparison.csv`, üreten betik `scripts/run_e5g_gini_baseline.py`.

## Sınırlamalar

XGBoost (E7) modeli **dört sınıfta F1 < 0.85** ile zorlanır. Tüm bu sınıfların test support değeri **n ≤ 2,941** — sınırlama büyük ölçüde modelin değil, verinin istatistiksel güç sınırlarındandır.

| Sınıf | F1 | Test Support |
|---|---:|---:|
| Recon_VulScan | 0.45 | 973 |
| Recon_OS_Scan | 0.69 | 2,941 |
| ARP_Spoofing | 0.76 | 1,744 |
| Recon_Ping_Sweep | 0.78 | 169 |

Otoenkoder tarafında **MQTT_DoS_Publish_Flood detection rate'i %6.7**'ye düşmektedir — içerik (payload) tabanlı bir anomali türü olduğundan akış istatistikleri (CICIoMT2024'ün özellik kümesi) bu sınıfı yetersiz temsil eder.

Test edilmemiş alanlar: adversarial robustness, uzun-dönem drift toleransı, paket-paket streaming senaryosu (yalnızca batch-mode 22.8 ms / 1000 örnek ölçülmüştür).

Kaynak: `results/supervised/metrics/E7_classification_report_test.json`, `results/unsupervised/metrics/per_class_detection_rates.csv`.

## Reproducibility

| Betik / dosya | Amaç |
|---|---|
| `scripts/run_e5g_gini_baseline.py` | RF-Gini A/B baseline (E5G); ~3 dakikada yeniden üretilebilir |
| `scripts/verify_report_numbers.py` | Kalıcı QA aracı — raporun **her** numerical claim'ini canonical dosyalara karşı çapraz doğrular (~2,200 entry registry) |
| `scripts/verification_report.md` | Son doğrulama denetiminin sonucu (GREEN/YELLOW/RED triage) |
| `notebooks/course_project_demo.ipynb` | 54-hücreli sunum notebook'u; kernel-restart-clean |

Tüm canonical metric dosyaları `results/` ve `eda_output/` dizinlerinde versiyon kontrolündedir.

## Notebook Çalıştırma

```bash
cd ~/IoMT-Project
source venv/bin/activate
jupyter notebook course-project/notebooks/course_project_demo.ipynb
# UI: Kernel → Restart & Run All  (yaklaşık 7 dakika sürer; TF model yükleme dahil)
```

## Bilimsel Dürüstlük Notu

Final rapor, yazım sürecinde **ardışık sayısal-doğruluk denetimlerinden** geçmiştir; tüm sayısal iddialar `results/` ve `eda_output/` dizinlerindeki kanonik dosyalardan otomatik olarak yüklenmektedir. Erken sürümlerde tahminî olarak yer alan ifadeler — örneğin entropy vs Gini criterion karşılaştırmasında **ölçülen 0.47 pp farkın** "yaklaşık 26 puan" olarak verildiği §5.5 — kontrollü A/B deneyleri (`scripts/run_e5g_gini_baseline.py`) ve doğrulama betiği (`scripts/verify_report_numbers.py`) ile gerçek ölçümler kullanılarak yenilenmiştir. Beklenmedik null sonuçlar gizlenmek yerine *metodolojik disiplin örneği* olarak şeffafça raporlanmıştır.

Bu yaklaşım, Ioannidis (2005) "*Why Most Published Research Findings Are False*" makalesindeki **negatif sonuç raporlama** ve **hipotez ön-kayıt** ilkelerini takip etmektedir. Doğrulama denetiminin tam izi `scripts/verification_report.md` dosyasındadır; bağımsız bir reviewer aynı betiği çalıştırarak sonuçları yeniden üretebilir.

## Tezle İlişki

Bu ders projesi, AMRO MOUSA ISMAIL BASEET'in Y. Lisans tezinin (*Hibrit Denetimli-Denetimsiz IoMT Anomali Tespit Çerçevesi*) **Faz 4 (XGBoost ablation)** ve **Faz 5 (Otoenkoder + Isolation Forest)** kısımlarını kapsar. Tezde ek olarak fusion logic, multi-seed validation ve sıfır-gün simülasyon protokolü incelenmektedir; bu bileşenler ders projesinin kapsamı dışındadır ve Mayıs 2026'da ayrıca tez savunmasında sunulacaktır.
