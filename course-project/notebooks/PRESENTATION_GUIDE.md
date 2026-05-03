# Sunum Rehberi — 15 Dakikalık Akış

**Notebook:** `course_project_demo_v2.ipynb` (54 hücre toplam)
**Sunum süresi:** 15 dakika + 5 dakika Q&A
**Sunum modu:** Jupyter, "Restart Kernel & Run All" → tüm çıktılar hazır

Bu rehber, sunum sırasında **8-10 anahtar hücre** üzerinde yoğunlaşmanızı önerir. Diğer hücreler **ekran-altı destek** olarak kalır (profesör isterse açabilirsiniz).

---

## Önerilen Akış (15 dk)

### **Açılış (1 dk)** — Cell 0 + Cell 1
- **Cell 0** (title): proje başlığı, ders, tarih, yazarlar
- **Cell 1** (roadmap): "10 bölüm × 15 dk" tablosu — izleyiciye ne göreceğini söyler
- **Konuş:** "Bu notebook PowerPoint yerine geçer; tüm sayılar gerçek dosyalardan yüklenir."

### **EDA Headline (1.5 dk)** — Cell 15 (Cohen's d top 10 chart)
- Görsel: 4 bar olağanüstü (d > 2.0) eşiğinin üstünde
- **Konuş:** "Top 4 özellik d > 2.0 — RST/PSH/ACK flag sayaçları + Variance. TCP davranış imzası saldırı tespitinin anchor sinyali."
- **Yumuşak köprü:** "Bu güçlü univariate sinyal, tree modelleri için ideal taban."

### **Faz 4 Headline (3 dk)** — Cell 25 (E1-E8 ablation + SMOTETomek viz)
- ⭐ **PRIMARY HEADLINE**
- Tablo: 8 konfigürasyon, E7 vurgulu (final, 0.9076 macro F1)
- **Görsel: SMOTETomek 4 oklu degradation chart** — 4 konfigürasyonda 4 düşüş oku
- **Konuş:** "Faktoriyel tasarım: 2 model × 2 özellik × 2 dengesizlik = 8 deney. **Beklenmedik bulgu**: SMOTETomek dört konfigürasyonun **dördünde de** macro F1'i düşürdü (1.1-4.5 pp). Bu, IDS literatüründe yeterince vurgulanmamış bir gözlemdir."
- **Profesör sorusu beklentisi:** "SMOTETomek niye?" → "Yüksek-bilgi-içerikli IDS verisinde sentetik azınlık örnekleri sınır gürültüsünü artırabilir."

### **Faz 4 Null Sonuç (1.5 dk)** — Cell 26 (Entropy vs Gini A/B)
- ⭐ **DÜRÜSTLÜK ANLAR**
- Görsel: 4 metrikte küçük (~0.5 pp) entropy avantajı
- **Konuş:** "Cohen's d'nin >2.0 olması entropy'nin avantajını **teorik olarak** öngördü. Ama A/B testimiz farkı **0.47 pp** olarak ölçtü — gürültü mertebesinde. Bu, **scikit-learn dokümantasyonunu doğrular**: 'iki criterion benzer ağaçlar üretir'. Hipotezimiz uygulamada doğrulanmadı; bu null sonucu raporladık (Bölüm 5.5'te detayda)."
- **Etkisi:** Bilimsel disiplin sergilenir. Profesör hipotezi+raporlama döngüsünü görür.

### **E7 Final Sonuç (1 dk)** — Cell 27 (E7 big metrics)
- Çıktı: %99.27 / 0.9076 / 0.9906 (büyük rakamlar)
- **Konuş:** "XGBoost (E7) bilinen 19 sınıfta — accuracy %99.27, macro F1 0.9076, MCC 0.9906. Tripwire assertion'lar geçti, kanonik tez değerleriyle eşleşti."

### **Faz 4 Limitasyonlar (1 dk)** — Cell 28 (Per-class F1, sorted asc)
- Görsel: 4 zor sınıf kırmızıyla işaretli, n ≤ 2,941 cutoff doğal
- **Konuş:** "Limitasyonu **gizlemek yerine işaret ediyoruz**: Recon_VulScan 0.45, Recon_OS_Scan 0.69, ARP_Spoofing 0.76, Recon_Ping_Sweep 0.78. **Hepsi support ≤ 2,941**. Sınırlama **modelin değil, verinin** istatistiksel güç sınırlarından."

### **Faz 5 Headline (2 dk)** — Cell 36 (AE vs IF ROC)
- ⭐ **PRIMARY EMPIRICAL HEADLINE**
- Görsel: AE eğrisi sol-üstte, IF eğrisi belirgin altta
- **Konuş:** "Otoenkoder AUC = **0.9892**, Isolation Forest AUC = **0.8612** — **12.8 puan AE üstünlüğü**. Per-class detection: AE %80, IF %16.3 — IF azınlık sınıflarında neredeyse hiç tespit yapamaz. Bu hibrit IDS'de AE seçiminin **ampirik gerekçesidir**."
- **Bug fix vurgu:** "Var olan kodda bu karşılaştırma yön ters yapılmıştı (IF AUC 0.13 görünüyordu — gizli bir bug). Düzelttik, gerçek değer 0.86."

### **Canlı Demo (2 dk)** — Cell 43 (Hybrit inference tablosu)
- ⭐ **VITAMIN D MOMENT**
- Tablo: 10 örnek, XGB confidence + AE recon error yan yana
- Sarı satır(lar): "REVIEW" işareti — XGB tereddütlü, AE yardımıyla işaretlendi
- **Konuş:** "10 stratejik örnek: 1 Benign + 5 kolay sınıf + 4 zor sınıf + 1 hibrit-tamamlayıcılık örneği. Her biri için iki katman birlikte karar veriyor. **Sarı satırlar**: XGB confidence < 0.90 + AE anomalik = 'REVIEW' (insan analiste yönlendir)."

### **Sonuç (1 dk)** — Cell 46 (5 anahtar bulgu)
- 5 madde markdown listesi
- **Konuş:** "Beş anahtar bulgu — özetle: AE +12.8 pp IF'e karşı (PRIMARY), SMOTETomek 4/4 düşürdü (counter-intuitive), n=2,941 sample-size cutoff, model swap +25 pp ARP, entropy vs Gini ~0.5 pp null. Ders projesinin kapsamı bunlar — fusion mantığı tezde."

---

## Akış Tablosu

| # | Hücre | Süre | Slot Tipi |
|---|---|---|---|
| 1 | 0 + 1 | 1.0 dk | Açılış |
| 2 | 15 | 1.5 dk | EDA headline |
| 3 | **25** | **3.0 dk** | **PRIMARY: SMOTETomek bulgusu** |
| 4 | 26 | 1.5 dk | Dürüstlük: entropy vs Gini null |
| 5 | 27 | 1.0 dk | E7 final metrikleri |
| 6 | 28 | 1.0 dk | Per-class limitasyonlar |
| 7 | **36** | **2.0 dk** | **PRIMARY: AE vs IF 12.8 pp** |
| 8 | **43** | **2.0 dk** | **Canlı hibrit demo** |
| 9 | 46 | 1.0 dk | Sonuç |
| **Toplam** | **9 cells** | **14.0 dk** | (1 dk buffer) |

---

## Q&A Hazırlığı

### Beklenen sorular ve hücreler:

| Soru | Cevap-hücresi |
|---|---|
| "AE adversarial saldırılara dayanıklı mı?" | **Cell 51** (Q&A backup #1) |
| "Production'da kaç ms latency?" | **Cell 52** (Q&A backup #2) |
| "Drift handling stratejiniz ne?" | **Cell 53** (Q&A backup #3) |
| "Yacoubi paper'ın 99.87%'ni geçtiniz mi?" | Cell 14'teki tablodaki Yacoubi karşılaştırması |
| "Confusion matrix?" | Cell 29 |
| "Recon error histogram?" | Cell 34 |
| "AE training overfit etmedi mi?" | Cell 33 (train-val gap +0.013, no overfitting) |

### Önemli noktalar:

- **Q&A backup hücreleri (50-53)** sunum sırasında varsayılan olarak **gizli** — açıkça soru gelirse açılır.
- **Cell 39'daki SHAP-Cohen's d "kısmi örtüşme"** açıklaması: dürüstlük thread'i. Profesör "neden farklı sıralama?" derse: "Univariate vs multivariate ölçtükleri farklı; ortak nokta: TCP flag sinyali her ikisinde de güçlü."

---

## Hızlı Çalıştırma

```bash
cd ~/IoMT-Project
source venv/bin/activate
jupyter notebook course-project/notebooks/course_project_demo.ipynb
# UI: Kernel → Restart & Run All
# ~5-8 dk içinde tüm hücreler hazır olur (TF model load + AE inference dahil)
```

Hata olursa: `course-project/notebooks/test_log.txt` dosyasında son başarılı çalıştırmanın log'u var.

---

**Yedek strateji:** Eğer sunum esnasında bir hücre tekrar çalıştırmaya gerek olursa, kerneli yeniden başlatma yerine yalnızca o hücreye gidip **Shift+Enter**. Tüm değişkenler önceki hücrelerden kalmış olacak.

