# Course Project Notebooks — Klasör Özeti

Bu klasör dönem projesi sunum-hazır demo notebook'unu ve onun denetim kayıtlarını içerir.

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `course_project_demo.ipynb` | **Sunulacak ana notebook** (Stage 4 sonunda v1 → v1_backup'a, v2 → demo'ya rename edilecek) |
| `course_project_demo_v2.ipynb` | Sunum-hazır yeni notebook (yapım aşamasında — şu an Stage 1 tamamlandı) |
| `course_project_demo_v1_backup.ipynb` | (Stage 4 sonunda) Eski 46-hücreli notebook'un yedeği |
| `test_log.txt` | Son `nbconvert --execute` çalıştırmasının log'u |
| `README.md` | Bu dosya — klasör içeriği ve **v1 → v2 hata raporu** |

## Çalıştırma

```bash
cd ~/IoMT-Project
source venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 \
    course-project/notebooks/course_project_demo_v2.ipynb
```

Beklenen süre: ~5-10 dakika (model eğitimi yok, sadece kayıtlı sonuçların yüklenmesi).

---

# v1 → v2 Hata Raporu

**Tarih:** 2026-05-02
**Bağlam:** `course_project_demo.ipynb` (v1, ~46 hücre) sunum-hazır 50+ hücreli `course_project_demo_v2.ipynb`'ye dönüştürülürken yapılan kod denetimi sırasında üç sessiz hata tespit edildi. Bu rapor hataları, nedenlerini ve v2'deki düzeltmeleri belgeler.

> **Not:** v1, hatalar tespit edildiğinde de teknik olarak "çalışıyordu" — hiçbiri exception fırlatmıyordu. Hepsi *sessiz hatalardı*: yanlış sonuç gösterip hata vermiyorlardı. Bu da neden kod-denetiminin yalnızca CI yeşil ışığına güvenmemek gerektiğinin örneğidir.

---

## Hata #1 — `accuracy = %0.00` (yanlış JSON anahtarı)

### Konum
v1, Hücre 24 (Faz 4 özet hücresi).

### Belirti
v1 çalıştırıldığında şu çıktı veriyor:

```
Genel Performans (E7 — En İyi Konfigürasyon):
  • Accuracy   : 0.00%
  • Macro F1   : 0.0000
  • MCC        : 0.0000
```

### Kök Neden
Kod, JSON dosyasından metrikleri yanlış anahtar adlarıyla okuyordu:

```python
# v1 (hatalı)
e7_metrics.get('accuracy', 0)   # JSON'da bu anahtar YOK
e7_metrics.get('macro_f1', 0)   # JSON'da bu anahtar YOK
e7_metrics.get('mcc', 0)        # JSON'da bu anahtar YOK
```

`results/supervised/metrics/E7_multiclass.json` dosyasının gerçek anahtarları:

```json
{
  "test_accuracy": 0.9926557939991124,
  "test_f1_macro": 0.907626622882394,
  "test_mcc": 0.9906169668153739
}
```

`.get(key, 0)` çağrısı, anahtar bulunamadığında **sessizce 0 döndürür** — bu yüzden hücre exception fırlatmadan tamamen yanlış sayılar gösteriyordu.

### v2 Düzeltmesi (Stage 2'de uygulanacak)

```python
# v2 (doğru)
e7_metrics['test_accuracy']   # 0.9927
e7_metrics['test_f1_macro']   # 0.9076
e7_metrics['test_mcc']        # 0.9906
```

Ek olarak **tripwire assert** eklenecek (kanonik değerlerle eşleşmeyi doğrular):

```python
assert abs(e7_metrics['test_f1_macro'] - 0.9076) < 1e-3, \
    "E7 macro_F1 kanonik değerle uyuşmuyor — yeniden eğitim mi yapıldı?"
assert abs(e7_metrics['test_mcc'] - 0.9906) < 1e-3, \
    "E7 MCC kanonik değerle uyuşmuyor"
```

### Çıkarım
**Sözlük indeksleme `[]` ile yapılmalı**, eksik anahtarda KeyError fırlatsın. `.get(key, default)` kullanımı yalnızca varsayılan değer **gerçekten geçerli bir fallback** ise tercih edilmelidir; metrik okuma için değil.

---

## Hata #2 — AE eşik = `0.2000` (varsayılan değere düşüş)

### Konum
v1, Hücre 30 (Otoenkoder reconstruction error analizi).

### Belirti
v1 çıktısı:

```
AE p90 Eşik Değeri: 0.2000
```

Beklenen: `0.2013` (rapordaki ve thresholds.json'daki gerçek değer).

### Kök Neden
v1 kodu thresholds.json'u yanlış nested yapıda okumaya çalışıyordu:

```python
# v1 (hatalı)
ae_threshold = thresholds.get('ae_p90', thresholds.get('ae_threshold', 0.20))
```

İlk anahtar (`ae_p90`) ve ikinci anahtar (`ae_threshold`) JSON'da yok. Üçüncü fallback olan `0.20` her seferinde döndürülüyordu. JSON'un gerçek yapısı **iç içe**:

```json
{
  "thresholds": {
    "p90": 0.20127058029174805,
    "p95": 0.37264156341552734
  },
  "selected": {
    "name": "p90",
    "value": 0.20127058029174805
  }
}
```

Yine `.get(..., default)` deseni hatayı sessize aldı.

### v2 Düzeltmesi (Stage 3'te uygulanacak)

```python
# v2 (doğru) — explicit nested erişim, fallback yok
with open(f'{RESULTS}/unsupervised/thresholds.json') as f:
    thresholds_data = json.load(f)
ae_threshold = thresholds_data['thresholds']['p90']  # 0.20127
# veya alternatif olarak:
ae_threshold = thresholds_data['selected']['value']  # 0.20127
```

### Çıkarım
**Yapı değişiklikleri `.get()` ile saklanır.** İç içe konfigürasyon dosyalarına erişim, doğrudan `[key]` indeksleme ile yapılmalıdır; bu hem belge görevi görür hem de yapı değişiklikleri yüksek sesle başarısız olur.

---

## Hata #3 — Isolation Forest AUC = `0.1388` (skor yönü ters)

### Konum
v1, Hücre 36 (AE vs IF ROC karşılaştırma hücresi).

### Belirti
v1 çıktısı:

```
ROC AUC Skorları (Binary: Saldırı vs Benign)
  • Otoenkoder      : 0.9892
  • Isolation Forest: 0.1388

  → AE üstünlüğü: 85.04 puan
```

`0.1388` rastgele bir sınıflandırıcıdan **kötü** (0.5 altı) — bu, skorun **ters yönde** kullanıldığının işaretidir. Beklenen değer: `0.8612` (`model_comparison.csv`'de doğru saklanıyor).

### Kök Neden
sklearn'in `IsolationForest.decision_function()`'u şu konvansiyonu kullanır:

> **Daha YÜKSEK skor = daha NORMAL örnek.**
> Anomali skoru için bu işareti çevirmek gerekir.

v1 kodu skoru tersi-çevirmeden ROC AUC'ye geçiriyordu:

```python
# v1 (hatalı)
if_test_scores = np.load(f'{RESULTS}/unsupervised/scores/if_test_scores.npy')
y_binary = (y_test['label'] != 'Benign').astype(int).values  # saldırı=1
if_auc = roc_auc_score(y_binary, if_test_scores)  # YÜKSEK skor = NORMAL → ters!
```

`roc_auc_score`, "y_score" parametresinin **pozitif sınıf için yüksek** olmasını bekler. IF anomali (`saldırı=1`) tespit etmek için skoru ters çevirmemiz gerekir.

### v2 Düzeltmesi (Stage 3'te uygulanacak)

```python
# v2 (doğru) — IF skoru saldırı için ters çevriliyor
if_auc = roc_auc_score(y_binary, -if_test_scores)
# Veya ekvivalan: 1 - roc_auc_score(y_binary, if_test_scores)

# Doğrulama: model_comparison.csv'deki kayıtlı değerle eşleşmeli
expected_if_auc = pd.read_csv(f'{RESULTS}/unsupervised/metrics/model_comparison.csv') \
    .query('metric == "AUC-ROC (test)"')['IsolationForest'].iloc[0]
assert abs(if_auc - expected_if_auc) < 1e-3, \
    f"IF AUC kayıtlı değerden ({expected_if_auc:.4f}) sapıyor: {if_auc:.4f}"
```

### Çıkarım
**sklearn anomali tespit modellerinde skor yönü modele göre değişir** (`IsolationForest.decision_function` → yüksek=normal; `LocalOutlierFactor.score_samples` → yüksek=normal; `OneClassSVM.decision_function` → yüksek=normal). ROC AUC'ye geçirmeden önce işaret kontrolü yapılmalı, ve sonuçlar kayıtlı CSV/JSON'larla **çapraz doğrulanmalı**. AUC < 0.5 her zaman bir uyarı işaretidir.

---

## Tezi Korumak İçin Çıkarımlar

Bu üç hata aynı meta-şikayeti gösterir: **`.get()` ve fallback değerler hataları sessizleştirir.** Tezde aynı kalıbın olmaması için:

1. **Konfigürasyon yükleme:** Yapısal anahtarlara `dict[key]` ile eriş, `dict.get(key, default)` kullanma — eksik anahtar yapısal değişiklik demektir, sessize alınmamalı.
2. **Metrik karşılaştırma:** Kanonik değerlerle eşleşme **assert** ile zorlanır. Sayı kayar (örn. yeniden eğitim sonrası) ise hata yüksek sesle çıkar.
3. **Skor yönü:** sklearn anomali modellerinde decision_function/score_samples konvansiyonunu RTFM ile doğrula; AUC < 0.5 daima ters-skor için kontrol edilmelidir.
4. **Çapraz doğrulama:** Hesaplanan değer yanında **kayıtlı CSV** varsa, ikisinin eşleşmesi assert edilir.

Bu üç hata v1'de **silently incorrect** ama notebook çalıştı — bu yüzden CI yeşil ışığı **doğruluk garantisi vermez**. Code review + canonical-value asserts birlikte gereklidir.

---

## v2 Düzeltme Zaman Çizelgesi

| Hata | v1 Hücre | v2 Hücre (planlanan) | Aşama |
|---|---:|---:|---|
| #1 accuracy = 0% | 24 | ~27 (E7 metrik gösterimi) | Stage 2 |
| #2 threshold = 0.2000 | 30 | ~35 (AE eşik açıklaması) | Stage 3 |
| #3 IF AUC = 0.1388 | 36 | ~36 (AE vs IF ROC) | Stage 3 |

v1 dosyası **dokunulmadan kalır** ve Stage 4 sonunda `course_project_demo_v1_backup.ipynb` olarak yedeklenir. v2, hatalar bu rapora göre düzeltilerek `course_project_demo.ipynb` olarak yerleştirilir.
