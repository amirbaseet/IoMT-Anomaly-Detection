"""Additional Turkish thesis figures: architecture + fusion diagrams and 8 data plots."""
from __future__ import annotations
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path("/Users/amoorabaseet/code/IoMT-Project")
OUT = ROOT / "thesis/manuscript/figures"
C_MAIN, C_ALT, C_MUTED, C_WARN = "#0E6E60", "#B07C22", "#5C6A66", "#A33B3B"
C_SOFT, C_ALTSOFT = "#E2EEEA", "#F4EBD9"
GRID = "#DFE5E2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                     "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .8,
                     "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight"})

def save(fig, name):
    fig.savefig(OUT / f"{name}.png"); plt.close(fig); print("  ", name)

def box(ax, x, y, w, h, text, fc, ec, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#1C2422", linespacing=1.45)

def arrow(ax, p1, p2, style="-|>", color=C_MUTED, lw=1.3, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13,
                                 color=color, lw=lw, linestyle=ls, shrinkA=2, shrinkB=2))

# ---------- Şekil 3.1 — four-layer architecture (NEW; no counterpart anywhere) ----------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8.4); ax.axis("off"); ax.grid(False)

    box(ax, 0.3, 7.4, 9.4, 0.75, "CICIoMT2024 — Wi-Fi + MQTT akışları\n8.775.013 ham kayıt → tekilleştirme → 44 öznitelik, 19 sınıf",
        "#F2F0EA", C_MUTED, 9)

    box(ax, 0.3, 5.85, 4.5, 1.05, "KATMAN 1 — Denetimli\nXGBoost (E7), 19 sınıf\nçıktı: softmax olasılık vektörü", C_SOFT, C_MAIN, 9, True)
    box(ax, 5.2, 5.85, 4.5, 1.05, "KATMAN 2 — Denetimsiz\nOtokodlayıcı  ·  Isolation Forest\nyalnızca Benign ile eğitilir", C_ALTSOFT, C_ALT, 9, True)
    arrow(ax, (2.55, 7.4), (2.55, 6.9)); arrow(ax, (7.45, 7.4), (7.45, 6.9))

    box(ax, 1.6, 3.95, 6.8, 1.0, "KATMAN 3 — Entropi kapılı beş durumlu karar füzyonu\nhiçbir model yeniden eğitilmez; kayıtlı çıktılar birleştirilir",
        C_SOFT, C_MAIN, 9, True)
    arrow(ax, (2.55, 5.85), (3.5, 4.95), color=C_MAIN)
    arrow(ax, (7.45, 5.85), (6.5, 4.95), color=C_ALT)
    ax.text(2.75, 5.42, "softmax vektörü\n+ Shannon entropisi", ha="center", fontsize=7.8, color=C_MAIN,
            bbox=dict(fc="white", ec="none", pad=1.5))
    ax.text(7.25, 5.42, "yeniden-oluşturma\nhatası (AE kararı)", ha="center", fontsize=7.8, color=C_ALT,
            bbox=dict(fc="white", ec="none", pad=1.5))
    ax.text(9.68, 5.50, "IF: yalnızca ablasyon kanalı —\nnihai karar zincirinde yok", ha="right", fontsize=7.2,
            color=C_MUTED, style="italic")

    states = [("1 Onaylanmış\nSaldırı", C_MAIN), ("2 Sıfır-Gün\nUyarısı", C_WARN),
              ("3 Düşük\nGüven", C_ALT), ("4 Temiz", C_MUTED), ("5 Belirsiz\nAlarm", C_ALT)]
    for i, (t, c) in enumerate(states):
        x = 0.35 + i * 1.94
        box(ax, x, 2.35, 1.75, 0.95, t, "white", c, 8)
        arrow(ax, (5.0, 3.95), (x + 0.875, 3.32), color=c, lw=1.0)
    ax.text(5.0, 2.06, "her durum bir operatör eylemine karşılık gelir (engelle · karantina · izle · izin ver · incele)",
            ha="center", fontsize=8, color=C_MUTED, style="italic")

    box(ax, 1.6, 0.75, 6.8, 0.95, "KATMAN 4 — Sınıf başına TreeSHAP açıklanabilirliği\nE7 kararlarını çevrimdışı açıklar; füzyona geri besleme yapmaz",
        C_SOFT, C_MAIN, 9, True)
    ax.set_title("Şekil 3.3  Dört katmanlı tespit çerçevesi", fontsize=12.5, pad=12)
    save(fig, "sekil_3_3_mimari")

# ---------- Şekil 3.4 — five-case decision table (NEW) ----------
def fig_fusion_states():
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    ax.set_xlim(0, 10.1); ax.set_ylim(0, 6); ax.axis("off"); ax.grid(False)
    ax.text(5, 5.6, "Şekil 3.4  Füzyonun beş durumlu karar tablosu", ha="center", fontsize=12.5)

    ax.text(3.1, 4.75, "Katman 2 — otokodlayıcı", ha="center", fontsize=9.5, color=C_ALT, fontweight="bold")
    ax.text(1.9, 4.35, "ANOMALİ", ha="center", fontsize=9, color=C_ALT)
    ax.text(4.3, 4.35, "NORMAL", ha="center", fontsize=9, color=C_ALT)
    ax.text(0.42, 2.55, "Katman 1 — E7", rotation=90, va="center", fontsize=9.5, color=C_MAIN, fontweight="bold")
    ax.text(0.95, 3.45, "SALDIRI", rotation=90, va="center", ha="center", fontsize=9, color=C_MAIN)
    ax.text(0.95, 1.85, "BENIGN", rotation=90, va="center", ha="center", fontsize=9, color=C_MAIN)

    cells = [(1.25, 2.85, "Durum 1\nOnaylanmış Saldırı\n→ engelle", C_MAIN, C_SOFT),
             (3.65, 2.85, "Durum 3\nDüşük Güven\n→ izle", C_ALT, "white"),
             (1.25, 1.25, "Durum 2\nSıfır-Gün Uyarısı\n→ karantina", C_WARN, "#F7E9E9"),
             (3.65, 1.25, "Durum 4\nTemiz\n→ izin ver", C_MUTED, "white")]
    for x, y, t, ec, fc in cells:
        box(ax, x, y, 2.2, 1.4, t, fc, ec, 8.5)

    box(ax, 6.75, 1.25, 3.0, 3.0, "Durum 5 — Belirsiz Alarm\n→ operatör incelemesi\n\nTetikleyici: yüksek\nsoftmax entropisi\n(otokodlayıcı anomali\ndemese bile)\n\nEşik: Benign-doğrulama\nkesitinin yüzdelikleri\n(n = 38.546) · p95 = 0,3946",
        C_ALTSOFT, C_ALT, 8.2)
    arrow(ax, (5.9, 2.55), (6.75, 2.55), color=C_ALT)
    ax.text(6.32, 2.72, "entropi kapısı", ha="center", fontsize=7.5, color=C_ALT)
    ax.text(5.0, 0.72, "Durum 2, sıfır-gün kurtarmasının kapısıdır: sınıflandırıcı «temiz» derken denetimsiz katman itiraz eder.",
            ha="center", fontsize=8.5, color="#1C2422", style="italic")
    save(fig, "sekil_3_4_bes_durum")

if __name__ == "__main__":
    print("diagrams →", OUT)
    fig_architecture(); fig_fusion_states()

# ---------- data figures ----------
def _ablation():
    return list(csv.DictReader(open(ROOT / "results/enhanced_fusion/metrics/ablation_table.csv")))

def fig_grid():
    """E1–E8 grid + E5G, 19-class macro-F1."""
    exp = {}
    for e in ["E1","E2","E3","E4","E5","E5G","E6","E7","E8"]:
        f = ROOT / f"results/supervised/metrics/{e}_multiclass.json"
        if f.exists(): exp[e] = json.load(open(f))["test_f1_macro"]
    lbl = {"E1":"RF / 28 / Orij.","E2":"RF / 28 / SMOTE","E3":"XGB / 28 / Orij.","E4":"XGB / 28 / SMOTE",
           "E5":"RF / 44 / Orij.","E5G":"RF-gini / 44 / Orij.","E6":"RF / 44 / SMOTE",
           "E7":"XGB / 44 / Orij.","E8":"XGB / 44 / SMOTE"}
    items = sorted(exp.items(), key=lambda kv: kv[1])
    names = [f"{k} — {lbl[k]}" for k, _ in items]; vals = [v for _, v in items]
    cols = [C_MAIN if k == "E7" else (C_ALT if "SMOTE" in lbl[k] else C_MUTED) for k, _ in items]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.barh(range(len(names)), vals, color=cols, height=.68)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlim(0.80, 0.95); ax.set_xlabel("macro-F1 (19 sınıflı test görevi, seed 42)")
    ax.set_title("Şekil 4.1  E1–E8 deney ızgarası — üretim yapılandırması E7")
    ax.grid(axis="y", visible=False)
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    for i, (v, (k, _)) in enumerate(zip(vals, items)):
        ax.text(v + .0015, i, f"{v:.4f}".replace(".", ","), va="center", fontsize=8,
                color=C_MAIN if k == "E7" else "#1C2422",
                fontweight="bold" if k == "E7" else "normal")
    ax.text(.949, .04, "turuncu = SMOTETomek kolları (dördü de taban çizgisinin altında)",
            transform=ax.transAxes, ha="right", fontsize=8, color=C_ALT, style="italic")
    save(fig, "sekil_4_1_deney_izgarasi")

def fig_smote():
    """SMOTETomek deltas, 4 paired configs."""
    pairs = [("RF / 28", "E1", "E2"), ("RF / 44", "E5", "E6"),
             ("XGB / 28", "E3", "E4"), ("XGB / 44", "E7", "E8")]
    labs, deltas = [], []
    for lab, a, b in pairs:
        fa = json.load(open(ROOT / f"results/supervised/metrics/{a}_multiclass.json"))["test_f1_macro"]
        fb = json.load(open(ROOT / f"results/supervised/metrics/{b}_multiclass.json"))["test_f1_macro"]
        labs.append(lab); deltas.append(fb - fa)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(range(4), deltas, color=C_WARN, width=.55)
    ax.axhline(0, color="#1C2422", lw=1)
    ax.axhspan(-0.0168, 0.0168, color=C_MUTED, alpha=.13, zorder=0)
    ax.text(3.42, 0.0125, "±1σ tohum gürültüsü", ha="right", fontsize=8, color=C_MUTED)
    ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel("macro-F1 değişimi (SMOTETomek − Orijinal)")
    ax.set_title("Şekil 4.3  SMOTETomek etkisi — dört yapılandırmanın dördü de negatif")
    ax.grid(axis="x", visible=False)
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    for i, d in enumerate(deltas):
        ax.text(i, d - .0035, f"{d:+.4f}".replace(".", ","), ha="center", va="top", fontsize=8.5)
    save(fig, "sekil_4_3_smote_etkisi")

def fig_detection_heatmap():
    """Per-class AE detection rate across thresholds."""
    rows = [r for r in csv.DictReader(open(ROOT / "results/unsupervised/metrics/per_class_detection_rates.csv"))
            if r["model"] == "Autoencoder"]
    rows = sorted(rows, key=lambda r: -float(r["p90"]))
    names = [r["class"].replace("_", " ") for r in rows]
    cols = ["p90", "p95", "p99"]
    M = np.array([[float(r[c]) for c in cols] for r in rows])
    fig, ax = plt.subplots(figsize=(5.6, 6.4))
    im = ax.imshow(M, cmap="BuGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(cols, fontsize=9)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("otokodlayıcı eşiği"); ax.grid(visible=False)
    ax.set_title("Şekil 4.5  Sınıf başına tespit oranı\n(denetimsiz katman, eşik başına)", fontsize=11.5)
    for i in range(len(names)):
        for j in range(3):
            ax.text(j, i, f"{M[i,j]:.2f}".replace("0.", ",").replace(".", ","), ha="center", va="center",
                    fontsize=7, color="white" if M[i, j] > .55 else "#1C2422")
    cb = fig.colorbar(im, ax=ax, fraction=.06, pad=.04); cb.set_label("tespit oranı", fontsize=9)
    save(fig, "sekil_4_5_sinif_basina_tespit")

def fig_case_dist():
    """Four-case fusion distribution at AE p90."""
    r = [x for x in csv.DictReader(open(ROOT / "results/fusion/metrics/case_distribution.csv"))
         if x["variant"] == "AE_p90"][0]
    labs = ["Durum 1\nOnaylanmış\nSaldırı", "Durum 2\nSıfır-Gün\nUyarısı", "Durum 3\nDüşük\nGüven", "Durum 4\nTemiz"]
    n = [int(r[f"case{i}_n"]) for i in range(1, 5)]
    pct = [float(r[f"case{i}_pct"]) for i in range(1, 5)]
    cols = [C_MAIN, C_WARN, C_ALT, C_MUTED]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    b = ax.bar(range(4), pct, color=cols, width=.6)
    ax.set_yscale("log"); ax.set_ylim(.1, 200)
    ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=8.5)
    ax.set_ylabel("test kümesindeki pay (%, log ölçek)")
    ax.set_title("Şekil 4.7  Füzyon durum dağılımı (AE p90, n = 892.268)")
    ax.grid(axis="x", visible=False)
    for i, (p, c) in enumerate(zip(pct, n)):
        ax.text(i, p * 1.25, f"%{p:.2f}".replace(".", ",") + f"\n{c:,}".replace(",", "."),
                ha="center", fontsize=8)
    ax.text(.5, .04, "Sıfır-gün adayları Durum 2'dedir: trafiğin binde 7'si — yönetilebilir bir inceleme kuyruğu",
            transform=ax.transAxes, ha="center", fontsize=8, color=C_MUTED, style="italic")
    save(fig, "sekil_4_7_durum_dagilimi")

def fig_per_target():
    """Per-target rescue: baseline vs entropy gate."""
    rows = list(csv.DictReader(open(ROOT / "results/enhanced_fusion/metrics/per_target_results.csv")))
    base = {r["target"]: r for r in rows if r["variant"] == "baseline_ae_p90"}
    ent = {r["target"]: r for r in rows if r["variant"] == "entropy_benign_p95"}
    tg = [t for t in base if base[t]["h2_strict_rescue_recall"] and ent.get(t, {}).get("h2_strict_rescue_recall")]
    tg = sorted(tg, key=lambda t: float(ent[t]["h2_strict_rescue_recall"]))
    b = [float(base[t]["h2_strict_rescue_recall"]) for t in tg]
    e = [float(ent[t]["h2_strict_rescue_recall"]) for t in tg]
    y = np.arange(len(tg))
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.barh(y + .19, e, height=.36, color=C_MAIN, label="entropi kapısı (p95)")
    ax.barh(y - .19, b, height=.36, color=C_MUTED, label="taban çizgisi (AE p90)")
    ax.axvline(.70, color=C_WARN, lw=1.6, ls="--")
    ax.text(.715, len(tg) - .45, "H2-katı eşiği 0,70", color=C_WARN, fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels([t.replace("_", " ") for t in tg], fontsize=9)
    ax.set_xlabel("katı kurtarma geri çağırması"); ax.set_xlim(0, 1.08)
    ax.set_title("Şekil 4.8  Hedef başına sıfır-gün kurtarması (seed 42)")
    ax.grid(axis="y", visible=False)
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    for i, (bb, ee) in enumerate(zip(b, e)):
        ax.text(ee + .012, i + .19, f"{ee:.3f}".replace(".", ","), va="center", fontsize=7.5, color=C_MAIN)
        ax.text(bb + .012, i - .19, f"{bb:.3f}".replace(".", ","), va="center", fontsize=7.5, color=C_MUTED)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    save(fig, "sekil_4_8_hedef_basina_kurtarma")

def fig_multiseed():
    """Multi-seed strict rescue distribution."""
    rows = [r for r in csv.DictReader(open(ROOT / "results/enhanced_fusion/multi_seed_summary.csv"))
            if r["variant"] == "entropy_benign_p95"]
    r = rows[0]
    mean, sd = float(r["h2_strict_avg_mean"]), float(r["h2_strict_avg_std"])
    lo, hi = float(r["h2_strict_avg_min"]), float(r["h2_strict_avg_max"])
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.axhspan(mean - sd, mean + sd, color=C_MAIN, alpha=.14, zorder=0)
    ax.axhline(mean, color=C_MAIN, lw=1.8)
    ax.axhline(.70, color=C_WARN, lw=1.5, ls="--")
    pts = [lo, mean, hi, 0.8035264623662012]
    ax.scatter([1, 2, 3, 4], pts, s=70, color=[C_MUTED, C_MAIN, C_MUTED, C_ALT], zorder=5,
               edgecolor="white", linewidth=1.4)
    for x, v, lab in zip([1, 2, 3, 4], pts, ["en düşük", "ortalama", "en yüksek", "tohum 42\n(kanonik)"]):
        ax.text(x, v + .006, f"{v:.3f}".replace(".", ","), ha="center", fontsize=8.5)
        ax.text(x, .688, lab, ha="center", fontsize=8, color=C_MUTED)
    ax.set_xlim(.4, 4.6); ax.set_ylim(.68, .87); ax.set_xticks([])
    ax.set_ylabel("katı kurtarma ortalaması")
    ax.set_title("Şekil 4.9  Çoklu-tohum doğrulama — 0,799 ± 0,023 (5 tohum)")
    ax.text(4.55, .706, "H2-katı eşiği 0,70", ha="right", fontsize=8, color=C_WARN)
    ax.grid(axis="x", visible=False)
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    save(fig, "sekil_4_9_coklu_tohum")

def fig_shap_methods():
    """Top-10 Jaccard similarity across four feature-importance methods."""
    rows = list(csv.DictReader(open(ROOT / "results/shap/metrics/method_jaccard.csv")))
    meths = ["Our SHAP", "Cohen's d", "RF Importance", "Yacoubi SHAP"]
    tr = {"Our SHAP": "Bu tez\n(SHAP)", "Cohen's d": "Cohen's d", "RF Importance": "RF önemi\n(E5)",
          "Yacoubi SHAP": "Yacoubi vd.\n(SHAP)"}
    M = np.zeros((4, 4))
    for i, a in enumerate(meths):
        r = [x for x in rows if x["method"] == a][0]
        for j, b in enumerate(meths):
            M[i, j] = float(r[b])
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = ax.imshow(M, cmap="BuGn", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels([tr[m] for m in meths], fontsize=8.5)
    ax.set_yticks(range(4)); ax.set_yticklabels([tr[m] for m in meths], fontsize=8.5)
    ax.grid(visible=False)
    ax.set_title("Şekil 4.10  Öznitelik-önem yöntemleri arası\nilk-10 Jaccard benzerliği", fontsize=11.5)
    for i in range(4):
        for j in range(4):
            v = M[i, j]
            ax.text(j, i, f"{v:.3f}".replace(".", ","), ha="center", va="center", fontsize=9,
                    color="white" if v > .55 else "#1C2422",
                    fontweight="bold" if (i, j) in [(0, 1), (1, 0)] else "normal")
    ax.text(.5, -.16, "Bu tezin SHAP sıralaması ile Cohen's d arasında ilk-10'da tek bir ortak öznitelik yoktur (0,000).",
            transform=ax.transAxes, ha="center", fontsize=8, color=C_WARN, style="italic")
    save(fig, "sekil_4_10_yontem_karsilastirmasi")

def fig_session_gap():
    """Session-disjoint vs random-pooled split gap."""
    labels = ["macro-F1\n(denetimli)", "MCC\n(denetimli)", "AUC\n(otokodlayıcı)"]
    pooled = [0.734, 0.991, 0.908]
    disj   = [0.475, 0.556, 0.831]
    x = np.arange(3); w = .34
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.bar(x - w/2, pooled, w, color=C_MUTED, label="rastgele havuzlanmış bölme (sızıntılı)")
    ax.bar(x + w/2, disj,  w, color=C_MAIN,  label="oturum-ayrık bölme (dürüst)")
    for i, (p, d) in enumerate(zip(pooled, disj)):
        ax.text(i - w/2, p + .015, f"{p:.3f}".replace(".", ","), ha="center", fontsize=8.5)
        ax.text(i + w/2, d + .015, f"{d:.3f}".replace(".", ","), ha="center", fontsize=8.5)
        ax.annotate("", (i + w/2, d), (i - w/2, p),
                    arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.3))
        ax.text(i, (p + d) / 2, f"−{p-d:.3f}".replace(".", ","), ha="center", fontsize=8.5,
                color=C_WARN, bbox=dict(fc="white", ec="none", pad=1.5))
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.12); ax.set_ylabel("değer")
    ax.set_title("Şekil 4.12  Oturum-ayrık değerlendirmenin bedeli (5 tohum)")
    ax.grid(axis="x", visible=False)
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.text(.5, .035, "Denetimsiz katman oturum kaymasına belirgin biçimde daha dayanıklıdır — hibrit mimarinin ölçülmüş gerekçesi.",
            transform=ax.transAxes, ha="center", fontsize=8, color=C_MUTED, style="italic")
    save(fig, "sekil_4_12_oturum_ayrik")
