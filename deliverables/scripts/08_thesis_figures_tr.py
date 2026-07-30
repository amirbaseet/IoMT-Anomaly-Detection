"""
Turkish thesis figures (Sekil 3.1 - 4.7) for thesis/manuscript.

Reads committed artifacts only; writes PNG pairs into thesis/manuscript/figures/.
Labels follow thesis/manuscript/terminoloji.md. Style matches 07_generate_figures.py.
Every value plotted traces to numbers_map.md or the named artifact.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/amoorabaseet/code/IoMT-Project")
OUT = ROOT / "thesis/manuscript/figures"
DPI = 200

# Categorical palette: fixed order, never cycled (dataviz rule).
# Teal / amber / slate / rose — validated separation on light surface.
C_MAIN = "#0E6E60"
C_ALT = "#B07C22"
C_MUTED = "#5C6A66"
C_WARN = "#A33B3B"
GRID = "#DFE5E2"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
})


def tr_ticks(ax, axis: str = "x") -> None:
    """Turkish decimal comma on numeric ticks."""
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  wrote figures/{name}.png")


def fig_dedup_perclass() -> None:
    """Sekil 3.1 - per-class float32 within-class duplicate rate (DR-6b)."""
    d = json.load(open(ROOT.parent / "iomt-pcap-experiments/dr6_out/dr6b_perclass_f32.json"))
    items = sorted(((k, v["train"]["rate"]) for k, v in d.items()), key=lambda x: -x[1])
    names = [k.replace("TCP_IP-", "").replace("_", " ") for k, _ in items]
    rates = [v * 100 for _, v in items]
    colors = [C_MAIN if r >= 30 else (C_ALT if r > 1 else C_MUTED) for r in rates]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(names)), rates, color=colors, height=0.72)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Sınıf içi yinelenen kayıt oranı (%, eğitim bölmesi, float32)")
    ax.set_title("Şekil 3.1  Yinelenen kayıtların sınıf başına dağılımı")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    for i, r in enumerate(rates):
        if r > 0.5:
            ax.text(r + 1.2, i, f"%{r:.1f}".replace(".", ","), va="center", fontsize=7.5, color="#1C2422")
    ax.text(0.98, 0.03, "Taşkın sınıfları eğitim yinelenme kütlesinin %99,5'ini taşır",
            transform=ax.transAxes, ha="right", fontsize=8, color=C_MUTED, style="italic")
    save(fig, "sekil_3_1_yinelenen_kayit_sinif_bazinda")


def fig_imbalance() -> None:
    """Sekil 3.2 - class imbalance (log scale), 2,374:1."""
    rows = list(csv.DictReader(open(ROOT / "eda_output/imbalance_table.csv")))
    rows = sorted(rows, key=lambda r: -int(r["train"]))
    names = [r["class"].replace("_", " ") for r in rows]
    counts = [int(r["train"]) for r in rows]
    colors = [C_MAIN if c == max(counts) or c == min(counts) else C_MUTED for c in counts]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(names)), counts, color=colors, width=0.72)
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=70, ha="right", fontsize=7.5)
    ax.set_ylabel("Eğitim kaydı sayısı (log ölçek)")
    ax.set_title("Şekil 3.2  Sınıf dengesizliği — en büyük/en küçük oranı 2.374:1")
    ax.grid(axis="x", visible=False)
    ax.annotate(f"{counts[0]:,}".replace(",", "."), (0, counts[0]), textcoords="offset points",
                xytext=(0, 5), ha="center", fontsize=8, color=C_MAIN)
    ax.annotate(f"{counts[-1]:,}".replace(",", "."), (len(counts) - 1, counts[-1]),
                textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8, color=C_MAIN)
    save(fig, "sekil_3_2_sinif_dengesizligi")


def fig_ablation() -> None:
    """Sekil 4.3 - 11-variant ablation, strict rescue average."""
    rows = list(csv.DictReader(open(ROOT / "results/enhanced_fusion/metrics/ablation_table.csv")))
    rows = sorted(rows, key=lambda r: float(r["h2_strict_avg"]))
    names = [r["variant"].replace("_", " ") for r in rows]
    vals = [float(r["h2_strict_avg"]) for r in rows]
    passes = [r["h2_strict_pass"] for r in rows]
    colors = [C_MAIN if p.startswith("4/") else (C_ALT if p.startswith(("2/", "3/")) else C_MUTED)
              for p in passes]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(range(len(names)), vals, color=colors, height=0.7)
    ax.axvline(0.70, color=C_WARN, linewidth=1.6, linestyle="--")
    ax.text(0.705, -0.6, "H2-katı eşiği 0,70", color=C_WARN, fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Katı kurtarma ortalaması (strict rescue avg)")
    ax.set_title("Şekil 4.3  On bir varyantlı ablasyon — entropi kapısı belirleyicidir")
    ax.grid(axis="y", visible=False)
    tr_ticks(ax, "x")
    for i, (v, p) in enumerate(zip(vals, passes)):
        ax.text(v + 0.012, i, p, va="center", fontsize=7.5, color="#1C2422")
    save(fig, "sekil_4_3_ablasyon")


def fig_tau_curve() -> None:
    """Sekil 4.6 - accuracy-coverage operating characteristic (tau sweep)."""
    rows = list(csv.DictReader(open(ROOT / "results/tausweep/tau_sweep_curve.csv")))
    ent = [(float(r["coverage"]), float(r["retained_accuracy"])) for r in rows
           if r["sweep"] == "entropy_ceiling"]
    conf = [(float(r["coverage"]), float(r["retained_accuracy"])) for r in rows
            if r["sweep"] == "confidence_floor"]
    ent.sort(); conf.sort()

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot([c * 100 for c, _ in conf], [a * 100 for _, a in conf], color=C_ALT,
            linewidth=2, label="Güven tabanı (max softmax)")
    ax.plot([c * 100 for c, _ in ent], [a * 100 for _, a in ent], color=C_MAIN,
            linewidth=2, label="Entropi tavanı (normalize)")
    ax.scatter([98.324], [99.8588], s=70, color=C_MAIN, zorder=5,
               edgecolor="white", linewidth=1.5)
    ax.annotate("Tez işletim noktası (ent_p95)\nkapsama %98,3 · doğruluk %99,86",
                (98.324, 99.8588), textcoords="offset points", xytext=(-186, 34),
                fontsize=8, color="#1C2422",
                arrowprops=dict(arrowstyle="-", color=C_MUTED, linewidth=0.8))
    ax.axhline(99.2656, color=C_MUTED, linewidth=1, linestyle=":")
    ax.text(99.9, 99.285, "kapısız doğruluk %99,27", fontsize=8, color=C_MUTED, ha="right")
    ax.set_xlabel("Otomatik karara bağlanan akış oranı — kapsama (%)")
    ax.set_ylabel("Tutulan doğruluk (%)")
    ax.set_title("Şekil 4.6  Entropi kapısının işletim karakteristiği")
    ax.set_xlim(90, 100.4)
    tr_ticks(ax, "x"); tr_ticks(ax, "y")
    ax.legend(frameon=False, loc="lower left", fontsize=9, bbox_to_anchor=(0.0, 0.06))
    save(fig, "sekil_4_6_tau_isletim_karakteristigi")


def fig_transfer() -> None:
    """Sekil 4.7 - per-family cross-dataset transfer collapse (Design B)."""
    d = json.load(open(ROOT / "thesis/crossdataset/raw_results.json"))
    fam = d["cross"]["design_b"]["per_family"]
    order = ["DDoS", "Recon", "Benign", "Spoofing", "DoS"]
    order = [f for f in order if f in fam] + [f for f in fam if f not in order]
    vals = [fam[f]["f1"]["mean"] for f in order]
    errs = [fam[f]["f1"]["std"] for f in order]
    labels = {"Recon": "Keşif (Recon)", "Benign": "Benign", "Spoofing": "Sahtecilik",
              "DoS": "DoS", "DDoS": "DDoS"}

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar(range(len(order)), vals, yerr=errs, color=C_MAIN, width=0.6,
           error_kw=dict(ecolor=C_MUTED, capsize=3, lw=1))
    ax.axhline(0.9037, color=C_MUTED, linewidth=1.4, linestyle="--")
    ax.text(len(order) - 0.5, 0.915, "veri kümesi içi macro-F1 0,9037", ha="right",
            fontsize=8, color=C_MUTED)
    ax.axhline(0.19654, color=C_ALT, linewidth=1.4, linestyle=":")
    ax.text(len(order) - 0.5, 0.155, "çapraz veri kümesi macro-F1 0,1965", ha="right", fontsize=8, color=C_ALT)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([labels.get(f, f) for f in order], fontsize=9)
    ax.set_ylabel("F1 (CICIoT2023 üzerinde)")
    ax.set_title("Şekil 4.7  Aile düzeyinde çapraz veri kümesi aktarımı (üç tohum)")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="x", visible=False)
    tr_ticks(ax, "y")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.2f}".replace(".", ","), ha="center", fontsize=8.5)
    save(fig, "sekil_4_7_capraz_veri_kumesi_aktarimi")


if __name__ == "__main__":
    print("Turkish thesis figures ->", OUT)
    fig_dedup_perclass()
    fig_imbalance()
    fig_ablation()
    fig_tau_curve()
    fig_transfer()
    print("done")
