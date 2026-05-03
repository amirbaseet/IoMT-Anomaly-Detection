"""
verify_report_numbers.py — Cross-check numbers in the course-project report
against canonical results files.

Purpose
-------
After Waves 1-5 of fabrication cleanup, this script provides a comprehensive
final pass: every labeled numeric claim in
Course_Project_FULL_REPORT_FINAL.md is matched against canonical sources
(JSON metric files, CSV tables, computed AUCs).

Output
------
scripts/verification_report.md — three buckets:
  GREEN  : value matches canonical source within tolerance
  YELLOW : value matches some canonical source but entity context unclear
  RED    : no canonical source contains this value (likely fabrication)

Tolerances
----------
F1/MCC/AUC      : ±0.001 base + rounding-aware match (handles 2-decimal report values)
Accuracy %      : ±0.05 percentage points
Cohen's d       : ±0.01
Detection rate %: ±1.0 percentage point
Support counts  : exact match
Pearson r       : ±0.005

Usage
-----
    cd ~/IoMT-Project
    source venv/bin/activate
    python scripts/verify_report_numbers.py

Reads:
    course-project/Course_Project_FULL_REPORT_FINAL.md
    results/supervised/metrics/E*_multiclass.json
    results/supervised/metrics/E*_classification_report_test.json
    results/supervised/metrics/E5_vs_E5G_comparison.csv
    eda_output/{imbalance_table,feature_target_cohens_d,high_correlation_pairs}.csv
    results/unsupervised/metrics/{per_class_detection_rates,model_comparison}.csv
    results/unsupervised/scores/{ae_test_mse,if_test_scores}.npy
    preprocessed/full_features/y_test.csv
"""
from __future__ import annotations

import json
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path.home() / "IoMT-Project"
REPORT_PATH = ROOT / "course-project" / "Course_Project_FULL_REPORT_FINAL.md"
OUT_PATH = ROOT / "scripts" / "verification_report.md"


# ============================================================================
# 1. LOAD CANONICAL REGISTRY
# ============================================================================
def load_canonical_registry() -> list[dict]:
    """Build a flat registry of (entity, metric, value, source) tuples."""
    reg: list[dict] = []
    sup_metrics = ROOT / "results" / "supervised" / "metrics"

    # ---- E*_multiclass.json (overall metrics per experiment) ----
    for f in sorted(sup_metrics.glob("E*_multiclass.json")):
        with open(f) as fp:
            data = json.load(fp)
        eid = data.get("experiment", f.stem.split("_")[0])
        src = str(f.relative_to(ROOT))
        for k, v in data.items():
            if isinstance(v, (int, float)) and not k.startswith("n_"):
                reg.append({"entity": eid, "metric": k, "value": float(v), "source": src})
                # Also unscoped, so "macro F1 0.9076" without explicit eid still matches
                if k in ("test_f1_macro", "test_mcc", "test_accuracy",
                         "test_f1_weighted", "test_recall_macro",
                         "test_precision_macro"):
                    reg.append({"entity": "*", "metric": k, "value": float(v), "source": src})

    # ---- E*_classification_report_test.json (per-class metrics) ----
    for f in sorted(sup_metrics.glob("E*_classification_report_test.json")):
        with open(f) as fp:
            data = json.load(fp)
        eid = f.stem.split("_")[0]
        src = str(f.relative_to(ROOT))
        for cls, vals in data.items():
            if cls in ("accuracy", "macro avg", "weighted avg") or not isinstance(vals, dict):
                continue
            for vk, vv in vals.items():
                # vk is one of: precision, recall, f1-score, support
                reg.append({"entity": cls, "metric": f"perclass.{vk}.{eid}",
                            "value": float(vv), "source": src})
                # Unscoped (no experiment) for forgiving matches
                reg.append({"entity": cls, "metric": f"perclass.{vk}",
                            "value": float(vv), "source": src})

    # ---- E5_vs_E5G_comparison.csv ----
    f = sup_metrics / "E5_vs_E5G_comparison.csv"
    if f.exists():
        df = pd.read_csv(f)
        src = str(f.relative_to(ROOT))
        for _, row in df.iterrows():
            metric = str(row["metric_label"])
            reg.append({"entity": "E5G", "metric": f"comparison_{metric}",
                        "value": float(row["rf_gini_E5G"]), "source": src})
            reg.append({"entity": "E5", "metric": f"comparison_{metric}",
                        "value": float(row["rf_entropy_E5"]), "source": src})

    # ---- imbalance_table.csv ----
    df = pd.read_csv(ROOT / "eda_output" / "imbalance_table.csv")
    src = "eda_output/imbalance_table.csv"
    for _, row in df.iterrows():
        cls = str(row["class"])
        reg.append({"entity": cls, "metric": "support_train",
                    "value": int(row["train"]), "source": src})
        reg.append({"entity": cls, "metric": "support_test",
                    "value": int(row["test"]), "source": src})
        reg.append({"entity": cls, "metric": "train_pct",
                    "value": float(row["train_%"]), "source": src})
        reg.append({"entity": cls, "metric": "imbalance_ratio",
                    "value": float(row["ratio_vs_largest"]), "source": src})

    # ---- feature_target_cohens_d.csv ----
    df = pd.read_csv(ROOT / "eda_output" / "feature_target_cohens_d.csv")
    src = "eda_output/feature_target_cohens_d.csv"
    for _, row in df.iterrows():
        feat = str(row.iloc[0])
        d_val = float(row.iloc[1])
        reg.append({"entity": feat, "metric": "cohens_d", "value": d_val, "source": src})
        reg.append({"entity": "*", "metric": "cohens_d", "value": d_val, "source": src})

    # ---- high_correlation_pairs.csv ----
    df = pd.read_csv(ROOT / "eda_output" / "high_correlation_pairs.csv")
    src = "eda_output/high_correlation_pairs.csv"
    for _, row in df.iterrows():
        pair = f"{row['feature_a']}<->{row['feature_b']}"
        reg.append({"entity": pair, "metric": "pearson_abs_corr",
                    "value": float(row["abs_corr"]), "source": src})
        reg.append({"entity": "*", "metric": "pearson_abs_corr",
                    "value": float(row["abs_corr"]), "source": src})

    # ---- per_class_detection_rates.csv ----
    f = ROOT / "results" / "unsupervised" / "metrics" / "per_class_detection_rates.csv"
    if f.exists():
        df = pd.read_csv(f)
        src = str(f.relative_to(ROOT))
        for _, row in df.iterrows():
            cls = str(row["class"])
            model = str(row["model"])
            for thresh in ["p90", "p95", "p99", "mean_2std", "mean_3std"]:
                v = row.get(thresh)
                if pd.notna(v):
                    # Detection rate as fraction
                    reg.append({"entity": cls, "metric": f"detection_{model}_{thresh}",
                                "value": float(v), "source": src})
                    # Detection rate as percent (%55, %62 in report style)
                    reg.append({"entity": cls, "metric": f"detection_pct_{model}_{thresh}",
                                "value": float(v) * 100, "source": src})

    # ---- model_comparison.csv (AE vs IF) ----
    f = ROOT / "results" / "unsupervised" / "metrics" / "model_comparison.csv"
    if f.exists():
        df = pd.read_csv(f)
        src = str(f.relative_to(ROOT))
        for _, row in df.iterrows():
            metric = str(row["metric"])
            for model in ["Autoencoder", "IsolationForest"]:
                v = row.get(model)
                if pd.notna(v):
                    reg.append({"entity": model, "metric": metric,
                                "value": float(v), "source": src})

    # ---- AUC computed live from saved scores ----
    try:
        ae_scores = np.load(ROOT / "results" / "unsupervised" / "scores" / "ae_test_mse.npy")
        if_scores = np.load(ROOT / "results" / "unsupervised" / "scores" / "if_test_scores.npy")
        y_test = pd.read_csv(ROOT / "preprocessed" / "full_features" / "y_test.csv")
        y_binary = (y_test["label"] != "Benign").astype(int).values
        ae_auc = roc_auc_score(y_binary, ae_scores)
        if_auc = roc_auc_score(y_binary, -if_scores)  # IF: higher=normal → negate
        reg.append({"entity": "Autoencoder", "metric": "AUC_computed",
                    "value": float(ae_auc),
                    "source": "computed from ae_test_mse.npy + y_test.csv"})
        reg.append({"entity": "IsolationForest", "metric": "AUC_computed",
                    "value": float(if_auc),
                    "source": "computed from if_test_scores.npy (negated) + y_test.csv"})
    except Exception as e:
        print(f"  [warn] could not compute AUC: {e}")

    # ---- thresholds.json ----
    f = ROOT / "results" / "unsupervised" / "thresholds.json"
    if f.exists():
        with open(f) as fp:
            data = json.load(fp)
        src = str(f.relative_to(ROOT))
        for tname, tval in data.get("thresholds", {}).items():
            reg.append({"entity": "AE_threshold", "metric": tname,
                        "value": float(tval), "source": src})

    # ---- Hand-curated facts from findings.md ----
    reg.append({"entity": "PCA", "metric": "k_95_var", "value": 22, "source": "eda_output/findings.md"})
    reg.append({"entity": "PCA", "metric": "k_99_var", "value": 28, "source": "eda_output/findings.md"})
    reg.append({"entity": "dataset", "metric": "n_features_raw", "value": 45, "source": "eda_output/findings.md"})
    reg.append({"entity": "dataset", "metric": "n_features_after_drop", "value": 44, "source": "eda_output/findings.md"})
    reg.append({"entity": "dataset", "metric": "n_classes", "value": 19, "source": "eda_output/findings.md"})

    return reg


# ============================================================================
# 2. PARSE REPORT
# ============================================================================
# Patterns: (regex, metric_label, base_tolerance, value_extractor)
PATTERNS = [
    # F1 score
    (re.compile(r"F1\s*[:=]\s*(\d+\.\d+)", re.IGNORECASE), "F1", 0.001, float),
    # macro F1 / weighted F1 (with optional separator)
    (re.compile(r"macro\s*F1[\s:=]+(\d+\.\d+)", re.IGNORECASE), "macro_F1", 0.001, float),
    (re.compile(r"weighted\s*F1[\s:=]+(\d+\.\d+)", re.IGNORECASE), "weighted_F1", 0.001, float),
    # MCC
    (re.compile(r"MCC[\s:=]+(\d+\.\d+)"), "MCC", 0.001, float),
    # AUC
    (re.compile(r"AUC[\s:=]*(\d+\.\d+)"), "AUC", 0.001, float),
    # accuracy as %
    (re.compile(r"%\s*(\d+(?:\.\d+)?)\s*(?:doğruluk|accuracy|acc\.)", re.IGNORECASE), "accuracy_pct", 0.05, float),
    (re.compile(r"(?:doğruluk|accuracy)\D{0,5}%?\s*(\d+(?:\.\d+)?)\s*%", re.IGNORECASE), "accuracy_pct", 0.05, float),
    # Cohen's d
    (re.compile(r"Cohen.{0,3}d[\s:=>]+(\d+\.\d+)", re.IGNORECASE), "cohens_d", 0.01, float),
    # Detection rate %
    (re.compile(r"detection\s*rate[\s:=]+%?\s*(\d+(?:\.\d+)?)", re.IGNORECASE), "detection_pct", 1.0, float),
    # Pearson r
    (re.compile(r"\br\s*=\s*(\d+\.\d+)"), "pearson_r", 0.005, float),
    # Recall / Precision (per-class)
    (re.compile(r"Recall[\s:=]+(\d+\.\d+)"), "recall", 0.005, float),
    (re.compile(r"Precision[\s:=]+(\d+\.\d+)"), "precision", 0.005, float),
]

# Class names known a priori
CLASS_NAMES = [
    "ARP_Spoofing", "Benign",
    "DDoS_ICMP", "DDoS_SYN", "DDoS_TCP", "DDoS_UDP",
    "DoS_ICMP", "DoS_SYN", "DoS_TCP", "DoS_UDP",
    "MQTT_DDoS_Connect_Flood", "MQTT_DDoS_Publish_Flood",
    "MQTT_DoS_Connect_Flood", "MQTT_DoS_Publish_Flood", "MQTT_Malformed_Data",
    "Recon_OS_Scan", "Recon_Ping_Sweep", "Recon_Port_Scan", "Recon_VulScan",
]


def find_entity_in_context(context: str) -> str | None:
    """Identify which entity the surrounding context refers to."""
    for cls in CLASS_NAMES:
        if cls in context:
            return cls
    if "Otoenkoder" in context or "Autoencoder" in context:
        return "Autoencoder"
    if "Isolation" in context or "IsolationForest" in context:
        return "IsolationForest"
    for eid in ["E1", "E2", "E3", "E4", "E5G", "E5", "E6", "E7", "E8"]:
        if re.search(rf"\b{eid}\b", context):
            return eid
    return None


def parse_report_numbers(report_text: str) -> list[dict]:
    """Find all labeled numeric claims with surrounding context."""
    found: list[dict] = []
    for lineno, line in enumerate(report_text.splitlines(), 1):
        if len(line.strip()) < 5 or line.strip().startswith("|---"):
            continue
        for regex, label, tol, extractor in PATTERNS:
            for m in regex.finditer(line):
                try:
                    val = extractor(m.group(1))
                except (ValueError, TypeError):
                    continue
                # Capture context: 50 chars before/after match
                start = max(0, m.start() - 60)
                end = min(len(line), m.end() + 60)
                ctx = line[start:end].strip()
                found.append({
                    "line": lineno,
                    "metric": label,
                    "value": val,
                    "tolerance": tol,
                    "context": ctx,
                })

        # Table-row pattern: |  ClassName  |  1,234  |  0.9876  |  ...
        table_match = re.match(
            r"\|\s*\*?\*?([A-Za-z_][A-Za-z_0-9\s]*?)\*?\*?\s*\|\s*\*?\*?([\d,]+)\*?\*?\s*\|\s*\*?\*?(\d+\.\d+)\*?\*?\s*\|",
            line,
        )
        if table_match:
            cls_name = table_match.group(1).strip()
            try:
                support_val = int(table_match.group(2).replace(",", ""))
                f1_val = float(table_match.group(3))
                # Heuristic: only treat as a per-class table row if class name is known
                if cls_name in CLASS_NAMES:
                    found.append({
                        "line": lineno, "metric": "table_support", "value": support_val,
                        "tolerance": 0, "context": f"table row | {cls_name}",
                        "entity_hint": cls_name,
                    })
                    found.append({
                        "line": lineno, "metric": "table_F1", "value": f1_val,
                        "tolerance": 0.001, "context": f"table row | {cls_name}",
                        "entity_hint": cls_name,
                    })
            except (ValueError, TypeError):
                pass

    return found


# ============================================================================
# 3. CLASSIFY MATCHES
# ============================================================================
def values_match(canonical: float, found: float, base_tol: float) -> bool:
    """Match within base tolerance OR within rounding tolerance of `found`'s precision."""
    if abs(canonical - found) <= base_tol:
        return True
    # Rounding-aware: if `found` is e.g. 0.76 (2 decimals), allow 0.755-0.764999 to match
    s = repr(float(found))
    if "." in s:
        decimals = len(s.split(".")[1].rstrip("0"))
        if decimals > 0:
            rounding_tol = 0.5 * (10 ** -decimals) + 1e-9
            if abs(canonical - found) <= rounding_tol:
                return True
            # Or strict round-to-precision equality
            if round(canonical, decimals) == round(found, decimals):
                return True
    return False


def classify_match(found: dict, registry: list[dict]) -> tuple[str, dict | None]:
    """Match a found number against canonical registry. Return (status, best_match)."""
    metric = found["metric"]
    val = found["value"]
    tol = found["tolerance"]
    ctx = found["context"]
    entity_hint = found.get("entity_hint") or find_entity_in_context(ctx)

    # Stage 1: collect all canonical entries whose value matches.
    # For percent-style report values (accuracy_pct, detection_pct), also try
    # comparing against canonical_fraction * 100 — canonical files store accuracies
    # as fractions like 0.9852, while the report cites them as percents like 98.52%.
    candidates = []
    for r in registry:
        if values_match(r["value"], val, tol):
            candidates.append(r)
        elif metric in ("accuracy_pct", "detection_pct") and 0.0 <= r["value"] <= 1.0:
            if values_match(r["value"] * 100, val, tol):
                candidates.append(r)

    if not candidates:
        return "RED", None

    # Stage 2: refine by entity
    if entity_hint:
        narrowed = [c for c in candidates if c.get("entity") in (entity_hint, "*")]
        if narrowed:
            return "GREEN", narrowed[0]
        # Value matches, but for a different entity than context suggests
        return "YELLOW", candidates[0]

    # No entity hint — accept first match
    if len(candidates) <= 3:
        return "GREEN", candidates[0]
    return "YELLOW", candidates[0]


# ============================================================================
# 4. WRITE VERIFICATION REPORT
# ============================================================================
def write_report(found_with_status: list[dict]) -> None:
    green = [x for x in found_with_status if x["status"] == "GREEN"]
    yellow = [x for x in found_with_status if x["status"] == "YELLOW"]
    red = [x for x in found_with_status if x["status"] == "RED"]

    lines = [
        "# Verification Report",
        "",
        f"**Source:** `course-project/Course_Project_FULL_REPORT_FINAL.md`",
        f"**Generated:** {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"**Summary:** {len(green)} GREEN · {len(yellow)} YELLOW · **{len(red)} RED**",
        "",
        "Legend:",
        "- ✅ GREEN — number matches a canonical source within tolerance",
        "- ⚠️ YELLOW — value matches *some* canonical source but entity context unclear",
        "- 🚩 RED — no canonical source contains this value (likely fabrication)",
        "",
        "---",
        "",
        "## 🚩 RED — Unmatched (likely fabrications)",
        "",
    ]

    if not red:
        lines.append("_None — all cited numbers traceable to canonical files._\n")
    else:
        lines.append("| Line | Metric | Value | Context |")
        lines.append("|---:|---|---:|---|")
        for x in red:
            ctx = x["context"].replace("|", "\\|").replace("\n", " ")[:80]
            v = x["value"]
            v_str = f"{v}" if isinstance(v, int) else f"{v:.4f}".rstrip("0").rstrip(".")
            lines.append(f"| {x['line']} | {x['metric']} | {v_str} | {ctx} |")
        lines.append("")

    lines += ["", "## ⚠️ YELLOW — Ambiguous (needs human triage)", ""]
    if not yellow:
        lines.append("_None._\n")
    else:
        # Show first 30 to keep file readable
        lines.append("| Line | Metric | Value | Best Match (entity.metric=value) | Context |")
        lines.append("|---:|---|---:|---|---|")
        for x in yellow[:50]:
            m = x.get("match", {}) or {}
            match_str = f"{m.get('entity', '?')}.{m.get('metric', '?')}={m.get('value', '?')}"
            ctx = x["context"].replace("|", "\\|").replace("\n", " ")[:60]
            v = x["value"]
            v_str = f"{v}" if isinstance(v, int) else f"{v:.4f}".rstrip("0").rstrip(".")
            lines.append(f"| {x['line']} | {x['metric']} | {v_str} | {match_str} | {ctx} |")
        if len(yellow) > 50:
            lines.append(f"\n_...and {len(yellow) - 50} more YELLOW items_")
        lines.append("")

    lines += [
        "",
        f"## ✅ GREEN — Verified ({len(green)} matches)",
        "",
        "_Verified matches not listed individually for brevity. "
        "First 10 shown for spot-check:_\n",
        "| Line | Metric | Value | Source |",
        "|---:|---|---:|---|",
    ]
    for x in green[:10]:
        m = x.get("match", {}) or {}
        v = x["value"]
        v_str = f"{v}" if isinstance(v, int) else f"{v:.4f}".rstrip("0").rstrip(".")
        lines.append(f"| {x['line']} | {x['metric']} | {v_str} | {m.get('source', '?')} |")
    lines.append("")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")


# ============================================================================
# 5. MAIN
# ============================================================================
def main() -> None:
    print("Loading canonical registry ...")
    registry = load_canonical_registry()
    print(f"  {len(registry):,} canonical entries loaded")

    print("Parsing report ...")
    report_text = REPORT_PATH.read_text(encoding="utf-8")
    found = parse_report_numbers(report_text)
    print(f"  {len(found)} numbers extracted")

    print("Classifying ...")
    results: list[dict] = []
    for f in found:
        status, match = classify_match(f, registry)
        f["status"] = status
        f["match"] = match
        results.append(f)

    n_green = sum(1 for r in results if r["status"] == "GREEN")
    n_yellow = sum(1 for r in results if r["status"] == "YELLOW")
    n_red = sum(1 for r in results if r["status"] == "RED")

    print()
    print(f"  ✅ GREEN  : {n_green}")
    print(f"  ⚠️  YELLOW : {n_yellow}")
    print(f"  🚩 RED    : {n_red}")
    print()

    write_report(results)
    print(f"Report written: {OUT_PATH.relative_to(ROOT)}")

    if n_red > 0:
        print()
        print(f"!!! {n_red} RED items require investigation !!!")


if __name__ == "__main__":
    main()
