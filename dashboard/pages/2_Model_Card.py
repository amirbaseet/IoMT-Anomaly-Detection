"""Page 5 — Model Card.

Static page: intended use, training data, headline performance, 8-row
limitations table with closure status, ethical considerations, BibTeX.
All numerical claims are hard-coded with citations to README sections;
no CSV reads.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.data_loader import GITHUB_URL, TEST_SET_SIZE

st.set_page_config(
    page_title="IoMT IDS — Model Card",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Header --------------------------------------------------------------
st.title("Model Card: IoMT Hybrid IDS")
st.caption(
    "Path B Tier 1 + Tier 2 hardened · Tier 3 (this dashboard) in progress · "
    "Thesis defense pending"
)

st.divider()

# --- Intended use --------------------------------------------------------
st.subheader("Intended use")
st.markdown(
    "This system is a **hybrid intrusion-detection model for medical IoT (IoMT) "
    "networks**: it screens 44-feature CICIoMT2024-schema network flows and flags "
    "anomalous traffic via a per-class entropy-and-confidence fusion rule "
    "(§15D-published operating point). It is **not** a clinical decision-support "
    "tool, does **not** identify patients, and is **not** a replacement for "
    "perimeter security or layered defence-in-depth — it operates on already-"
    "permitted intra-LAN traffic and triages anomalies for human review."
)

st.divider()

# --- Training data -------------------------------------------------------
st.subheader("Training data")
st.markdown(
    f"""
**Dataset:** CICIoMT2024 (Canadian Institute for Cybersecurity, 2024) — 19 attack
classes spanning reconnaissance, MQTT-targeted DoS / DDoS, ARP spoofing, and benign
traffic.

**Volume:** ~3.6M flows post-cleaning · 70 / 15 / 15 train / val / test split
(test set = **{TEST_SET_SIZE:,} flows**).

**Acknowledged duplication:** ~37% of flows are exact duplicates within and across
splits in the source dataset. This is documented in README §3; the per-fold KS
framework (§15C.10) confirms benign-distribution stability across splits despite
the duplication.
"""
)

st.divider()

# --- Performance ---------------------------------------------------------
st.subheader("Performance (test set)")
m1, m2, m3 = st.columns(3)
m1.metric("F1-macro (E7, 19-class)", "0.9076")
m2.metric("Accuracy", "99.27%")
m3.metric("H2-strict (eligible LOO targets)", "4 / 4")
st.caption(
    "Operating point: entropy percentile p93.0, confidence floor 0.70, "
    "FPR 0.247 (within 0.25 budget). H2-strict source: §15D continuous sweep. "
    "See the **Threshold Tuning** page for live exploration."
)

st.divider()

# --- Known limitations ---------------------------------------------------
st.subheader("Known limitations & closure status")

limitations = pd.DataFrame(
    [
        {"Limitation": "Single random seed",          "Closed by / Status": "Closed by §15B (5-seed multi-seed; σ_strict_avg = 0.022, 0/19 cells fail)"},
        {"Limitation": "Discrete-grid threshold",     "Closed by / Status": "Closed by §15D (continuous sweep, p93.0 refined as new optimum)"},
        {"Limitation": "Aggregate-only KS",           "Closed by / Status": "Closed by §15C.10 update (per-fold KS table)"},
        {"Limitation": "Test-drawn SHAP background",  "Closed by / Status": "Closed by §16.7B (train-bg → test-bg Kendall τ_top10 = 0.927)"},
        {"Limitation": "VAE replacement open",        "Closed by / Status": "Closed by §15E (β-VAE β=0.5 substitution-equivalent: Δ_strict_avg = −0.0001)"},
        {"Limitation": "DDoS vs DoS boundary",        "Closed by / Status": "Open: §16.4 cosine = 0.991 documented; future per-class layer (§17 roadmap)"},
        {"Limitation": "Profiling data unused",       "Closed by / Status": "Open: §17 future work — vendor / device-fingerprint signals not yet integrated"},
        {"Limitation": "Recon_Ping_Sweep eligibility", "Closed by / Status": "Open: dataset-driven, n_test = 169 (small sample → high recall variance)"},
    ]
)
st.table(limitations)
st.caption(
    "Five Tier-1 / Tier-2 limitations are closed by the corresponding hardening "
    "sections; three remain open and are routed to §17 future work."
)

st.divider()

# --- Ethical considerations ----------------------------------------------
st.subheader("Ethical considerations")
st.markdown(
    "Operational deployment in a medical-IoT environment carries two material risks. "
    "**(1) False-positive cost.** The published §15D operating point has a 0.247 "
    "false-alert rate on the LOO-benign rescue path; a deployed clinic running "
    "thousands of flows per minute would surface a substantial alert volume that "
    "must be triaged with care. False alerts on life-critical telemetry "
    "(infusion pumps, patient monitors, ventilators) cannot be auto-suppressed "
    "without explicit operator review. **(2) Distribution drift.** The model is "
    "trained on CICIoMT2024 (a 2024 capture); deployment-time traffic from new "
    "device firmware, new MQTT brokers, or post-2024 attack tooling may degrade "
    "detection without warning. The §15C.10 per-fold KS framework provides one "
    "drift signal but is not a substitute for periodic re-validation against "
    "fresh captures, and the Recon_Ping_Sweep n=169 limit (open) means low-volume "
    "attack classes cannot be statistically separated from sampling noise without "
    "additional collection."
)

st.divider()

# --- Citation ------------------------------------------------------------
st.subheader("Citation")
st.markdown(
    f"If you reference this work, please cite the public repository "
    f"({GITHUB_URL}). A BibTeX template is below — replace the author field "
    "with your preferred attribution before citing."
)
st.code(
    r"""@misc{iomt_hybrid_ids_2026,
  author       = {{amirbaseet}},
  title        = {{IoMT Hybrid IDS: Path B Hybrid Anomaly Detection for Medical IoT}},
  year         = {2026},
  howpublished = {\url{https://github.com/amirbaseet/IoMT-Anomaly-Detection}},
  note         = {Path B Tier 1 + Tier 2 hardened; thesis defense pending}
}""",
    language="bibtex",
)

st.caption(
    "Numerical claims on this page are hard-coded with citations to README sections; "
    "see `Project_Journey_Complete.md` for full methodology."
)
