"""
Label mapping for both datasets -> {binary, 5-family} label spaces.

Design A (binary):  benign vs attack.
Design B (5 families): Benign, Spoofing, Recon, DoS, DDoS.

CICIoT2023 classes with no CICIoMT2024 counterpart are OUT OF SCOPE for
Design B and dropped from the B test set (mapped to ``None``); they remain
'attack' for Design A.

All functions are pure: they read a raw label string and return a mapped
label (or ``None``), never mutating inputs.
"""
from __future__ import annotations

# Canonical 5-family names + binary names (single source of truth).
BENIGN = "Benign"
SPOOFING = "Spoofing"
RECON = "Recon"
DOS = "DoS"
DDOS = "DDoS"
FAMILIES = (BENIGN, SPOOFING, RECON, DOS, DDOS)   # fixed order for reports

ATTACK = "attack"
BINARY_CLASSES = (BENIGN.lower(), ATTACK)          # ("benign", "attack")


# --------------------------------------------------------------------------- #
# CICIoT2023 (test) — raw uppercase strings observed in the merged CSVs.
# --------------------------------------------------------------------------- #
# Out-of-scope raw labels: no CICIoMT2024 counterpart (design doc §4).
CICIOT2023_OUT_OF_SCOPE = frozenset({
    "MIRAI-GREETH_FLOOD", "MIRAI-GREIP_FLOOD", "MIRAI-UDPPLAIN",
    "SQLINJECTION", "XSS", "BACKDOOR_MALWARE", "BROWSERHIJACKING",
    "COMMANDINJECTION", "UPLOADING_ATTACK", "DICTIONARYBRUTEFORCE",
})


def normalize_ciciot2023(raw) -> str:
    """
    Strip the trailing CRLF (TRAP 2) and surrounding whitespace, upper-case.

    Robust to non-string cells: a missing/NaN label (some merged CSV rows carry
    an empty last field) normalizes to "" so it can be recognised as invalid and
    excluded, rather than crashing or being mislabelled 'attack'.
    """
    if not isinstance(raw, str):
        if raw is None or (isinstance(raw, float) and raw != raw):  # NaN
            return ""
        raw = str(raw)
    return raw.replace("\r", "").replace("\n", "").strip().upper()


def is_known_ciciot2023(raw) -> bool:
    """True if the label is a recognised CICIoT2023 class (benign, a shared
    family, or a named out-of-scope attack).  Malformed/empty labels -> False."""
    lab = normalize_ciciot2023(raw)
    if lab == "":
        return False
    return ciciot2023_family(raw) is not None or lab in CICIOT2023_OUT_OF_SCOPE


def ciciot2023_binary(raw: str) -> str:
    """Design A: BENIGN -> 'benign', everything else -> 'attack'."""
    return BENIGN.lower() if normalize_ciciot2023(raw) == "BENIGN" else ATTACK


def ciciot2023_family(raw: str) -> str | None:
    """
    Design B: map a CICIoT2023 label to one of the 5 shared families, or
    ``None`` if the class is out of scope (to be dropped from the B test set).

    Prefix-based so it captures every DDOS-* member the design doc intends
    (incl. DDOS-SLOWLORIS and the *_FRAGMENTATION variants), not just the
    ones named in the table.
    """
    lab = normalize_ciciot2023(raw)
    if lab == "BENIGN":
        return BENIGN
    if lab in CICIOT2023_OUT_OF_SCOPE:
        return None
    if lab in ("MITM-ARPSPOOFING", "DNS_SPOOFING"):
        return SPOOFING
    if lab.startswith("RECON-") or lab == "VULNERABILITYSCAN":
        return RECON
    if lab.startswith("DDOS-"):
        return DDOS
    if lab.startswith("DOS-"):
        return DOS
    # Unknown label -> out of scope, but the caller logs it so it never
    # silently vanishes into 'attack'/dropped without notice.
    return None


# --------------------------------------------------------------------------- #
# CICIoMT2024 (train) — the label is the per-class file's prefix, e.g.
# "TCP_IP-DDoS-SYN_train.pcap.csv" -> DDoS.  We derive the 6-class 'category'
# used by the frozen thesis (Benign, DDoS, DoS, MQTT, Recon, Spoofing), then
# reduce to the 5 shared families (dropping MQTT for Design B).
# --------------------------------------------------------------------------- #
def ciciomt2024_category(filename: str) -> str:
    """Map a CICIoMT2024 per-class filename to its 6-class thesis category."""
    stem = filename.split("_test.pcap")[0].split("_train.pcap")[0]
    if stem == "Benign":
        return BENIGN
    if stem == "ARP_Spoofing":
        return SPOOFING
    if stem.startswith("Recon-"):
        return RECON
    if stem.startswith("MQTT-"):
        return "MQTT"                      # IoMT-specific — out of scope for B
    if stem.startswith("TCP_IP-DDoS-"):
        return DDOS
    if stem.startswith("TCP_IP-DoS-"):
        return DOS
    raise ValueError(f"Unrecognised CICIoMT2024 class file: {filename!r}")


def ciciomt2024_binary(filename: str) -> str:
    """Design A train label: Benign -> 'benign', all attack files -> 'attack'."""
    return BENIGN.lower() if ciciomt2024_category(filename) == BENIGN else ATTACK


def ciciomt2024_family(filename: str) -> str | None:
    """
    Design B train label: keep the 5 shared families, drop MQTT (``None``).
    """
    cat = ciciomt2024_category(filename)
    return cat if cat in FAMILIES else None
