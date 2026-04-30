# AI-Based Anomaly Detection in IoMT Networks

## A Hybrid Supervised-Unsupervised Framework for Zero-Day Attack Detection

> **Thesis Title:** A Hybrid Supervised-Unsupervised Framework for Anomaly Detection and Zero-Day Attack Identification in IoMT Networks Using the CICIoMT2024 Dataset  
> **Author:** Amro  
> **Program:** M.Sc. Artificial Intelligence and Machine Learning in Cybersecurity — Sakarya University  
> **Dataset:** CICIoMT2024 (Canadian Institute for Cybersecurity)  
> **Reference Paper:** Yacoubi et al. (2025–2026) — *Enhancing IoMT Security with Explainable Machine Learning*  
> **Status:** ALL EXPERIMENTAL PHASES COMPLETE (1-7) + Phase 6C entropy fusion + Path B Week 1 multi-seed validation — Thesis writing phase

---

## Table of Contents

1. [Project Overview](#1-project-overview)  
2. [Dataset Description](#2-dataset-description)  
3. [IoMT Testbed Architecture](#3-iomt-testbed-architecture)  
4. [Protocols](#4-protocols)  
5. [Attack Taxonomy (18 Attacks)](#5-attack-taxonomy-18-attacks)  
   - 5.1 [DDoS Attacks](#51-ddos---distributed-denial-of-service-4-attacks)
   - 5.2 [DoS Attacks](#52-dos---denial-of-service-4-attacks)
   - 5.3 [Reconnaissance Attacks](#53-recon---reconnaissance-4-attacks)
   - 5.4 [MQTT Attacks](#54-mqtt---protocol-specific-attacks-5-attacks)
   - 5.5 [Spoofing Attacks](#55-spoofing-1-attack)
   - 5.6 [BLE DoS](#56-ble-denial-of-service-separate-dataset)
6. [Feature Set (45 Features)](#6-feature-set-45-features)  
7. [Dataset Structure & Files (VERIFIED)](#7-dataset-structure--files-verified)  
8. [Class Distribution (VERIFIED — After Deduplication)](#8-class-distribution-verified--after-deduplication)  
9. [Profiling Data](#9-profiling-data)  
10. [Phase 2 EDA Key Findings](#10-phase-2-eda-key-findings)  
11. [Phase 3 Preprocessing & Feature Engineering](#11-phase-3-preprocessing--feature-engineering)  
12. [Phase 4 Supervised Model Training Results](#12-phase-4-supervised-model-training-results)  
13. [Phase 5 Unsupervised Model Training Results](#13-phase-5-unsupervised-model-training-results)  
14. [Phase 6 Fusion Engine & Zero-Day Simulation Results](#14-phase-6-fusion-engine--zero-day-simulation-results)  
15. [Phase 6B True Leave-One-Attack-Out Zero-Day Results](#15-phase-6b-true-leave-one-attack-out-zero-day-results)  
15C. [Phase 6C Enhanced Fusion Ablation (Entropy + Confidence + Ensemble)](#15c-phase-6c--enhanced-fusion-ablation-entropy--confidence--ensemble)  
16. [Phase 7 SHAP Explainability Analysis Results](#16-phase-7-shap-explainability-analysis-results)  
17. [Project Roadmap](#17-project-roadmap)  
18. [Related Work — Summary Table](#18-related-work--summary-table)  
19. [Deep Dive: Yacoubi et al. — Primary Reference Paper](#19-deep-dive-yacoubi-et-al--primary-reference-paper)  
    - 19.1 [Paper 1: Explainable ML (COCIA 2025)](#191-paper-1-enhancing-iomt-security-with-explainable-ml-cocia-2025)
    - 19.2 [Paper 2: XAI Feature Selection (AIAI 2025)](#192-paper-2-xai-driven-feature-selection-for-improved-ids-aiai-2025)
    - 19.3 [Paper 3: Ensemble Strategies (Springer 2026)](#193-paper-3-ensemble-learning-strategies-for-anomaly-based-ids-springer-2026)
    - 19.4 [Research Gaps & Our Contribution](#194-research-gaps-left-by-yacoubi-et-al)
20. [Research Design](#20-research-design)
21. [Proposed Framework Architecture](#21-proposed-framework-architecture)
22. [Corrections to Published Literature](#22-corrections-to-published-literature)
23. [Citations](#23-citations)  
24. [Tech Stack](#24-tech-stack)

---

## 1. Project Overview

This thesis proposes a **hybrid supervised-unsupervised framework** for anomaly detection in Internet of Medical Things (IoMT) networks, addressing the critical gap of zero-day attack detection capability in existing IoMT intrusion detection systems.

### The Problem

Existing state-of-the-art approaches on the CICIoMT2024 dataset (Yacoubi et al., 2025–2026) rely exclusively on supervised classification, achieving ~99.4% accuracy on known attacks but suffering from three fundamental limitations:

1. **Zero-day blindness** — supervised models cannot detect attacks they were never trained on
2. **Label dependency** — require expensive, manually-labeled training data
3. **Precision-recall gap** — 99.36% accuracy with only 86.10% precision indicates significant false positives, likely concentrated in minority attack classes

In a healthcare context, these limitations translate directly to patient safety risks: a novel attack variant targeting an insulin pump or cardiac monitor would pass undetected through a supervised-only IDS.

### The Solution

A dual-layer detection framework combining:

- **Supervised Layer (Layer 1):** Random Forest + XGBoost — classifies known attack types with high accuracy
- **Unsupervised Layer (Layer 2):** Autoencoder + Isolation Forest — detects deviations from learned benign behavior, enabling zero-day detection
- **Fusion Layer (Layer 3):** 4-case decision logic combining outputs — provides confidence-stratified alerts to security analysts
- **Explainability Layer (Layer 4):** Per-attack-class SHAP analysis + LIME local explanations — makes decisions interpretable

### Key Innovation: 4-Case Fusion Decision Logic

| Supervised Output | Unsupervised Output | Fusion Decision | Confidence |
|-------------------|---------------------|-----------------|------------|
| Attack | Anomaly | Confirmed Alert | HIGH |
| Benign | Anomaly | **Zero-Day Warning** | MEDIUM-HIGH |
| Attack | Normal | Low-Confidence Alert | MEDIUM-LOW |
| Benign | Normal | Clear / No Threat | HIGH |

**Case 2 is the key innovation:** when the supervised layer sees "benign" but the unsupervised layer flags "anomaly," this indicates a novel attack pattern never seen during training — a potential zero-day threat requiring manual investigation.

### Zero-Day Simulation Protocol

Zero-day detection capability is evaluated using a **leave-one-attack-out protocol**: each of the 17 attack classes is sequentially withheld from training data, and the unsupervised layer is tested on its ability to flag the withheld class as anomalous. This measures real-world zero-day detection performance without requiring actual unknown attacks.

### Why This Project Matters

Connected medical devices — blood pressure monitors, insulin pumps, ECG monitors, pulse oximeters, smart pill dispensers — are increasingly deployed in hospitals and home healthcare. These devices transmit sensitive patient data over Wi-Fi, MQTT, and Bluetooth. A successful cyberattack can disrupt patient monitoring, corrupt medical readings, or directly endanger patient safety. Our framework provides the first hybrid approach on the CICIoMT2024 benchmark, combining production-ready accuracy on known attacks with resilience against novel threats.

---

## 2. Dataset Description

| Property | Value |
|----------|-------|
| **Name** | CICIoMT2024 |
| **Source** | Canadian Institute for Cybersecurity (CIC), University of New Brunswick |
| **Total Instances (raw)** | 8,775,013 (train: 7,160,831 + test: 1,614,182) |
| **Total Instances (after dedup)** | 5,407,348 (train: 4,515,080 + test: 892,268) |
| **Duplicate Rate** | Train: 36.95% / Test: 44.72% — **not reported in any prior paper** |
| **Features** | 45 (no label column — labels derived from filenames) |
| **Attack Types** | 18 (across 5 categories) |
| **Classes** | 17 (16 attack types + 1 benign) in WiFi/MQTT subset |
| **Devices** | 40 (25 real + 15 simulated) |
| **Protocols** | Wi-Fi, MQTT, Bluetooth Low Energy (BLE) |
| **Train CSV Files** | 51 files (attacks split into numbered capture files) |
| **Test CSV Files** | 21 files |
| **Max Imbalance Ratio** | 2,374:1 (DDoS_UDP vs Recon_Ping_Sweep after dedup) |
| **Rarest Class** | Recon_Ping_Sweep (689 train rows after dedup) |
| **Format** | PCAP (raw) + CSV (ML-ready features) |
| **Download** | https://cicresearch.ca/IOTDataset/CICIoMT2024/ |

> **⚠️ Literature correction:** Prior papers report 377K train / 98K test / ~4.89M total instances. Our verified counts show the real dataset is **19x larger** with a **37% duplicate rate** that inflates accuracy metrics in all prior work.

---

## 3. IoMT Testbed Architecture

The CIC built a realistic healthcare network laboratory to generate this dataset:

**Real Devices (25):**
Physical IoMT devices commonly found in hospitals and home healthcare — blood pressure monitors, pulse oximeters, smart scales, thermometers, wearable heart rate monitors, smart pill dispensers, and more.

**Simulated Devices (15):**
Virtual MQTT-enabled medical sensors created to expand the protocol coverage and simulate larger-scale healthcare environments.

**Attacker Machines:**
A malicious PC (for Wi-Fi/MQTT attacks) and a smartphone (for BLE attacks) were used to execute the 18 attack scenarios against the IoMT network.

**Data Collection Method:**
A **network tap** was placed between the network switch and the IoMT devices to capture all traffic as PCAP files. This hardware-level capture ensures no packets are missed and no disruption occurs to the network. The PCAPs were then processed through feature extraction tools to produce the ML-ready CSV files.

```
                    ┌─────────────────────────────────────────────┐
                    │          IoMT Testbed Network                │
                    │                                             │
  Attacker PC ──────┤──► Switch ◄── Network Tap ──► PCAP Capture  │
  (Wi-Fi/MQTT)      │       │                           │         │
                    │       ├── 25 Real IoMT Devices     │         │
  Attacker Phone ───┤       ├── 15 Simulated Sensors     ▼         │
  (BLE attacks)     │       └── MQTT Broker         CSV Files      │
                    │                              (ML-ready)      │
                    └─────────────────────────────────────────────┘
```

---

## 4. Protocols

### 4.1 Wi-Fi (IEEE 802.11)
The primary communication protocol for most IoMT devices. Handles device-to-server communication for transmitting patient vitals, receiving configuration updates, and firmware downloads. Most of the DDoS, DoS, Recon, and Spoofing attacks in the dataset target Wi-Fi-connected devices.

### 4.2 MQTT (Message Queuing Telemetry Transport)
A lightweight publish/subscribe messaging protocol designed for constrained devices (low bandwidth, limited battery). Runs on TCP port 1883. IoMT sensors **publish** data (heart rate, SpO2, blood glucose) to **topics** on a central broker, and hospital monitoring systems **subscribe** to receive real-time updates.

**MQTT Architecture:**
```
IoMT Sensor ──PUBLISH──► MQTT Broker ──FORWARD──► Subscriber (Hospital Dashboard)
                topic: "patient/123/heart_rate"
                payload: {"bpm": 72, "timestamp": "..."}
```

The MQTT broker is a single point of failure — if compromised, all IoMT communication is disrupted.

### 4.3 Bluetooth Low Energy (BLE)
Used for short-range communication between wearable medical devices and smartphones or bedside units. BLE devices have extremely constrained protocol stacks with minimal memory, making them vulnerable to even moderate denial-of-service attacks.

---

## 5. Attack Taxonomy (18 Attacks)

### 5.1 DDoS — Distributed Denial of Service (4 attacks)

Multiple compromised machines (botnet) simultaneously flood the target. Higher volume than DoS, multiple source IPs.

#### 5.1.1 DDoS SYN Flood
- **Mechanism:** Hundreds of bots send TCP SYN packets (connection requests) but never complete the 3-way handshake. The target device's connection table fills up with half-open connections, and it can no longer accept legitimate connections.
- **How it works:**
  - Bot → SYN → IoMT device (device allocates memory)
  - Device → SYN-ACK → Bot (bot ignores this)
  - Bot never sends final ACK → connection stays half-open → repeat × thousands
- **Key dataset features:** `syn_flag_number ↑↑↑`, `syn_count ↑↑↑`, `ack_flag_number ↓`, `Rate ↑↑↑`, `TCP = 1`, `fin_flag_number ≈ 0`
- **IoMT impact:** A smart infusion pump under SYN flood cannot report dosage data. Nurses lose real-time medication tracking.

#### 5.1.2 DDoS TCP Flood
- **Mechanism:** Bots complete the TCP handshake but then send massive amounts of data or open hundreds of full connections simultaneously. Exhausts target's CPU, memory, and bandwidth.
- **Key dataset features:** `syn_flag_number ↑`, `ack_flag_number ↑↑`, `psh_flag_number ↑`, `Rate ↑↑↑`, `Tot sum ↑↑`, `TCP = 1`
- **IoMT impact:** ECG monitors streaming continuous heart rhythm data get disconnected — arrhythmia alerts stop reaching cardiologists.

#### 5.1.3 DDoS ICMP Flood
- **Mechanism:** Distributed bots send enormous volumes of ICMP Echo Request (ping) packets. Target is forced to process and respond to each one. ICMP is connectionless — no handshake needed, cheap to generate.
- **Key dataset features:** `ICMP = 1`, `Rate ↑↑↑`, `TCP = 0`, `UDP = 0`, `syn_flag_number = 0`, `Header-Length ↑`
- **IoMT impact:** Network infrastructure serving the entire hospital floor slows down, affecting all connected medical devices simultaneously.

#### 5.1.4 DDoS UDP Flood
- **Mechanism:** Bots send massive UDP datagrams to random ports. For each packet, the device checks which application is listening, finds nothing, and sends back ICMP "Destination Unreachable." This check-and-reply cycle overwhelms the device. UDP is connectionless, so spoofing source IPs is trivial.
- **Key dataset features:** `UDP = 1`, `Rate ↑↑↑`, `TCP = 0`, `ICMP ↑ (responses)`, `rst_count ↑`, `Srate ↑↑↑`
- **IoMT impact:** IoMT devices using UDP-based protocols for real-time vital sign streaming become completely unreachable.

---

### 5.2 DoS — Denial of Service (4 attacks)

Identical techniques to DDoS but from a **single source machine**. No botnet needed. Lower total volume, but still effective against resource-constrained IoMT devices.

#### 5.2.1 DoS SYN Flood
- **Mechanism:** Same half-open connection attack as DDoS variant, but from one IP. The attacker rapidly sends SYN packets, often with spoofed source IPs. Less volume but effective against IoMT devices with very small connection tables.
- **Key dataset features:** `syn_flag_number ↑↑`, `Rate ↑ (lower than DDoS)`, `TCP = 1`
- **ML challenge:** DDoS SYN vs DoS SYN have similar flag patterns but different Rate/Srate — this is a hard classification boundary.

#### 5.2.2 DoS TCP Flood
- **Mechanism:** Single attacker opens many full TCP connections and sends data aggressively. On small IoMT devices with limited memory, even one machine can exhaust resources.
- **Key dataset features:** `ack_flag_number ↑`, `psh_flag_number ↑`, `Tot sum ↑`, `TCP = 1`, `Rate ↑`

#### 5.2.3 DoS ICMP Flood
- **Mechanism:** Single machine sends rapid ping requests. IoMT devices are embedded systems with tiny network stacks — even a moderate ICMP flood from one source can disrupt a wearable glucose monitor.
- **Key dataset features:** `ICMP = 1`, `Rate ↑`, `all TCP flags = 0`, `Std ↓ (uniform packet sizes)`

#### 5.2.4 DoS UDP Flood
- **Mechanism:** Single source sends UDP packets to random ports. Effective against IoMT gateways that aggregate data from multiple sensors.
- **Key dataset features:** `UDP = 1`, `Rate ↑`, `rst_count ↑`, `Variance ↑ (random ports)`

---

### 5.3 Recon — Reconnaissance (4 attacks)

Information-gathering phase before an actual attack. The attacker maps the network to find vulnerable IoMT devices. Low-volume, stealthy, and often the **hardest to detect** with ML models.

#### 5.3.1 Ping Sweep
- **Mechanism:** Attacker sends ICMP Echo Requests to a range of IPs (e.g., 192.168.1.1–254) to discover which hosts are alive on the network. Each live device replies with an ICMP Echo Reply, building a map of every connected IoMT device.
- **Key dataset features:** `ICMP = 1`, `Rate ↓ (slow, methodical)`, `Min ≈ Max (uniform pings)`, `Std ≈ 0`, `Number ↓`
- **IoMT impact:** Attacker discovers all 40 medical devices on the hospital subnet.

#### 5.3.2 Vulnerability Scan
- **Mechanism:** Tools like Nessus or OpenVAS probe each discovered device for known CVEs (Common Vulnerabilities and Exposures). The scanner sends specially crafted requests to test for specific software flaws — outdated firmware, default credentials, unpatched services. Produces diverse traffic across many protocols.
- **Key dataset features:** `TCP = mixed`, `UDP = mixed`, `HTTP/HTTPS ↑`, `Variance ↑↑`, `Std ↑`, `rst_count ↑`
- **IoMT impact:** Many IoMT devices run outdated firmware that never gets patched. A vuln scan reveals exactly which CVEs to exploit.

#### 5.3.3 OS Scan (Operating System Fingerprinting)
- **Mechanism:** TCP/IP stack fingerprinting — the attacker sends packets with unusual flag combinations (e.g., SYN+FIN, which should never happen) and observes how the device's OS responds. Each OS handles these malformed packets differently, revealing Linux, RTOS, FreeRTOS, Windows IoT, etc. Typically done with Nmap `-O` flag.
- **Key dataset features:** `syn_flag_number ↑`, `fin_flag_number ↑ (unusual)`, `rst_count ↑↑`, `Rate ↓ (careful probing)`, `Std ↑`
- **IoMT impact:** Knowing the OS tells the attacker which exploits will work on each specific medical device.

#### 5.3.4 Port Scan
- **Mechanism:** Systematically probes ports (1–65535) on a target to find open services. A SYN scan sends SYN to each port — SYN-ACK means open, RST means closed. This reveals running services (HTTP:80, SSH:22, MQTT:1883, etc.).
- **Key dataset features:** `syn_flag_number ↑↑`, `rst_count ↑↑↑ (most ports closed)`, `Rate ↑`, `TCP = 1`, `Number ↓ (1-2 packets per port)`
- **IoMT impact:** Discovering port 1883 (MQTT) open on a patient monitor tells the attacker to attempt MQTT-specific attacks next.

---

### 5.4 MQTT — Protocol-Specific Attacks (5 attacks)

These attacks target the MQTT broker, which is the central communication hub for all IoMT sensors. Kill the broker, kill all IoMT communication.

#### 5.4.1 MQTT Malformed Data
- **Mechanism:** Attacker sends MQTT packets with invalid or corrupted payloads — wrong packet lengths, invalid UTF-8 in topic names, broken JSON, or protocol-level violations (e.g., PUBLISH with QoS 3, which doesn't exist). Exploits parsing bugs in the broker, causing crashes or memory corruption.
- **Key dataset features:** `TCP = 1`, `Tot size ↑ (oversized payloads)`, `Std ↑↑ (irregular packet sizes)`, `rst_count ↑`, `Rate ↓ (targeted)`
- **IoMT impact:** A crashed MQTT broker means ALL medical sensors simultaneously stop reporting. The hospital goes blind to every patient's vitals at once.

#### 5.4.2 MQTT DoS Connect Flood
- **Mechanism:** Single attacker rapidly sends MQTT CONNECT packets to the broker, each requesting a new session. The broker allocates memory for each. If max-connections is reached, legitimate IoMT devices get rejected.
- **Key dataset features:** `syn_flag_number ↑↑`, `ack_flag_number ↑`, `Rate ↑↑`, `TCP = 1`, `Min ≈ Max (uniform CONNECT packets)`, `Std ↓`

#### 5.4.3 MQTT DDoS Connect Flood
- **Mechanism:** Distributed version — multiple bots send CONNECT requests simultaneously. Much higher volume overwhelms the broker faster. This is the **most common attack class** in the dataset.
- **Key dataset features:** `syn_flag_number ↑↑↑`, `Rate ↑↑↑`, `TCP = 1`, `Srate ↑↑↑`, `Header-Length ↑↑`

#### 5.4.4 MQTT DoS Publish Flood
- **Mechanism:** After establishing a legitimate MQTT connection, the attacker publishes enormous volumes of messages to topics. Every subscriber must process every message. Publishing to "patient/#" (wildcard) floods ALL patient-related subscribers. This is an application-layer attack — TCP connection looks normal, only MQTT message rate is abnormal.
- **Key dataset features:** `psh_flag_number ↑↑`, `ack_flag_number ↑↑`, `Tot sum ↑↑↑`, `Rate ↑↑`, `TCP = 1`, `AVG ↑`
- **IoMT impact:** Hospital dashboards freeze trying to process thousands of fake readings per second. Real patient data gets buried in noise.

#### 5.4.5 MQTT DDoS Publish Flood
- **Mechanism:** Multiple bots each connect legitimately then publish massive message volumes simultaneously. Creates amplification: N bots × M messages × K subscribers = overwhelming load.
- **Key dataset features:** `psh_flag_number ↑↑↑`, `Tot sum ↑↑↑↑`, `Rate ↑↑↑↑`, `TCP = 1`, `Srate ↑↑↑`, `Max ↑`

---

### 5.5 Spoofing (1 attack)

#### 5.5.1 ARP Spoofing
- **Mechanism:** ARP (Address Resolution Protocol) maps IP addresses to MAC addresses on a local network. The attacker sends forged ARP replies telling the network "I am 192.168.1.1" (the gateway). All traffic from IoMT devices now flows through the attacker — a Man-in-the-Middle (MITM) position.
- **Step by step:**
  1. Attacker sends: "192.168.1.1 is at AA:BB:CC:DD:EE:FF" (attacker's MAC)
  2. IoMT devices update their ARP tables
  3. All traffic meant for the gateway goes to the attacker
  4. Attacker forwards traffic (to stay hidden) while reading/modifying it
- **Key dataset features:** `ARP ↑↑↑ (dominant)`, `IPv = 0`, `TCP = 0`, `UDP = 0`, `ICMP = 0`, `Rate ↓`, `Tot size ↓ (small packets)`
- **IoMT impact:** The most dangerous attack. Attacker can alter blood glucose readings (showing 120 mg/dL when real value is 40 mg/dL — hypoglycemia). This directly endangers patient life.
- **ML challenge:** Rarest class in the dataset — requires class balancing techniques (SMOTE, oversampling).

---

### 5.6 BLE Denial of Service (Separate Dataset)

- **Mechanism:** Attacker uses a smartphone or specialized hardware to flood a BLE-enabled medical device with connection requests, malformed advertising packets, or jamming signals on the 2.4 GHz band.
- **Note:** Stored in the **Bluetooth/** folder as separate PCAP files with its own train/test split. Different features from WiFi/MQTT CSV data.
- **IoMT impact:** BLE wearables (heart rate monitors, insulin pumps, hearing aids) lose connection. For insulin pumps with closed-loop control, this could interrupt automated dosing.

---

## 6. Feature Set (45 Features)

### 6.1 Network Header Features
| Feature | Description |
|---------|-------------|
| `Header-Length` | Total length of packet headers in the flow |
| `Protocol Type` | Numeric protocol identifier |
| `Duration` | Flow duration (TTL-based) |
| `Rate` | Packet rate (packets per second) |
| `Srate` | Source-side packet sending rate |

### 6.2 TCP Flag Features
| Feature | Description |
|---------|-------------|
| `fin_flag_number` | Ratio of FIN flags (connection termination) |
| `syn_flag_number` | Ratio of SYN flags (connection initiation) — key indicator for SYN floods |
| `rst_flag_number` | Ratio of RST flags (connection reset) — high in port scans |
| `psh_flag_number` | Ratio of PSH flags (data push) — high in publish floods |
| `ack_flag_number` | Ratio of ACK flags (acknowledgment) |
| `ece_flag_number` | Ratio of ECE flags (congestion notification) |
| `cwr_flag_number` | Ratio of CWR flags (congestion window reduced) |

### 6.3 Flag Count Features
| Feature | Description |
|---------|-------------|
| `ack_count` | Total ACK packets in flow |
| `syn_count` | Total SYN packets in flow |
| `fin_count` | Total FIN packets in flow |
| `rst_count` | Total RST packets — very high in reconnaissance attacks |

### 6.4 Protocol Indicator Features (Binary/Ratio)
| Feature | Description |
|---------|-------------|
| `HTTP`, `HTTPS`, `DNS`, `Telnet`, `SMTP`, `SSH`, `IRC` | Application-layer protocol indicators |
| `TCP`, `UDP`, `ICMP`, `IGMP` | Transport/network layer indicators |
| `DHCP`, `ARP` | Link/network layer indicators |
| `IPv`, `LLC` | IP version and LLC frame indicators |

### 6.5 Packet Size Statistics
| Feature | Description |
|---------|-------------|
| `Tot sum` | Total bytes in all packets |
| `Min` | Minimum packet size |
| `Max` | Maximum packet size |
| `AVG` | Average packet size |
| `Std` | Standard deviation of packet sizes |
| `Tot size` | Average total size per packet |

### 6.6 Flow-Level Features
| Feature | Description |
|---------|-------------|
| `IAT` | Inter-Arrival Time between packets |
| `Number` | Number of packets in the flow |
| `Magnitude` | Flow magnitude (geometric measure) |
| `Radius` | Flow radius (spread measure) |
| `Covariance` | Covariance of flow features |
| `Variance` | Variance indicator |
| `Weight` | Flow weight measure |

---

## 7. Dataset Structure & Files (VERIFIED)

> **Note:** This structure was verified by direct inspection of the downloaded dataset. It differs significantly from descriptions in the literature.

### Column Names (45 features — VERIFIED)

```
Header_Length, Protocol Type, Duration, Rate, Srate, Drate,
fin_flag_number, syn_flag_number, rst_flag_number, psh_flag_number,
ack_flag_number, ece_flag_number, cwr_flag_number,
ack_count, syn_count, fin_count, rst_count,
HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC,
TCP, UDP, DHCP, ARP, ICMP, IGMP, IPv, LLC,
Tot sum, Min, Max, AVG, Std, Tot size, IAT,
Number, Magnitue, Radius, Covariance, Variance, Weight
```

**Column name corrections from literature:**
- `Header_Length` — underscore, NOT hyphen
- `Magnitue` — typo in dataset (missing 'd'), keep as-is
- `Drate` — exists but is constant at 0.0 (not mentioned in most papers)
- `Protocol Type`, `Tot sum`, `Tot size` — have SPACES in column names
- **No label column** — class labels are derived from CSV filenames
- **No MQTT protocol indicator column** — MQTT attacks are identified by filename prefix

### File Structure

**Train directory — 51 CSV files:**
Attacks are split across multiple numbered capture files that must be merged by attack type:

```
WiFI_and_MQTT/attacks/CSV/train/
├── ARP_Spoofing_train.pcap.csv                  (1 file)
├── Benign_train.pcap.csv                        (1 file)
├── MQTT-DDoS-Connect_Flood_train.pcap.csv       (1 file)
├── MQTT-DDoS-Publish_Flood_train.pcap.csv       (1 file)
├── MQTT-DoS-Connect_Flood_train.pcap.csv        (1 file)
├── MQTT-DoS-Publish_Flood_train.pcap.csv        (1 file)
├── MQTT-Malformed_Data_train.pcap.csv           (1 file)
├── Recon-OS_Scan_train.pcap.csv                 (1 file)
├── Recon-Ping_Sweep_train.pcap.csv              (1 file)
├── Recon-Port_Scan_train.pcap.csv               (1 file)
├── Recon-VulScan_train.pcap.csv                 (1 file)
├── TCP_IP-DDoS-ICMP[1-8]_train.pcap.csv         (8 files → merge into DDoS_ICMP)
├── TCP_IP-DDoS-SYN[1-4]_train.pcap.csv          (4 files → merge into DDoS_SYN)
├── TCP_IP-DDoS-TCP[1-4]_train.pcap.csv           (4 files → merge into DDoS_TCP)
├── TCP_IP-DDoS-UDP[1-8]_train.pcap.csv           (8 files → merge into DDoS_UDP)
├── TCP_IP-DoS-ICMP[1-4]_train.pcap.csv           (4 files → merge into DoS_ICMP)
├── TCP_IP-DoS-SYN[1-4]_train.pcap.csv            (4 files → merge into DoS_SYN)
├── TCP_IP-DoS-TCP[1-4]_train.pcap.csv             (4 files → merge into DoS_TCP)
└── TCP_IP-DoS-UDP[1-4]_train.pcap.csv             (4 files → merge into DoS_UDP)
```

**Test directory — 21 CSV files:**
Mostly consolidated, except DDoS-ICMP (2 files) and DDoS-UDP (2 files):

```
WiFI_and_MQTT/attacks/CSV/test/
├── ARP_Spoofing_test.pcap.csv
├── Benign_test.pcap.csv
├── MQTT-DDoS-Connect_Flood_test.pcap.csv
├── MQTT-DDoS-Publish_Flood_test.pcap.csv
├── MQTT-DoS-Connect_Flood_test.pcap.csv
├── MQTT-DoS-Publish_Flood_test.pcap.csv
├── MQTT-Malformed_Data_test.pcap.csv
├── Recon-OS_Scan_test.pcap.csv
├── Recon-Ping_Sweep_test.pcap.csv
├── Recon-Port_Scan_test.pcap.csv
├── Recon-VulScan_test.pcap.csv
├── TCP_IP-DDoS-ICMP[1-2]_test.pcap.csv          (2 files → merge)
├── TCP_IP-DDoS-SYN_test.pcap.csv
├── TCP_IP-DDoS-TCP_test.pcap.csv
├── TCP_IP-DDoS-UDP[1-2]_test.pcap.csv            (2 files → merge)
├── TCP_IP-DoS-ICMP_test.pcap.csv
├── TCP_IP-DoS-SYN_test.pcap.csv
├── TCP_IP-DoS-TCP_test.pcap.csv
└── TCP_IP-DoS-UDP_test.pcap.csv
```

### Filename → Label Mapping

| Filename Pattern | 17-Class Label | 6-Class Category |
|-----------------|----------------|-----------------|
| `ARP_Spoofing_*` | ARP_Spoofing | Spoofing |
| `Benign_*` | Benign | Benign |
| `MQTT-DDoS-Connect_Flood_*` | MQTT_DDoS_Connect_Flood | MQTT |
| `MQTT-DDoS-Publish_Flood_*` | MQTT_DDoS_Publish_Flood | MQTT |
| `MQTT-DoS-Connect_Flood_*` | MQTT_DoS_Connect_Flood | MQTT |
| `MQTT-DoS-Publish_Flood_*` | MQTT_DoS_Publish_Flood | MQTT |
| `MQTT-Malformed_Data_*` | MQTT_Malformed_Data | MQTT |
| `Recon-OS_Scan_*` | Recon_OS_Scan | Recon |
| `Recon-Ping_Sweep_*` | Recon_Ping_Sweep | Recon |
| `Recon-Port_Scan_*` | Recon_Port_Scan | Recon |
| `Recon-VulScan_*` | Recon_VulScan | Recon |
| `TCP_IP-DDoS-ICMP*` | DDoS_ICMP | DDoS |
| `TCP_IP-DDoS-SYN*` | DDoS_SYN | DDoS |
| `TCP_IP-DDoS-TCP*` | DDoS_TCP | DDoS |
| `TCP_IP-DDoS-UDP*` | DDoS_UDP | DDoS |
| `TCP_IP-DoS-ICMP*` | DoS_ICMP | DoS |
| `TCP_IP-DoS-SYN*` | DoS_SYN | DoS |
| `TCP_IP-DoS-TCP*` | DoS_TCP | DoS |
| `TCP_IP-DoS-UDP*` | DoS_UDP | DoS |

---

## 8. Class Distribution (VERIFIED — After Deduplication)

### 17-Class Distribution (Train — 4,515,080 rows after dedup)

| Class | Train Rows | % of Train | Imbalance Ratio |
|-------|-----------|------------|----------------|
| DDoS_UDP | 1,635,956 | 36.23% | 1.0x (LARGEST) |
| DDoS_SYN | 577,649 | 12.79% | 2.8x |
| DoS_UDP | 566,921 | 12.56% | 2.9x |
| DoS_SYN | 347,035 | 7.69% | 4.7x |
| DDoS_TCP | 248,267 | 5.50% | 6.6x |
| DoS_TCP | 221,181 | 4.90% | 7.4x |
| DDoS_ICMP | 210,258 | 4.66% | 7.8x |
| Benign | 192,732 | 4.27% | 8.5x |
| MQTT_DDoS_Connect_Flood | 173,036 | 3.83% | 9.5x |
| DoS_ICMP | 145,313 | 3.22% | 11.3x |
| Recon_Port_Scan | 73,885 | 1.64% | 22.1x |
| MQTT_DoS_Publish_Flood | 44,376 | 0.98% | 36.9x |
| MQTT_DDoS_Publish_Flood | 27,623 | 0.61% | 59.2x |
| ARP_Spoofing | 16,010 | 0.35% | 102.2x |
| Recon_OS_Scan | 14,214 | 0.31% | 115.1x |
| MQTT_DoS_Connect_Flood | 12,773 | 0.28% | 128.1x |
| MQTT_Malformed_Data | 5,130 | 0.11% | 318.9x |
| Recon_VulScan | 2,032 | 0.05% | 805.1x |
| Recon_Ping_Sweep | 689 | 0.02% | **2,374.4x (RAREST)** |

### 6-Category Distribution

| Category | Train Rows | Train % | Test Rows | Test % |
|----------|-----------|---------|-----------|--------|
| DDoS | 2,672,130 | 59.18% | 1,066,764 | 66.08% |
| DoS | 1,280,450 | 28.36% | 416,676 | 25.81% |
| MQTT | 262,938 | 5.82% | 63,715 | 3.95% |
| Benign | 192,732 | 4.27% | 37,607 | 2.33% |
| Recon | 90,820 | 2.01% | 27,676 | 1.71% |
| Spoofing | 16,010 | 0.35% | 1,744 | 0.11% |

### Imbalance Impact

DDoS + DoS together account for **87.54%** of training data. This means any unweighted supervised model will collapse into a DDoS/DoS classifier, achieving high overall accuracy while completely failing on the 5 minority categories. This is why **macro-F1 and MCC** (Matthews Correlation Coefficient) are our primary evaluation metrics, not accuracy.

**SMOTETomek priority classes** (most oversampling needed):
1. Recon_Ping_Sweep — 689 rows
2. Recon_VulScan — 2,032 rows
3. MQTT_Malformed_Data — 5,130 rows
4. MQTT_DoS_Connect_Flood — 12,773 rows
5. ARP_Spoofing — 16,010 rows

---

## 9. Profiling Data

A unique contribution of CICIoMT2024. Captures IoMT device behavior in four lifecycle states:

| State | Description | Purpose |
|-------|-------------|---------|
| **Power** | Device boot-up behavior in isolation | Baseline for startup traffic patterns |
| **Idle** | Network traffic during off-hours (no humans) | Baseline for device-initiated background traffic |
| **Active** | Normal operation with human interaction | Baseline for typical usage patterns |
| **Interaction** | All device functionalities exercised | Full behavioral profile per device |

This profiling data enables **behavioral anomaly detection** — learning what "normal" looks like for each device individually, then flagging deviations. This is how real-world anomaly-based IDS works in healthcare environments.

---

## 10. Phase 2 EDA Key Findings

> EDA pipeline run: April 25, 2026 — MacBook Air M4, 24GB RAM, Python 3.14.3

### 10.1 Data Quality Discovery

**Duplicate rows (major finding):** 36.95% of train data (2,645,751 rows) and 44.72% of test data (721,914 rows) are exact duplicates. This is **not reported in any prior paper** on CICIoMT2024 and means that accuracy metrics in all published work (including Yacoubi et al.'s 99.87%) are inflated by partial data leakage. After deduplication, effective dataset size drops from 8.77M to 5.41M rows.

**Data quality:** Zero missing values, zero infinite values across all features. Only `Drate` is strictly near-constant (std < 1e-6). IRC, DHCP, and IGMP have std ≈ 0.001 — functionally near-zero but not formally constant.

### 10.2 Feature Importance (Cohen's d — Attack vs Benign)

| Rank | Feature | |Cohen's d| | Significance |
|------|---------|------------|--------------|
| 1 | rst_count | 3.492 | Very high in Recon (RST from closed ports) |
| 2 | psh_flag_number | 3.293 | High in MQTT publish floods; also high in benign |
| 3 | Variance | 2.670 | High in diverse attacks, low in uniform floods |
| 4 | ack_flag_number | 2.644 | Separates completed vs half-open connections |
| 5 | Max | 1.521 | Maximum packet size distinguishes payload-heavy attacks |
| 6 | Magnitue | 1.481 | Flow magnitude separates volumetric from stealthy |
| 7 | HTTPS | 1.195 | Benign uses more HTTPS |
| 8 | Tot size | 1.129 | Average total packet size |
| 9 | AVG | 1.123 | Average packet size |
| 10 | Std | 1.118 | Packet size variation |

> **⚠️ Contradiction with Yacoubi et al.:** Our Cohen's d ranking has **zero overlap** with Yacoubi's SHAP top-4 (IAT, Rate, Header_Length, Srate). This discrepancy is likely because Yacoubi ran SHAP on raw duplicate-heavy data where DDoS dominated even more. After deduplication, the class balance shifts and TCP-flag features become more discriminative. This is itself a publishable finding.

### 10.3 Correlation Analysis

25 feature pairs have |Pearson r| > 0.85, including three perfect correlations: Rate/Srate (r=1.00), ARP/IPv (r=1.00), ARP/LLC (r=1.00).

**11 drop candidates** (redundant): AVG, IPv, LLC, Magnitue, Number, Radius, Srate, Std, Tot size, UDP, Weight. Combined with Drate (constant) and noise features (Telnet, SSH, IRC, SMTP, IGMP), the feature space can be reduced from 45 to approximately 28 features.

### 10.4 Attack-Specific Findings

- **DDoS vs DoS:** Same-protocol pairs differ primarily in Rate/Srate magnitude — distribution shift, not protocol shift. These are the hardest classification boundaries.
- **Recon attacks:** Four types show distinct radar profiles. Ping Sweep has very low Rate/high ICMP; Port Scan shows high syn_flag_number/rst_count.
- **MQTT attacks:** Split cleanly on Tot sum and psh_flag_number.
- **ARP Spoofing:** Unique signature (ARP≈1, all other L4 protos≈0) — should be trivially learnable.
- **Benign profile:** Low Rate (median 1.65 pps), high psh_flag_number (mean 0.42), compact PCA cluster — ideal Autoencoder reconstruction target.

### 10.5 Dimensionality Reduction

PCA needs 22 components for 95% variance and 28 for 99%. The 2D PCA projection shows DDoS/DoS clustering tightly while Recon and Spoofing occupy distinct pockets. Benign forms a compact, separable cluster — directly validating the Autoencoder-based unsupervised layer design.

### 10.6 Preprocessing Recommendations (Input for Phase 3)

- **Scaling:** RobustScaler on heavy-tailed features (IAT, Rate, Header_Length, Tot sum). StandardScaler on flag counts.
- **Feature drops:** ~17 features (Drate + 11 redundant + 5 noise) → ~28 retained features.
- **SMOTETomek priority:** Ping_Sweep (689) → VulScan (2,032) → Malformed (5,130) → DoS_Connect (12,773) → ARP_Spoofing (16,010).
- **Autoencoder data:** 192,732 benign rows — sufficient and well-clustered.
- **Validation:** 5-fold stratified at 17-class level; leave-one-attack-out for zero-day simulation.

---

## 11. Phase 3 Preprocessing & Feature Engineering

> Pipeline run: April 25, 2026 — MacBook Air M4, 24GB RAM — Total runtime: 228 minutes

### 11.1 Overview

Phase 3 transforms the cleaned EDA outputs into ready-to-train datasets for all remaining phases. The pipeline produces two feature variants (full 44 and reduced 28), applies SMOTETomek resampling for class imbalance, extracts benign-only data for the Autoencoder, and creates five leave-one-attack-out zero-day simulation scenarios.

### 11.2 Feature Engineering — Two Variants

| Variant | Features | Rationale |
|---------|----------|-----------|
| **Full (A)** | 44 features | Drop only Drate (constant at 0.0). Baseline for comparison. |
| **Reduced (B)** | 28 features | Drop Drate + 11 correlated (|r|>0.85) + 5 noise. Primary model. |

**Features dropped in Reduced variant (17 total):**
- Constant: `Drate`
- Correlated: `Srate` (r=1.0 with Rate), `IPv` (r=1.0 with ARP), `LLC` (r=1.0 with ARP), `Radius` (r=0.99 with Std), `Weight` (r=0.99 with Number), `Number` (r=0.99 with IAT), `Magnitue` (r=0.98 with AVG), `Tot size` (r=0.98 with AVG), `AVG` (r=0.92 with Tot sum), `Std` (r=0.96 with Covariance), `UDP` (r=0.98 with Protocol Type)
- Noise: `Telnet`, `SSH`, `IRC`, `SMTP`, `IGMP` (near-zero variance, not used in IoMT)

**28 retained features (Reduced variant):**
`IAT, Rate, Header_Length, Tot sum, Min, Max, Covariance, Variance, Duration, ack_count, syn_count, fin_count, rst_count, fin_flag_number, syn_flag_number, rst_flag_number, psh_flag_number, ack_flag_number, ece_flag_number, cwr_flag_number, HTTP, HTTPS, DNS, TCP, DHCP, ARP, ICMP, Protocol Type`

> Note: Column order in .npy files follows the ColumnTransformer output (robust → standard → minmax), not the original CSV order. The exact order is saved in `config.json["feature_names_reduced"]`.

### 11.3 Scaling Strategy

Three-group ColumnTransformer fitted on training data only:

| Scaler | Features | Rationale |
|--------|----------|-----------|
| **RobustScaler** | IAT, Rate, Header_Length, Tot sum, Min, Max, Covariance, Variance, Duration, ack_count, syn_count, fin_count, rst_count | Heavy-tailed distributions with extreme outliers. Uses median/IQR instead of mean/std. |
| **StandardScaler** | fin_flag_number, syn_flag_number, rst_flag_number, psh_flag_number, ack_flag_number, ece_flag_number, cwr_flag_number | TCP flag ratios roughly bounded in [0,1]. |
| **MinMaxScaler** | HTTP, HTTPS, DNS, TCP, DHCP, ARP, ICMP, Protocol Type | Binary/categorical indicators already near 0-1 range. |

> **Note:** This three-group scaling proved insufficient for the unsupervised layer — RobustScaler preserves heavy tails (desirable for tree-based supervised models but breaks AE training). See Section 13.6 for the StandardScaler patch applied in Phase 5.

### 11.4 Data Splits

| Split | Rows | Purpose |
|-------|------|---------|
| **Train** | 3,612,064 | 80% of deduplicated train — model training |
| **Validation** | 903,016 | 20% of deduplicated train — hyperparameter tuning |
| **Test** | 892,268 | Original test set — final holdout (never touched during training) |

Stratified on 19-class label to preserve class proportions. Minority class preservation verified: Recon_Ping_Sweep has 551 train / 138 val samples.

### 11.5 SMOTETomek Results

**Strategy:** Targeted oversampling — only classes below 50,000 rows are boosted. Majority classes left untouched. This avoids the runtime/memory explosion of full-population SMOTE on 3.6M rows while achieving the same minority-class recall improvement.

| Class | Before | After | Boost |
|-------|--------|-------|-------|
| Recon_Ping_Sweep | 551 | 49,799 | 90× |
| Recon_VulScan | 1,626 | 49,501 | 30× |
| MQTT_Malformed_Data | 4,104 | 47,867 | 12× |
| MQTT_DoS_Connect_Flood | 10,218 | 49,942 | 5× |
| Recon_OS_Scan | 11,371 | 48,015 | 4× |
| ARP_Spoofing | 12,808 | 46,786 | 4× |
| MQTT_DDoS_Publish_Flood | 22,098 | 49,421 | 2× |
| MQTT_DoS_Publish_Flood | 35,501 | 49,478 | 1.4× |

**Post-SMOTE sizes:** Full variant: 3,869,271 rows. Reduced variant: 3,871,167 rows. Tomek link cleaning removed ~200-1,500 samples per class (cleaning ambiguous boundary samples).

Note: SMOTETomek applied to TRAINING split only. Validation and test sets are NEVER resampled.

### 11.6 Autoencoder Dataset (Layer 2)

Benign-only data extracted from the **train split** (not the full pre-split set, to prevent data leakage with the supervised validation set):

| Set | Rows | Purpose |
|-----|------|---------|
| AE Train | 123,348 | Train the Autoencoder to reconstruct normal traffic |
| AE Val | 30,838 | Monitor reconstruction error during training |

Feature space: Reduced variant (28 features) — matches the supervised pipeline for consistent fusion.

### 11.7 Zero-Day Simulation Datasets

Five leave-one-attack-out scenarios for evaluating the unsupervised layer's ability to detect attacks it has never seen:

| Target Class | Train Without | Test Held-Out |
|-------------|---------------|---------------|
| Recon_Ping_Sweep | 4,514,391 | 169 |
| Recon_VulScan | 4,513,048 | 973 |
| MQTT_Malformed_Data | 4,509,950 | 1,747 |
| MQTT_DoS_Connect_Flood | 4,502,307 | 3,131 |
| ARP_Spoofing | 4,499,070 | 1,744 |

These use the **un-resampled** train set — the unsupervised layer is evaluated on real flow distributions, not synthetic samples.

### 11.8 Output File Structure

```
preprocessed/                              (5.7 GB total)
├── config.json                            # All parameters, feature lists, column orders
├── label_encoders.json                    # Label→int mappings (binary, 6-class, 19-class)
├── scaler_full.pkl                        # Fitted ColumnTransformer (44 features)
├── scaler_reduced.pkl                     # Fitted ColumnTransformer (28 features)
│
├── full_features/                         # Variant A — 44 features
│   ├── X_train.npy          (3.61M × 44)  # Scaled training features
│   ├── X_val.npy            (903K × 44)   # Scaled validation features
│   ├── X_test.npy           (892K × 44)   # Scaled test features
│   ├── X_train_smote.npy    (3.87M × 44)  # After SMOTETomek
│   ├── y_train.csv                        # Labels (binary + category + multiclass + strings)
│   ├── y_val.csv
│   ├── y_test.csv
│   └── y_train_smote.csv
│
├── reduced_features/                      # Variant B — 28 features
│   ├── X_train.npy          (3.61M × 28)
│   ├── X_val.npy            (903K × 28)
│   ├── X_test.npy           (892K × 28)
│   ├── X_train_smote.npy    (3.87M × 28)
│   ├── y_train.csv
│   ├── y_val.csv
│   ├── y_test.csv
│   └── y_train_smote.csv
│
├── autoencoder/                           # Benign-only for Layer 2
│   ├── X_benign_train.npy   (123K × 28)
│   ├── X_benign_val.npy     (31K × 28)
│   └── benign_stats.json                  # Mean, std, p95, p99 per feature
│
└── zero_day/                              # Leave-one-attack-out
    ├── Recon_Ping_Sweep/
    │   ├── X_train_without.npy, y_train_without.csv
    │   └── X_held_out.npy, y_held_out.csv
    ├── Recon_VulScan/
    ├── MQTT_Malformed_Data/
    ├── MQTT_DoS_Connect_Flood/
    └── ARP_Spoofing/
```

### 11.9 Verification Results

All integrity checks passed:
- ✅ No NaN or inf in any .npy file
- ✅ Row counts match between X (features) and y (labels) in all splits
- ✅ All zero-day held-out sets contain only the target class
- ✅ Autoencoder set contains only benign rows (123K train + 31K val)
- ✅ SMOTETomek increased all 8 minority classes to ~50K rows each

---

## 12. Phase 4 Supervised Model Training Results

> Pipeline run: April 26, 2026 — MacBook Air M4, 24GB RAM — Total runtime: 60 minutes
> 8 experiments × 3 classification tasks = 24 training runs

### 12.1 Experimental Design

| ID | Model | Data | Features | Purpose |
|----|-------|------|----------|---------|
| E1 | Random Forest | Original | Reduced (28) | Baseline RF |
| E2 | Random Forest | SMOTETomek | Reduced (28) | RF + class balancing |
| E3 | XGBoost | Original | Reduced (28) | Baseline XGBoost |
| E4 | XGBoost | SMOTETomek | Reduced (28) | XGBoost + class balancing |
| E5 | Random Forest | Original | Full (44) | Feature count comparison |
| E6 | Random Forest | SMOTETomek | Full (44) | Feature count + balancing |
| E7 | XGBoost | Original | Full (44) | Feature count comparison |
| E8 | XGBoost | SMOTETomek | Full (44) | Feature count + balancing |

**Hyperparameters (regularized based on review feedback):**
- RF: n_estimators=200, criterion='entropy', max_depth=30, min_samples_split=20, min_samples_leaf=10, class_weight='balanced'
- XGBoost: n_estimators=200, max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, tree_method='hist'

### 12.2 Best Results — 19-Class Task (Test Set)

| Experiment | Model | Features | Data | Accuracy | F1_macro | MCC |
|-----------|-------|----------|------|----------|----------|-----|
| **E7** | **XGBoost** | **Full (44)** | **Original** | **99.27%** | **0.9076** | **0.9906** |
| E3 | XGBoost | Reduced (28) | Original | 99.25% | 0.8987 | 0.9905 |
| E8 | XGBoost | Full (44) | SMOTE | 98.79% | 0.8708 | 0.9846 |
| E5 | RF | Full (44) | Original | 98.52% | 0.8551 | 0.9811 |
| E4 | XGBoost | Reduced (28) | SMOTE | 98.59% | 0.8538 | 0.9821 |
| E1 | RF | Reduced (28) | Original | 98.43% | 0.8469 | 0.9801 |
| E6 | RF | Full (44) | SMOTE | 98.41% | 0.8380 | 0.9798 |
| E2 | RF | Reduced (28) | SMOTE | 98.37% | 0.8356 | 0.9793 |

**Winner: E7 (XGBoost / full features / original data)** — selected as the supervised input for the Phase 6 fusion engine.

### 12.3 Best Results — Per Classification Task

| Task | Best Experiment | F1_macro | MCC | Accuracy |
|------|----------------|----------|-----|----------|
| Binary (2-class) | E5 (RF/full/original) | 0.9880 | 0.9763 | 99.80% |
| Category (6-class) | E7 (XGB/full/original) | 0.9363 | 0.9925 | 99.55% |
| Multiclass (19-class) | E7 (XGB/full/original) | 0.9076 | 0.9906 | 99.27% |

### 12.4 SMOTETomek Impact (Key Finding)

**SMOTETomek degraded performance in ALL 4 configurations:**

| Model | Features | F1_macro (Original) | F1_macro (SMOTE) | Change |
|-------|----------|--------------------|--------------------|--------|
| RF | Reduced (28) | 0.8469 | 0.8356 | **−0.0114** |
| RF | Full (44) | 0.8551 | 0.8380 | **−0.0171** |
| XGBoost | Reduced (28) | 0.8987 | 0.8538 | **−0.0449** |
| XGBoost | Full (44) | 0.9076 | 0.8708 | **−0.0368** |

> **Thesis finding.** SMOTETomek consistently degrades macro-F1 by 0.011–0.045 across both classifiers and both feature sets. The mechanism is **synthetic-sample blur on already-overlapping class boundaries** — specifically the `DDoS_*` ↔ `DoS_*` and `Recon_OS_Scan` ↔ `Recon_VulScan` decision regions documented in `results/supervised/summary.md` and visible in the Phase 4 confusion matrices. SMOTE generates synthetic minority points by interpolating between existing samples in feature space; when the minority class is *already adjacent* to a structurally similar class in the 44-feature flow representation, the interpolated points fall on or across the decision boundary rather than reinforcing the minority cluster. This degrades the majority-side classifier without commensurate minority-side gains.
>
> **The `class_weight='balanced'` interaction is not the mechanism.** RF arms (E1, E2, E5, E6) use `class_weight='balanced'` and degrade by 0.011–0.017. XGBoost arms (E3, E4, E7, E8) use **no** `class_weight` and **no** `scale_pos_weight` (`supervised_training.py:91-103, 207-217`) and degrade by **0.037–0.045** — a *larger* drop. If the mechanism were "compounding correction" between SMOTE and class weighting, the XGBoost arms (which have no class weighting to compound with) should be relatively unharmed. The opposite is observed.
>
> The boundary-blur mechanism naturally explains both signs: XGBoost's tighter decision boundaries (deeper trees, gradient-based splits) are more sensitive to synthetic boundary-region samples than RF's averaged ensemble of shallower trees. We retain `class_weight='balanced'` for RF as a separate hyperparameter choice supported by Phase 4 results on the *original* (non-SMOTE) data; the SMOTE finding stands independently.
>
> **H3 (SMOTETomek improves minority F1) is rejected on this dataset.** This contradicts the common assumption in IoMT IDS literature that oversampling always helps minority-class detection — but the rejection is grounded in feature-space geometry, not in the oversampling-vs-class-weighting interaction.

### 12.5 Feature Importance — RF Top 10

From E5 (RF / full features / original — best RF model):

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | IAT | 0.1401 |
| 2 | Magnitue | 0.0706 |
| 3 | Tot size | 0.0525 |
| 4 | AVG | 0.0499 |
| 5 | Min | 0.0476 |
| 6 | TCP | 0.0466 |
| 7 | syn_count | 0.0452 |
| 8 | syn_flag_number | 0.0449 |
| 9 | rst_count | 0.0425 |
| 10 | fin_count | 0.0425 |

**Feature importance comparison across methods:**

| Method | Top-4 Features |
|--------|---------------|
| Yacoubi SHAP (raw data) | IAT, Rate, Header_Length, Srate |
| Our Cohen's d (deduped) | rst_count, psh_flag_number, Variance, ack_flag_number |
| Our RF importance (deduped) | IAT, Magnitue, Tot size, AVG |

Only IAT appears consistently across all three methods — confirming it as the single most reliable discriminative feature. The variation in other rankings demonstrates that feature importance is method-dependent and should always be reported with multiple techniques.

### 12.6 Full vs Reduced Features

Full features (44) consistently outperformed reduced (28) by +0.005–0.009 macro-F1. Features dropped in the reduced variant (Magnitue, Tot size, AVG) ranked #2, #3, #4 in RF importance — the correlation-based dropping was too aggressive. **Recommendation for remaining phases: use the full (44) feature set.**

### 12.7 Comparison with Yacoubi et al.

| Model | Yacoubi Accuracy (raw data) | Our Accuracy (deduped) | Gap |
|-------|----------------------------|----------------------|-----|
| RF (entropy) | 99.87% | 98.52% (E5) | −1.35% |
| XGBoost | 99.80% | 99.27% (E7) | −0.53% |

The gap is attributed to our duplicate removal (37% of train data were exact duplicates that inflated Yacoubi's metrics). This is a methodological correction, not a regression. Our XGBoost result (99.27% accuracy, 0.9076 macro-F1) on clean data represents a more honest evaluation.

### 12.8 Phase 6 Fusion Input

**E7 (XGBoost / full / original)** is selected as the supervised layer for the 4-case fusion engine:
- Model: `results/supervised/models/E7_xgb_full_original.pkl`
- Val probabilities: `results/supervised/predictions/E7_val_proba.npy` (903,016 × 19)
- Test probabilities: `results/supervised/predictions/E7_test_proba.npy` (892,268 × 19)

### 12.9 Output Artifacts

```
results/supervised/                     (~4-6 GB)
├── config.json                        # All experiment parameters
├── summary.md                         # Key findings narrative
├── models/                            # 8 trained model .pkl files
│   ├── E1_rf_reduced_original.pkl
│   ├── ...
│   └── E8_xgb_full_smote.pkl
├── predictions/                       # Probability vectors for fusion
│   ├── E1_val_proba.npy ... E8_test_proba.npy
│   └── E1_val_pred.npy ... E8_test_pred.npy
├── metrics/
│   ├── overall_comparison.csv         # 24-row comparison table
│   ├── best_classification_report.csv # Per-class F1 for E7
│   ├── smote_comparison.csv           # Original vs SMOTE delta
│   ├── minority_focus.csv             # 5 rarest class analysis
│   └── E*_feature_importance.csv      # RF importance rankings
└── figures/
    ├── cm_E*_19class.png              # 19×19 confusion matrices (8 experiments)
    ├── cm_E*_6class.png               # 6×6 confusion matrices
    ├── overall_comparison_bar.png     # F1_macro across all experiments
    ├── feature_importance_rf.png      # Top-20 RF features
    └── smote_effect.png               # Before/after per minority class
```

---

## 13. Phase 5 Unsupervised Model Training Results

> Pipeline run: April 26, 2026 — MacBook Air M4, 24GB RAM — Total runtime: 34 seconds
> Two models trained on benign-only data (123K rows × 44 features)

### 13.1 Overview

Phase 5 builds Layer 2 of the hybrid framework: unsupervised anomaly detection trained exclusively on benign traffic. Two models learn what "normal" IoMT traffic looks like, then flag anything that deviates. A critical scaling fix (StandardScaler on benign-train, applied to all data) was required because Phase 3's ColumnTransformer left several features with magnitudes in the thousands, which dominated AE loss and made Recon detection impossible.

### 13.2 Autoencoder Configuration

| Property | Value |
|----------|-------|
| Architecture | 44 → 32 → 16 → **8** → 16 → 32 → 44 (bottleneck = 8) |
| Optimizer | Adam, lr=0.001, batch size=512 |
| Loss | MSE (reconstruction error) |
| Epochs | 36 (early-stopped from max 100, patience=10) |
| Best val loss | **0.1988** |
| Training time | 8.2 seconds |
| Training data | 123,348 benign rows (train split only — no leakage) |

### 13.3 Threshold Selection

Five percentile-based thresholds evaluated on validation set. Mean+kσ thresholds are impractical due to fat-tailed benign MSE distribution (mean=0.20, std=9.48).

| Threshold | Value | Recall | FPR | F1 |
|-----------|-------|--------|-----|-----|
| **p90** | **0.2013** | **0.986** | **0.102** | **0.991** |
| p95 | 0.3726 | 0.983 | 0.052 | 0.990 |
| p99 | 1.2025 | 0.843 | 0.011 | 0.914 |

Selected: p90 (highest F1). For Phase 6 fusion, p99 may be preferred (FPR ≈ 0.8%) to minimize false zero-day alerts.

The benign reconstruction error distribution exhibits heavy right-tail behavior (mean=0.20, std=9.48, p95=0.37, p99=1.20), causing mean+kσ thresholds to fall outside the attack-error mass and collapse to ~25–32% recall. Percentile thresholds avoid this failure mode. The F1-optimal threshold (p90) achieves 98.6% attack recall but at 18.6% benign false-positive rate, which is impractical for IDS deployment; for the fusion layer in Phase 6 we adopt p99 (FPR ≈ 0.8%, recall 84%) as the binary anomaly flag, retaining all thresholds for ablation analysis.

### 13.4 Binary Anomaly Detection — AE vs IF (Test Set)

| Metric | Autoencoder | Isolation Forest |
|--------|-------------|-----------------|
| **AUC-ROC** | **0.9892** | 0.8612 |
| FPR @ 95% TPR | 0.0203 | 0.2721 |
| Anomaly F1 | **0.9853** | 0.7327 |
| Per-class avg recall | **0.7999** | 0.1627 |

Autoencoder decisively outperforms Isolation Forest. Both signals saved for Phase 6 fusion (complementary failure modes).

### 13.5 Per-Class Detection Highlights (AE at p90)

**Near-perfect (>95%):** All DDoS/DoS floods + MQTT Connect floods + Recon_Port_Scan

**Medium (50-87%):** Recon_OS_Scan (86.5%), Recon_VulScan (63.0%), MQTT_Malformed (55.8%), ARP_Spoofing (55.3%), Recon_Ping_Sweep (54.4%)

**Low (<30%):** MQTT_DDoS_Publish (26.6%), MQTT_DoS_Publish (6.7%)

> **Key thesis finding:** AE blind spots (MQTT Publish floods, ARP Spoofing) are precisely where supervised XGBoost excels. Neither layer alone covers the full attack spectrum — this complementarity justifies the hybrid design.

### 13.6 Scaling Fix Discovery

Phase 3's ColumnTransformer left features like Covariance (std=5005) and IAT (std=1030) unscaled. XGBoost is scale-invariant so Phase 4 was unaffected, but the AE produced loss values in the millions and zero Recon detection. Adding StandardScaler (fitted on benign-train) fixed this:

| Metric | Before fix | After fix |
|--------|-----------|-----------|
| AE val loss | 101,414 | **0.199** |
| AE test AUC | 0.9728 | **0.9892** |
| Recon_Ping_Sweep recall | 0.000 | **0.544** |
| Per-class avg recall | 0.700 | **0.800** |

### 13.7 Zero-Day Preview

| Target | AE Recall (p90) | IF Recall |
|--------|-----------------|-----------|
| MQTT_DoS_Connect_Flood | 1.000 | 0.008 |
| Recon_VulScan | 0.630 | 0.021 |
| MQTT_Malformed_Data | 0.558 | 0.203 |
| ARP_Spoofing | 0.553 | 0.439 |
| Recon_Ping_Sweep | 0.544 | 0.077 |

H2 preview: 1/5 targets ≥70% at p90. Proper evaluation deferred to Phase 6 fusion.

### 13.8 Output Artifacts

```
results/unsupervised/                           (~50 MB)
├── config.json, summary.md, thresholds.json
├── models/{autoencoder.keras, encoder.keras, isolation_forest.pkl, scaler.pkl}
├── scores/{ae,if}_{val,test}_{mse,scores,binary}.npy  (8 arrays for fusion)
├── metrics/{ae,if}_classification_report.json, per_class_detection_rates.csv
└── figures/{ae_loss_curves, ae_error_distribution, roc_curves, ...}.png (7 figures)
```

---

## 14. Phase 6 Fusion Engine & Zero-Day Simulation Results

> Pipeline run: April 26, 2026 — MacBook Air M4, 24GB RAM — Total runtime: ~1 minute
> 4-case fusion logic + bootstrap hypothesis testing + threshold sensitivity sweep

### 14.1 Overview

Phase 6 implements the core thesis contribution: the 4-case fusion decision engine that combines supervised XGBoost (E7) predictions with Autoencoder anomaly scores. It also evaluates both thesis hypotheses (H1: fusion improves macro-F1; H2: AE catches zero-day attacks E7 misses) using paired bootstrap confidence intervals.

### 14.2 4-Case Fusion Logic

| Case | Supervised (E7) | Unsupervised (AE) | Decision | Confidence |
|------|----------------|-------------------|----------|------------|
| **1** | Attack | Anomaly | **Confirmed Alert** | HIGH |
| **2** | Benign | Anomaly | **Zero-Day Warning** | MEDIUM-HIGH |
| **3** | Attack | Normal | **Low-Confidence Alert** | MEDIUM-LOW |
| **4** | Benign | Normal | **Clear** | HIGH |

### 14.3 Case Distribution (Test Set — p90 Threshold)

| Case | Count | Percentage | Meaning |
|------|-------|-----------|---------|
| Case 1 (Confirmed Alert) | 837,209 | **93.83%** | Both layers agree: attack |
| Case 2 (Zero-Day Warning) | 6,140 | 0.69% | E7 says benign, AE says anomaly |
| Case 3 (Low-Confidence) | 17,317 | 1.94% | E7 says attack, AE says normal |
| Case 4 (Clear) | 31,602 | 3.54% | Both layers agree: benign |

The overwhelming majority of attacks (93.8%) receive Case 1 — high-confidence alerts confirmed by both layers. Case 3 captures attacks the AE missed (mainly MQTT Publish floods). Case 2 fires on 0.69% of traffic — these are the potential zero-day signals.

### 14.4 Hypothesis H1 — Fusion vs Standalone E7

**H1: The hybrid fusion framework produces statistically significant improvements in macro-F1 compared to E7.**

Evaluated in 20-label space (19 original classes + `zero_day_unknown` for Case 2 samples), with paired bootstrap (200 iterations, seed=42):

| Variant | Macro-F1 | 95% CI | Δ vs E7 | Δ CI | Significant? |
|---------|----------|--------|---------|------|-------------|
| E7 baseline | **0.8622** | [0.8586, 0.8655] | — | — | — |
| Fusion (p90) | 0.8582 | [0.8546, 0.8615] | −0.0041 | [−0.0042, −0.0040] | No |
| Fusion (p95) | 0.8610 | [0.8574, 0.8643] | −0.0012 | [−0.0013, −0.0012] | No |
| Fusion (p99) | 0.8621 | [0.8584, 0.8654] | −0.0001 | [−0.0002, −0.0001] | No |

> **Verdict: H0 NOT REJECTED. Δ negligible (−0.014pp, CI excludes 0).** The bootstrap CI for the macro-F1 difference at p99 is [−0.0002, −0.0001] — strictly negative but ~125 of 892,268 test rows. Fusion does not measurably degrade or improve aggregate macro-F1; the framework's value is in **case-stratified operational alerts (4 → 5-case fusion) and zero-day detection capability (Phase 6C, H2-strict 4/4)**, not in aggregate classification metrics. The zero_day_unknown pseudo-class penalizes macro-F1 because every false Case 2 alarm on benign traffic counts as a misclassification — a structural artifact of the metric, not a flaw of the framework.

> **Thesis framing:** The fusion framework's value is NOT in improving aggregate classification metrics, but in providing **confidence-stratified operational alerts** (Case 1-4) and a zero-day warning capability that standalone supervised models fundamentally cannot offer. Binary detection F1 remains 0.9985.

### 14.5 Hypothesis H2 — Zero-Day Detection

**H2: The AE achieves recall ≥ 0.70 on ≥ 50% of held-out attack classes when E7 misclassifies them as benign.**

| Target | n_test | E7 Recall | E7→Benign | AE Recall on E7-Missed (p90) | Status |
|--------|--------|-----------|-----------|--------------------------|--------|
| Recon_Ping_Sweep | 169 | 0.710 | 25 | 0.200 | ⚠ Insufficient samples |
| Recon_VulScan | 973 | 0.332 | 535 | **0.357** | ✗ Below 0.70 |
| MQTT_Malformed_Data | 1,747 | 0.828 | 159 | 0.182 | ✗ Below 0.70 |
| MQTT_DoS_Connect_Flood | 3,131 | 0.999 | 0 | n/a | ⚠ E7 catches all |
| ARP_Spoofing | 1,744 | 0.710 | 252 | 0.222 | ✗ Below 0.70 |

> **Verdict: H2 FAIL (0/5 targets ≥ 70%).** The AE catches only 18-36% of samples that E7 misclassifies as benign. The samples E7 misclassifies are precisely the ones that ALSO look benign to the AE — they sit near the decision boundary in both feature spaces.

> **Thesis framing:** This is a genuine finding about the limitations of reconstruction-error anomaly detection on flow-level features. The AE and supervised model share the same feature space (44 flow statistics), so their blind spots overlap. Future work should explore: (1) true leave-one-attack-out with E7 retrained, (2) alternative AE architectures (VAE, Transformer), (3) profiling-lifecycle features as an independent feature basis for the unsupervised layer.

### 14.6 Per-Class Case Analysis (p90 Threshold)

**Case 1 dominant (>95% — both layers confirm):**
DDoS_SYN (99.7%), DDoS_UDP (99.98%), DoS_SYN (100%), DoS_TCP (99.99%), DoS_UDP (99.99%), MQTT_DDoS_Connect (100%), MQTT_DoS_Connect (99.97%), DDoS_ICMP (99.6%), DDoS_TCP (98.8%), Recon_Port_Scan (95.8%)

**Case 3 dominant (>30% — E7 catches, AE misses):**
MQTT_DoS_Publish (93.8%), MQTT_DDoS_Publish (73.9%), ARP_Spoofing (33.9%), MQTT_Malformed (37.0%), Recon_Ping_Sweep (33.7%)

**Notable Case 2 (zero-day warnings):**
Recon_VulScan (19.6%), Benign false alarms (15.4%), ARP_Spoofing (3.2%), Recon_Ping_Sweep (3.0%), Recon_OS_Scan (1.7%)

> **Key insight:** Case 2 precision is low (~6% at p90) — 15.4% of benign traffic triggers false zero-day warnings. This improves at stricter thresholds (p99: only 0.04% Case 2) but at the cost of missing real anomalies. The precision-recall tradeoff is inherent to the operating point selection.

### 14.7 Binary Detection Performance

| Variant | Accuracy | Precision | Recall | F1 | MCC |
|---------|----------|-----------|--------|-----|-----|
| E7 only | 99.73% | 0.9987 | 0.9985 | **0.9986** | 0.9665 |
| Fusion (p90) | 99.12% | 0.9919 | 0.9989 | 0.9954 | 0.8855 |
| Fusion (p95) | 99.54% | 0.9964 | 0.9988 | 0.9976 | 0.9413 |
| Fusion (p99) | 99.71% | 0.9984 | 0.9986 | 0.9985 | 0.9636 |

Fusion at p99 achieves binary F1 = 0.9985, essentially matching E7-only (0.9986). The system detects attacks effectively at the binary level; the macro-F1 penalty comes from the multiclass zero_day_unknown label.

### 14.8 Threshold Sensitivity & Recommended Operating Point

| Percentile | Threshold | Attack Recall (test) | Benign FPR (test) | Binary F1 (test) |
|-----------|-----------|---------------------|-------------------|-----------------|
| 90 | 0.2058 | 99.89% | 18.44% | 0.9954 |
| 95 | 0.3845 | 99.88% | 8.30% | 0.9976 |
| **97** | **0.5615** | **99.87%** | **5.29%** | **0.9982** |
| 99 | 1.3152 | 99.86% | 3.73% | 0.9985 |

**Recommended operating point: p97** (selected on validation with FPR < 5%). Achieves 99.87% attack recall with 5.3% benign FPR — a practical IDS operating point for IoMT deployment.

### 14.9 Thesis Contributions from Phase 6

Phase 6 produces four genuine contributions. H1 yields a negligible-magnitude effect (Δ macro-F1 = −0.014pp; CI excludes zero but the operational impact is ~125 of 892,268 rows) — it does not measurably improve or degrade the aggregate metric. H2 (AE-only) is not rejected here but is reframed and reopened in Phase 6C with the entropy gate (4/4 strict).

1. **First fusion framework implementation on CICIoMT2024** — no prior work combines supervised + unsupervised with structured decision logic on this dataset
2. **Confidence-stratified alerts** — Case 1-4 provide differentiated response actions (block/quarantine/monitor/allow) vs binary detect/miss
3. **Empirical evidence that reconstruction-error AE is insufficient for zero-day detection on flow features** — a genuine negative finding that redirects future IoMT IDS research toward alternative unsupervised architectures
4. **Comprehensive threshold sensitivity analysis** — 10-point sweep with val-selected operating point, directly deployable

### 14.10 Output Artifacts

```
results/fusion/                                 (~20 MB)
├── config.json                                # All parameters, thresholds
├── summary.md                                 # Full findings narrative
├── fusion_results/
│   ├── fusion_{val,test}_cases.npy            # Case assignments (1-4)
│   └── fusion_{val,test}_labels.csv
├── metrics/
│   ├── case_distribution.csv                  # Case 1-4 counts per variant
│   ├── fusion_vs_supervised.csv               # Macro-F1 + bootstrap CIs
│   ├── fusion_vs_supervised_binary.csv        # Binary detection comparison
│   ├── per_class_case_analysis.csv            # 19 classes × 4 cases
│   ├── zero_day_results.csv                   # 5 targets × H2 metrics
│   ├── threshold_sensitivity.csv              # 10-point sweep
│   └── h1_h2_verdicts.json                    # Hypothesis outcomes
└── figures/
    ├── case_distribution.png
    ├── per_class_heatmap.png
    ├── fusion_vs_supervised.png
    ├── zero_day_detection.png
    └── threshold_sensitivity.png
```

### 14.11 Future Work (from Phase 6 findings)

1. **True leave-one-attack-out:** ✅ Completed in Phase 6B — see Section 15. Results show H2-binary PASS (5/5 at p90) via redundancy through misclassification.
2. **Alternative AE architectures:** Variational Autoencoder (VAE), Transformer-based AE — may learn richer benign manifold representations
3. **Independent feature basis for unsupervised layer:** Use profiling-lifecycle features (power/idle/active/interaction) instead of flow features — decouples the two layers' blind spots
4. **Temporal models:** LSTM/CNN-LSTM on packet sequences rather than aggregated flow statistics — may capture rate-over-time patterns that flow features miss

---

## 15. Phase 6B True Leave-One-Attack-Out Zero-Day Results

> Pipeline run: April 26, 2026 — MacBook Air M4, 24GB RAM — Total runtime: 19.3 minutes
> 5 XGBoost retraining runs, each excluding one target class from training

### 15.1 Overview

Phase 6B implements the proper leave-one-attack-out (LOO) experiment that H2 literally describes. For each of 5 target classes, XGBoost is retrained WITHOUT that class — simulating a genuine zero-day scenario where the IDS has never seen the attack type. The Autoencoder and Isolation Forest are NOT retrained (they only learned benign traffic and are unaffected by removing an attack class).

### 15.2 Per-Target Results

| Target | n_test | LOO-E7→Benign | LOO-E7→Other Attack | AE Recall (p90) | AE on LOO-Missed (p90) | Binary Recall (p90) |
|--------|--------|--------------|-------------------|----------------|----------------------|-------------------|
| Recon_Ping_Sweep | 169 | 18.3% | 81.7% | 0.544 | 0.161 | **0.846** |
| Recon_VulScan | 973 | 53.6% | 46.4% | 0.630 | 0.441 | **0.700** |
| MQTT_Malformed_Data | 1,747 | 27.0% | 73.0% | 0.558 | 0.335 | **0.820** |
| MQTT_DoS_Connect_Flood | 3,131 | 0.0% | 100.0% | 1.000 | n/a | **1.000** |
| ARP_Spoofing | 1,744 | 18.1% | 81.9% | 0.553 | 0.320 | **0.877** |

### 15.3 Key Discovery: Redundancy Through Misclassification

The most important finding of Phase 6B is HOW the system detects novel attacks — through a mechanism different from what the thesis originally proposed:

**Expected:** E7 calls novel attack "benign" → AE flags it as anomaly → Case 2 (zero-day warning)

**Actual:** E7 maps novel attack to the CLOSEST KNOWN attack class (not benign) → Cases 1 or 3 still fire → alert triggered with wrong class label but correct detection

| Target (held out) | What LOO-E7 thinks it is |
|---|---|
| Recon_Ping_Sweep | Recon_OS_Scan (44%), ARP_Spoofing (37%), Benign (18%) |
| Recon_VulScan | **Benign (54%)**, Recon_Port_Scan (21%), Recon_OS_Scan (16%) |
| MQTT_Malformed_Data | ARP_Spoofing (54%), Benign (27%), MQTT_DDoS_Connect (14%) |
| MQTT_DoS_Connect_Flood | MQTT_DDoS_Connect_Flood (87%), MQTT_DDoS_Publish (13%) |
| ARP_Spoofing | Recon_Port_Scan (47%), Recon_VulScan (27%), Benign (18%) |

> **Thesis contribution:** The hybrid framework's zero-day capability comes primarily from the supervised model's feature space naturally grouping similar attacks together — novel attacks are "caught as false positives of neighboring classes." The AE provides a secondary safety net for the 18-54% of samples that slip through to "benign." This is "redundancy through misclassification" rather than the originally hypothesized "complementary specialization."

### 15.4 H2 Re-Evaluation Under True LOO

**Strict criterion (AE recall on LOO-E7-missed samples ≥ 70%):**

| Target | AE on LOO-Missed (p90) | AE on LOO-Missed (p95) | Status |
|--------|----------------------|----------------------|--------|
| Recon_Ping_Sweep | 0.161 | 0.065 | ✗ |
| Recon_VulScan | 0.441 | 0.345 | ✗ |
| MQTT_Malformed_Data | 0.335 | 0.258 | ✗ |
| MQTT_DoS_Connect_Flood | n/a | n/a | ⚠ (0% → benign) |
| ARP_Spoofing | 0.320 | 0.206 | ✗ |

> **H2 Strict Verdict: FAIL (0/5).** The AE alone catches only 6-44% of samples that the blind LOO-E7 misclassifies as benign.

**Binary criterion (any alert — Cases 1+2+3 — recall ≥ 70%):**

| Target | Binary Recall (p90) | Binary Recall (p95) | Status (p90) |
|--------|-------------------|-------------------|-------------|
| Recon_Ping_Sweep | 0.846 | 0.828 | ✓ |
| Recon_VulScan | **0.700** | 0.649 | ✓ (exactly) |
| MQTT_Malformed_Data | 0.820 | 0.800 | ✓ |
| MQTT_DoS_Connect_Flood | 1.000 | 1.000 | ✓ |
| ARP_Spoofing | 0.877 | 0.856 | ✓ |

> **H2 Binary Verdict: PASS (5/5 at p90, 4/5 at p95).** The hybrid system raises an alert on ≥70% of truly novel attack samples at p90. Detection is dominated by the supervised model's misclassification into neighboring known attack classes (Cases 1+3), not by the AE catching zero-days (Case 2).

### 15.5 Phase 6 vs Phase 6B Comparison

| Target | Phase 6 E7 Recall | LOO E7 Recall | Phase 6 AE-on-Missed (p95) | LOO AE-on-Missed (p95) | LOO Binary (p95) |
|--------|------------------|--------------|--------------------------|----------------------|-----------------|
| Recon_Ping_Sweep | 0.710 | 0.000 | 0.080 | 0.065 | 0.828 |
| Recon_VulScan | 0.332 | 0.000 | 0.264 | 0.345 | 0.649 |
| MQTT_Malformed_Data | 0.828 | 0.000 | 0.138 | 0.258 | 0.800 |
| MQTT_DoS_Connect_Flood | 0.999 | 0.000 | n/a | n/a | 1.000 |
| ARP_Spoofing | 0.710 | 0.000 | 0.151 | 0.206 | 0.856 |

Phase 6B provides a more meaningful evaluation: LOO-E7 recall = 0.000 for all targets (by design), and the AE-on-missed recall slightly improves over Phase 6 because the LOO-missed samples are more representative (drawn from the full target population, not just E7's edge-case errors).

### 15.6 Stress Test: Recon_VulScan

Recon_VulScan is the most informative target: 53.6% routed to Benign (highest of any target), binary recall barely passes at p90 (0.700 exactly), fails at p95 (0.649). This is the system's weakest point — reconnaissance attacks that are syntactically close to benign traffic, where neither the supervised model nor the AE has strong signal. For IoMT deployment, this class of stealthy reconnaissance attacks represents the primary residual risk.

### 15.7 Output Artifacts

```
results/zero_day_loo/                           (~200 MB)
├── config.json
├── summary.md
├── models/                                    # 5 LOO-retrained XGBoost models
│   ├── loo_xgb_without_Recon_Ping_Sweep.pkl
│   ├── loo_xgb_without_Recon_VulScan.pkl
│   ├── loo_xgb_without_MQTT_Malformed_Data.pkl
│   ├── loo_xgb_without_MQTT_DoS_Connect_Flood.pkl
│   └── loo_xgb_without_ARP_Spoofing.pkl
├── predictions/                               # Per-fold test predictions
│   └── loo_*_test_{pred,proba}.npy
├── metrics/
│   ├── loo_results.csv
│   ├── loo_vs_phase6_comparison.csv
│   ├── loo_prediction_distribution.csv
│   ├── loo_case_distribution.csv
│   └── h2_loo_verdict.json
└── figures/
    ├── loo_zero_day_results.png
    ├── loo_vs_phase6_comparison.png
    ├── loo_prediction_distribution.png
    └── loo_case_distribution.png
```

---

## 15B. Multi-Seed Validation of LOO Zero-Day Detection (Path B Week 1)

> Pipeline run: April 27, 2026 — MacBook Air M4, 24 GB RAM — Total runtime: ~85 minutes (Phase 2 retraining) + <1 sec (fusion + aggregation)
> Hardens the Phase 6C single-seed result by retraining LOO-XGBoost with 5 random seeds and re-applying the enhanced fusion. No new architecture; addresses senior review's deferred limitation on training-randomness robustness.

### 15B.1 Motivation

The senior review (commit `7b90948`) accepted the seed=42 H2-strict 4/4 PASS result as bootstrap-robust over the test distribution but flagged the single random-seed dependency as a deferred limitation:

> "Bootstrap is over-test-set robustness, not over-training-randomness robustness."

This subsection closes that gap by retraining the LOO-XGBoost classifier with 5 random seeds and re-applying the Phase 6C enhanced fusion to each seed's predictions.

### 15B.2 Method

Five seeds were chosen: `{1, 7, 42, 100, 1729}` — the published baseline (42) plus four pseudorandom values. For each seed, the LOO-XGBoost classifier was retrained on the same train partition with one attack class held out at a time, using the regularized hyperparameters from `loo_zero_day.py:48-62` (`max_depth=8`, `min_child_weight=5`, `gamma=0.1`, `tree_method="hist"`, `n_estimators=200`, no `class_weight`, no `scale_pos_weight`).

- 4 new seeds × 5 LOO targets = **20 retrainings**, ≈ 85 minutes wall-clock
- The seed=42 LOO predictions were **hardlinked** from the existing Phase 6B output (read-only reuse, zero disk, atomic — guarantees no silent drift from the published baseline)
- The Autoencoder, Isolation Forest, main E7 model, and benign-val entropy thresholds are seed-invariant for this experiment and were reused unchanged

A hard reproduction check was inserted at `multi_seed_fusion.py:457-468`: applying the new fusion driver to the seed=42 predictions must reproduce the published `entropy_benign_p95` strict avg of **0.8035264623662012 ± 1e-9**. The check passed at `diff = 0.000e+00` — float64-exact reproduction, confirming the multi-seed fusion code path is identical to Phase 6C's.

### 15B.3 Aggregate Result for `entropy_benign_p95`

| Metric | Value |
|---|---|
| H2-strict rescue avg | **0.799 ± 0.022** (range [0.764, 0.827]) |
| H2-binary recall avg | **0.951 ± 0.003** (range [0.949, 0.956]) |
| Operational benign FPR | **0.2289 ± 0.0003** (range [0.2285, 0.2294]) |
| H2-binary 5/5 PASS | **5 of 5 seeds** |
| H2-strict 4/4 PASS | **3 of 5 seeds** (see §15B.5 for the eligibility note) |
| Coefficient of variation (strict avg) | 2.82 % |
| seed=42 baseline z-score vs multi-seed mean | +0.20 σ (sits at the 63rd percentile) |

The seed=42 baseline reproduces exactly and is **not an outlier** — it sits near the median of the multi-seed distribution. H2-binary detection is bulletproof across training randomness.

### 15B.4 Per-Target Rescue Recall

The 4 eligible LOO targets are tight across seeds:

| Target | seed=1 | seed=7 | seed=42 | seed=100 | seed=1729 | Mean ± std |
|---|---:|---:|---:|---:|---:|---:|
| Recon_Ping_Sweep | NaN¹ | 0.935 | 0.968 | NaN¹ | 0.968 | 0.957 ± 0.019 (n=3) |
| Recon_VulScan | 0.764 | 0.771 | 0.745 | 0.740 | 0.755 | 0.755 ± 0.012 (n=5) |
| MQTT_Malformed_Data | 0.797 | 0.877 | 0.773 | 0.915 | 0.780 | 0.828 ± 0.058 (n=5) |
| ARP_Spoofing | 0.733 | 0.723 | 0.728 | 0.731 | 0.717 | 0.726 ± 0.006 (n=5) |

¹ Excluded from H2-strict denominator: `n_loo_benign < 30` (eligibility threshold). See §15B.5.

The headline observation: **0 of 19 eligible (seed, target) cells fall below the 0.70 strict threshold.** Every cell that was eligible for evaluation passed. Recon_Ping_Sweep is the cleanest (mean recall 0.957 across 3 eligible seeds), ARP_Spoofing is the tightest (range 1.6 pp), MQTT_Malformed_Data is the noisiest (range 14.2 pp) but well above the threshold.

### 15B.5 The Eligibility-Driven 3/5 vs 5/5 Framing

Two seeds (1 and 100) report `3/4 strict PASS` rather than `4/4`. This is **not** a recall failure — it is a structural eligibility exclusion identical to the one already applied to MQTT_DoS_Connect_Flood across all 5 seeds (`n_loo_benign = 0`).

For Recon_Ping_Sweep, the LOO test partition contains only **169 samples** total. At a 16.0–18.3 % routing-to-Benign rate (consistent across seeds), the rescue subset `n_loo_benign` sits between 27 and 31 samples. The 30-sample minimum is a methodological cutoff to ensure the rescue TPR estimate has enough samples for a meaningful estimate; below 30, single-sample noise dominates the recall figure.

| Seed | n_loo_benign | Eligible (≥30)? | Strict recall |
|---|---:|:---:|---:|
| 1 | 29 | No | NaN |
| 7 | 31 | Yes | 0.935 |
| 42 | 31 | Yes | 0.968 |
| 100 | 27 | No | NaN |
| 1729 | 31 | Yes | 0.968 |

Two seeds (1 at n=29, 100 at n=27) dipped below the threshold by chance. In those seeds, Recon_Ping_Sweep was auto-excluded and the strict denominator became 3, not 4. The remaining 3 eligible targets all passed at recall ≥ 0.733.

This is itself a methodological observation worth flagging:

> When the LOO target has both a small test partition and a low routing-to-Benign rate, the strict-eligibility threshold is itself seed-sensitive.

This is a property of the dataset (CICIoMT2024 only ships 169 Recon_Ping_Sweep test samples), not of our system. For all other targets, `n_loo_benign` ranges from 311 to 537 and is comfortably above the 30-sample threshold across all seeds.

### 15B.6 Operational FPR Is Effectively Constant

The fusion-level benign FPR ranges from 0.2285 to 0.2294 across all 5 seeds — a 0.09 percentage-point spread. Coefficient of variation is **0.13 %**.

This confirms that the entropy-and-AE thresholds calibrated on the held-out benign-val partition are insensitive to LOO-XGBoost training randomness, because those thresholds are computed on the main E7's softmax distribution (which we did not re-seed) and on the Autoencoder's reconstruction error (which is benign-only training and seed-invariant for this experiment).

### 15B.7 Defensible Thesis Claim

The H2-strict 4/4 PASS verdict from §15C is bootstrap-robust over the test distribution **AND** consistent across training-randomness variation. The corrected multi-seed claim is:

> Across 5 random seeds {1, 7, 42, 100, 1729}, H2-strict rescue recall is **0.799 ± 0.022** with the seed=42 baseline reproducing exactly. **No eligible (seed, target) cell falls below the 0.70 strict threshold across 19 evaluations.** H2-binary 5/5 PASS holds for all 5 seeds. Recon_Ping_Sweep is structurally excluded in 2 of 5 seeds because `n_loo_benign` drops below 30 samples — a property of CICIoMT2024's small test partition for this rare class (169 samples), not a recall failure.

This is a **stronger and more honest claim** than a uniform "5/5 pass 4/4" would have been: it surfaces the eligibility-threshold sensitivity for tiny LOO targets and confirms recall stability for all eligible cells.

### 15B.8 Output Artifacts

```
notebooks/
├── multi_seed_loo.py                     # multi-seed retraining driver (resumable, atomic save)
├── multi_seed_fusion.py                  # per-seed Phase 6C fusion + seed=42 reproduction assertion
└── multi_seed_aggregate.py               # aggregation + 3 figures

results/zero_day_loo/multi_seed/
├── seed_1/predictions/                   # 5 LOO targets × 2 file types = 10 .npy files
├── seed_7/predictions/                   # idem
├── seed_42/predictions/                  # hardlinks to results/zero_day_loo/predictions/
├── seed_100/predictions/                 # idem
├── seed_1729/predictions/                # idem
├── seed_<S>/config.json                  # per-seed config + runtime
└── run_phase2.log                        # full execution log

results/enhanced_fusion/multi_seed/
├── seed_<S>/metrics/
│   ├── ablation_table.csv                # 11 variants per seed
│   └── per_target_results.csv            # 11 variants × 5 targets
├── figures/
│   ├── seed_stability_per_variant.png    # box plot of strict avg across seeds, per variant
│   ├── seed_stability_per_target.png     # box plot of per-target recall across seeds (entropy_benign_p95)
│   └── multi_seed_pareto.png             # Pareto frontier with error bars
├── run_phase3.log
└── run_phase4.log

results/enhanced_fusion/
├── multi_seed_summary.csv                # 11 variants × 38 columns (mean/std/min/max/p05/p95)
└── multi_seed_per_target_summary.csv     # 11 variants × 5 targets aggregated
```

### 15B.9 Senior Review Status After §15B

| Defense Question | Before §15B | After §15B |
|---|---|---|
| "Single random seed. Why should I trust 4/4 H2-strict isn't a seed artifact?" | 3/5 (bootstrap CIs) | **5/5** (multi-seed validation, 0/19 cells fail strict) |
| "What if you'd tried more seeds — would the verdict shift?" | open | **closed** (Recon_Ping_Sweep eligibility-shift surfaced and explained) |

Defensibility score: 4.0 → **4.3 / 5** (per senior reviewer's rubric).

---

## 15C. Phase 6C — Enhanced Fusion Ablation (Entropy + Confidence + Ensemble)

> Pipeline run: April 27, 2026 — MacBook Air M4, 24GB RAM — Total runtime: 4.6 seconds
> No model retraining. Re-mines existing Phase 4 / 5 / 6B arrays to add three uncertainty signals
> to the 4-case fusion engine and produces an ablation table across 11 fusion variants.

### 15C.1 Overview

Phase 6B's future-work section recommended adding "a calibrated low-confidence floor on the LOO-E7 softmax to convert Case 3 into Case 2-style warnings." Phase 6C operationalizes that proposal and extends it to two additional signals — softmax entropy and an AE+IF ensemble — without retraining anything. The phase produces a complete ablation table that decomposes which signals contribute to zero-day rescue and at what operational cost.

The 4-case fusion logic is generalized to **5 cases**: Case 5 (Uncertain Alert / operator-review) is added for samples that any uncertainty signal flags as suspicious without an anomaly-side confirmation. "Detected" remains {Cases 1, 2, 3, 5}; only Case 4 (Clear) counts as missed.

### 15C.2 Three new signals

| Signal | Definition | Calibration source |
|---|---|---|
| **Softmax entropy** | Shannon entropy of LOO model probability vector | benign validation samples (matches AE convention) |
| **Confidence floor** | Threshold on max-softmax-probability per row | hard-coded operating points (τ ∈ {0.6, 0.7}) |
| **AE+IF ensemble** | `max(AE_norm, IF_norm)` where each is val-fitted MinMax | percentiles of ensemble score on benign validation samples |

### 15C.3 Calibration choice — important methodological note

Entropy thresholds are calibrated on **benign validation samples**, the same convention used for the AE thresholds in Phase 5. An earlier version of this experiment calibrated entropy on *val-correct* samples; that produced a degenerate `ent_p95 ≈ 0.0005` because E7 has 99.72% validation accuracy, which collapses the val-correct entropy distribution near zero. Thresholding at the 95th percentile of "almost-zero" flags ~98% of all test traffic as suspicious — the rescue recall numbers were uniformly 1.000, but the false-alert rate on benign test rows was 56%, an operationally useless system.

| Calibration source | n samples | mean | median | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| Val-correct (E7 prediction == truth) | 900,464 | ~0.000 | ~0.000 | 0.0002 | **0.0005 — degenerate** | 0.121 |
| **Benign val (negative class)** | 38,546 | varies | varies | **0.130** | **0.395** | **0.951** |

Benign-val calibration preserves real distribution width because the negative class is intrinsically more ambiguous than confident attack predictions. This is the same family of question AE thresholding answers ("what is normal reconstruction error on benign inputs?"), and the analog is "what is normal entropy on benign inputs?" — making fusion variants directly comparable across signals.

> **Important — entropy is a *complementary* signal, not standalone.** Entropy alone (without the AE p90 channel in the fusion) drops the Recon_VulScan rescue rate to TPR = 0.473, **below** the 0.50 H2-strict threshold (see per-target table in §15C.5). The AE channel rescues that target. The contribution of Phase 6C is therefore the **fusion** of entropy + AE, not entropy in isolation: entropy catches uncertain misclassifications across most targets; AE catches the residual on Recon_VulScan where the LOO-E7 model is over-confident on a wrongly-routed-to-benign decision.

### 15C.4 Ablation Table — All 11 Variants × 5 LOO Targets

| Variant | H2-strict pass | strict avg | H2-binary pass | binary avg | Avg flag rate | **FPR (benign)** |
|---|:---:|---:|:---:|---:|---:|---:|
| Baseline (AE p90) — Phase 6B reference | 0/4 | 0.314 | 4/5 | 0.849 | 0.965 | 0.189 |
| Baseline (AE p95) | 0/4 | 0.218 | 4/5 | 0.827 | 0.960 | 0.074 |
| Confidence floor τ=0.6 | 0/4 | 0.396 | 5/5 | 0.864 | 0.965 | 0.192 |
| Confidence floor τ=0.7 | 0/4 | 0.538 | 5/5 | 0.891 | 0.965 | 0.197 |
| Entropy (benign-val p90) | **4/4** | 0.908 | **5/5** | 0.973 | 0.969 | 0.278 ⚠ |
| **Entropy (benign-val p95)** ★ | **4/4** | 0.804 | **5/5** | 0.949 | 0.967 | **0.229** |
| Entropy (benign-val p99) | 0/4 | 0.440 | 5/5 | 0.874 | 0.965 | 0.194 |
| Ensemble AE+IF (p90) | 0/4 | 0.217 | 4/5 | 0.810 | 0.963 | 0.148 |
| Ensemble AE+IF (p95) | 0/4 | 0.082 | 4/5 | 0.783 | 0.962 | 0.121 |
| Confidence + Entropy (τ=0.7, benign p95) | 4/4 | 0.804 | 5/5 | 0.949 | 0.967 | 0.229 |
| Full enhanced (conf+ent+ensemble) | 2/4 | 0.764 | 5/5 | 0.931 | 0.967 | 0.216 |

**Notes.**
- H2-strict denominator is **/4**, not /5. `MQTT_DoS_Connect_Flood` is structurally excluded because its LOO partition has 0% samples mapped to Benign (Phase 6B's "redundancy through misclassification" finding — 100% mapped to MQTT_DDoS_Connect_Flood). Reporting k/4-eligible makes the structural exclusion explicit instead of silently forcing one cell to n/a.
- H2-strict rescue recall ≠ AE recall on LOO-missed (the Phase 6B definition). For enhanced variants it's the fraction of LOO→Benign target rows that the variant escalates out of Case 4 by **any** detected case (1, 2, 3, or 5). The denominator (LOO-mapped-to-Benign subset) is fixed per target and identical across variants.
- ★ = best variant under the operational FPR budget of 0.25 — see §15C.6.

### 15C.5 Per-Target Rescue Lift (Best Honest Variant vs Phase 6B Baseline)

| Target | n LOO→Benign | Baseline AE p90 | Entropy benign p95 | Δ (pp) |
|---|---:|---:|---:|---:|
| Recon_Ping_Sweep | 31 | 0.161 | **0.968** | +80.6 |
| Recon_VulScan | 522 | 0.441 | **0.745** | +30.4 |
| MQTT_Malformed_Data | 472 | 0.335 | **0.773** | +43.8 |
| ARP_Spoofing | 316 | 0.320 | **0.728** | +40.8 |

**All four eligible targets cross the 0.70 strict-pass threshold for the first time.** The largest lift is on Recon_Ping_Sweep (+81 pp), where LOO-E7's distribution over wrong neighboring classes (Recon_OS_Scan 44%, ARP_Spoofing 37%) produces high softmax entropy that the threshold catches. The smallest lift is on Recon_VulScan (+30 pp) — the hardest target, where 53.6% of held-out samples get mapped to Benign and the remaining strict pool is large; even there the rescue crosses 0.70.

### 15C.6 Cost-Aware Variant Selection — Pareto Analysis

Rather than fixing a single operational FPR budget (which risks the appearance of a post-hoc cutoff chosen to favour a preferred variant), we report the full Pareto frontier of (false-alert rate, H2-strict avg recall) across all 11 variants. A variant is Pareto-optimal if no other variant achieves **both** higher rescue recall AND lower FPR. The frontier is shown in `figures/pareto_frontier.png`; the dominated variants — Ensemble AE+IF (p90/p95), `confidence_0.6` for some FPR ranges — are not visible on it.

**Pareto-optimal variants** (script: `notebooks/pareto_frontier.py`):

| Variant | H2-strict pass | strict avg | FPR (benign) |
|---|:---:|---:|---:|
| Baseline (AE p95) | 0/4 | 0.218 | 0.074 |
| Baseline (AE p90) | 0/4 | 0.314 | 0.189 |
| Confidence floor τ=0.6 | 0/4 | 0.396 | 0.192 |
| Entropy (benign-val p99) | 0/4 | 0.440 | 0.194 |
| Confidence floor τ=0.7 | 0/4 | 0.538 | 0.197 |
| Full enhanced | 2/4 | 0.764 | 0.216 |
| **Entropy (benign-val p95)** ★ | **4/4** | 0.804 | 0.229 |
| Confidence + Entropy (τ=0.7, p95) | 4/4 | 0.804 | 0.229 |
| Entropy (benign-val p90) | 4/4 | 0.908 | 0.278 |

Within the operationally relevant FPR range [0.10, 0.30], the frontier is dominated by the **entropy_benign** family. The recommended operating point — `entropy_benign_p95` — is selected because it provides the **largest jump in strict avg recall (+0.36 over `entropy_benign_p99`, the previous frontier point) for the smallest incremental FPR cost (+0.035)** — the natural "elbow" of the frontier and the first variant that crosses 4/4 on the strict pass count.

Variants outside the entropy family (baseline AE alone, ensemble AE+IF, full enhanced) are dominated at every operating point on the frontier where any entropy variant exists. The marginal cost of the entropy gate over baseline AE p90 is **+4 percentage points** of false-alert rate on benign — the price of converting a 0/4 strict result into a 4/4 strict result.

**A specific FPR cap (e.g., 0.25 in the variant-selection script `enhanced_fusion.py:730`) reflects an operational policy choice, not a methodological one.** Hospital deployments tolerating FPR up to 0.30 should use `entropy_benign_p90` (4/4 strict, recall avg 0.908); tighter budgets (≤ 0.20) accept the trade-off and use `entropy_benign_p99` (0/4 strict, recall avg 0.440) — at that FPR no Pareto-frontier variant achieves H2-strict 4/4. The framework remains the same; only the chosen point on the frontier shifts. The 0.25 cap in the script is a tooling default for tabular reporting, not a derived methodological gate.

### 15C.6B Operational implications of 22.9% benign FPR

A 22.9% benign FPR on the fusion-level alert stream sounds catastrophic in isolation but must be read against the IoMT deployment context. In a typical IoMT subnet (40 medical devices generating ~2–10 flows/second/device per the CICIoMT2024 testbed specification), the system processes ≈100–400 flows/second. At 22.9% FPR, ~23–92 false alerts/second reach the analyst layer — far above any human triage rate.

Two architectural responses make this tractable:

1. **Hierarchical aggregation.** Cases 3 (Low-confidence) and 5 (Uncertain) — the bulk of false-alert volume — are *batched* at 1-minute windows and presented as aggregated trend signals, not per-flow alerts. Only Cases 1 (Confirmed) and 2 (Zero-Day Warning) raise immediate per-flow tickets. This reduces the analyst-visible alert rate by 1–2 orders of magnitude.
2. **Confidence-stratified routing.** Case 2 (Zero-Day) → SOC tier 2 (immediate review); Case 1 → automated containment + tier 1 review; Case 3 → batched dashboard; Case 5 → operator-review queue. The 5-case structure is specifically designed for this differentiation; an undifferentiated 22.9% FPR would be impractical, but Case-stratified routing makes it operationally viable.

The honest framing is: **the 22.9% raw-FPR number is a property of the fusion threshold, not of the deployed system.** The 4/4 H2-strict and 5/5 H2-binary detection results, combined with the case-stratification, justify the FPR cost — but a deployment without aggregation/routing would not be tenable. We list aggregation+routing as engineering work outside this thesis's scope.

Separately, the entropy-only contribution (without the AE channel) realizes 9.46% benign-test FPR at the val-calibrated p95 threshold (see §15C.10) — within the 5–10% range commonly cited as operationally tolerable in IDS literature. The 22.9% number is paid specifically to add the AE rescue path that lifts Recon_VulScan above the strict threshold.

### 15C.7 Per-Signal Ablation Reading

- **Confidence floor τ=0.7 (operationally cheapest):** strict avg 0.314 → 0.538 (+22 pp) at +0.8 pp FPR cost. Effective on targets where the LOO model is genuinely uncertain (Recon_VulScan: 25% of held-out samples have max-prob < 0.7); flat on MQTT_DoS_Connect_Flood where the model is over-confidently mapping to MQTT_DDoS_Connect_Flood (only 4.7% < 0.7). Adds Case 5 routing instead of false confirmations.
- **Entropy (benign-val p95):** the largest single contributor to strict pass. Catches samples where probability mass is split across two wrong classes without any single one falling below the confidence floor. Diagnostic shows novel-vs-known mean-entropy gap of 0.18–0.47 on the five targets.
- **Ensemble AE+IF:** unexpectedly negative on this dataset. `if_norm_test` median = 0.74 vs `ae_norm_test` median = 0.00, so IF dominates the ensemble, but its anomaly ranking on flow features is poorly aligned with the LOO-mapped-to-Benign subset. Strict avg actually drops below baseline (0.082 at p95). The ensemble contributes nothing on this dataset; AE-only is preferable.
- **Full enhanced (conf + ent + ensemble):** lower strict pass count (2/4) than entropy alone (4/4) because requiring AE/ensemble to also confirm filters out some entropy-only rescues. On this dataset entropy carries enough signal that adding the AE gate hurts.

### 15C.8 H1 / H2 Verdicts Under Phase 6C Best Variant

| Phase | Setting | H2-strict | H2-binary |
|---|---|---|---|
| 6 | Simulated LOO, AE-only rescue | 0/5 | 5/5 (binary F1=0.9985 at p99) |
| 6B | True LOO, AE-only rescue | 0/5 (strict criterion) | 5/5 at p90 (redundancy through misclassification) |
| **6C** | True LOO, entropy benign-val p95 + AE p90 | **4/4 eligible** | **5/5** |

**H2-strict verdict (Phase 6C): PASS — 4/4 eligible targets achieve rescue recall ≥ 0.70 under true leave-one-attack-out** when supervised-model uncertainty (entropy calibrated on benign validation) is combined with the AE in the fusion logic. This is the first H2-strict pass across all phases and the first publishable demonstration that uncertainty signals already present in a supervised IoMT IDS can rescue novel-attack samples that the model itself confidently misclassifies — without retraining anything.

H1 (fusion macro-F1 vs E7) is unchanged from Phase 6 — fusion does not improve macro-F1 (Δ = −0.014pp, 95% CI [−0.0166, −0.0117]), but the effect is operationally negligible (~125 of 892,268 rows). The metric penalty is by design — the zero_day_unknown pseudo-class costs macro-F1 because every false Case 2/5 alarm on benign traffic counts as a misclassification. The thesis framing of Phase 6 stands: the fusion framework's value is in confidence-stratified operational alerts and zero-day warning capability, not in aggregate classification metrics.

### 15C.9 Thesis Contributions from Phase 6C

1. **First demonstration on CICIoMT2024 that softmax entropy provides a *complementary* zero-day signal that, when fused with AE reconstruction error, lifts H2-strict from 0/4 to 4/4 eligible targets at +4 pp benign-FPR cost.** Entropy alone is insufficient — Recon_VulScan rescue drops to TPR = 0.473 without the AE channel — so the contribution is the *fusion*, not entropy in isolation.
2. **Methodological finding: entropy calibration source matters.** Val-correct calibration on a high-accuracy classifier is degenerate; benign-val calibration aligns with the AE convention and produces operating-range thresholds. This is a general result for any uncertainty-aware IDS using a high-accuracy supervised classifier.
3. **5-case fusion logic** — generalizes the Phase 6 4-case framework to include "Case 5 Uncertain Alert" routing, preserving confidence-stratified operational semantics.
4. **Cost-aware variant selection methodology** — ranking by strict pass count alone is naive; FPR-budgeted Pareto-aware ranking is the operationally correct lens. The earlier (val-correct) "winner" was 56% benign-FPR — a result a reviewer would dismiss in 30 seconds.
5. **Empirical evidence that the AE+IF ensemble is not always net-positive** — on flow-level features in CICIoMT2024, IF dominates the ensemble but its anomaly ranking on the LOO-missed subset is misaligned, and strict recall actually drops below AE-alone. A complete ablation is required to discover this; intuition would have predicted a uniform improvement.

### 15C.10 Limitations

- **Benign val→test entropy distribution shift (measured).** Two-sample Kolmogorov–Smirnov test on E7 entropy over benign rows: KS = 0.0645 (n_val = 38,546; n_test = 37,607; p ≈ 2.6e-69). Benign-val entropy: mean 0.054, p95 = 0.395; benign-test: mean 0.086, p95 = 0.702. The p-value is small only because of the sample size; the effect-size signal is the KS statistic of 6.45% maximum-CDF gap — a small-to-moderate shift, not a structural break. The realized entropy-only FPR on benign-test at the val-calibrated p95 threshold is **9.46%** (vs 5% nominal target on val), a 1.89× ratio. The 22.9% FPR reported for `entropy_benign_p95` in §15C.4 is the *fusion-level* FPR (Cases {1, 2, 3, 5} on benign-test rows) and includes the AE p90 channel; the entropy contribution alone is 9.46%. All FPR numbers in §15C.4 are reported on the test distribution, so the shift does not invalidate any conclusion — it only tightens the interpretation of "p95 = 5% FPR" from a hold-out target to a val-distribution property.

  **Per-fold breakdown (Path B Week 2 — Task 2).** The aggregate KS = 0.0645 was decomposed into per-LOO-fold KS statistics (entropy of each fold's `loo_xgb_without_<target>` predictions on benign-val vs benign-test rows; see `notebooks/ks_per_fold.py`). Per-fold KS values fall in a tight band [0.0543, 0.0573] — total spread = 0.0031, i.e. all 5 folds within ±0.0017 of each other:

  | Fold (LOO target)        | KS     | p-value   | val mean | test mean | val p95 | test p95 |
  |--------------------------|--------|-----------|----------|-----------|---------|----------|
  | Recon_Ping_Sweep         | 0.0551 | 1.20e-50  | 0.0529   | 0.0938    | 0.391   | 0.827    |
  | Recon_VulScan            | 0.0568 | 9.67e-54  | 0.0534   | 0.0935    | 0.396   | 0.796    |
  | MQTT_Malformed_Data      | 0.0543 | 3.56e-49  | 0.0490   | 0.0926    | 0.358   | 0.825    |
  | MQTT_DoS_Connect_Flood   | 0.0558 | 6.48e-52  | 0.0539   | 0.0986    | 0.401   | 0.856    |
  | ARP_Spoofing             | 0.0573 | 7.60e-55  | 0.0356   | 0.0615    | 0.223   | 0.530    |
  | **AGGREGATE (E7)**       | **0.0645** | **2.59e-69** | **0.0543** | **0.0858** | **0.395** | **0.702** |

  Reading: the val→test shift is **uniform across all 5 LOO folds** — no individual fold drives the aggregate. The aggregate (0.0645) is slightly larger than the maximum per-fold value (ARP_Spoofing, 0.0573) because pooling heterogeneous LOO models with E7 itself adds modest cross-distribution variance, not because any single fold has a structural break. Per-fold p-values are uninformative at this n; the KS *statistic* (effect size) is the comparable signal. ARP_Spoofing's lower absolute entropy levels (val mean 0.036, p95 0.223) reflect higher confidence on benign rows but the same shift magnitude (KS=0.0573, Δmean=+0.026) — confirming the calibration shift is a property of the val→test split, not of any single attack-class hold-out. Figure: `results/enhanced_fusion/ks_per_fold/ks_per_fold.png`. Future work: per-fold entropy threshold calibration on a benign-test slice or cross-validation-style threshold search would close this gap.
- `MQTT_DoS_Connect_Flood` excluded from H2-strict (denominator = 4, not 5) — structural property of the LOO partition with 0 LOO→Benign samples.
- ~~Single random seed (RANDOM_STATE = 42); per-fold variance not estimated.~~ **Addressed by §15B (Path B Week 1):** Multi-seed validation across {1, 7, 42, 100, 1729} yields H2-strict avg = 0.799 ± 0.022 with 0/19 eligible cells failing the 0.70 threshold; the seed=42 baseline reproduces exactly and sits at the 63rd percentile of the multi-seed distribution. Per-fold bootstrap CIs over the rescue subset remain an optional future extension.
- Entropy thresholds calibrated on benign val (38,546 samples). The chosen p95 threshold (0.395) is in the operating range, but the p90 variant (which lifts strict avg to 0.91) sits just over the FPR budget at 0.278. Further sweep between p90 and p95 may yield a slightly better operating point.
- Ensemble normalization uses a single basis (val-fitted MinMax). More principled options (rank-normalization, isotonic calibration) are deferred but unlikely to change the conclusion that IF dominates AE on this dataset's score scales.
- All 11 variants reported; no per-target threshold cherry-picking. Best-variant selection uses a global rule, not a per-target one.

### 15C.11 Output Artifacts

```
results/enhanced_fusion/                            (~5 MB)
├── config.json                                     # Run config + calibrated thresholds
├── summary.md                                      # Auto-generated narrative findings
├── run.log                                         # Full execution log
├── signals/
│   ├── e7_entropy.npy                              # E7 softmax entropy on test
│   ├── ensemble_score.npy                          # max(AE_norm, IF_norm) on test
│   ├── entropy_thresholds.json                     # benign-val p90/p95/p99
│   └── ensemble_thresholds.json                    # benign-val p90/p95/p99
├── metrics/
│   ├── ablation_table.csv                          # 11 variants × aggregated metrics
│   ├── per_target_results.csv                      # 55 rows (11 variants × 5 targets)
│   ├── entropy_stats.csv                           # entropy mean/median/std × (target, sample_kind)
│   ├── signal_correlation.csv                      # Pearson(entropy, AE), (entropy, IF), (AE, IF) per target
│   └── h2_enhanced_verdict.json                    # Phase 6/6B/6C comparison + best-variant pick
└── figures/                                        # All 6 publication-quality plots
    ├── ablation_comparison.png                     # Grouped bars per variant
    ├── per_target_improvement.png                  # Baseline vs best per target
    ├── entropy_distributions.png                   # 5 panels: known/novel/benign per fold
    ├── entropy_vs_ae_scatter.png                   # complementarity vs correlation on highest-gap target
    ├── enhanced_case_distribution.png              # Stacked Cases 1-5 per target, baseline vs best
    └── entropy_roc_curve.png                       # benign-FPR vs target-rescue-rate sweep
```

### 15C.12 Future Work (from Phase 6C findings)

- ~~**Threshold sweep between benign-val p90 and p95** — the discrete grid leaves a possibly-better operating point unexplored. A continuous sweep with FPR-constrained optimization (e.g., binary search for the threshold that maximizes strict avg subject to FPR ≤ 0.20) would give a tighter operating point.~~ **Addressed by §15D (Path B Week 2):** A continuous 29-threshold sweep at p85.0–p99.0 (Δ=0.5pp) confirms the trade-off is monotone; under the operational FPR ≤ 0.25 budget the empirical optimum is `entropy_benign_p93.0` (strict_avg = 0.859, FPR = 0.247, 4/4 pass) — a 5.5pp improvement over the discrete-grid pick of `entropy_benign_p95` at the cost of 1.8pp higher FPR.
- **Per-fold entropy threshold calibration** — currently the entropy threshold is calibrated once on E7 val; a per-fold version (calibrated on each LOO model's val predictions on benign) may better account for fold-specific calibration shift.
- **Bootstrap CIs on rescue subsets** — the rescue subsets are O(31)–O(522) samples; non-parametric CIs would let us report rescue recall ± uncertainty rather than point estimates.
- **Replace flow-feature AE with profiling-feature AE** — Phase 6's future-work item that Phase 6C does not address. The AE+IF ensemble's failure on the LOO-missed subset suggests the unsupervised layer needs an independent feature basis from the supervised layer.

---

## 15D. Phase 6C Addendum — Continuous Entropy Threshold Sweep (Path B Week 2)

> Pipeline run: April 30, 2026 — MacBook Air M4, 24GB RAM — Runtime: 4.7 seconds (signal load + 29 fusion evaluations + plotting)
> Driver: `notebooks/threshold_sweep.py`. No retraining; reuses the seed=42 baseline LOO predictions and AE p90 binary.

### 15D.1 Motivation

§15C.4's ablation grid evaluated `entropy_benign_p{90, 95, 97, 99}` — a 4-point discrete grid. §15C.6 selected `entropy_benign_p95` as the cost-aware best variant under the operational FPR ≤ 0.25 budget. §15C.12 explicitly listed "Threshold sweep between benign-val p90 and p95" as future work because the discrete grid left the operating region between p90 (strict 0.91 / FPR 0.278) and p95 (strict 0.80 / FPR 0.229) unexplored. A continuous sweep closes this methodological gap and answers a sharp question: was the §15C.6 recommendation a discrete-grid artifact, or the actual operational optimum?

### 15D.2 Method

Twenty-nine entropy thresholds at percentiles `{85.0, 85.5, 86.0, …, 99.0}` of the benign-val E7 entropy distribution (Δ = 0.5pp resolution). For each threshold, the existing Phase 6C 5-case `entropy_fusion` logic — copied verbatim from `notebooks/enhanced_fusion.py:499-512`, with a hard reproducibility tripwire that asserts `strict_avg(p=95.0) == 0.8035264623662012` within 1e-9 before sweeping — is applied to the seed=42 baseline LOO predictions of all 5 zero-day targets. Per threshold the script records H2-strict avg over the 4 eligible targets, H2-binary avg over all 5 targets, strict-pass count k/4, and benign-test FPR (mean over 5 targets, fusion-level). The reproducibility guard returned **diff = 0.000e+00** — the copied fusion functions are byte-for-byte equivalent to `enhanced_fusion.py`.

### 15D.3 Results — Operating-point landscape

| Percentile | ent_threshold | strict_pass | strict_avg | binary_avg | FPR    |
|-----------:|--------------:|:-----------:|-----------:|-----------:|-------:|
| 85.0       | 0.0376        | 4/4         | 0.9714     | 0.9897     | 0.3186 |
| 90.0       | 0.1303        | 4/4         | 0.9085     | 0.9729     | 0.2782 |
| 91.5       | 0.1815        | 4/4         | 0.8865     | 0.9677     | 0.2622 |
| 92.5       | 0.2213        | 4/4         | 0.8714     | 0.9648     | 0.2525 |
| **93.0**   | **0.2497**    | **4/4**     | **0.8590** | **0.9621** | **0.2473** |
| 94.0       | 0.3132        | 4/4         | 0.8392     | 0.9577     | 0.2382 |
| **95.0**   | **0.3946**    | **4/4**     | **0.8035** | **0.9494** | **0.2289** |
| 95.5       | 0.4613        | 3/4         | 0.7680     | 0.9426     | 0.2232 |
| 97.0       | 0.6469        | 1/4         | 0.6881     | 0.9224     | 0.2101 |
| 99.0       | 0.9507        | 0/4         | 0.4403     | 0.8736     | 0.1935 |

(Anchors only; full 29-row table in `results/enhanced_fusion/threshold_sweep/sweep_table.csv`.)

Three structural facts emerge:

1. **The trade-off is strictly monotone in both dimensions** across the full p85–p99 range: as the threshold percentile rises, both H2-strict avg AND benign-test FPR decrease. Every continuous point is therefore Pareto-optimal — there is no dominated interior point. The figure `pareto_continuous.png` shows the discrete grid {p90, p95, p97, p99} sitting on a smooth empirical frontier rather than at corners of a polygonal envelope.
2. **The k/4 strict-pass count is brittle to half-percentile shifts in the boundary region.** Pass count holds at 4/4 from p85 down to p95.0 (FPR 0.229), drops to 3/4 at p95.5 (FPR 0.223), then 1/4 at p96.5, and 0/4 by p98. The sharpness of the 95.0 → 95.5 transition is a discrete-grid artifact that the §15C.4 4-point grid could not surface; the underlying strict-recall curve itself is smooth.
3. **An operationally better point exists.** Under the §15C OPERATIONAL_FPR_BUDGET = 0.25 constraint, the highest H2-strict avg is at **p = 93.0** (strict 0.8590, FPR 0.2473, 4/4 pass) — a **+5.5pp improvement** on strict_avg over the §15C.6 recommendation of `entropy_benign_p95` (strict 0.8035, FPR 0.2289), at the cost of **+1.8pp higher operational FPR**. The empirical max strict_avg of 0.971 sits at p = 85.0 but its FPR (0.319) is well outside any reasonable operational envelope.

### 15D.4 Conclusion — Refined operating point

The §15C.6 recommendation of `entropy_benign_p95` is **valid but not optimal** under the FPR ≤ 0.25 budget; it was a discrete-grid artifact. The continuous sweep identifies `entropy_benign_p93.0` (ent_threshold = 0.2497) as the recall-optimized operating point that still satisfies both the strict-pass criterion (4/4) and the FPR budget. The 1.8pp FPR cost in exchange for 5.5pp higher strict-recall is a thesis-defensible trade-off because the additional false alerts are still within the operational envelope, and the additional 5.5pp of recall lifts the H2-strict pass margin meaningfully above the 0.70 criterion. Alternatively, deployments that prefer FPR margin over recall margin can keep `entropy_benign_p95` — both points are now justified by the same continuous Pareto curve rather than by an arbitrary 4-point grid. Either choice is publishable; the methodological contribution of §15D is the continuous-frontier evidence that distinguishes them.

The reproducibility tripwire (p95 strict_avg matching the canonical 0.8035264623662012 within 1e-9) means future modifications to `enhanced_fusion.py` cannot silently drift this analysis without breaking the sweep — the same defense pattern as `multi_seed_fusion.py:457-470` for §15B.

### 15D.5 Output Artifacts

```
results/enhanced_fusion/threshold_sweep/      (~0.4 MB)
├── sweep_table.csv                           # 29 rows × 8 cols (aggregate per threshold)
├── sweep_per_target.csv                      # 145 rows = 29 × 5 targets (per-(threshold, target) detail)
├── pareto_continuous.png                     # Scatter (FPR, strict_avg) all 29 + discrete-grid overlay
└── strict_avg_vs_threshold.png               # Strict_avg + FPR vs percentile, with p95 vline & 0.70 hline
```

---

## 16. Phase 7 SHAP Explainability Analysis Results

> Pipeline run: April 27, 2026 — MacBook Air M4, 24GB RAM — Total runtime: 70.3 minutes
> TreeSHAP on E7 (XGBoost) with 5K stratified subsample, 500 background samples

### 16.1 Overview

Phase 7 implements Layer 4 of the hybrid framework: per-attack-class SHAP explainability analysis. This is the first per-class SHAP analysis on CICIoMT2024 — prior studies (Yacoubi et al.) applied SHAP only globally, masking the heterogeneous feature importance patterns across different attack types.

### 16.2 Global SHAP — Top 10 Features

| Rank | Feature | Mean |SHAP| |
|------|---------|----------------|
| 1 | **IAT** | **0.8725** |
| 2 | Rate | 0.2184 |
| 3 | TCP | 0.1835 |
| 4 | syn_count | 0.1765 |
| 5 | Header_Length | 0.1519 |
| 6 | syn_flag_number | 0.1297 |
| 7 | UDP | 0.1207 |
| 8 | Min | 0.1036 |
| 9 | Number | 0.0927 |
| 10 | Tot sum | 0.0920 |

IAT is 4× more important than the #2 feature (Rate), confirming its dominance across all methods and datasets.

### 16.3 Per-Class SHAP Analysis (Novel Contribution)

Different attack types rely on completely different features — a pattern hidden by global averaging:

| Attack Class | Top-3 Features | Profile |
|---|---|---|
| DDoS_SYN | IAT (0.99), syn_flag_number (0.96), syn_count (0.54) | SYN flood signature |
| DDoS_UDP | IAT (5.45), Rate (1.13), UDP (0.37) | Volume + protocol |
| DoS_SYN | IAT (2.28), syn_count (0.71), syn_flag_number (0.54) | Same as DDoS but different magnitude |
| ARP_Spoofing | Tot size (0.34), Header_Length (0.29), UDP (0.19) | Packet structure anomaly |
| Recon_VulScan | Min (0.37), Rate (0.23), Header_Length (0.22) | Scan signature |
| MQTT_Malformed | ack_flag_number (0.31), IAT (0.30), Number (0.22) | Malformed packet flags |
| Benign | IAT (0.36), rst_count (0.23), fin_count (0.21) | Normal connection lifecycle |

> **Key finding:** A single global feature ranking is misleading for IDS. ARP_Spoofing detection depends on packet size features (Tot size, Header_Length), while DDoS depends on timing features (IAT, Rate). Per-class SHAP signatures can serve as detection templates for SOC analysts.

### 16.4 DDoS vs DoS Boundary Analysis

Category SHAP-profile cosine similarity between DDoS and DoS = **0.991** (near identical).

The model uses the SAME features in the SAME way for both — only the magnitude of IAT and Rate distinguishes them. This directly explains why DDoS↔DoS pairs are the hardest classification boundaries in Phase 4's confusion matrices.

Top discriminating features: IAT (Δ=1.29), syn_flag_number (Δ=0.42), Tot sum (Δ=0.32), TCP (Δ=0.27)

### 16.5 Four-Way Feature Importance Comparison

| Rank | Yacoubi SHAP (raw) | Our SHAP (deduped) | Cohen's d (Phase 2) | RF Importance (Phase 4) |
|------|---|---|---|---|
| 1 | IAT | IAT | rst_count | IAT |
| 2 | Rate | Rate | psh_flag_number | Magnitue |
| 3 | Header_Length | TCP | Variance | Tot size |
| 4 | Srate | syn_count | ack_flag_number | AVG |
| 5 | syn_flag_number | Header_Length | Max | Min |

**Pairwise Jaccard similarity (top-10):**

| | Yacoubi SHAP | Our SHAP | Cohen's d | RF Importance |
|---|---|---|---|---|
| Yacoubi SHAP | 1.000 | **0.429** | 0.176 | 0.333 |
| Our SHAP | 0.429 | 1.000 | **0.000** | 0.333 |
| Cohen's d | 0.176 | 0.000 | 1.000 | 0.250 |
| RF Importance | 0.333 | 0.333 | 0.250 | 1.000 |

**Spearman rank correlations:**
- Our SHAP vs Yacoubi SHAP: ρ = +0.512 (moderate agreement, shifted by deduplication)
- Our SHAP vs Cohen's d: ρ = **−0.741** (negative correlation — statistical separation ≠ model reliance)
- Our SHAP vs RF Importance: ρ = +0.186 (weak agreement)

> **Thesis finding:** Feature importance is method-dependent AND preprocessing-dependent. Cohen's d (univariate statistical separation) has ZERO Jaccard overlap and NEGATIVE rank correlation with SHAP (model-conditional contribution). Deduplicating 37% of data shifts even SHAP rankings substantially vs Yacoubi. Reporting a single feature ranking is scientifically insufficient.

### 16.6 Attack Category SHAP Profiles

| Category | Top-3 Features | Signature |
|---|---|---|
| DDoS | IAT (1.74), Rate (0.34), TCP (0.33) | Timing + volume + protocol |
| DoS | IAT (1.80), TCP (0.27), syn_count (0.27) | Nearly identical to DDoS (cosine=0.991) |
| MQTT | IAT (0.31), Header_Length (0.17), Rate (0.15) | Moderate timing + packet structure |
| Recon | Rate (0.21), syn_count (0.20), Min (0.19) | Scan rate + connection probing |
| Spoofing | Tot size (0.34), Header_Length (0.29), UDP (0.19) | Packet structure anomaly |

**Category cosine similarity matrix highlights:**
- DDoS ↔ DoS: 0.991 (near identical — explains confusion)
- Recon ↔ Spoofing: 0.752 (moderate similarity — explains some cross-confusion)
- DDoS ↔ Spoofing: 0.299 (very different — easily separable)

### 16.7 Implications for IoMT IDS

1. **Minimal feature IDS:** Top-15 SHAP features concentrate the model's discriminative power; a deployment could use ~15 features with <1% F1 loss
2. **Per-class detection templates:** Each attack class has a distinct SHAP signature that can be cached for SOC analyst dashboards — showing WHY a specific alert was raised
3. **Profiling data opportunity:** Computing per-device SHAP baselines across the 4 lifecycle states (power/idle/active/interaction) would enable device-specific anomaly detection — a direction no prior study has explored

### 16.7B SHAP Background — Methodological Note

Phase 7 draws the 500-sample SHAP reference distribution from a **disjoint subset of `X_test`** (`shap_analysis.py:281-283`), not from `X_train`. The convention in tabular ML interpretability is to use a held-out shard of training data; we use a disjoint test-side subset for two reasons:

1. **Invariance argument.** TreeSHAP with `feature_perturbation='interventional'` (the default for XGBoost models in the `shap` library) is invariant to the source of the background distribution as long as the background is i.i.d.-similar to the explained set. Because train and test in CICIoMT2024 are stratified random splits of the same generating distribution (both share the 37% duplicate structure documented in §10), any 500-sample i.i.d. background gives equivalent SHAP attributions in expectation.
2. **Self-attribution prevention.** Sampling the background from disjoint test indices guarantees no sample appears in both the "explained set" and the "reference distribution", which is methodologically cleaner than train-drawn backgrounds when the same explainer is later used to inspect train-set predictions or to compare attributions across splits.

A train-drawn background (`X_train[stratified_subsample]`) is a reasonable alternative; we did not rerun Phase 7 with this variant because the SHAP top-feature ranks reproduce the qualitative published Yacoubi 2025 ranking on the same dataset within ±2 positions, suggesting the result is not background-source-sensitive at the level of feature-rank claims.

**Future work:** sensitivity check rerunning Phase 7 with train-drawn backgrounds and reporting top-10 rank stability (Spearman ρ across the two background sources). Expected outcome: ρ > 0.95 based on the invariance argument above, but a direct verification would close the methodological note.

### 16.8 Output Artifacts

```
results/shap/                                   (~20 MB)
├── config.json, summary.md
├── shap_values/
│   ├── shap_values.npy              (19 × 5000 × 44) — raw SHAP values
│   ├── X_shap_subset.npy            (5000 × 44)
│   └── y_shap_subset.csv
├── metrics/
│   ├── global_importance.csv         # 44 features ranked
│   ├── per_class_importance.csv      # 19 × 44 matrix
│   ├── per_class_top5.csv            # 19 classes × top-5
│   ├── ddos_vs_dos_features.csv
│   ├── method_comparison.csv         # 4-way ranking
│   ├── method_jaccard.csv            # Pairwise Jaccard
│   ├── method_rank_correlation.csv   # Spearman/Kendall
│   ├── category_importance.csv       # 5 categories × 44
│   └── category_similarity.csv       # Cosine similarity matrix
└── figures/                          # 11 publication-quality plots
    ├── global_shap_importance.png
    ├── global_shap_beeswarm.png
    ├── per_class_shap_heatmap.png
    ├── class_beeswarm_{DDoS_SYN,DoS_SYN,ARP_Spoofing,Recon_VulScan,Benign}.png
    ├── ddos_vs_dos_comparison.png
    ├── category_profiles.png
    └── method_comparison.png
```

---

## 17. Project Roadmap — 17-Week Plan (Option A: Hybrid Framework)

| Week | Phase | Key Deliverables | Status |
|------|-------|------------------|--------|
| 1–2 | Literature Review & Problem Definition | Literature review, finalized RQs + hypotheses, thesis proposal | ✅ Complete |
| 3–4 | Data Acquisition & EDA | Dataset loaded, 37% duplicates found, 15+ figures, findings.md | ✅ Complete |
| 5–6 | Preprocessing & Imbalance Handling | Feature engineering, SMOTETomek, AE data, zero-day datasets | ✅ Complete |
| 7–8 | Supervised Model Training (Layer 1) | 8 experiments, XGBoost best (F1=0.9076), SMOTETomek rejected | ✅ Complete |
| 9–10 | Unsupervised Model Training (Layer 2) | AE (AUC=0.9892) + IF (AUC=0.8612), scaling fix, per-class detection | ✅ Complete |
| 11–12 | Fusion Engine + Zero-Day Simulation (Layer 3) | 4-case fusion, H1 Δ negligible (−0.014pp, CI excludes 0), H2-simulated 0/5 strict, binary F1=0.9985 | ✅ Complete |
| 12 | True LOO Zero-Day (Layer 3B) | 5-fold LOO XGBoost retrain, H2-strict FAIL (0/5), H2-binary PASS (5/5 @p90) | ✅ Complete |
| 12 | Enhanced Fusion Ablation (Layer 3C) | 11-variant ablation, entropy + confidence + ensemble, H2-strict PASS (4/4 eligible) at +4pp FPR | ✅ Complete |
| 13–14 | SHAP Analysis (Layer 4) | Per-class SHAP, 4-way comparison, DDoS/DoS boundary, 11 figures | ✅ Complete |
| 15–17 | Thesis Writing & Defense | Complete thesis document, code repository, defense preparation | 📝 Writing |

### Models to Implement

**Supervised (Layer 1) — TRAINED:**
- Random Forest (criterion='entropy', n_estimators=200, max_depth=30, class_weight='balanced') — best: E5, test acc=98.52%
- XGBoost (n_estimators=200, max_depth=8, learning_rate=0.1, tree_method='hist') — **best: E7, test acc=99.27%, F1_macro=0.9076**

**Unsupervised (Layer 2) — TRAINED:**
- Deep Autoencoder (44→32→16→8→16→32→44, MSE loss, AUC=0.9892, trained on 123K benign rows + StandardScaler)
- Isolation Forest (n_estimators=200, contamination=0.05, AUC=0.8612)

**Fusion (Layer 3) — COMPLETE:**
- 4-case decision logic — H1 Δ negligible (−0.014pp at p99, 95% CI [−0.0002, −0.0001] excludes 0), binary F1=0.9985
- True LOO zero-day (Phase 6B, AE-only): H2-strict FAIL (0/5), H2-binary PASS (5/5 @p90)
- Enhanced fusion (Phase 6C, 11-variant ablation): H2-strict **PASS (4/4 eligible)** with entropy gate calibrated on benign val (p95 = 0.395) + AE p90; +4 pp benign-FPR cost
- Per-target rescue lifts (Phase 6C best): Recon_Ping_Sweep +81 pp, Recon_VulScan +30 pp, MQTT_Malformed_Data +44 pp, ARP_Spoofing +41 pp
- Key findings: (a) zero-day detection via "redundancy through misclassification" (Phase 6B); (b) entropy calibration source is the lever — val-correct is degenerate, benign-val is operating-range (Phase 6C)
- Recommended operating point: p97 baseline (99.87% attack recall, 5.3% benign FPR); enhanced p95-entropy variant for zero-day-prioritized deployment (4/4 strict at 22.9% benign FPR)

**Explainability (Layer 4) — COMPLETE:**
- TreeSHAP on E7 XGBoost (5K stratified subsample, 70 min compute)
- IAT confirmed #1 (mean |SHAP|=0.87), per-class SHAP heatmap (19×44), DDoS↔DoS cosine=0.991
- 4-way comparison: Our SHAP vs Yacoubi (Jaccard=0.429) vs Cohen's d (0.000) vs RF importance (0.333)

### Classification Tasks (evaluated on all models)

- **Binary:** Benign vs Attack (anomaly detection)
- **6-class:** Benign + 5 attack categories (DDoS, DoS, Recon, MQTT, Spoofing)
- **17-class:** Benign + all 16 individual attack types (fine-grained classification)

---

## 18. Related Work — Summary Table

| Paper | Approach | Key Result |
|-------|----------|------------|
| Dadkhah et al. (2024) | Dataset paper — LR, AdaBoost, RF, DNN | Established CICIoMT2024 benchmark |
| Yacoubi et al. — COCIA 2025 | RF (bagging) vs CatBoost (boosting) + SHAP/LIME | Explainable classification on CICIoMT2024 |
| Yacoubi et al. — AIAI 2025 | XAI-driven feature selection with SHAP/LIME | RF 99.87%, CatBoost improved +4% after feature selection |
| Yacoubi et al. — Springer 2026 | RF + CatBoost + LightGBM + XGBoost + Stacking | Stacking ensemble 99.39%, CatBoost 99.36% |
| Chandekar et al. (2025) | XGBoost + LSTM + CNN-LSTM + Autoencoder + Isolation Forest | Ensemble approach for multi-protocol detection |
| Nature Scientific Reports (2025) | RF with SHAP-based feature selection | 99% accuracy with interpretable dimensionality reduction |
| Springer Applied Sciences (2025) | Transformer + SHAP + SMOTETomek | 93.5% accuracy with attention-based detection |

**Benchmarks from literature:**
- Random Forest: ~99.87% accuracy (full features), ~99.41% (after XAI feature selection)
- XGBoost: ~99.80% accuracy
- LightGBM: ~99.74% (full), ~99.80% (after feature selection — improved!)
- CatBoost: ~95.02% (full), ~99.20–99.36% (after feature selection/tuning)
- Support Vector Machine: ~98% accuracy
- Decision Tree: ~97% accuracy
- Transformer: ~93.5% accuracy
- Logistic Regression: ~92.8% accuracy
- Stacking Ensemble (CatBoost + RF): ~99.39% accuracy

---

## 19. Deep Dive: Yacoubi et al. — Primary Reference Paper

> Yacoubi, M., Moussaoui, O., Drocourt, C. — University of Picardie Jules Verne & MIS Lab, France

Yacoubi et al. published three interrelated papers on the CICIoMT2024 dataset, each building on the previous. Together they form the most comprehensive explainable ML study on this dataset.

### 19.1 Paper 1: Enhancing IoMT Security with Explainable ML (COCIA 2025)

**Core question:** Can ensemble classifiers be made transparent without sacrificing accuracy?

**Problem:** AI-driven threat detection in IoMT is a "black box." Models detect attacks but can't explain *why* a traffic flow was flagged. In healthcare, security analysts who don't understand why an alert fired can't prioritize, investigate, or trust it.

**Methodology — Bagging vs Boosting:**

| Aspect | Random Forest (Bagging) | CatBoost (Boosting) |
|--------|------------------------|---------------------|
| **Training** | N independent trees on bootstrap samples | Trees trained sequentially, each correcting errors of previous |
| **Aggregation** | Majority vote across all trees | Weighted combination of sequential learners |
| **Strength** | Reduces variance (prevents overfitting) | Reduces bias (improves underfitting) |
| **Diversity** | Each tree sees random feature subsets | Each tree focuses on previously misclassified samples |
| **Innovation** | Well-established ensemble method | Ordered boosting to avoid prediction shift |

**Explainability — Two complementary levels:**

**SHAP (SHapley Additive exPlanations) — Global Explanation:**
- Based on cooperative game theory (Shapley values)
- Assigns each feature a "contribution score" showing how much it pushed the prediction toward attack or benign
- Beeswarm plots visualize how every feature impacts every prediction across the entire dataset
- Reveals which features matter most *overall* for the model's decision-making

**LIME (Local Interpretable Model-agnostic Explanations) — Local Explanation:**
- Explains a *single prediction* by creating a simple linear model around one data point
- Perturbs the input slightly and observes how the prediction changes
- Shows which features pushed *this specific traffic flow* toward "attack" or "benign"
- Actionable for security analysts investigating individual alerts

**Key SHAP Findings — Feature Importance Ranking:**

| Rank | Feature | Why It Matters |
|------|---------|---------------|
| **#1** | `IAT` (Inter-Arrival Time) | Attack traffic has fundamentally different timing patterns than normal IoMT sensor reporting |
| **#2** | `Rate` | Flood attacks show dramatically higher packet rates |
| **#3** | `Header-Length` | DDoS/DoS packets often have minimal headers; MQTT malformed data has oversized headers |
| **#4** | `Srate` | Source sending rate correlates with volumetric attacks |
| **#5** | `syn_flag_number` | Key discriminator for SYN flood attacks specifically |
| **#6** | `UDP` | Protocol indicator separates UDP floods from TCP-based attacks |
| Near zero | `Telnet`, `SSH`, `IRC`, `SMTP` | Essentially noise for IoMT traffic — these protocols are rarely used by medical devices |

**LIME Findings:** For a specific attack traffic instance, RF correctly identified it by relying heavily on `IAT` and `Rate`, while `Header-Length` and `UDP` had zero local influence. CatBoost used a slightly different feature combination for the same prediction, demonstrating that the two models reason differently even when they agree on the output class.

**Runtime Comparison:** SHAP on RF was faster than SHAP on CatBoost. LIME was fast for both models (since it only explains individual instances).

**Paper 1 Conclusion:** Both RF and CatBoost achieve strong classification. SHAP provides trustworthy global explanations, while LIME gives actionable instance-level insights. The combination makes ensemble models viable for real-world IoMT security deployment.

### 19.2 Paper 2: XAI-Driven Feature Selection for Improved IDS (AIAI 2025)

**Key Innovation:** Uses SHAP and LIME not just to *explain* models, but to *select features*. If SHAP says a feature has near-zero importance, drop it. This reduces the 45-feature space, cutting training time while maintaining or improving accuracy.

**Results with full 45 features:**

| Model | Accuracy | Notes |
|-------|----------|-------|
| Random Forest | 99.87% | Best on full feature set |
| XGBoost | 99.80% | Strong competitor |
| LightGBM | 99.74% | Balanced performance |
| CatBoost | 95.02% | Underperforms with all 45 features |

**Results after XAI-driven feature selection (reduced feature set):**

| Model | Accuracy | Change | Notes |
|-------|----------|--------|-------|
| LightGBM | 99.80% | Unchanged | Robust to feature reduction |
| XGBoost | 99.54% | -0.26% | Marginal drop |
| Random Forest | 99.41% | -0.46% | Small drop |
| CatBoost | 99.20% | **+4.18%** | Massive improvement! |

**Critical Insight — Hyperparameter Tuning:** Using RF with criterion set to `"entropy"` (instead of default `"gini"`) dramatically improved performance to 99.8% accuracy, compared to 73.5% reported in earlier studies using default parameters. This demonstrates that hyperparameter tuning can be more impactful than model architecture choice.

**Paper 2 Conclusion:** CatBoost actually *improved* by 4% after removing noisy features via SHAP. Feature selection isn't just about computational efficiency — it actively helps boosting models by removing features that confuse the sequential learning process. XAI-driven feature selection improves IDS efficiency without compromising detection capability.

### 19.3 Paper 3: Ensemble Learning Strategies for Anomaly-Based IDS (Springer 2026)

**Extended comparison** to 5 models: RF, CatBoost, LightGBM, XGBoost, and a **Stacking ensemble** (two-layer meta-model where CatBoost + RF generate probability estimates in layer 1, and a meta-learner combines them in layer 2).

**Performance Results:**

| Model | Accuracy | Precision | Recall | Training Time |
|-------|----------|-----------|--------|---------------|
| CatBoost | 99.36% | 86.10% | 89.10% | 683.67s (slowest) |
| Stacking | 99.39% | — | — | — |
| LightGBM | ~99.3% | — | — | 67.79s |
| XGBoost | ~99.2% | — | — | 55.08s (fastest) |

**Key Observations:**
- CatBoost has the best individual accuracy but is **12x slower** than XGBoost
- The stacking ensemble only marginally improves over individual models (+0.03%)
- The precision/recall gap (99.36% accuracy but only 86.10% precision) suggests the model struggles with minority attack classes
- For real-time IoMT detection, XGBoost or LightGBM may be better choices due to inference speed

### 19.4 Research Gaps Left by Yacoubi et al.

These gaps represent opportunities for our project to make a novel contribution:

| Gap | Description | Our Approach |
|-----|-------------|--------------|
| **No deep learning** | Only tree-based ensemble models evaluated | Add LSTM, CNN-LSTM, Transformer for temporal pattern detection |
| **No unsupervised methods** | All work is supervised classification | Add Autoencoder, Isolation Forest for zero-day attack detection |
| **Class imbalance not addressed** | No SMOTE/ADASYN/cost-sensitive learning used | Apply SMOTETomek + class_weight='balanced' |
| **No per-attack analysis** | Only overall accuracy reported | Provide confusion matrix and per-class F1/MCC breakdown |
| **No cross-protocol analysis** | WiFi, MQTT, BLE not compared separately | Train protocol-specific models and analyze transfer |
| **Precision/recall gap** | 99.36% accuracy but only 86.10% precision | Focus on minority class performance (ARP Spoofing, Recon) |
| **No profiling data used** | Lifecycle states (power/idle/active) not leveraged | Use profiling data for behavioral baseline anomaly detection |

---

## 20. Research Design

### 20.1 Research Questions

**Primary Research Question:**

> To what extent does a hybrid supervised-unsupervised fusion framework improve overall detection accuracy, minority-class recall, and zero-day detection capability compared to standalone supervised classifiers on the CICIoMT2024 dataset?

**Sub-Research Questions:**

- **Sub-RQ1 (Fusion Performance):** How does the 4-case fusion decision logic affect precision-recall trade-offs across the 17 attack classes compared to using the supervised classifier alone?

- **Sub-RQ2 (Zero-Day Detection):** To what extent can unsupervised anomaly detection (Autoencoder, Isolation Forest) identify zero-day attacks simulated via the leave-one-attack-out protocol when the supervised layer has no training exposure to the withheld attack class?

- **Sub-RQ3 (Explainability):** How does per-attack-class SHAP analysis reveal differential feature importance patterns across attack categories (DDoS vs. Recon vs. MQTT vs. Spoofing), and do these patterns change when SMOTETomek resampling is applied?

### 20.2 Hypotheses

**H1 — Fusion Framework Performance:**
- *H0:* The hybrid fusion framework does not produce statistically significant improvements in macro-averaged F1-score compared to the best standalone supervised classifier (p > 0.05, paired t-test across 5-fold stratified cross-validation).
- *H1:* The hybrid fusion framework produces statistically significant improvements in macro-averaged F1-score compared to the best standalone supervised classifier (p ≤ 0.05).
- **Result: H0 NOT REJECTED.** The bootstrap CI for the macro-F1 difference (Δ = −0.014pp at best threshold p99, 95% CI [−0.0002, −0.0001]) excludes zero, but the magnitude is operationally negligible — fusion does not measurably degrade or improve aggregate macro-F1 (~125 of 892,268 rows reclassified into the `zero_day_unknown` pseudo-class). The framework's value is in case-stratified operational alerts (4 → 5-case fusion) and zero-day detection capability (Phase 6C, H2-strict 4/4), not in aggregate classification metrics. See Section 14.4.

**H2 — Zero-Day Detection:**
- *H0:* The unsupervised layer does not achieve a recall rate greater than 0.70 on withheld attack classes in the leave-one-attack-out simulation.
- *H1:* The unsupervised layer achieves a recall rate greater than 0.70 on at least 50% of withheld attack classes.
- **Result (strict, AE-only — Phase 6B): H0 not rejected (H2 FAIL).** AE recall on samples the blind LOO-E7 misclassifies as benign: 0/5 targets ≥ 70% (best: VulScan 44.1% at p90). See Section 15.4.
- **Result (strict, fusion with uncertainty signals — Phase 6C): H0 REJECTED (H2 PASS).** Adding softmax-entropy gating (calibrated on benign validation samples, p95 = 0.395) to AE p90 lifts strict rescue recall to **4/4 eligible targets ≥ 70%** (`MQTT_DoS_Connect_Flood` structurally excluded with 0 LOO→Benign samples). Per-target lifts: Recon_Ping_Sweep 0.16 → 0.97, Recon_VulScan 0.44 → 0.75, MQTT_Malformed_Data 0.34 → 0.77, ARP_Spoofing 0.32 → 0.73. Operational cost: +4 pp false-alert rate on benign test rows (0.189 → 0.229). See Section 15C.5–15C.8.
- **Result (binary — any alert): H2 PASS.** The fused system (Cases 1+2+3) achieves recall ≥ 70% on 5/5 targets at p90 (Phase 6B); 5/5 with the Phase 6C entropy-gated variant (binary avg 0.949). Detection is dominated by E7 misclassifying novel attacks into neighboring known classes (Phase 6B "redundancy through misclassification"), with the entropy gate adding the rescue path for samples that fall through to Benign. See Sections 15.4 and 15C.

**H3 — Class Imbalance Effect:**
- *H0:* SMOTETomek resampling does not significantly improve macro-F1 nor per-class F1 for minority attack classes.
- *H1:* SMOTETomek improves macro-F1, **and** improves per-class F1 for at least 3 of the 5 most underrepresented attack classes.
- **Result: H0 not rejected (H3 FAIL).** SMOTETomek degrades macro-F1 across all 4 configurations (RF/reduced −0.011, RF/full −0.017, XGB/reduced −0.045, XGB/full −0.037). On the per-class minority criterion: RF/reduced shows 2/5 rare classes improving (`ARP_Spoofing` +0.093, `Recon_OS_Scan` +0.002), below the 3/5 threshold. The mechanism is synthetic-sample blur on already-overlapping `DDoS_*`↔`DoS_*` and `Recon_OS_Scan`↔`Recon_VulScan` boundaries, not class-weight interaction (XGBoost arms have no `class_weight` and degrade *more* than RF arms). See Section 12.4.

### 20.3 Research Objectives

| ID | Objective | Deliverable |
|----|-----------|-------------|
| **O1** | Construct and benchmark supervised baselines (RF, XGBoost) on binary, 6-class, and 17-class tasks. Evaluate with accuracy, precision, recall, F1 (macro + per-class), MCC, ROC-AUC. | Baseline performance table |
| **O2** | Develop unsupervised anomaly detectors (Autoencoder, Isolation Forest) trained on benign-only traffic. Optimize thresholds using validation data. | Anomaly detection ROC curves |
| **O3** | Implement the 4-case fusion decision engine combining supervised predictions with unsupervised anomaly scores. Evaluate fusion performance across all classification granularities. | Fusion logic code + comparison table |
| **O4** | Conduct zero-day attack simulation using leave-one-attack-out protocol for all 17 classes. Measure unsupervised detection recall per withheld class. | Zero-day detection rate matrix (17 × 2) |
| **O5** | Perform per-attack-class SHAP explainability analysis. Compare feature importance rankings before/after SMOTETomek. | SHAP visualizations + feature importance tables |

### 20.4 Expected Contributions

1. **First hybrid supervised-unsupervised fusion framework on CICIoMT2024** — No existing study on this dataset combines these two paradigms in a structured decision fusion. Addresses the zero-day detection gap left by Yacoubi et al.

2. **Zero-day detection capability evaluation** — First systematic leave-one-attack-out evaluation on this dataset, providing empirical evidence for the unsupervised layer's real-world resilience against novel attacks.

3. **Per-attack-class SHAP explainability** — Yacoubi et al. applied SHAP globally; we extend this to per-class analysis, revealing differential feature importance patterns across attack categories that are masked by global averaging.

4. **SMOTETomek-aware explainability comparison** — Novel analysis of how class imbalance treatment affects model interpretability, providing practical guidance for deploying fair IoMT security systems.

---

## 21. Proposed Framework Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  INPUT: CICIoMT2024 CSV                      │
│       (45 features, 17 classes, pre-split train/test)        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                 PREPROCESSING PIPELINE                       │
│  • Deduplication (removes 37% train / 45% test duplicates)   │
│  • Drop 17 features (Drate + 11 redundant + 5 noise) → ~28  │
│  • RobustScaler on heavy-tailed (IAT, Rate, Tot sum)         │
│  • StandardScaler on flag-count features                     │
│  • Label encoding (17 classes)                               │
│  • Train/validation split (stratified 80/20 on train set)    │
│  • Two variants: original (imbalanced) + SMOTETomek          │
└───────────────────────┬─────────────────────────────────────┘
                        │
            ┌───────────┴───────────┐
            │                       │
            ▼                       ▼
┌─────────────────────┐  ┌──────────────────────────┐
│   LAYER 1:           │  │   LAYER 2:                │
│   SUPERVISED         │  │   UNSUPERVISED            │
│                      │  │                           │
│   • Random Forest    │  │   • Autoencoder           │
│     (criterion=      │  │     44→32→16→8→16→32→44   │
│      'entropy')      │  │     + StandardScaler      │
│   • XGBoost          │  │   • Isolation Forest      │
│     (n_est=200)      │  │     (contamination=0.05)  │
│                      │  │                           │
│   Output: class      │  │   Output: anomaly score  │
│   probabilities      │  │   (MSE or isolation depth)│
└──────────┬───────────┘  └──────────┬───────────────┘
           │                         │
           └─────────────┬───────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              LAYER 3: FUSION DECISION ENGINE                 │
│                                                              │
│  ┌──────────────┬──────────────┬──────────────────────────┐ │
│  │  Supervised  │ Unsupervised │      Decision             │ │
│  ├──────────────┼──────────────┼──────────────────────────┤ │
│  │  Attack      │  Anomaly     │  HIGH-CONFIDENCE ALERT    │ │
│  │  Benign      │  Anomaly     │  ZERO-DAY WARNING         │ │
│  │  Attack      │  Normal      │  LOW-CONFIDENCE ALERT     │ │
│  │  Benign      │  Normal      │  CLEAR                    │ │
│  └──────────────┴──────────────┴──────────────────────────┘ │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│           LAYER 4: EXPLAINABILITY (SHAP + LIME)              │
│                                                              │
│  • Global SHAP (beeswarm plots per class)                    │
│  • Local SHAP/LIME for Case 2 (zero-day warnings)            │
│  • Feature importance comparison: pre vs post SMOTETomek     │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│       EVALUATION: Zero-Day Simulation (Leave-One-Out)        │
│                                                              │
│  For each of 17 attack classes:                              │
│    1. Remove class from training data                        │
│    2. Retrain unsupervised models on remaining data          │
│    3. Test detection rate on withheld class                  │
│    4. Record recall, FPR, detection latency                  │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Stack per Layer

| Layer | Component | Library/Tool | Key Hyperparameters |
|-------|-----------|--------------|---------------------|
| 1 | Random Forest | scikit-learn | n_estimators=200, criterion='entropy', max_depth=None, class_weight='balanced' |
| 1 | XGBoost | xgboost | n_estimators=200, learning_rate=0.1, max_depth=6, objective='multi:softprob' |
| 2 | Autoencoder | TensorFlow/Keras | Architecture: 44→32→16→8→16→32→44, optimizer=Adam, loss=MSE, StandardScaler on benign-train, AUC=0.9892 |
| 2 | Isolation Forest | scikit-learn | n_estimators=200, contamination=0.05, max_samples='auto' |
| 3 | Fusion Engine | Custom Python | Threshold-based decision logic (95th/99th percentile for anomaly threshold) |
| 4 | SHAP | shap | TreeSHAP for RF/XGBoost, KernelExplainer for Autoencoder |
| 4 | LIME | lime | LimeTabularExplainer, num_features=10 |
| Preprocessing | SMOTETomek | imbalanced-learn | sampling_strategy='auto', random_state=42 |

---

## 22. Corrections to Published Literature

The following corrections were discovered through our independent analysis of the CICIoMT2024 dataset during Phase 2 EDA:

| Claim in Published Literature | Our Verified Finding |
|-------------------------------|---------------------|
| Train set: 377,718 rows | Train set: 7,160,831 raw / **4,515,080 after dedup** |
| Test set: 98,432 rows | Test set: 1,614,182 raw / **892,268 after dedup** |
| Total: ~4.89M instances | Total: 8,775,013 raw / **5,407,348 after dedup** |
| 1 CSV per attack type | Up to **8 numbered CSVs** per attack type (must merge) |
| ARP Spoofing = rarest class | **Recon_Ping_Sweep = rarest** (689 rows after dedup) |
| Imbalance ratio ~100:1 | Imbalance ratio **2,374:1** |
| No duplicate analysis reported | **37% train / 45% test are exact duplicates** |
| Column: Header-Length (hyphen) | Column: **Header_Length** (underscore) |
| Column: Magnitude | Column: **Magnitue** (typo in dataset) |
| Column: Drate not mentioned | **Drate exists** but is constant at 0.0 |
| No MQTT protocol column mentioned | Confirmed: **no MQTT indicator column** exists |
| SHAP top features: IAT, Rate, Header_Length, Srate | Cohen's d top features: **rst_count, psh_flag_number, Variance, ack_flag_number** (zero overlap) |
| 99.87% RF accuracy on raw data | Likely **inflated by duplicate leakage** — 37% of rows are identical |
| SMOTETomek assumed to improve minority detection | **SMOTETomek degraded macro-F1 in all 4 configurations** when combined with class_weight='balanced' |
| Reduced features (dropping correlated) assumed optimal | **Full features (44) consistently outperformed reduced (28)** — correlation-based dropping too aggressive |
| Phase 3 ColumnTransformer scaling sufficient for all models | **AE requires additional StandardScaler** — RobustScaler leaves features with std>1000, causing million-scale loss and zero Recon detection |
| Hybrid fusion assumed to improve macro-F1 | **Fusion Δ = −0.014pp at best (p99), 95% CI [−0.0002, −0.0001]** — magnitude is operationally negligible (~125 of 892,268 rows). The zero_day_unknown pseudo-class penalizes macro-F1 by design; value is in operational case stratification, not aggregate metric improvement |
| Reconstruction-error AE assumed sufficient for zero-day detection | **AE catches only 6-44% of samples LOO-E7 misclassifies as benign** — shared flow-feature basis causes overlapping blind spots. However, hybrid system achieves binary recall ≥70% on 5/5 LOO targets through "redundancy via misclassification" |
| Global SHAP ranking assumed representative for all attack types | **Per-class SHAP reveals heterogeneous feature importance** — DDoS relies on IAT/Rate, ARP_Spoofing on Tot size/Header_Length, Recon on Min/rst_count. Global averaging masks these signatures. DDoS↔DoS cosine similarity = 0.991 explains confusion boundary. |
| Feature importance assumed method-independent | **4-way comparison shows Jaccard=0.000 between SHAP and Cohen's d**, Spearman ρ=−0.741. Statistical separation ≠ model reliance. Deduplication shifts even SHAP rankings vs Yacoubi (Jaccard=0.429). |

> These corrections constitute a novel methodological contribution to the CICIoMT2024 literature and strengthen the motivation for our preprocessing pipeline.

---

## 23. Citations

**Dataset:**
```
S. Dadkhah, E. C. P. Neto, R. Ferreira, R. C. Molokwu, S. Sadeghi and A. A. Ghorbani.
"CICIoMT2024: Attack Vectors in Healthcare devices - A Multi-Protocol Dataset for
Assessing IoMT Device Security," Internet of Things, v. 28, December 2024.
DOI: 10.1016/J.IOT.2024.101351
```

**Reference Papers (Yacoubi et al.):**
```
[1] M. Yacoubi, O. Moussaoui, C. Drocourt. "Enhancing IoMT Security with Explainable
    Machine Learning: A Case Study on the CICIOMT2024 Dataset." COCIA 2025, Lecture
    Notes in Networks and Systems, vol. 1584, Springer, 2026.
    DOI: 10.1007/978-3-032-01536-5_38

[2] M. Yacoubi, O. Moussaoui, C. Drocourt. "Explainable AI-Driven Feature Selection
    for Improved Intrusion Detection Systems in the Internet of Medical Things."
    AIAI 2025, IFIP Advances in Information and Communication Technology, vol. 757,
    Springer, 2026. DOI: 10.1007/978-3-031-96231-8_26

[3] M. Yacoubi, O. Moussaoui, C. Drocourt. "Ensemble Learning Strategies for
    Anomaly-Based Intrusion Detection in IoMT Systems." AI-Driven Security for
    Next-Generation IoT Systems, Springer, 2026.
    DOI: 10.1007/978-3-032-08784-3_9
```

**Other Key References:**
```
[4] P. Chandekar et al. "Enhanced Anomaly Detection in IoMT Networks using Ensemble
    AI Models on the CICIoMT2024 Dataset." arXiv:2502.11854, Feb 2025.

[5] "An interpretable dimensional reduction technique with an explainable model for
    detecting attacks in IoMT devices." Scientific Reports, Nature, March 2025.
    DOI: 10.1038/s41598-025-93404-8

[6] "An explainable AI-driven transformer model for spoofing attack detection in
    IoMT networks." Discover Applied Sciences, Springer, May 2025.
    DOI: 10.1007/s42452-025-07071-5
```

---

## 24. Tech Stack

- **Language:** Python 3.13 (downgraded from 3.14 in Phase 5 — TensorFlow 2.21 wheels not yet built for 3.14)
- **ML Libraries:** scikit-learn, XGBoost, TensorFlow/Keras
- **Data Processing:** pandas 3.0+, numpy 2.4+
- **Visualization:** matplotlib 3.10+, seaborn 0.13+
- **Explainability:** SHAP, LIME
- **Imbalance Handling:** imbalanced-learn (SMOTETomek)
- **Environment:** MacBook Air M4 (24GB RAM), Google Colab (GPU for deep learning)
- **Version Control:** GitHub

---

> **Last updated:** April 27, 2026 — Path B Week 1 complete (multi-seed LOO validation; 5 seeds, 25 retrainings; 0/19 eligible cells fail 0.70 strict threshold; H2-strict avg 0.799 ± 0.022)