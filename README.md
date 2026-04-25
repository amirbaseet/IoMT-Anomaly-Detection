# AI-Based Anomaly Detection in IoMT Networks

## A Hybrid Supervised-Unsupervised Framework for Zero-Day Attack Detection

> **Thesis Title:** A Hybrid Supervised-Unsupervised Framework for Anomaly Detection and Zero-Day Attack Identification in IoMT Networks Using the CICIoMT2024 Dataset  
> **Author:** Amro  
> **Program:** M.Sc. Artificial Intelligence and Machine Learning in Cybersecurity — Sakarya University  
> **Dataset:** CICIoMT2024 (Canadian Institute for Cybersecurity)  
> **Reference Paper:** Yacoubi et al. (2025–2026) — *Enhancing IoMT Security with Explainable Machine Learning*  
> **Status:** Phase 1-3 complete — Preprocessing done, supervised model training next (Phase 4)

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
12. [Project Roadmap](#12-project-roadmap)  
13. [Related Work — Summary Table](#13-related-work--summary-table)  
14. [Deep Dive: Yacoubi et al. — Primary Reference Paper](#14-deep-dive-yacoubi-et-al--primary-reference-paper)  
    - 14.1 [Paper 1: Explainable ML (COCIA 2025)](#141-paper-1-enhancing-iomt-security-with-explainable-ml-cocia-2025)
    - 14.2 [Paper 2: XAI Feature Selection (AIAI 2025)](#142-paper-2-xai-driven-feature-selection-for-improved-ids-aiai-2025)
    - 14.3 [Paper 3: Ensemble Strategies (Springer 2026)](#143-paper-3-ensemble-learning-strategies-for-anomaly-based-ids-springer-2026)
    - 14.4 [Research Gaps & Our Contribution](#144-research-gaps-left-by-yacoubi-et-al)
15. [Research Design](#15-research-design)
16. [Proposed Framework Architecture](#16-proposed-framework-architecture)
17. [Corrections to Published Literature](#17-corrections-to-published-literature)
18. [Citations](#18-citations)  
19. [Tech Stack](#19-tech-stack)

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

## 12. Project Roadmap — 17-Week Plan (Option A: Hybrid Framework)

| Week | Phase | Key Deliverables | Status |
|------|-------|------------------|--------|
| 1–2 | Literature Review & Problem Definition | Literature review, finalized RQs + hypotheses, thesis proposal | ✅ Complete |
| 3–4 | Data Acquisition & EDA | Dataset loaded, 37% duplicates found, 15+ figures, findings.md | ✅ Complete |
| 5–6 | Preprocessing & Imbalance Handling | Feature engineering, SMOTETomek, AE data, zero-day datasets | ✅ Complete |
| 7–8 | Supervised Model Training (Layer 1) | RF + XGBoost (original + resampled), baseline performance tables | 🔄 Next |
| 9–10 | Unsupervised Model Training (Layer 2) | Autoencoder + Isolation Forest on benign data, anomaly thresholds | ⏳ Planned |
| 11–12 | Fusion Engine + Zero-Day Simulation (Layer 3) | 4-case fusion logic, leave-one-attack-out results | ⏳ Planned |
| 13–14 | SHAP Analysis + Confusion Matrix (Layer 4) | Per-class SHAP plots, feature importance comparisons, confusion matrices | ⏳ Planned |
| 15 | Profiling Integration (Stretch Goal) | Delta features from profiling data (if time permits) | ⏳ Optional |
| 16–17 | Documentation & Defense | Complete thesis document, code repository, defense preparation | ⏳ Planned |

### Models to Implement

**Supervised (Layer 1):**
- Random Forest (criterion='entropy', n_estimators=200, class_weight='balanced')
- XGBoost (n_estimators=200, learning_rate=0.1, max_depth=6)

**Unsupervised (Layer 2):**
- Deep Autoencoder (architecture: ~28→20→12→6→12→20→~28, MSE loss, trained on 123K benign rows from train split)
- Isolation Forest (n_estimators=200, contamination=0.05)

**Fusion (Layer 3):**
- 4-case decision logic (custom Python implementation)

**Explainability (Layer 4):**
- SHAP (TreeSHAP for ensembles, KernelExplainer for Autoencoder)
- LIME (LimeTabularExplainer for individual predictions)

### Classification Tasks (evaluated on all models)

- **Binary:** Benign vs Attack (anomaly detection)
- **6-class:** Benign + 5 attack categories (DDoS, DoS, Recon, MQTT, Spoofing)
- **17-class:** Benign + all 16 individual attack types (fine-grained classification)

---

## 13. Related Work — Summary Table

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

## 14. Deep Dive: Yacoubi et al. — Primary Reference Paper

> Yacoubi, M., Moussaoui, O., Drocourt, C. — University of Picardie Jules Verne & MIS Lab, France

Yacoubi et al. published three interrelated papers on the CICIoMT2024 dataset, each building on the previous. Together they form the most comprehensive explainable ML study on this dataset.

### 14.1 Paper 1: Enhancing IoMT Security with Explainable ML (COCIA 2025)

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

### 14.2 Paper 2: XAI-Driven Feature Selection for Improved IDS (AIAI 2025)

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

### 14.3 Paper 3: Ensemble Learning Strategies for Anomaly-Based IDS (Springer 2026)

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

### 14.4 Research Gaps Left by Yacoubi et al.

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

## 15. Research Design

### 15.1 Research Questions

**Primary Research Question:**

> To what extent does a hybrid supervised-unsupervised fusion framework improve overall detection accuracy, minority-class recall, and zero-day detection capability compared to standalone supervised classifiers on the CICIoMT2024 dataset?

**Sub-Research Questions:**

- **Sub-RQ1 (Fusion Performance):** How does the 4-case fusion decision logic affect precision-recall trade-offs across the 17 attack classes compared to using the supervised classifier alone?

- **Sub-RQ2 (Zero-Day Detection):** To what extent can unsupervised anomaly detection (Autoencoder, Isolation Forest) identify zero-day attacks simulated via the leave-one-attack-out protocol when the supervised layer has no training exposure to the withheld attack class?

- **Sub-RQ3 (Explainability):** How does per-attack-class SHAP analysis reveal differential feature importance patterns across attack categories (DDoS vs. Recon vs. MQTT vs. Spoofing), and do these patterns change when SMOTETomek resampling is applied?

### 15.2 Hypotheses

**H1 — Fusion Framework Performance:**
- *H0:* The hybrid fusion framework does not produce statistically significant improvements in macro-averaged F1-score compared to the best standalone supervised classifier (p > 0.05, paired t-test across 5-fold stratified cross-validation).
- *H1:* The hybrid fusion framework produces statistically significant improvements in macro-averaged F1-score compared to the best standalone supervised classifier (p ≤ 0.05).

**H2 — Zero-Day Detection:**
- *H0:* The unsupervised layer does not achieve a recall rate greater than 0.70 on withheld attack classes in the leave-one-attack-out simulation.
- *H1:* The unsupervised layer achieves a recall rate greater than 0.70 on at least 50% of withheld attack classes.

**H3 — Class Imbalance Effect:**
- *H0:* SMOTETomek resampling does not significantly improve per-class F1-score for minority attack classes.
- *H1:* SMOTETomek significantly improves per-class F1-score for at least 3 of the 5 most underrepresented attack classes.

### 15.3 Research Objectives

| ID | Objective | Deliverable |
|----|-----------|-------------|
| **O1** | Construct and benchmark supervised baselines (RF, XGBoost) on binary, 6-class, and 17-class tasks. Evaluate with accuracy, precision, recall, F1 (macro + per-class), MCC, ROC-AUC. | Baseline performance table |
| **O2** | Develop unsupervised anomaly detectors (Autoencoder, Isolation Forest) trained on benign-only traffic. Optimize thresholds using validation data. | Anomaly detection ROC curves |
| **O3** | Implement the 4-case fusion decision engine combining supervised predictions with unsupervised anomaly scores. Evaluate fusion performance across all classification granularities. | Fusion logic code + comparison table |
| **O4** | Conduct zero-day attack simulation using leave-one-attack-out protocol for all 17 classes. Measure unsupervised detection recall per withheld class. | Zero-day detection rate matrix (17 × 2) |
| **O5** | Perform per-attack-class SHAP explainability analysis. Compare feature importance rankings before/after SMOTETomek. | SHAP visualizations + feature importance tables |

### 15.4 Expected Contributions

1. **First hybrid supervised-unsupervised fusion framework on CICIoMT2024** — No existing study on this dataset combines these two paradigms in a structured decision fusion. Addresses the zero-day detection gap left by Yacoubi et al.

2. **Zero-day detection capability evaluation** — First systematic leave-one-attack-out evaluation on this dataset, providing empirical evidence for the unsupervised layer's real-world resilience against novel attacks.

3. **Per-attack-class SHAP explainability** — Yacoubi et al. applied SHAP globally; we extend this to per-class analysis, revealing differential feature importance patterns across attack categories that are masked by global averaging.

4. **SMOTETomek-aware explainability comparison** — Novel analysis of how class imbalance treatment affects model interpretability, providing practical guidance for deploying fair IoMT security systems.

---

## 16. Proposed Framework Architecture

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
│   • Random Forest    │  │   • Autoencoder          │
│     (criterion=      │  │     (~28→20→12→6→12→20→28)│
│      'entropy')      │  │     Trained on benign    │
│   • XGBoost          │  │   • Isolation Forest     │
│     (n_est=200)      │  │     (contamination=0.05) │
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
| 2 | Autoencoder | TensorFlow/Keras | Architecture: ~28→20→12→6→12→20→~28, optimizer=Adam, loss=MSE, epochs=50 |
| 2 | Isolation Forest | scikit-learn | n_estimators=200, contamination=0.05, max_samples='auto' |
| 3 | Fusion Engine | Custom Python | Threshold-based decision logic (95th/99th percentile for anomaly threshold) |
| 4 | SHAP | shap | TreeSHAP for RF/XGBoost, KernelExplainer for Autoencoder |
| 4 | LIME | lime | LimeTabularExplainer, num_features=10 |
| Preprocessing | SMOTETomek | imbalanced-learn | sampling_strategy='auto', random_state=42 |

---

## 17. Corrections to Published Literature

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

> These corrections constitute a novel methodological contribution to the CICIoMT2024 literature and strengthen the motivation for our preprocessing pipeline.

---

## 18. Citations

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

## 19. Tech Stack

- **Language:** Python 3.14+
- **ML Libraries:** scikit-learn, XGBoost, TensorFlow/Keras
- **Data Processing:** pandas 3.0+, numpy 2.4+
- **Visualization:** matplotlib 3.10+, seaborn 0.13+
- **Explainability:** SHAP, LIME
- **Imbalance Handling:** imbalanced-learn (SMOTETomek)
- **Environment:** MacBook Air M4 (24GB RAM), Google Colab (GPU for deep learning)
- **Version Control:** GitHub

---