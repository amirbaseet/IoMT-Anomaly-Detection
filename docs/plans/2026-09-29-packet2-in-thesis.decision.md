# Decision — packet-2 paper results in the MSc thesis

- **Status:** FROZEN
- **Date:** 2026-09-29
- **Decided by:** Doç. Dr. İsmail Bütün (thesis supervisor, paper co-author)
- **Question asked (by Amro, 2026-09-29):** may the packet-2 paper's results be added to the thesis?
- **Verbatim answer:** "kabul edilirse evet ama kabul için iki ay daha var"
  (EN: "if it is accepted, yes, but there are two more months until acceptance")

## Rulings

1. **R1:** The packet-2 paper's results (the random vs unseen-device AUC table, Tablo 4.12, and any sentence citing
   Baseet & Bütün 2026) enter the thesis **only if the paper is accepted**. The paper was submitted 2026-09-29 to
   SecuPriv-IoT 2026 at FICN 2026 (Workshop Paper track, ID 2603375820). The decision is expected in about 2 months
   (≈ late November 2026).
2. **R2 (inference, not a ruling):** until acceptance the paper is **not** cited in the thesis. This follows from R1
   but was not asked separately.

## Not covered by this answer (open)

- **O1:** Does R1 also cover the thesis's **own** CICIoMT2024 measurement, which does not come from the paper? This is
  the finding that 0.9999 of 1,680,177 official-test flows share a source MAC with the same class's training captures
  (`iomt-pcap-experiments/track_a_out/device_overlap_train_test.json`).
  - **O1 closed by the owner (2026-09-29):** no question sent. The measurement uses only thesis data (first measured 2026-09-29, pcap `763a996`) and is not a paper result. It stays in the thesis, and Hoca sees it with the next draft.
- **O2:** Is the workshop double-blind? Should the public code repo `amirbaseet/iot-packet2-measurement` go private
  until the decision?

## Effect on the current manuscript

These commits currently contain paper-derived content that R1 holds back until acceptance:

- manuscript `8de466f`: §4.5 paper paragraph + Tablo 4.12, the limitation-5 sentence, ref #49
- manuscript `2428d0a`: the ref #49 venue
- lit-review `bb9b633`: the DR-8 sentence citing the paper
