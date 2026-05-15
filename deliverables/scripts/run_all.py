"""
Run all deliverable-production scripts in order, fail-fast on any non-zero exit.

WHAT WE FOUND
  Six scripts run end-to-end in well under a minute on the project machine
  (MacBook Air M4, 24 GB RAM, no GPU). Two reproducibility tripwires
  (entropy_benign_p95 and §15D anchor p93) assert bit-exactly. All 11
  ablation-table variants reproduce verbatim from on-disk CSVs.

WHY WE CHOSE THIS APPROACH
  Alternatives considered:
    - One monolithic script (loses phase boundaries; tracebacks point at
      the wrong place)
    - Makefile orchestration (requires Make; harder to embed in the
      notebook)
    - pytest harness (would conflate "this metric matches" with "the test
      framework is set up correctly")
  Decision criterion: linear 00 → 01 → ... → 06 dependency chain with
    fail-fast halting and a final summary table makes the artifact most
    useful to a human reader.
  Tradeoff accepted: stdout-grep summary collection is slightly less
    rigorous than a structured-JSON aggregation; mitigated by the parseable
    emit() format from _common.py (one metric per line).
  Evidence: deliverables/scripts/*.py, deliverables/run_all_summary.txt.

Run from project root:
  venv/bin/python -m deliverables.scripts.run_all
or:
  venv/bin/python deliverables/scripts/run_all.py
"""
from __future__ import annotations

import importlib
import subprocess
import sys
import time
from pathlib import Path

from deliverables.scripts._common import PROJECT_ROOT, banner

SCRIPTS = [
    ("Phase 0", "deliverables.scripts.00_env_check", "00_env_check"),
    ("Phase 2–3", "deliverables.scripts.01_data_preprocessing", "01_data_preprocessing"),
    ("Phase 4", "deliverables.scripts.02_supervised_phase4", "02_supervised_phase4"),
    ("Phase 5", "deliverables.scripts.03_unsupervised_phase5", "03_unsupervised_phase5"),
    ("Phase 6/6B/6C", "deliverables.scripts.04_fusion_phase6", "04_fusion_phase6"),
    ("Phase 7", "deliverables.scripts.05_explainability_phase7", "05_explainability_phase7"),
    ("Path B", "deliverables.scripts.06_path_b_hardening", "06_path_b_hardening"),
    ("Phase F", "deliverables.scripts.07_generate_figures", "07_generate_figures"),
]


def _runpy(module_path: str) -> tuple[int, str]:
    """Run a script as a subprocess so each call gets a clean import state.
    Returns (exit_code, captured_stdout). Stderr is streamed to console."""
    cmd = [sys.executable, "-m", module_path]
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=sys.stderr,  # forward to console
        text=True,
        bufsize=1,
    )
    chunks: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        chunks.append(line)
    rc = proc.wait()
    return rc, "".join(chunks)


def main() -> int:
    banner("run_all — orchestrator for the 7-script production deliverable pass")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"Python:       {sys.executable}  ({sys.version.split()[0]})")
    print()

    all_stdout: list[str] = []
    t_total = time.perf_counter()
    failures: list[tuple[str, int]] = []
    for label, module_path, _slug in SCRIPTS:
        print()
        banner(f"running {module_path}   ({label})")
        t0 = time.perf_counter()
        rc, out = _runpy(module_path)
        elapsed = time.perf_counter() - t0
        print(f"  ({label}: exit={rc}, {elapsed:.2f}s)")
        all_stdout.append(f"\n\n===== {module_path} ({label}) — exit {rc}, {elapsed:.2f}s =====\n")
        all_stdout.append(out)
        if rc != 0:
            failures.append((module_path, rc))
            break  # fail-fast

    elapsed_total = time.perf_counter() - t_total
    print()
    banner(f"run_all complete in {elapsed_total:.2f}s")

    # Write summary to the exempt path (.claude/plans/deliverables/run_all_summary.txt)
    summary_path = PROJECT_ROOT / ".claude" / "plans" / "deliverables" / "run_all_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as fp:
        fp.write(f"run_all_summary.txt — generated {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fp.write(f"PROJECT_ROOT: {PROJECT_ROOT}\n")
        fp.write(f"Python: {sys.version.split()[0]}\n")
        fp.write(f"Total elapsed: {elapsed_total:.2f}s\n\n")
        fp.write("================================================================\n")
        fp.write("  Per-script stdout (verbatim)\n")
        fp.write("================================================================\n")
        fp.write("".join(all_stdout))

        # Collect headline-number lines for the final summary section
        fp.write("\n\n================================================================\n")
        fp.write("  Headline-numbers summary (grep '[Phase' or '[Path B')\n")
        fp.write("================================================================\n")
        for line in "".join(all_stdout).splitlines():
            if line.startswith("[Phase") or line.startswith("[Path B"):
                fp.write(line + "\n")

    print(f"Summary written to: {summary_path}")

    if failures:
        print()
        print(f"!!! FAILURES: {failures}")
        return failures[0][1]

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
