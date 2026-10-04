# Agent cycle — IoMT-Project

How work moves through this repo. Roles map to the global agent roster (`~/.claude/agents/`) +
this repo's skills. Nothing auto-commits; a human merges.

1. **Brain (plan)** — `/plan-feature` (or the `planner` agent) turns a request into a
   house-format plan in `docs/plans/`. Grounds in the code; asks only what code can't answer.
2. **Builder (implement)** — a worker session executes ONE plan. Kickoff via `/worker-kickoff`.
   TDD is N/A (no test suite) — the builder must keep `run_all`'s tripwires green.
3. **Verifier (independent)** — `/verify-worker` re-runs the gates from scratch, never trusting
   the builder's report: `run_all` reproduces the tripwires + `scripts/verify_*.py` pass.
4. **Reviewer (rules)** — the `invariant-guardian` agent audits the diff against every DN-/INV-
   rule in `CLAUDE.md`; `pipeline-verifier` smoke-runs any changed pipeline stage and diffs
   output schema / row counts against the previous artifact.
5. **Human merge** — you review, approve, and merge.

Cross-cutting: `/fact-check` after any doc; `/log-arc` records a finished arc in `CLAUDE.md` §4;
`/experiment-log` appends a run to `RUNS.md`.
