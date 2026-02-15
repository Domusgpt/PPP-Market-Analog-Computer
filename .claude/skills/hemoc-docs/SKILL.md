---
name: hemoc-docs
description: Documentation reconciliation and audit for Phase F of the Recovery Plan. Use for checking doc accuracy, detecting contradictions, or generating the canonical status document.
argument-hint: "[audit|contradictions|generate-status|cross-reference <claim>]"
---

# HEMOC Documentation Reconciliation

Support Phase F of the Recovery Plan: make documentation accurate and navigable.
The exit criterion is "new contributor finds current truth in under 5 minutes."

## Repository Paths

```
PPP_ROOT=/home/user/PPP-Market-Analog-Computer
HEMOC_ROOT=/home/user/HEMOC-Stain-Glass-Flower
```

## Modes

### `audit` — Check All Referenced Docs

Verify every doc in the CLAUDE.md reading list (12 total):

**PPP docs (always on current branch):**

| # | File | Check |
|---|------|-------|
| 1 | `WHAT_THIS_DOES.md` | Exists, describes physical optical device |
| 2 | `CODING_AGENT_PROMPT.md` | Exists, has 24-cell math, Trinity Decomposition |
| 3 | `docs/HEMOC_PPP_CROSS_REPO_DIGEST.md` | Exists, has 6 phases, 7 gaps |
| 10 | `DEV_TRACK.md` | Exists, session count matches (38+) |
| 11 | `docs/refactor-plan.md` | Exists, Phase 0-5 structure |

**HEMOC docs (branch-dependent):**

| # | File | Branch | Check |
|---|------|--------|-------|
| 4 | `docs/CRITICAL_FINDING_AND_ROADMAP.md` | main | Centroid-zero discovery |
| 5 | `docs/PROJECT_STATUS.md` | main | 15 experiments listed |
| 6 | `FINAL_ANALYSIS_REPORT.md` | main | V4 Plastic benchmark |
| 7 | `docs/TESTING_REVIEW_AND_DEV_TRACK.md` | main | Testing audit |
| 8 | `docs/MUSICGEOMETRY_GAP_ANALYSIS.md` | main | 5 gaps identified |
| 9 | `docs/benchmark_contract.md` | hemoc-visual-system-init-* | Schema governance |
| 12 | `docs/STRUCTURAL_TRAINING_VISION.md` | main | Music curriculum |

For each doc:
1. Verify it exists on the expected branch
2. Check file size (non-empty)
3. Grep for key claims mentioned in CLAUDE.md
4. Flag any obviously stale references (dates, experiment numbers)

### `contradictions` — Detect All Doc Contradictions

Scan docs for factual claims and cross-reference them. Check beyond the 3 known contradictions.

**Known contradictions** (documented in CLAUDE.md "Known Doc Contradictions"):

| Claim | Wrong Doc | Right Source | Status |
|-------|-----------|-------------|--------|
| Domain-diverse "done, 0.73" | MUSICGEOMETRY_GAP_ANALYSIS | TESTING_REVIEW: "NOT RUN" | Known |
| `src/` paths exist | PROJECT_STATUS module inventory | Actual repo: no `src/` | Known |
| `src/core/plastic_encoder.py` | FINAL_ANALYSIS_REPORT | Actual: `demos/dual_decoder.py` | Known |

**Additional claims to check:**
- Experiment result numbers: does PROJECT_STATUS match the JSON artifacts in `results/`?
- Test counts: does TESTING_REVIEW match actual test file count?
- Architecture claims: does HEMOC_ARCHITECTURE_DETAILED match the visual system code?
- Phase status claims: do any docs claim phases are done that aren't?
- File path references: grep all `src/` references across all docs

Output:
```
| Contradiction | Doc A | Doc B | Resolution |
|--------------|-------|-------|------------|
| Known: domain-diverse | GAP_ANALYSIS | TESTING_REVIEW | Trust TESTING_REVIEW |
| NEW: [if any found] | ... | ... | ... |
```

### `generate-status` — Create STATUS_SINGLE_SOURCE.md

Generate the canonical status document using the Ontology Blueprint three-table structure
(`docs/ONTOLOGY_BLUEPRINT_ICE_ECT_DRAFT.md` on hemoc-visual-system-init branch).

**Claims Table** — What the project asserts:

| ID | Claim | Status | Evidence | Confidence |
|----|-------|--------|----------|------------|
| C1 | Encoding is injective (0 collisions / 1000) | Proven | E1 | High |
| C2 | CNN decodes 5/6 angles from moire patterns | Proven | E2 | High |
| C3 | HybridEncoder required (pure hypercube fails) | Proven | E3 | High |
| C4 | Audio pipeline achieves 0.916 correlation | Proven | E4 | High |
| C5 | More training data helps (8K→15K: 0.894→0.916) | Proven | E4, E5 | High |
| C6 | Cross-domain zero-shot transfer fails | Proven | E6 | High |
| C7 | V4 Plastic Encoder is optimal | Proven | E7 | High |
| C8 | Centroid-zero kills MLP decoders | Proven | E8 | High |
| C9 | Phillips matrix has golden-ratio block structure | Proven | E9 | High |
| C10 | Domain-diverse training improves transfer | Untested | — | — |
| C11 | Golden ratio outperforms alternatives | Untested | — | — |
| C12 | HEMOC outperforms direct-feature baselines | Untested | — | — |

**Evidence Table** — What experiments produced:

| ID | Experiment | Result | Seed | Date | Reference |
|----|-----------|--------|------|------|-----------|
| E1 | Exp 1-6 injectivity | 0 collisions / 1000 | 42 | 2026-01-27 | PROJECT_STATUS |
| E2 | Exp 7 CNN decoder | 0.657 avg, 5/6 passing | 42 | 2026-01-28 | PROJECT_STATUS |
| E3 | Exp 8 Pure V3 + CNN | 0.090 avg, 0/6 | 42 | 2026-01-28 | PROJECT_STATUS |
| E4 | Exp 13 Scaled CNN 15K | 0.916 avg, 6/6 | 42 | 2026-02-01 | PROJECT_STATUS |
| E5 | Exp 12 Audio pipeline 8K | 0.894 avg, 6/6 | 42 | 2026-01-31 | PROJECT_STATUS |
| E6 | Exp 14-15 Cross-domain | -0.027 avg, 0/6 | 42 | 2026-02-02 | PROJECT_STATUS |
| E7 | V4 Plastic benchmark | 0.6261 cosine sim | 42 | 2026-02-03 | FINAL_ANALYSIS |
| E8 | Centroid-zero analysis | Tesseract centroid = origin | — | 2026-01-27 | CRITICAL_FINDING |
| E9 | Phillips matrix verification | U_R = φ·U_L confirmed | — | 2026-02-10 | CODING_AGENT_PROMPT |

**Reproducibility Table** — How to verify claims:

| ID | Command | Expected | Last Verified |
|----|---------|----------|---------------|
| R1 | `pytest tests/unit/test_encoder.py` | PASS | [check date] |
| R2 | `python demos/dual_decoder.py` | Generates 64x64x3 pattern | [check date] |
| R3 | `python scripts/prove_system.py` | V4 validation passes | [check date] |

Write this to `docs/STATUS_SINGLE_SOURCE.md` in the HEMOC repo.
Include a header noting it supersedes conflicting claims in other docs.

### `cross-reference <claim>` — Trace a Specific Claim

For the given claim (e.g., "domain-diverse training"), find every doc that mentions it:
```bash
grep -ril "domain.diverse" docs/ FINAL_ANALYSIS_REPORT.md README.md 2>/dev/null
```

For each mention, extract the exact claim text and compare across docs.
Report whether all mentions are consistent or contradictory.

## Output Format

```
## Documentation Audit Report — [DATE]

### Doc Existence & Freshness
| # | Document | Exists | Branch | Size | Last Modified |
|---|----------|--------|--------|------|---------------|
| 1 | WHAT_THIS_DOES.md | YES | PPP main | 4.2KB | 2026-02-10 |
| ... | ... | ... | ... | ... | ... |

### Contradictions
| # | Status | Description |
|---|--------|-------------|
| 1 | Known | Domain-diverse claim (GAP_ANALYSIS vs TESTING_REVIEW) |
| 2 | Known | src/ paths (PROJECT_STATUS vs actual repo) |
| 3 | Known | plastic_encoder.py location (FINAL_ANALYSIS vs actual) |
| 4+ | NEW | [if any found] |

### Phase F Readiness
- STATUS_SINGLE_SOURCE.md: [exists / not yet created]
- Claims table: [N claims documented]
- Evidence table: [N evidence entries]
- Reproducibility table: [N reproducibility entries]

### Recommendation
[What to do next for Phase F completion]
```
