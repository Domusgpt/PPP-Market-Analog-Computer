# Research Archaeology: Chronological Timeline of the Phillips Matrix Discovery

**Compiled:** February 8, 2026
**Method:** Git history analysis across PPP-Market-Analog-Computer and merged repositories
**Status:** Phase 1 (git evidence only; conversation history and Google Docs pending)

---

## Repository Inventory

### Directly Analyzed (this repo)
| Repo | Role | First Commit | Commits |
|------|------|-------------|---------|
| PPP-Market-Analog-Computer | Main spine, synergized system | Oct 11, 2025 | 40+ |

### Merged Into _SYNERGIZED_SYSTEM (Feb 6, 2026)
| Repo | Package Name | Branch Ref | Role |
|------|-------------|------------|------|
| PPP Market Analog Computer | - | codex/2026-02-06/... | Frontend telemetry spine |
| HEMOC-SGF (via Rotorary) | - | claude/analyze-repos-1fUCE | Python physics engine |
| HEMAC Holographic | @hemac | claude/integrate-chronomorphic-e8-engines-P9F3Q | Geometric Algebra, E8 viz |
| Chronomorphic Polytopal Engine (CPE) | @clear-seas/cpe | claude/integrate-cpe-engine-CdsEN | Topology, 600-cell, E8 projection |
| HEMOC-Rotorary | - | claude/analyze-repos-1fUCE | Pipeline wrapper |

### Referenced But Not In Repo (CRITICAL — provenance gap)
| Repo | Evidence | Key File |
|------|----------|----------|
| **e8codec-hyperrendering** | Referenced in E8Renderer.ts | `utils/e8.ts` |
| **ppp-info-site** | Referenced in E8Renderer.ts | `lib/topology/E8H4Folding.ts` |
| HEMOC-Stain-Glass-Flower | Referenced in docs | WebGL moiré encoder |

---

## Chronological Timeline

### Phase 0: Foundation (Oct 2025 – Jan 2026)

**Oct 11, 2025** — `9610a6a` — PPP Information & Marketing Site initialized
- First commit by Paul Phillips (Domusgpt)
- PPP concept exists as a market data visualization system

**Oct 31, 2025** — `5d479b3` — Hypercube demo shipped
- First 4D visualization code
- Entry point into polytopal thinking

**Nov 1–2, 2025** — Various docs commits
- Documentation of controls, metrics, use-cases
- PPP positioned as "internal tool"

### Phase 1: Musical Geometry & 24-Cell (Jan 2026)

**Jan 10, 2026** — `a432788` — Paul uploads two critical documents:
1. **MusicGeometryDomain-Design.md** — Maps 24-cell to musical structures
   - 24 vertices → 24 major/minor keys
   - 3 inscribed 16-cells → thesis/antithesis/synthesis triad
   - Package: `@clear-seas/cpe` (Chronomorphic Polytopal Engine)
   - THIS is the 24-cell decomposition that Paul describes in his email to Moxness
2. **Polytope Decompositions for Musical Mapping.txt** — Framework for polytope-based music AI

> *Paul to Moxness: "I find use in splitting the icositetrachoron into three 16 cells allows me to map semantic concepts as thesis & antithesis with their overlaps forming a synthesis third 16 cell."*

**Jan 10, 2026** — `1528c8f` — "Add pusher to Tester.txt" (infrastructure)

### Phase 2: Phase-Locked Stereoscopy (Jan 24–27, 2026)

**Jan 24, 2026** — `28ea05a` — Phase-locked stereoscopy system implemented
- Core PPP visualization engine
- TypeScript: StereoscopicFeed, TimeBinder, DataPrism

**Jan 24–26** — Multiple commits refining phase-lock system
- Tests, PhaseLockEngine.js, GitHub Pages deployment
- Full PPP data pipeline: RawApiTick → MarketTick → SyncedFrame → StereoscopicFrame

**Jan 27** — PR #1 merged: Phase-locked market data

### Phase 3: The Grand Unification (Feb 6, 2026)

**Feb 6, 2026 12:42** — `8d8e504` — Adapter metrics reset (codex branch)

**Feb 6, 2026 21:33** — `682ddcc` — **GRAND UNIFICATION**
- Created `_SYNERGIZED_SYSTEM/` by merging 5 repositories
- **Cell600.ts** enters the repo (from CPE: `@clear-seas/cpe`)
  - Contains the E8→H4 projection with EXACT Phillips constants:
    ```typescript
    (a[0] + PHI * a[4]) / 2     // coefficient: PHI/2 = c = 0.809
    (a[0] - PHI_INV * a[4]) / 2  // coefficient: PHI_INV/2 = b = 0.309
    ```
  - **These are the three entry constants {1/2, (φ-1)/2, φ/2} of the Phillips matrix**
  - But in diagonal form: each output dim depends on only 2 input dims
- **E8Renderer.ts** enters (from HEMAC Holographic)
  - Contains a 4×8 "Moxness matrix" with entries {0, 1, PHI, PHI_INV}
  - Different from both the Phillips matrix and the actual Moxness C600
  - References external files: `e8codec-hyperrendering/utils/e8.ts` and
    `ppp-info-site/lib/topology/E8H4Folding.ts`

### Phase 4: E8→H4 Projection & Matrix Discovery (Feb 7, 2026)

**Feb 7, 2026 00:33** — `3e96207` — Comprehensive coding agent prompt added
- 24-cell decomposition architecture for coding agents
- System design using polytopal concepts

**Feb 7, 2026 05:32** — `9c54f49` — Base system fixed, 143 tests passing
- Python backend operational with geometry, topology, physics

**Feb 7, 2026 12:30** — `ebc5655` — Import/compile fixes, system README

**Feb 7, 2026 16:30** — `ea6c1c2` — **E8→H4 PROJECTION CREATED**
- Created `e8_projection.py` with Baez 4×8 matrix ONLY
- Trilatic decomposition fixed
- Feature extraction upgraded
- **The Phillips matrix does NOT yet exist at this point**

**Feb 7, 2026 17:33** — `a628f70` — **THE PHILLIPS MATRIX APPEARS**
⚡ **62 minutes after the Baez-only version**
- Phillips 8×8 matrix added alongside Baez
- 50 verification tests, all passing
- Entry constants: a=1/2, b=(φ-1)/2, c=φ/2
- **Source: User conversation input** (not derived from any file in the repo)
- This is where the dense 8×8 matrix with {±a, ±b, ±c} entries materializes

> **PROVENANCE GAP**: The matrix entered via conversation with Claude.
> The entry constants {1/2, (φ-1)/2, φ/2} match Cell600.ts projection
> coefficients exactly, but the sign pattern and dense structure are new.
> The external files `E8H4Folding.ts` and `e8.ts` may contain intermediate
> forms but cannot be accessed from this repo.

**Feb 7, 2026 18:03** — `9042bc5` — Deep exploration of Phillips kernel
- Kernel characterization, collision analysis, eigenstructure
- Discovery: rank 4, single collision vector, eigenvalue 5
- 33 new tests added (now 281 total)

**Feb 7, 2026 18:46** — `f2c2fef` — Letter to Moxness drafted
- (Initially with incorrect attribution — later corrected)

**Feb 7, 2026 22:44** — `b837059` — Paper draft v0.1 completed
- 7 theorems with proofs
- Full academic paper structure

### Phase 5: Attribution Correction (Feb 8, 2026)

**Feb 8, 2026 03:47** — `fa4228c` — Definitive C600 vs Phillips comparison
- Proved matrices are completely different objects

**Feb 8, 2026 06:31** — `0687c51` — Attribution corrected everywhere
- Paper, letter, code comments all fixed

**Feb 8, 2026 13:11** — `d71e43f` — Moxness U-analysis.pdf uploaded
- Moxness provided his actual corrected matrix

**Feb 8, 2026 18:20** — `576ab02` — Comparison with corrected Moxness matrix
- Confirmed: completely different matrices
- Orthogonal row spaces, different projection geometry

---

## The Provenance Chain

### What We Can Prove From Git Evidence

```
Oct 2025    PPP concept → hypercube visualization
                ↓
Jan 2026    24-cell musical mapping (thesis/antithesis/synthesis)
                ↓
            Phase-locked stereoscopy (PPP data pipeline)
                ↓
Feb 6       Grand Unification merges 5 repos including:
            • Cell600.ts (CPE) — diagonal E8→H4 with constants {1/2, PHI/2, PHI_INV/2}
            • E8Renderer.ts (HEMAC) — 4×8 Moxness matrix {0, 1, PHI, PHI_INV}
                ↓
Feb 7 16:30 e8_projection.py created with Baez 4×8 only
                ↓
Feb 7 17:33 ⚡ Phillips 8×8 matrix appears (via conversation)
            Constants {1/2, (φ-1)/2, φ/2} = Cell600.ts coefficients
            Dense 8×8 structure = NEW (not in any merged file)
```

### The Key Question: How Did the Dense Sign Pattern Arise?

The three constants {a, b, c} = {1/2, (φ-1)/2, φ/2} come from the
Cell600.ts diagonal projection. But the SIGN PATTERN of the Phillips
matrix (which entries get +a, +b, -a, -b in each row) is not derivable
from any file in this repository. It likely emerged from:

1. **A conversation with Claude or Gemini** where Paul discussed
   generalizing the diagonal Cell600 projection to a dense 8×8 matrix
2. **The external file `E8H4Folding.ts`** in the ppp-info-site repo
   (cannot be accessed from here)
3. **A manual construction** by Paul while implementing PPP's
   stereoscopic rendering

### What We Still Need

To complete the provenance:

1. **`ppp-info-site` repo** — Contains `lib/topology/E8H4Folding.ts`
   which likely has the Phillips matrix or an intermediate form
2. **`e8codec-hyperrendering` repo** — Contains `utils/e8.ts`
3. **Claude/Gemini conversation history** — The matrix was introduced
   "via conversation input" at commit a628f70
4. **Google Docs** — Paul may have working notes

---

## Related Research Context

### Moxness's Work (cited in paper)
- 2014: viXra:1411.0130 — C600 matrix, E8 visualization
- 2018: viXra:1808.0107 — Fourfold H4 600-cell mapping
- 2019: Unimodular variant (det=1)
- 2023: arXiv:2311.11918 — Hadamard isomorphism
- 2026: U-analysis.pdf (personal communication) — corrected matrix

### Paul's Research Path (from emails to Moxness)
1. Building PPP for geometric market data visualization
2. Quaternion rotations for machine cognition
3. 24-cell → thesis/antithesis/synthesis (musical geometry)
4. 24-cell inscribed in 600-cell (5 copies)
5. 600-cell connection to E8 (via Moxness's work)
6. Implemented E8→H4 projection
7. Discovered the Phillips matrix spectral properties

---

## Phase 2 Plan: Expanding the Evidence Base

### Getting Google Docs Access
Paul needs to:
1. Export relevant Google Docs as PDF/Markdown
2. OR set up Google Docs MCP server for Claude Code
3. Key documents to find: working notes on matrix construction,
   E8/H4 exploration notes, PPP design documents

### Getting Conversation History
1. **Claude conversations**: Export from claude.ai settings
2. **Gemini conversations**: Export from Google Takeout
3. **ChatGPT conversations** (if any): Export from settings
4. Key conversations: any discussing E8→H4 projection matrix
   construction, golden ratio entry constants, 8×8 dense matrices

### Accessing Other Repos
The repos `ppp-info-site` and `e8codec-hyperrendering` contain
critical provenance evidence. Options:
1. Paul grants repo access to this environment
2. Paul exports the relevant files manually
3. Paul shares the git history of those repos
