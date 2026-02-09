# Research Archaeology: Complete Chronological Timeline of Paul Phillips's Mathematical Research

**Compiled:** February 8, 2026 (v2.0 — comprehensive multi-repo analysis)
**Method:** Git history analysis across 7+ GitHub repositories, ALL branches (55+ in ppp-info-site alone)
**Status:** Phase 2 complete — all repos and branches analyzed; conversation history and Google Docs pending

---

## Executive Summary

The Phillips matrix (8×8, dense, rank 4, entries {1/2, (φ-1)/2, φ/2}) was **first created on January 13, 2026** in `ppp-info-site/3-body-proof` as `E8H4Folding.ts`. The matrix was part of a 3-body problem proof framework, initially mislabeled as the "Moxness matrix." The attribution was corrected on **January 17, 2026** (MATRIX_PROVENANCE.md). The same matrix was independently reimplemented in Python on **February 7, 2026**, with the same attribution error repeated and corrected again on **February 8, 2026**.

The matrix emerged from a 4-month research arc: PPP market visualization (Oct 2025) → 4D polytope rendering (Oct 2025) → geometric cognition engine (Jan 2026) → 24-cell musical mapping (Jan 10) → HDC Voyage AI experiments (Jan 10) → 3-body proof with E8→H4 projection (Jan 13) → Phillips matrix discovery.

---

## Repository Inventory

### All Repositories Analyzed

| Repo | Branches | Key Content | Role |
|------|----------|-------------|------|
| **ppp-info-site** | 55+ | E8H4Folding.ts, MusicGeometryDomain, HDCEncoder, 3-body proof | **Primary** — contains matrix birth |
| **PPP-Market-Analog-Computer** | 80+ | _SYNERGIZED_SYSTEM, e8_projection.py, paper finalization | Main spine, Python reimplementation |
| **Chronomorphic-Polytopal-Engine (CPE)** | Multiple | Cell600.ts, topology modules | 600-cell, diagonal projection constants |
| **HEMAC / hemac-holographic** | Multiple | E8Renderer.ts, GeometricAlgebra | E8 visualization, 4×8 "Moxness" matrix |
| **e8codec-hyperrendering** | 21 | 4D text/torus rendering (utils/math.ts) | Not E8-related despite name |
| **HEMOC-Stain-Glass-Flower** | 18 | WebGL moiré encoder | Visual pipeline |
| **Rotorary** | Multiple | Python physics engine | Physics backend |

### Critical Finding: e8codec-hyperrendering
Despite being referenced in E8Renderer.ts as containing E8 code, the `e8codec-hyperrendering` repo's `utils/math.ts` contains only 4D text rendering (font maps, space-time manifold, torus knot) — NO E8 projection code. The file `utils/e8.ts` referenced in E8Renderer.ts does not exist in any branch.

---

## Complete Chronological Timeline

### Phase 0: Foundation (Oct 2025)

**Oct 11, 2025** — PPP Information & Marketing Site initialized
- First commit by Paul Phillips (Domusgpt)
- PPP concept as market data visualization system

**Oct 31, 2025** — Hypercube demo shipped
- First 4D visualization code
- Entry point into polytopal thinking

**Nov 1–2, 2025** — Documentation of controls, metrics, use-cases

---

### Phase 1: Hash-Chained Telemetry & Geometric Encoding (Jan 3–8, 2026)

**Jan 3–4, 2026** — 12 codex branches in ppp-info-site:
`2026-01-03/20-16-40integrate-hash-chained-telemetry-with-geometric-encodingcodex` through
`2026-01-04/04-28-56...`
- `SpinorTopologyWeave.js` — first topology-aware rendering
- `samples/telemetry/topology.json` — geometric telemetry data
- First experiments with geometric encoding of data

**Jan 7–8, 2026** — 7 more codex branches refining telemetry integration

---

### Phase 2: CPE Architecture & HDCEncoder (Jan 9, 2026)

**ppp-info-site** branches: `codex/2026-01-09/add-concept-note-for-ambiguity-metrics`, `update-theoretical-foundation-docs`

Key commits on `3-body-proof` and `geometric-cognition-engine-0sffy` branches:
- `6e8cf1d` — **CausalReasoningEngine** (CPE Phase 3)
- `95c5d27` — **Epistaorthognition** validation module (CPE Phase 4)
- `00fd83d` — **HDCEncoder** neural-geometric bridge (CPE Phase 5)
- `602c0a0` — **CPERendererBridge** WebGL integration (CPE Phase 6)
- `5928e5b` — **HDCEncoder v2.0** with API embeddings, attention, domain archetypes
- `1ae5988` — HDCEncoder adds Gemini and Anthropic/Voyage as embedding providers
- `5c26fad` — HDCEncoder deep dive analysis
- `Lattice24.ts` first appears — 24-cell implementation

---

### Phase 3: HDC Voyage Music Experiments & 24-Cell Musical Mapping (Jan 10, 2026)

**THIS IS THE CONCEPTUAL FOUNDATION OF THE PHILLIPS MATRIX**

**Jan 10, 2026** — On `codex/2026-01-12/design-and-implement-ppp-for-cognitive-modeling` branch:

**`e7d2a0f`** — **HDCEncoder E2E Test with Voyage AI**
- Full pipeline: Text → Voyage embedding (1024D) → Johnson-Lindenstrauss projection → 4D force vector
- **24 Concept Archetypes mapped to 24-Cell vertices:**
  ```
  existence, negation, causation, time, space, quantity, quality, relation,
  action, passion, identity, difference, whole, part, necessity, possibility,
  actuality, substance, attribute, mode, truth, falsity, good, evil
  ```
- Cosine similarity in 4D for archetype matching
- Rate-limited Voyage API calls (3 RPM free tier)

**`fea6134`** — **MusicGeometryDomain for CPE calibration**
- `docs/MusicGeometryDomain-Design.md` — comprehensive 10-page design document
- `lib/domains/MusicGeometryDomain.ts` — 800+ line implementation
- **Core mapping: 24 vertices of 24-cell → 24 major/minor keys**
  ```typescript
  // 24-Cell vertices (±1, ±1, 0, 0) and permutations = 24 vertices
  // Major keys: C, G, D, A, E, B, F#, Db, Ab, Eb, Bb, F
  // Minor keys: Am, Em, Bm, F#m, C#m, G#m, D#m, Bbm, Fm, Cm, Gm, Dm
  ```
- Circle of fifths as rotation in 4D
- Chord geometry: major triads form D3 symmetry, diminished 7th forms tetrahedron
- 4th dimension = time (temporal notes)
- Integration with HDCEncoder for semantic-musical bridge

> **Why the 24-Cell?** (from design doc):
> 1. 24 Vertices = 24 major/minor keys
> 2. 24 Octahedral Cells = 24 diatonic modes
> 3. 96 Edges = Interval relationships
> 4. Self-Dual = Major/minor duality
> 5. 4 Dimensions = Pitch, Time, Timbre, Dynamics

**`ee587a5`** — **SemanticHarmonicBridge**
- Integrates HDCEncoder embeddings with MusicGeometryDomain
- 8 emotion archetypes with fallback 4D vectors:
  ```typescript
  joy: [0.8, -0.3, 0.7, 0.2], sadness: [-0.6, 0.3, -0.4, -0.3],
  tension: [0.1, 0.9, 0.5, 0.1], peace: [0.4, -0.8, -0.5, -0.2], ...
  ```
- Chord suggestion engine based on semantic input

**`c61c18f`** — **GeminiAudioOracle** for multimodal hypothesis validation
- Gemini 3 Pro audio analysis for CPE calibration
- Structured prompts for tension rating, emotion classification, interval analysis

**`c61c18f`** — **ExperimentalFramework.md** — formal research design
- RQ1: "Is Western tonal music structurally isomorphic to the 24-Cell?"
- Three studies: tension correlation, Bach chorale geodesics, Pythagorean comma
- Power analysis, statistical methods, calibration protocol
- Budget: ~$55 (Gemini API + Voyage)

**Also Jan 10–11:**
- 8 codex branches for `evaluate-current-build-and-architecture`
- Topology modules: `Hypercube8.ts`, `Lattice24.ts`, `Simplex5.ts`, `Lattice24Provider.ts`
- `metamorphic-topology-spec.md` — topology system design

---

### Phase 4: Rust/WebGPU Engine & Cognitive Modeling (Jan 11–12, 2026)

**Jan 11** — `d2a51a2` — Geometric Cognition Simulation Engine in Rust/WebGPU
- `e9b1169` — GitHub Pages deployment

**Jan 12** — 6 codex branches: `design-and-implement-ppp-for-cognitive-modeling` variants
- Multiple Rust/WASM compilation iterations
- WebGL render debugging

---

### ⚡ Phase 5: 3-Body Proof & PHILLIPS MATRIX BIRTH (Jan 13, 2026)

**Jan 13, 2026 18:59 EST** — `0599df6` — **"New files for 3 body simulation and proof"**

**The Phillips matrix enters the codebase for the first time** as `E8H4Folding.ts`:

```typescript
const a = 0.5;                           // 1/2
const b = 0.5 * PHI_INV;                 // 1/(2φ) = (φ-1)/2
const c = 0.5 * PHI;                     // φ/2

// Row 0-3: H4L projection
// [a,  a,  a,  a,  b,  b, -b, -b]
// [a,  a, -a, -a,  b, -b,  b, -b]
// [a, -a,  a, -a,  b, -b, -b,  b]
// [a, -a, -a,  a,  b,  b, -b, -b]
// Row 4-7: H4R projection
// [c,  c,  c,  c, -a, -a,  a,  a]
// [c,  c, -c, -c, -a,  a, -a,  a]
// [c, -c,  c, -c, -a,  a,  a, -a]
// [c, -c, -c,  c, -a, -a,  a,  a]
```

**Initially mislabeled as "Moxness 8×8 rotation matrix"** — but the coefficients {0.5, 0.309, 0.809} are completely different from Moxness's {0, ±1, ±φ, ±1/φ, ±φ²}.

Files created in this commit:
- `E8H4Folding.ts` — The Phillips matrix
- `TrinityDecomposition.ts` — Standard Model particle mapping to 24-cell, with **"Phillips Synthesis"** concept in docstring:
  > "The Phillips Synthesis emerges: combining any two 16-cell projections geometrically reveals the third, encoding QCD's color confinement."
- `Lattice600.ts` — 600-cell implementation
- `e8-three-body-demo.html` — browser demo
- `Chronomorphic Polytopal Engine – Provisional Patent Application Outline.docx`
- `Geometric Physics: 3-Body to Subatomic.pdf`
- `Sensor-Driven 4D Polytopal Projection Engine Architecture.pdf`

---

### Phase 6: Matrix Fix & Python Proof (Jan 14, 2026)

**Jan 14, 00:30 UTC** — `788d4c0` — **"Fix E8→H4 proof simulation: Moxness matrix and Phillips Synthesis"**

Critical commit message:
```
CRITICAL FIXES:
- Moxness matrix now orthogonal with det=1, rank=8 (was det=0, rank=7)
- Phillips Synthesis tests all 512 (α,β,γ) combinations (was testing only 1)
- Found 32 valid color-neutral triads with best balance=0.0

ALL TESTS NOW PASSING:
- E8 Lattice: 240 roots ✓
- Moxness Matrix: det=1.0, rank=8 ✓
- 600-Cell: 120 vertices ✓
- Trinity: 24=8+8+8 ✓
- Energy Conservation: 99.999976% ✓
- Phillips Synthesis: 32 valid triads ✓
```

- `e8_three_body_proof_v2.py` — SVD-corrected orthogonal version
- **"Phillips Synthesis" named in commit message**
- `e8_h4_analysis.txt` — analysis output: det=-16, rank=8, 24 vertices at φ scale

**Jan 14, 04:01–05:10** — Multiple commits reorganizing topology modules
- `d9f96a6` — "Restore proper E8/H4 framework from 3-body-proof branch"
- `1e92c0a` — "Move 3-body modules to lib/topology, use nested 24-cell trialectic"

---

### Phase 7: Integration & Musical Trialectic (Jan 15–16, 2026)

**Jan 15** — `34608cc`/`666263a` — Restore original files, integrate into lib/topology/
- `5863f80` (Jan 16 02:49) — **"Add Phillips musical trialectic system to TrinityDecomposition"**
- `57d5d92` (Jan 16 02:09) — "Add trialectic/ folder with organized Phillips original research"

**Jan 16** — Deep mathematical verification:
- `d97ddb5` (15:19) — Moxness matrix verification script
- `18e8ed6` (15:51) — **E8H4Folding_Orthonormal.ts** — parallel implementation for comparison:
  - Normalized coefficients: a = 0.5/√(3-φ), b = 0.5*(φ-1)/√(3-φ), c = 0.5*φ/√(φ+2), d = 0.5/√(φ+2)
  - Also includes Coxeter projection matrix using cos/sin(kπ/30) angles
- `0836f79` (16:31) — "Deep analysis verifying φ-relationships are geometric"
- `624142f` (16:48) — Comprehensive research report

**Jan 16, 19:40** — `ecc98a8` — **arXiv paper written: "Golden Ratio Coupling in the E₈ → H₄ Folding Matrix"**
- LaTeX source ready for arXiv submission (math-ph)
- Already contains Theorems 1–3:
  - Row norms: √(3-φ) and √(φ+2)
  - √5 Identity: √(3-φ)·√(φ+2) = √5
  - Golden Coupling: ⟨Row₀, Row₄⟩ = φ − 1/φ = 1
- Author: Paul Joseph Phillips (sole author, set in `688874f`)

**Jan 16, 19:46–22:04** — Red team analysis, comprehensive improvements, tautology analysis

---

### ⚡ Phase 8: FIRST Attribution Correction (Jan 17, 2026)

**Jan 17** — Multiple mathematical investigations:
- `db6ee45` (01:09) — Fix three verified errors in arXiv paper
- `8be7929` (12:30 EST) — **"Research Guidance: E8 to H4 Folding Matrices.pdf"** uploaded
- `681e65e` (14:32) — Independent verification notes for non-peer-reviewed citations
- `9538e05` (17:31) — Extended paper with circularity defense
- `3a029be` (18:07) — φ-family classification theorem
- `c90f026` (20:08) — "Track A: Complete - Clifford algebra framework, icosahedral construction"

**Jan 17, 23:37 UTC** — `8d9e409` — **"Document PPP matrix provenance - clarify it is NOT the Moxness matrix"**

Created `docs/MATRIX_PROVENANCE.md`:
```
Key findings:
- PPP "Moxness matrix" uses different coefficients {0.5, (φ-1)/2, φ/2}
- Actual Moxness matrix uses {0, ±1, φ, 1/φ, φ²} with sparse structure
- PPP matrix is dense Hadamard-like, NOT symmetric, det=0
- Moxness matrix is sparse, symmetric, has unimodular (det=1) version

The PPP matrix has valid interesting properties (√5 coupling, rank-7)
but should not be called "Moxness matrix" - it's a separate construction.
```

---

### Phase 9: Empirical Testing — PPP Matrix vs Moxness (Jan 18, 2026)

**Jan 18, 15:55 UTC** — `398d918` — **"Test E8→H4 projection: PPP matrix FAILS, actual Moxness WORKS"**

Definitive empirical comparison (from MATRIX_PROVENANCE.md):

| Property | Actual Moxness | PPP/Phillips Matrix |
|----------|---------------|---------------------|
| Unique 4D points | 240 | 226 |
| Norm shells | 2 (ratio = φ) | 11 |
| 600-cell match | 240/240 = **100%** | 36/226 = **16%** |
| Valid E8→H4? | **YES** | **NO** |

> "The PPP matrix has interesting algebraic properties (√5-coupling, rank-7) but these are unrelated to E8→H4 geometry."

Recommended terminology: **"PPP projection matrix"** or **"φ-coupled Hadamard matrix"**

---

### Phase 10: Harmonic Alpha & Market Prediction (Jan 22–24, 2026)

**Jan 22** — 3 codex branches: `refactor-repo-for-4d-geometric-cognition`
- `cra-pom-v2/src/core/v3/geometry/cell600.ts` — alternative Cell600 with musical significance annotations

**Jan 23** — Harmonic Alpha Market Larynx system (on `harmonic-alpha-implementation-zLKGc`):
- `60ec503` — Implement Harmonic Alpha
- `b9546fe` — Rust CLI prediction tool
- `ba8eaa6` — Automated API monitoring
- `1baab22` — S&P 500 index harmonic research with Yahoo Finance data
- `ee0f445` — **Blind historical prediction test — HONEST RESULTS**

**Jan 24** — Stereoscopic model, geometric decision agent framework

---

### Phase 11: Phase-Locked Market Data (Jan 24–27, 2026)

**PPP-Market-Analog-Computer** repo:
- Jan 24 — Phase-locked stereoscopy implemented (StereoscopicFeed, TimeBinder, DataPrism)
- Jan 24–26 — Tests, PhaseLockEngine.js, GitHub Pages
- Jan 27 — PR #1 merged

---

### Phase 12: Grand Unification (Feb 6, 2026)

**PPP-Market-Analog-Computer** — `682ddcc`:
- Created `_SYNERGIZED_SYSTEM/` merging 5 repositories
- **Cell600.ts** (from CPE) enters with diagonal projection constants matching Phillips
- **E8Renderer.ts** (from HEMAC) enters with 4×8 matrix referencing `ppp-info-site/lib/topology/E8H4Folding.ts`

---

### Phase 13: Python Reimplementation (Feb 7, 2026)

**Feb 7, 16:30** — `ea6c1c2` — e8_projection.py with Baez 4×8 matrix ONLY

**Feb 7, 17:33** — `a628f70` — **Phillips matrix re-enters** (Python form)
- Same matrix reconstructed via conversation with Claude
- **Same attribution error**: labeled as related to Moxness
- Entry constants {1/2, (φ-1)/2, φ/2} — **IDENTICAL to Jan 13 E8H4Folding.ts**

**Feb 7, 18:03** — Deep kernel/eigenstructure analysis
- Now correctly finds: rank 4, 14 collisions (226 unique), 21 shells
- Eigenvalues {0⁴, λ₁, 5², λ₂}
- U_R = φ · U_L (pure scaling)

**Feb 7, 22:44** — Paper draft v0.1 (WRONG attribution again)

---

### Phase 14: SECOND Attribution Correction (Feb 8, 2026)

- `fa4228c` (03:47) — Definitive C600 vs Phillips comparison
- `0687c51` (06:31) — Attribution corrected in paper, letter, code
- `d71e43f` (13:11) — Moxness U-analysis.pdf uploaded (personal communication)
- `576ab02` (18:20) — Comparison with corrected Moxness matrix from PDF
  - Stacked row-space rank = 8 (completely orthogonal subspaces)
  - Confirmed: fundamentally different mathematical objects

---

## The Complete Provenance Chain

```
Oct 2025    PPP concept → hypercube visualization
                ↓
Jan 3-8     Hash-chained telemetry with geometric encoding
                ↓
Jan 9       CPE Phases 3-6: CausalReasoning, HDCEncoder, Epistaorthognition
            Lattice24.ts first appears
                ↓
Jan 10      ⚡ MusicGeometryDomain: 24-cell → 24 musical keys
            ⚡ HDCEncoder E2E with Voyage AI: 24 archetypes → 24-cell
            ⚡ SemanticHarmonicBridge: neural + musical + geometric
            ⚡ ExperimentalFramework: formal research design
                ↓
Jan 11-12   Rust/WebGPU engine, cognitive modeling architecture
                ↓
Jan 13      ⚡⚡⚡ E8H4Folding.ts created: PHILLIPS MATRIX BORN
            TrinityDecomposition.ts: "Phillips Synthesis" named
            3-body proof framework, Lattice600.ts
                ↓
Jan 14      Matrix fixed (SVD correction), Python proof v2
            All tests passing including Phillips Synthesis
                ↓
Jan 16      E8H4Folding_Orthonormal.ts for comparison
            arXiv paper written (Theorems 1-3: Row Norms, √5, Golden Coupling)
            Author: Paul Joseph Phillips
                ↓
Jan 17      FIRST ATTRIBUTION CORRECTION
            MATRIX_PROVENANCE.md: "NOT the Moxness matrix"
                ↓
Jan 18      Empirical test: PPP matrix FAILS as E8→H4 projection
            Actual Moxness: 100% match; Phillips: 16% match
                ↓
Jan 22-24   Harmonic Alpha, market prediction, stereoscopic models
                ↓
Jan 24-27   Phase-locked stereoscopy in PPP-Market-Analog-Computer
                ↓
Feb 6       Grand Unification: 5 repos merged into _SYNERGIZED_SYSTEM
                ↓
Feb 7       Phillips matrix RE-IMPLEMENTED in Python (same error repeated)
            Deeper analysis: rank 4, eigenvalue 5, 14 collisions
                ↓
Feb 8       SECOND ATTRIBUTION CORRECTION
            Moxness provides U-analysis.pdf (corrected matrix)
            Definitive comparison: orthogonal row spaces
```

---

## Key Discovery: The Same Error Twice

The Phillips matrix was mislabeled as "Moxness matrix" TWICE:

1. **January 13–17, 2026** — In E8H4Folding.ts (ppp-info-site)
   - Corrected Jan 17 in MATRIX_PROVENANCE.md
   - Empirically disproved Jan 18

2. **February 7–8, 2026** — In e8_projection.py (PPP-Market-Analog-Computer)
   - Same matrix reconstructed from memory/conversation
   - Same error repeated: labeled as Moxness-derived
   - Corrected Feb 8 after direct correspondence with Moxness

This pattern suggests the matrix construction is deeply associated with Moxness's work in Paul's mind (he was reading Moxness's papers while building it), even though the mathematical object is entirely different.

---

## The Rank Discrepancy: 7 (January) vs 4 (February)

The January analysis found rank 7; the February analysis found rank 4. Investigation:

- **January** (MATRIX_PROVENANCE.md): "Singular: det = 0, rank = 7. Null space: [0,0,0,0,1,1,1,1]ᵀ"
- **February** (e8_projection.py): Rank 4 confirmed, 4-dimensional kernel

The January analysis appears to have a bug (likely floating-point tolerance issue in the rank computation). The correct rank is **4**, as proven in the February analysis with exact symbolic verification:
- The left block U_L has rank 4 but the right block satisfies U_R = φ·U_L
- This means the full matrix has rank = rank(U_L) = 4, not 7

Similarly, January found "11 norm shells" while February found "21 shells" — likely different quantization precision.

---

## The Construction Path

Based on the git evidence, the Phillips matrix was constructed through this intellectual path:

1. **PPP market data** → 4D visualization need → polytope rendering
2. **24-cell** chosen for 24 keys (musical mapping) and thesis/antithesis/synthesis
3. **600-cell** as container (5 × 24-cell) → E8 connection via Moxness's work
4. **Moxness's papers** studied for E8→H4 projection methods
5. **Dense 8×8 matrix** constructed using golden ratio constants:
   - Constants {a, b, c} = {1/2, (φ-1)/2, φ/2} form a geometric progression with ratio φ
   - Hadamard-like sign pattern applied to create orthogonal-ish blocks
   - Left block uses {±a, ±b}, Right block uses {±c, ±a}
   - **This is NOT the Moxness matrix** — it's Paul's own construction
6. **Spectral properties discovered** (√5 coupling, golden ratio norms, rank 4, eigenvalue 5)

### What Makes It Original

The Phillips matrix differs from ALL known E8→H4 projections:
- **Moxness C600**: Sparse, symmetric, rank 8, entries {0, ±1, ±φ, ±1/φ, ±φ²}
- **Baez/Coxeter**: Sparse 4×8, uses cos/sin angles
- **Phillips**: Dense, non-symmetric, rank 4, entries from {±1/2, ±(φ-1)/2, ±φ/2}

Its unique property U_R = φ·U_L means the 8×8 matrix is really a 4×8 matrix with a built-in golden scaling between left and right H4 copies.

---

## Files of Primary Importance

### In ppp-info-site (Jan 2026)

| File | Branch | Date | Content |
|------|--------|------|---------|
| `E8H4Folding.ts` | 3-body-proof | Jan 13 | **Phillips matrix (FIRST APPEARANCE)** |
| `E8H4Folding_Orthonormal.ts` | run-simulation-module | Jan 16 | Normalized version + Coxeter comparison |
| `TrinityDecomposition.ts` | 3-body-proof | Jan 13 | Phillips Synthesis, Standard Model mapping |
| `Lattice600.ts` | 3-body-proof | Jan 13 | 600-cell implementation |
| `MusicGeometryDomain-Design.md` | cognitive-modeling | Jan 10 | 24-cell musical mapping design |
| `MusicGeometryDomain.ts` | cognitive-modeling | Jan 10 | 800+ line implementation |
| `SemanticHarmonicBridge.ts` | cognitive-modeling | Jan 10 | HDC + music + geometry bridge |
| `ExperimentalFramework.md` | cognitive-modeling | Jan 10 | Formal research design |
| `MATRIX_PROVENANCE.md` | harmonic-alpha | Jan 17-18 | Attribution correction + empirical test |
| `arxiv_paper_phi_coupled_matrix.md/.tex` | run-simulation-module | Jan 16 | First academic paper |
| `e8_three_body_proof_v2.py` | harmonic-alpha | Jan 14 | Python proof simulation |
| `test-e2e-voyage.ts` | cognitive-modeling | Jan 10 | HDCEncoder + Voyage AI test |
| `Research Guidance- E8 to H4 Folding Matrices.pdf` | run-simulation-module | Jan 17 | Research context |

### In PPP-Market-Analog-Computer (Feb 2026)

| File | Branch | Date | Content |
|------|--------|------|---------|
| `e8_projection.py` | claude/design-coding-agent-prompt | Feb 7 | Phillips matrix (Python reimplementation) |
| `Cell600.ts` | _SYNERGIZED_SYSTEM | Feb 6 | Diagonal projection with Phillips constants |
| `E8Renderer.ts` | _SYNERGIZED_SYSTEM | Feb 6 | 4×8 matrix, references E8H4Folding.ts |
| `PAPER_DRAFT_v0.1.md` | this branch | Feb 7-8 | Academic paper v0.2 |
| `compare_moxness_pdf_vs_phillips.py` | this branch | Feb 8 | Definitive comparison |

---

## Remaining Evidence to Collect

### Conversation History (CRITICAL)
The matrix construction likely happened in a Claude or Gemini conversation around Jan 12-13, 2026. Paul needs to:
1. **Claude**: claude.ai/settings → Export Data
2. **Gemini**: takeout.google.com → Gemini Apps
3. Search exports for: "E8", "H4", "folding matrix", "golden ratio", "projection"

### Google Docs
Paul may have working notes with matrix construction details.
Integration method: Google Docs MCP server or manual export.

### The Key Unanswered Question
The constants {a, b, c} have a clear origin (golden ratio geometric progression). But how was the specific SIGN PATTERN chosen? The Hadamard-like pattern in each 4×4 block is systematic but not the standard Walsh-Hadamard matrix. Was it:
- Derived from quaternionic multiplication rules?
- Inspired by the Kronecker product structure?
- Generated by a conversation with AI?
- Hand-crafted for orthogonality properties?

The conversation history from Jan 12-13 likely answers this.

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Repositories analyzed | 7+ |
| Total branches examined | 150+ |
| Time span | Oct 11, 2025 – Feb 8, 2026 (120 days) |
| Phillips matrix first appearance | Jan 13, 2026 (`0599df6`) |
| Phillips matrix Python reimplementation | Feb 7, 2026 (`a628f70`) |
| Days between first and second version | 25 |
| Attribution corrections | 2 (Jan 17 and Feb 8) |
| Total tests (current) | 281 pass, 0 fail |
| Papers written | 2 (Jan 16 arXiv draft, Feb 7 paper v0.2) |
