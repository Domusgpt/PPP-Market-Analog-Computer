# Cross-Repo Research Digest: HEMOC + PPP

**Date**: 2026-02-15
**Scope**: All recently active HEMOC branches + PPP main
**Branches read**: main, hemoc-visual-system-init-4592349023335104422, claude/project-digest, claude/restore-linear-path-decoding-EELHW, claude/review-testing-progress-F6a3C
**Purpose**: Digest the general research, identify gaps, suggest continued focus

---

## 1) The Research Arc (What Actually Happened, Chronologically)

### Phase 1: Origin (Jan 24-27) — `claude/atom-of-thoughts` → `claude/review-testing-progress`

HEMOC started as a physics simulation of an optical kirigami moire encoder. The original `OpticalKirigamiEncoder` in `main.py` used hexagonal lattice mechanics, tristable cells, Talbot resonance, and reservoir computing (ESN). 14 tests existed, 13 passed.

**Key finding — The Centroid-Zero Problem** (Jan 27, `docs/CRITICAL_FINDING_AND_ROADMAP.md`):
- Tesseract vertices sum to (0,0,0,0). Rotation is linear → sum preserved → centroid always at origin
- Aggregate statistics (mean, std) destroy information that IS present in individual vertex positions
- This is WHY all MLP decoders failed (experiments 1-4)
- This is a genuine mathematical insight that redirected the whole project

### Phase 2: Decoder Breakthrough (Jan 27 - Feb 2) — `claude/review-testing-progress`

15 experiments ran, documented in `docs/PROJECT_STATUS.md`:

| Exp | What | Result | Lesson |
|-----|------|--------|--------|
| 1-4 | MLP decoders | 0.001-0.15 | MLPs can't learn spatial structure from moire |
| 5 | Dual decoder (Ridge+MLP) | 0.405 (3/6) | Hybrid encoder helped |
| 7 | CNN decoder | 0.657 (5/6) | **CNN breakthrough** — spatial decoder works |
| 8 | Pure V3 + CNN | 0.090 (0/6) | Hypercube-only encoding fails; hybrid needed |
| 9-11 | ViT attempts | 0.14-0.35 | ViT consistently underperforms CNN on this task |
| 12 | Audio pipeline | 0.894 (6/6) | Audio features map naturally to moire |
| 13 | Scaled CNN (15K) | **0.916 (6/6)** | More data helps; this is the peak result |
| 14-15 | Cross-domain | 0.077 / -0.027 | **Zero-shot transfer fails** |

**What was proven**: Encoding IS injective (0 collisions). CNN DOES decode (0.916). Audio features map naturally. Cross-domain transfer FAILS without domain-specific training.

### Phase 3: Architecture Convergence (Feb 2-3) — `claude/project-digest`

V4 Plastic Encoder identified as the winner via head-to-head benchmark:
- V4 Plastic: 0.6261 cosine sim (WINNER)
- V3.1 600-cell: 0.4558
- Hybrid: 0.4168
- Trinity: 0.0774

**Why V4 won**: Plastic Ratio (ρ≈1.32, Pisot number) scaling creates frequency components that remain distinct across scales, which the CNN can "read" as depth vs orientation.

The "Digital Moire" insight: aliasing at 64x64 is deterministic and continuous. The pixel grid acts as the second diffraction grating. The interference is not error — it IS the computational mechanism.

### Phase 4: Visual System + Cross-Repo Integration (Feb 6-11) — `claude/restore-linear-path-decoding`

Major pivot: move from numpy-generated patterns to **real WebGL shader rendering** using VIB3 SDK components. The Visual System Plan (`docs/HEMOC_ARCHITECTURE_DETAILED.md`) specifies:

- Phillips Matrix ported to TypeScript for CPU-side E8→H4 projection
- 2-4-2 column weight mapping: EXPANDED (dims 0,4 → rotation, phase), STABLE (dims 1,2,5,6 → per-petal freq/tilt), CONTRACTED (dims 3,7 → sharpness, transmission)
- Fragment shader runs 4-petal moire with explicit YW signal isolation at freq=3
- Performance target: 60+ FPS, <16ms total frame time
- Training data via `readPixels()` capture from real GPU-rendered patterns

This is architecturally sound. The shader spec is concrete (full GLSL code in the architecture doc). VIB3's WebGL backend with state caching handles the performance side.

### Phase 5: MusicGeometry Integration (Jan 31 - Feb 11)

The MusicGeometry gap analysis (`docs/MUSICGEOMETRY_GAP_ANALYSIS.md`) is one of the most honest and useful docs in the project. It identifies exactly where HEMOC falls short of the MusicGeometryDomain framework:

- **Gap 1 (Circle of Fifths)**: CLOSED — implemented `freq_to_fifths_position()`
- **Gap 2 (Chords collapse to scalars)**: PARTIALLY CLOSED — chord quality detection added (major/minor/dim/aug) but full chord geometry still collapsed
- **Gap 3 (Temporal trajectories)**: OPEN — still frame-by-frame, streaming modules exist but disconnected
- **Gap 4 (Key awareness)**: OPEN — no tonal center detection
- **Gap 5 (Tesseract vs 24-cell)**: RESOLVED — CPE paper clarifies this is an intentional hierarchy, not a mistake

The Trinity Decomposition from the CPE paper is the key theoretical bridge: 3 petals = 3 inscribed 16-cells (Alpha/Beta/Gamma). This means the existing 3-petal HEMOC architecture ALREADY embodies the 24-cell structure implicitly. Making it explicit is Phase E work, not a redesign.

### Phase 6: Recovery & Systematization (Feb 11-14) — `hemoc-visual-system-init-4592349023335104422`

The Recovery Plan was created to convert fragmented research into a unified, test-gated system:
- Phase A: Benchmark contract (DONE — `docs/benchmark_contract.md`, `scripts/validate_results_schema.py`)
- Phase B: Protect working pipeline with tests
- Phase C: Operationalize domain-diverse training
- Phase D: Baselines + ablations
- Phase E: Unify Python + Visual System via shared schema
- Phase F: Documentation reconciliation

Malformed `numerical_results_20260125_182240.json` was repaired from the paired methodology report (deterministic reconstruction, not inference). Recovery audit documented. Ontology blueprint drafted for publication.

---

## 2) PPP System — What It Actually Is

The PPP repo (`PPP-Market-Analog-Computer`) is NOT just a concept doc. It's a **working runtime** with 38 development sessions, built through systematic iteration:

**Core Pipeline**:
```
Data Sources → Adapter Layer → TimeBinder (phase-lock) → GeometricLerp (SLERP)
→ StereoscopicFeed (left/right eye) → Renderer + Telemetry
```

**What exists and works**:
- MarketQuoteAdapter, HemocOddsAdapter with field mapping, metrics, and bridge utilities
- Phase-Locked Stereoscopy: irregular data ticks → smooth 60fps 4D rotation via SLERP interpolation
- Sonic Geometry Engine: 4-voice resonant lattice mapping channel clusters + rotation uniforms to harmonic telemetry
- Complete quaternion spinor bridge: double-quaternion decomposition → Hopf fiber modulation → carrier matrices
- 8 layers of telemetry: analysis, signal, transduction, manifold, topology, continuum, lattice, spinor
- Test suite covering phase-lock, adapters, and bridge behaviors
- Full refactor plan (Phases 0-5) with Phase 2 (adapter layer) in progress

**The PPP ↔ HEMOC Bridge**:
- `HemocPythonBridge.ts` translates physics telemetry (12 channels: moire_contrast, frequency, lattice_stress, reservoir_entropy, etc.) into the PPP tick format
- The `WHAT_THIS_DOES.md` doc is the single clearest explanation of the entire system anywhere

---

## 3) What's Actually Critical That's Missing

### A. Domain-Diverse Training Has NEVER BEEN RUN

This is the #1 gap. Code exists (`demos/domain_diverse_training.py`). Results don't exist. Cross-domain zero-shot transfer fails (-0.027), but 100-sample calibration already shows improvement (-0.03 → 0.22). The domain-diverse experiment is the single most important thing that hasn't been done. It requires GPU — this is where the cloud-scale strategy becomes critical.

### B. The Working Pipeline Has ZERO Automated Tests

The testing infrastructure tests the OLD `OpticalKirigamiEncoder`, not the `HybridEncoder` + `AngleRecoveryCNN` that achieves 0.916. This is explicitly identified in `docs/TESTING_REVIEW_AND_DEV_TRACK.md` and addressed by Phase B of the Recovery Plan. The exact test specifications are already designed (hybrid encoder unit tests, CNN decoder tests, encode-decode roundtrip tests). They just need to be written and run.

### C. No Baseline Comparisons Exist

Without baselines, it's impossible to know if the moire encoding adds value over simpler approaches:
- Direct-feature MLP (audio features → predictions with no visual encoding) — expected ~0.93 (the ceiling)
- CNN on raw spectrograms (standard audio ML)
- Random pattern control (random images instead of moire patterns)

Phase D of the Recovery Plan covers this. It hasn't been started.

### D. Golden Ratio / Plastic Ratio Ablations Don't Exist

The system uses φ extensively (grating phase offsets, 600-cell scaling) and ρ (V4 plastic scaling). Neither has been compared against alternatives (√2, e, learned constants, simple geometric progressions). This is Phase D ablation work.

### E. The Visual System Plan Needs Execution

The shader architecture spec in `docs/HEMOC_ARCHITECTURE_DETAILED.md` is detailed and correct. It specifies exactly how to port PhillipsMatrix to TypeScript, how the 2-4-2 uniform mapping works, and what the fragment shader does per-pixel. The hemoc-visual-system directory exists with initial scaffolding. This needs to be completed so training data comes from real GPU-rendered patterns, not numpy approximations.

### F. Temporal Trajectories (Gap 3) Remain Open

Frame-by-frame processing means temporal structure (progressions, cadences, sequences) is lost. The streaming modules (`src/streaming/`) exist and implement temporal encoding with kirigami memory. They're not connected to the audio pipeline. This is the hardest remaining gap and the one with the most potential payoff — it's what separates "processing sound" from "understanding music."

### G. The 21 Open PRs Need Resolution

Branches: main ← atom-of-thoughts ← review-testing ← restore-linear-path ← hemoc-visual-system-init, plus 6 codex branches forking off restore-linear-path. This is a linear chain with side branches. The valuable work needs to be merged down and the stale branches closed.

---

## 4) What's NOT Missing (Corrections to My Earlier Assessment)

### The Phillips Matrix Was NOT "Constructed to Have U_R = φ·U_L"

I was wrong. The Phillips matrix came from Paul Phillips' work on hyper-dimensional audio analysis and quaternion-based system needs (the CPE paper, "Advanced Architectures in Chronomorphic Polytopal Engines"). The relationship U_R = φ·U_L is a DISCOVERED property of a matrix that was built for specific geometric projection purposes, not a design constraint imposed from the start. This matters because it means the golden-ratio properties are emergent from the projection geometry, not tautological.

### The Recovery Plan Already Addresses the Testing/Validation Gaps

The plan on the hemoc-visual-system-init branch is comprehensive and well-structured. It identifies the exact same issues (no tests on working pipeline, no baselines, no cross-domain operationalization) and has concrete phases with exit criteria. It doesn't need to be replaced — it needs to be executed.

### The Cloud Scale Strategy Is Sound

The project-digest branch pivots to cloud-first ("Refactor for Cloud Scale: Packaging, Docker, and Strategy Docs"). This addresses the compute bottleneck directly: all the GPU-dependent work (domain-diverse training, ViT experiments, large-scale ablations) is blocked by running on a laptop. The strategy to verify scaling laws cheaply on cloud, then use those results for resource justification, is correct.

### The Ontology Blueprint Is Publication Infrastructure

The ICE/ECT ontology draft (`docs/ONTOLOGY_BLUEPRINT_ICE_ECT_DRAFT.md`) provides a claims → evidence → reproducibility graph. This is real publication infrastructure, not documentation busywork. The three-table structure (Claims Table, Evidence Table, Reproducibility Table) maps directly to how peer review works.

---

## 5) Suggested Continued Focus (Priority Order)

### Priority 1: Execute Phase B — Test the Working Pipeline

Write the tests specified in `docs/TESTING_REVIEW_AND_DEV_TRACK.md`:
- `test_hybrid_encoder.py`: determinism, sensitivity, injectivity, all-angles-affect
- `test_cnn_decoder.py`: architecture, forward pass, gradient flow
- `test_encode_decode_roundtrip.py`: known-angle recovery, correlation threshold >0.8

This gates everything else. Without regression tests, any future change risks silently breaking the 0.916 result.

### Priority 2: Run Domain-Diverse Training (Requires Cloud)

This is the #1 open experimental question. The code exists. Get it onto a cloud GPU and run it. The result determines the entire cross-domain strategy:
- If diverse training fixes transfer (>0.3 improvement): proceed with multi-domain vision
- If moderate improvement (0.05-0.3): combine with few-shot calibration
- If no improvement: scope claims to domain-specific encoding

### Priority 3: Complete the Visual System Shader Pipeline

The architecture spec is done. The TypeScript implementation needs:
1. Port `PhillipsMatrix.ts` from the spec in the architecture doc
2. Implement `E8ToUniforms.ts` with 2-4-2 mapping
3. Port the `hemoc_4petal.frag.glsl` shader
4. Build the capture pipeline for training data generation

This gives you REAL patterns from GPU rendering instead of numpy approximations. It also directly feeds into the PPP integration (Phase E of Recovery Plan).

### Priority 4: Run Baseline Comparisons

Even without cloud, some baselines can run on CPU:
- Direct-feature ridge regression (no visual encoding) — this is the ceiling test
- Random pattern control — this is the floor test
- If HEMOC beats random significantly but approaches the direct-feature ceiling, the encoding IS adding value but IS lossy (which is expected at 64x64)

### Priority 5: Merge the Branch Chain

Consolidate the linear chain: main ← atom ← testing ← restore ← visual-init. Merge down what's stable, close stale codex branches. Get to ONE canonical branch with all the good work unified.

### Priority 6: Connect Streaming Modules to Audio Pipeline (Gap 3)

This is the highest-risk, highest-reward remaining work. The `TemporalStreamEncoder` and kirigami memory modules exist in `src/streaming/`. Connecting them to the audio pipeline would give temporal trajectory processing. Start with a simple experiment: can a CNN+LSTM decode angle SEQUENCES better than a CNN decodes single frames?

---

## 6) The Big Picture

The project has two genuinely solid pillars:

1. **A mathematically characterized projection** (Phillips Matrix, 6 metric theorems, E8→H4→Trinity decomposition) that provides deterministic, structure-preserving encoding

2. **An experimentally validated encode-decode pipeline** (0.916 correlation on audio, CNN decoder, 15 experiments documenting the progression from 0.008)

The gap between these pillars is **validation infrastructure** (tests, baselines, ablations, cross-domain experiments). The Recovery Plan addresses exactly this gap. The cloud-scale strategy addresses the compute bottleneck blocking the most important experiments.

The PPP system provides the runtime visualization and telemetry infrastructure that makes the mathematical theory tangible — real-time 4D rotation, phase-locked rendering, adapter-based data ingestion, and 8 layers of sonic geometry telemetry.

The path forward is execution of the Recovery Plan phases, in order, with cloud GPU access for the experiments that need it. Theory expansion should wait until the validation infrastructure catches up.
