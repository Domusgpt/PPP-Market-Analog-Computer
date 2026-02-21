# HEMOC + PPP Status: Single Source of Truth

**Last updated**: 2026-02-18
**Canonical location**: `docs/STATUS_SINGLE_SOURCE.md`

> This document supersedes all prior status documents, project digests,
> and experiment summaries.  When in doubt, THIS file is current.
> All prior status documents are retained for archival provenance but
> should not be cited as the current state.

---

## 1. System Identity

| Property | Value |
|----------|-------|
| **Project** | HEMOC (Hexagonal Emergent Moire Optical Cognition) + PPP (Polytopal Projection Processing) |
| **Purpose** | Physics-based optical computing via moire interference patterns governed by 4D polytope geometry |
| **Core operator** | Phillips 8x8 projection matrix (E8 -> H4_L + H4_R decomposition) |
| **Peak result** | 0.916 Pearson correlation on audio-to-angle recovery (15K samples, CNN decoder) |
| **Repos** | `PPP-Market-Analog-Computer` (this repo), `HEMOC-Stain-Glass-Flower` (sibling) |

---

## 2. Claims Table

Every claim in the research program is listed below with its evidence
level and the experiment(s) that support or refute it.

| # | Claim | Evidence Level | Supporting Experiment | Notes |
|---|-------|---------------|----------------------|-------|
| C1 | Moire encoding is injective (0 collisions across 1000 samples) | **Proven** | exp-01 through exp-06 | Tested on multiple angle configurations |
| C2 | CNN decodes 5/6 angles from moire patterns | **Proven** | exp-07 | Spatial inductive bias is critical |
| C3 | HybridEncoder required (pure hypercube fails at 0.09) | **Proven** | exp-08 | Linear/direct channels for dims 0-2, hypercube for 3-5 |
| C4 | Audio pipeline achieves 0.916 correlation | **Proven** | exp-13 | Current peak; all 6 angles recovered |
| C5 | More training data helps (8K->15K: 0.894->0.916) | **Proven** | exp-12, exp-13 | Logarithmic diminishing returns expected |
| C6 | Cross-domain zero-shot transfer fails (-0.027) | **Proven** | exp-14 | Distribution shift, not encoding failure |
| C7 | Few-shot calibration (100 samples) improves to 0.22 | **Proven** | exp-15 | Strong improvement from minimal data |
| C8 | V4 Plastic Encoder > Hybrid > V3.1 > Trinity | **Proven** | FINAL_ANALYSIS_REPORT.md | Plastic ratio scaling (1.32^L) creates distinguishable scales |
| C9 | Centroid-zero kills MLP decoders | **Proven** | CRITICAL_FINDING_AND_ROADMAP.md | E8 roots sum to zero -> aggregate stats destroy info |
| C10 | Phillips matrix has 14 collision pairs among 240 E8 roots | **Proven** | exp-26, hemoc verify | Fuzz harness: 14 is stable across ALL entry perturbations |
| C11 | Phillips U_R = phi * U_L (exact block scaling) | **Proven** | exp-26, hemoc verify | All 12 Phillips theorems pass |
| C12 | Frobenius norm^2 = 20 (600-cell vertex valence) | **Proven** | exp-26, hemoc verify | Verified to machine epsilon |
| C13 | Collision direction d = (0,1,0,1,0,1,0,1)/2 is in kernel | **Proven** | exp-26, hemoc verify | Image under U is exactly zero |
| C14 | Phillips matrix satisfies all 5 Golden Hadamard axioms | **Proven** | exp-28, hemoc golden-hadamard | Dense, golden-ring entries, block scaling, rank deficient, Q(phi) eigenvalues |
| C15 | Renderer fix removes artificial ceiling on continuous tasks | **Re-verified** | exp-22, exp-23 | Proxy: 0.42 ceiling; Physics: 0.519 visual, 0.550 audio |
| C16 | Foveation reveals geometry-dependent detail (+50%) | **Re-verified** | exp-23 | Global 0.28 -> foveated 0.42 cosine similarity |
| C17 | Domain-diverse training improves transfer | **Untested** | exp-16 | Code exists, never run (needs GPU) |
| C18 | Golden ratio is architecturally necessary (not just convenient) | **Untested** | exp-19 | Ablation: phi vs sqrt(2) vs e vs learned |
| C19 | Direct-feature MLP is ceiling for HEMOC | **Untested** | exp-17 | No visual encoding baseline |
| C20 | Adaptive attention outperforms fixed foveation | **Untested** | exp-24 | First run shows attention collapse |

---

## 3. Evidence Table

| Experiment ID | Metric | Value | Renderer | Date | Artifact |
|--------------|--------|-------|----------|------|----------|
| exp-01 | Pearson correlation | 0.001-0.15 | proxy-safe | 2026-01 | results/exp01_mlp_baseline.json |
| exp-05 | Pearson correlation | 0.405 | proxy-safe | 2026-01 | results/exp05_dual_decoder.json |
| exp-07 | Pearson correlation | 0.657 | proxy-safe | 2026-01 | results/exp07_cnn_decoder.json |
| exp-08 | Pearson correlation | 0.090 | proxy-safe | 2026-01 | -- |
| exp-12 | Pearson correlation | 0.894 | proxy-safe | 2026-02 | results/exp12_audio_8k.json |
| exp-13 | Pearson correlation | 0.916 | proxy-safe | 2026-02 | results/exp13_audio_15k.json |
| exp-14 | Pearson correlation | -0.027 | proxy-safe | 2026-02 | -- |
| exp-15 | Pearson correlation | 0.22 | proxy-safe | 2026-02 | -- |
| exp-20 | Accuracy | 1.00 | proxy-safe | 2026-02 | -- |
| exp-22 | Cosine similarity | 0.519 (visual) | physics | 2026-02 | -- |
| exp-23 | Cosine similarity | 0.42 (foveated) | physics | 2026-02 | -- |
| exp-26 | All theorems pass | 12/12 | N/A | 2026-02-18 | hemoc verify output |
| exp-28 | All axioms pass | 5/5 | N/A | 2026-02-18 | hemoc golden-hadamard output |

---

## 4. Reproducibility Table

| Component | Reproducibility Status | Seed | Dependencies | Command |
|-----------|----------------------|------|-------------|---------|
| E8 root generation | Deterministic | N/A | numpy | `python -c "from hemoc.core import generate_e8_roots; print(len(generate_e8_roots()))"` |
| Phillips matrix verification | Deterministic | N/A | numpy | `python -m hemoc verify` |
| Phillips fuzz harness | Seeded | 42 | numpy | `python -m hemoc verify --fuzz --fuzz-trials 1000` |
| 600-cell generation | Deterministic | N/A | numpy | `from hemoc.core import generate_600_cell_vertices` |
| Golden Hadamard check | Deterministic | N/A | numpy | `python -m hemoc golden-hadamard` |
| Dual-channel renderer | Seeded | 42 | numpy | `python -m hemoc render-test` |
| Experiment registry | Static YAML | N/A | pyyaml | `python -m hemoc registry` |
| Audio pipeline (exp-13) | Seeded | recorded | torch, librosa | See `demos/option_e_scaled_cnn.py` |

---

## 5. Experimental Ontology (Reconstructed)

The HEMOC program bifurcates into two tracks that share mathematical
primitives but have distinct experimental surfaces:

### Track A: Feature Decoding / Structural Training

The "project status" thread: hybrid encoder, CNN decoder, audio-to-moire
pipeline, cross-domain transfer, scaling experiments.

**Working pipeline**: HybridEncoder (demos/dual_decoder.py) + AngleRecoveryCNN
(demos/option_e_scaled_cnn.py) on 15K audio samples.

**Key insight**: Spatial decoding (CNN) works because moire patterns have
geometric structure that MLPs cannot exploit (centroid-zero problem).

**Status**: Peak at 0.916 correlation.  Next: domain-diverse training,
baselines, ablations.

### Track B: Phillips Matrix / E8-H4 / 600-Cell / Cognitive Ontology

The "jules track": differentiable 600-cell renderer, cognitive trials
suite (navigation, manifold regression, foveation, adaptive attention),
Phillips matrix verification.

**Working pipeline**: DualChannelGaloisRenderer (hemoc/render/) with
full 4D rotation + projection and phi-coupling verification.

**Key insight**: The renderer fix (proxy -> physics) removes an artificial
ceiling on continuous tasks and restores the meaningfulness of attention
strategies.

**Status**: Renderer fixed; foveation validated (+50%); attention collapse
identified as next bottleneck.

### Shared Spine

Both tracks import from `hemoc.core`, the canonical math spine:
- E8 root system (240 roots, 112+128 construction)
- Phillips 8x8 matrix with all verified theorems
- H4_L + H4_R decomposition with phi-coupling
- 600-cell vertices and 5x24-cell partition
- Kernel basis with collision direction and phason channels

---

## 6. Renderer Fix: What It Invalidates

The pre-fix "proxy" renderer used a heuristic approximation instead of
full 600-cell geometry.  Impact assessment:

| Experiment Type | Proxy Impact | Re-verification Status |
|----------------|-------------|----------------------|
| Discrete-state tasks (exp-01 to exp-15, exp-20) | **Robust** | No re-run needed |
| Noise robustness (exp-21) | **At risk** | Partially re-verified |
| Continuous manifold regression (exp-22) | **Invalidated** under proxy | Re-verified: 0.519 |
| Foveation (exp-23) | **Invalidated** under proxy | Re-verified: +50% gain |
| Adaptive attention (exp-24) | **Invalidated** under proxy | Untested with physics |

---

## 7. Package Architecture

```
hemoc/                           # Canonical Python package (v0.1.0)
  core/                          # Mathematical primitives
    e8_roots.py                  # 240 roots, 112+128 construction
    phillips_matrix.py           # 8x8 operator, all constants and theorems
    h4_decomposition.py          # H4_L + H4_R projection functions
    cell_600.py                  # 120 vertices, 5x24 partition
    kernel_basis.py              # 4D null space, collision direction, phason
  theory/                        # Verification and conjectures
    invariant_verifier.py        # 12 theorem checks + fuzz harness
    galois_verifier.py           # Dual-channel phi-coupling detection
    golden_hadamard.py           # 5-axiom Golden Hadamard class check
  render/                        # Rendering layer
    renderer_contract.py         # Abstract interface all renderers satisfy
    dual_channel_renderer.py     # Galois-verified dual h_L/h_R renderer
    renderer_test_suite.py       # Contract validation functions
  experiments/                   # Experiment registry
    experiments.yaml             # 23 experiments, tagged by renderer dependence
    registry.py                  # Typed access to experiment definitions
  cli.py                         # CLI: hemoc verify / registry / render-test
```

---

## 8. What to Do Next (Priority Order)

1. **Run domain-diverse training on GPU** (exp-16) -- code exists, never run.
2. **Implement baselines** (exp-17, exp-18) -- ceiling and floor tests.
3. **Run golden ratio ablation** (exp-19) -- the key architectural question.
4. **Fix adaptive attention collapse** (exp-24) -- entropy regularization.
5. **Run Collision Universality fuzz at scale** (exp-27) -- 10K+ trials.
6. **Port Phillips matrix to TypeScript** -- for PPP visual system integration.

---

## 9. Superseded Documents

The following documents are retained for provenance but are NO LONGER
the current source of truth:

| Document | Reason Superseded |
|----------|------------------|
| `docs/HEMOC_PPP_CROSS_REPO_DIGEST.md` | Incorporated into this document |
| `docs/refactor-plan.md` | Phase A complete; phases B-F tracked here |
| HEMOC `docs/PROJECT_STATUS.md` | Experiments now in `experiments.yaml` |
| HEMOC `docs/CRITICAL_FINDING_AND_ROADMAP.md` | Findings in Claims Table above |
| HEMOC `FINAL_ANALYSIS_REPORT.md` | V4 Plastic winner in Claims Table |
