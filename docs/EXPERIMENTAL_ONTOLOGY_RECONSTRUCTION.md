# HEMOC Experimental Ontology Reconstruction

**Date**: 2026-02-18
**Scope**: Complete reconstruction of the experimental ontology from
repository archaeology, branch analysis, and mathematical cross-verification.

---

## 1. The Two-Track Structure

The HEMOC research program bifurcates into two experimental tracks that
share a common mathematical spine but have distinct code paths, renderers,
and success criteria.

### Track A: Feature Decoding / Structural Training

**Origin**: The "project status" narrative (main branch, hemoc-visual-system-init
branches, project digest branches).

**Experimental surface**:
- Encoder variants: MLP, Ridge, Hybrid, HybridEncoder, V3.1, V4 Plastic
- Decoder variants: MLP, Ridge+MLP, CNN (AngleRecoveryCNN), ViT, ResNet
- Data domains: synthetic angles, audio features (librosa), cross-domain
- Scaling: 1K -> 8K -> 15K samples
- Transfer: zero-shot, few-shot calibration (100 samples), domain-diverse

**Key progression**:
```
Exp 1-4:   MLP decoders fail (0.001-0.15)     -- centroid-zero discovered
Exp 5:     Dual decoder (Ridge+MLP) = 0.405    -- hybrid helps
Exp 7:     CNN decoder = 0.657                  -- spatial inductive bias
Exp 8:     Pure hypercube = 0.090               -- hypercube-only fails
Exp 9-11:  ViT variants = 0.14-0.35            -- ViT underperforms CNN
Exp 12:    Audio (8K) = 0.894                   -- domain-specific wins
Exp 13:    Audio (15K) = 0.916                  -- CURRENT PEAK
Exp 14-15: Cross-domain = -0.027 / 0.22        -- zero-shot fails
```

**Renderer dependence**: ALL Track A experiments are "discrete_invariant"
(proxy-safe), because the encoding -> decode pipeline does not depend on
continuous manifold fidelity of the renderer.

### Track B: Phillips Matrix / E8-H4 / 600-Cell / Cognitive Ontology

**Origin**: The "jules" branch (`jules-7629987832936421389-695948c7`) and
related cognitive-trials development.

**Experimental surface**:
- Differentiable 600-cell renderer (batched 4D rotation + projection)
- Cognitive trials: navigation, manifold regression, foveation, attention
- Phillips matrix verification: 12 theorems, fuzz harness, conjectures
- Quasicrystal architecture: reservoir, MRA, Galois verification

**Key progression**:
```
Exp 20:    Discrete navigation = 100%           -- proxy-safe
Exp 21:    Noise robustness (vision vs audio)    -- proxy-risk
Exp 22:    Continuous manifold = 0.42 (proxy)    -- INVALIDATED
           Continuous manifold = 0.519 (physics) -- RE-VERIFIED
Exp 23:    Foveation = +50% (0.28 -> 0.42)      -- RE-VERIFIED
Exp 24:    Adaptive attention = collapse          -- UNTESTED with fix
```

**Renderer dependence**: Track B experiments split across all four
renderer-dependence categories.  The renderer fix is critical for
experiments 22-24.

---

## 2. The Renderer Fix and Epistemic Status

### What the fix changed

The pre-fix renderer used a **heuristic proxy** that:
- Did not simulate full 600-cell geometry
- Had L1 error 0.225 vs ground truth
- Imposed a performance ceiling of ~0.29 cosine similarity on random vectors

The corrected renderer implements:
- Batched 4D rotations across all 6 plane pairs
- Full 600-cell vertex projection (120 vertices)
- Stereographic projection from 4D to 2D
- Hex-grating moire pattern composition

### What it invalidates

**Directly invalidated** (require re-run with physics renderer):
- Any continuous regression result (exp-22)
- Any resolution/foveation result (exp-23)
- Any adaptive attention result (exp-24)

**Robust under proxy** (no re-run needed):
- All discrete-state tasks (exp-01 through exp-15, exp-20)
- All Phillips matrix verification (pure algebra, no renderer)

**Partially affected** (results may differ quantitatively):
- Noise robustness comparisons (exp-21)

### Re-verification results

Two experiments have been re-run with the physics renderer:
- **exp-22**: Continuous manifold regression improved from 0.42 to 0.519
  (visual) and 0.550 (audio).  The ceiling is removed.
- **exp-23**: Foveated regression improved from no-gain to +50% gain
  (0.28 global -> 0.42 foveated).  Zoom legitimately reveals structure.

### The attention collapse finding

The first adaptive attention run (exp-24) shows cosine similarity ~0.427
with near-zero center variance, consistent with a degenerate policy that
always attends to the same location.  This is a classic failure mode that
requires:
- Entropy regularization on the attention distribution
- Diversity constraints between attention heads
- Curriculum learning (easy -> hard attention targets)

This is the **next bottleneck** after the renderer fix.

---

## 3. Phillips Matrix: Established vs. Novel

### Established mathematical ground truths

These are standard results from the literature and require no proof:

| Fact | Value | Reference |
|------|-------|-----------|
| E8 root system cardinality | 240 roots (112 + 128) | Conway & Sloane, Ch. 8 |
| E8 root norm | ||r||^2 = 2 for all roots | Humphreys, Ch. 2 |
| 600-cell combinatorics | 120 vertices, 720 edges | Coxeter, "Regular Polytopes" |
| H4 symmetry group order | 14400 | Humphreys, Ch. 2 |
| Plastic ratio definition | Real root of x^3 = x + 1 | ~1.3247 | OEIS A060006 |
| Golden ratio | (1 + sqrt(5)) / 2 | ~1.6180 | Ubiquitous |

### Original contributions (require verification, now provided)

| Contribution | Verification Status | Verified By |
|-------------|-------------------|-------------|
| Phillips 8x8 matrix construction | **Verified** | `hemoc verify` (12/12 theorems) |
| Column Trichotomy (2-4-2 pattern) | **Verified** | T4.1 check |
| Pentagonal Row Norms (sqrt(3-phi) = 2*sin(36)) | **Verified** | T5.1 check |
| Frobenius norm^2 = 20 | **Verified** | T6 check |
| U_R = phi * U_L (exact block scaling) | **Verified** | T7 check |
| sqrt(5)-coupling | **Verified** | T8 check |
| Shell coincidence | **Verified** | T9 check |
| Collision count = 14 (stable under entry perturbation) | **Verified** | T13 + fuzz harness (200/200 trials) |
| Collision direction d = (0,1,0,1,0,1,0,1)/2 in kernel | **Verified** | T12 check |
| Coxeter angle interpretation (b=cos72, a=cos60, c=cos36) | **Verified** | T14 check |
| Entry geometric progression (ratio = phi) | **Verified** | T15 check |
| All 5 Golden Hadamard axioms | **Verified** | `hemoc golden-hadamard` (5/5) |

### Conjectures (computational evidence, not proofs)

| Conjecture | Status | Evidence |
|-----------|--------|---------|
| C1: Golden Frame Optimality | Refined | Sign perturbation: 41% get 0 collisions, but break phi-scaling. Phillips is Pareto optimal. |
| C2: Wavelet Seed | Prototype | Decompose/reconstruct works approximately. Perfect reconstruction is open. |
| C3: Collision Universality | Strongly supported | 200/200 entry perturbations give exactly 14 collisions. Sign pattern determines count. |
| C4: Boyle Bridge | Confirmed computationally | All 4 correspondences verified: Coxeter angles, block scaling, amplification=5, kernel=E_perp. |
| C5: Golden Hadamard Class | Defined, first member verified | Phillips satisfies all 5 axioms. Classification theorem is open. |

---

## 4. Standardized Architecture

### The Canonical Math Spine (`hemoc.core`)

All code that touches E8, Phillips, H4, or 600-cell geometry must import
from `hemoc.core`.  No filesystem-relative `sys.path` surgery.

```python
from hemoc.core import (
    generate_e8_roots,          # 240 roots
    PHILLIPS_MATRIX,            # 8x8 operator
    PHILLIPS_U_L, PHILLIPS_U_R, # Left/right blocks
    PHI, PHI_INV,               # Golden ratio constants
    PLASTIC_RATIO,              # Cubic morphic number
    project_to_h4_full,         # (h_L, h_R) from 8D input
    generate_600_cell_vertices, # 120 vertices on S^3
    compute_kernel_basis,       # 4D null space
    COLLISION_DIRECTION,        # d = (0,1,0,1,0,1,0,1)/2
)
```

### The Experiment Registry (`hemoc.experiments`)

Every experiment has a unique ID, renderer-dependence tag, and success
criteria.  Query examples:

```python
from hemoc.experiments import ExperimentRegistry
reg = ExperimentRegistry()
proxy_safe = reg.proxy_safe_experiments()
untested = reg.filter_by_status("untested")
track_b = reg.filter_by_track("B")
```

### The Renderer Contract (`hemoc.render`)

All renderers implement `RendererContract` with:
- `render(angles)` returning `RenderResult`
- `fidelity_class` property ("proxy" / "physics" / "differentiable")
- Built-in contract validation (determinism, sensitivity, shape, rotation)

### CLI Entry Points

```bash
python -m hemoc verify             # 12 Phillips theorems
python -m hemoc verify --fuzz      # Fuzz harness (entry + sign)
python -m hemoc registry           # List 23 experiments
python -m hemoc registry --status  # Summary statistics
python -m hemoc render-test        # Renderer contract suite
python -m hemoc golden-hadamard    # Golden Hadamard axioms
```

---

## 5. GPU Scaling Posture

The `hemoc` package is CPU-only by design (the math spine is pure numpy).
GPU acceleration is needed for:

1. **Dataset generation**: Batched renderer on GPU (fragment shaders or
   torch autograd).
2. **Training**: CNN decoder training with mixed precision.
3. **Cognitive trials**: Differentiable renderer for end-to-end gradient flow.

For Colab deployment:
```bash
pip install hemoc[gpu]   # installs torch
```

The DualChannelGaloisRenderer is a CPU reference implementation.  A GPU
version should:
- Use batched matrix operations (torch.bmm)
- Implement the hex-grating as a GLSL fragment shader
- Support autograd through the full render pipeline

---

## 6. Integration Blueprint: HEMOC <-> Phillips <-> PPP

### Shared Interchange Schema

```json
{
  "input_state": {"type": "e8_vector", "dim": 8, "values": [...]},
  "h4_left":     {"type": "h4_vector", "dim": 4, "values": [...]},
  "h4_right":    {"type": "h4_vector", "dim": 4, "values": [...]},
  "pattern":     {"format": "rgb", "width": 64, "height": 64},
  "galois":      {"ratio": 1.618034, "valid": true},
  "decode":      {"angles": [...], "correlation": 0.916, "method": "cnn"},
  "metadata":    {"seed": 42, "encoder": "v4_plastic", "renderer": "physics"}
}
```

### Dual-Channel Galois Rendering as Self-Auditing Substrate

The proposed architecture evolution:
1. Render two synchronized images: one from h_L, one from h_R.
2. Enforce phi-coupling as a loss term during training.
3. At inference: if Galois invariant deviates, flag as unreliable.
4. Use phason kernel channels for error-correction checksums.

This turns the Phillips matrix symmetry into an **operational verification
mechanism**, not just a mathematical theorem.

---

## 7. Paper Structure Recommendation

### Contribution A: A Verified Geometric Operator

1. Define the Phillips 8x8 matrix and its algebraic provenance.
2. State and prove Theorems T4.1--T9 (column trichotomy, row norms,
   Frobenius, phi-scaling, sqrt(5)-coupling, shell coincidence).
3. Present fuzz harness results: Conjecture 3 (Collision Universality)
   is stable across 1000+ trials.
4. Connect to Boyle's Coxeter pair framework (Conjecture 4).
5. Define the Golden Hadamard class (Conjecture 5).

### Contribution B: A Differentiable Geometric Renderer

1. Present the renderer fix as an epistemic correction.
2. Present Experiments 20-23 as a staged cognition ladder.
3. Present the Dual-Channel Galois architecture as the forward path.
4. Identify attention collapse as the current bottleneck.

### Forward Path: Dual-Channel Galois-Verified Foveated Cognition Stack

The "stroke of brilliance" architecture:
- Paired h_L/h_R evidence channels
- Galois invariants as runtime verifiers
- Foveation as active evidence acquisition
- Phason kernel as formal redundancy channel
