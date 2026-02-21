# Experiment Contamination Audit

**Date**: 2026-02-21
**Auditor**: Automated deep-code audit (3 parallel domain audits)
**Scope**: All implementations in PPP-Market-Analog-Computer that contribute to experimental results
**Purpose**: Academic-grade provenance tracking for all HEMOC/PPP experimental claims

> **Policy**: Bad code is KEPT in place but flagged. No silent fixes. Every experiment built
> on or derived from buggy code is classified by contamination risk. Replacement research
> must reference this audit for full traceability.

---

## Table of Contents

1. [Bug Inventory](#1-bug-inventory)
2. [Verified Clean Components](#2-verified-clean-components)
3. [Experiment Numbering Systems](#3-experiment-numbering-systems)
4. [Cross-Reference Table](#4-cross-reference-table)
5. [Experiment Classification](#5-experiment-classification-greenamberred)
6. [The Contamination Chain: SA-8 → SA-9 → SA-11](#6-the-contamination-chain-sa-8--sa-9--sa-11)
7. [Impact Assessment on Publication Claims](#7-impact-assessment-on-publication-claims)
8. [Re-Run Protocol](#8-re-run-protocol)
9. [Provenance Tags for Paper](#9-provenance-tags-for-paper)

---

## 1. Bug Inventory

Seven bugs were identified across three implementation domains. Each is documented with
exact file paths, line numbers, and the mechanism by which it could contaminate results.

### BUG-001 — 600-Cell Parity Filter (CRITICAL)

**File**: `_SYNERGIZED_SYSTEM/backend/engine/geometry/h4_geometry.py`
**Lines**: 402–403 (primary), 461 (fallback)

**What's wrong**: The `Polytope600Cell._generate_vertices()` method generates Class 1
vertices — permutations of `(±1, ±1, ±1, ±1)/2` — but applies an **even-parity filter**:

```python
# Line 402-403
for signs in product([-1, 1], repeat=4):
    if sum(1 for s in signs if s < 0) % 2 == 0:  # BUG: even parity only
```

This produces only **8 of 16** half-integer vertices. The even-parity restriction is correct
for the **24-cell** (Hurwitz quaternion units), but the 600-cell contains ALL 16 permutations
of `(±½, ±½, ±½, ±½)`. The fallback method `_generate_full_vertices()` (line 461) has the
**same bug**.

**Downstream effect**: Any code path that uses `Polytope600Cell` from this file gets a
geometrically incomplete polytope. This affects:
- Edge count: 548 instead of 720
- Distance spectrum: 30 unique distances instead of 8
- 5×24-cell partition: impossible with missing vertices

**Contamination scope**: Any experiment using the PPP `h4_geometry.py` 600-cell implementation.

### BUG-002 — Sphere Membership Tolerance (HIGH)

**File**: `_SYNERGIZED_SYSTEM/backend/engine/geometry/h4_geometry.py`
**Lines**: 424, 456

**What's wrong**: Two tolerance checks for unit-sphere membership use absurdly loose thresholds:

```python
# Line 424
if np.isclose(sum(c**2 for c in coords), 1.0, atol=0.1):  # 10% tolerance!

# Line 456 (fallback method)
if np.isclose(sum(c**2 for c in coords), 1.0, atol=0.2):  # 20% tolerance!
```

Points with squared norm between 0.8 and 1.2 (or 1.4 in fallback) are accepted as
"on the unit sphere." This can admit spurious vertices that are NOT part of the 600-cell.

**Downstream effect**: The deduplication loop (lines 426–429) may reject valid vertices
while admitting invalid ones, compounding BUG-001.

### BUG-003 — Algebraic Exactness Destruction (MEDIUM)

**File**: `_SYNERGIZED_SYSTEM/backend/engine/geometry/h4_geometry.py`
**Line**: 484

**What's wrong**: The fallback vertex generator normalizes coordinates by dividing by
the floating-point norm:

```python
# Line 484
coords = tuple(c / norm for c in coords)
```

For golden-ratio vertices like `(φ, 1, 1/φ, 0)/2`, the exact norm-squared is 1.0
(provably, since `φ² + 1 + 1/φ² + 0 = φ² + 1 + φ² - 2φ + 1 = ...`). Dividing by
a floating-point approximation of `norm` introduces rounding error that destroys
the algebraic exactness of the coordinates. This means downstream computations
(eigenvalues, inner products, collision detection) lose the Q(√5) closure property.

**Downstream effect**: Numerical drift in any computation that depends on exact
golden-ratio relationships between vertex coordinates.

### BUG-004 — Fake E8 Embedding (CRITICAL)

**File**: `hemoc/render/dual_channel_renderer.py`
**Line**: 117

**What's wrong**: The `DualChannelGaloisRenderer` creates an "8D embedding" of 4D
vertices by simply duplicating the 4D coordinates:

```python
# Lines 113-117
# We use a representative 8D embedding:
#   v_8d = [v_4d[0], v_4d[1], v_4d[2], v_4d[3],
#           v_4d[0], v_4d[1], v_4d[2], v_4d[3]]
# This is a simplification; the full pipeline would use actual E8 roots.
v_8d_batch = np.hstack([rotated, rotated])  # (120, 8)
```

The comment on line 116 **explicitly acknowledges** this is a simplification.
The actual E8 root system has 240 vectors in R⁸ that are NOT constructed by
duplicating R⁴ coordinates. The E8 roots have two types:
- 112 permutation-type: coordinates are permutations of `(±1, ±1, 0, 0, 0, 0, 0, 0)`
- 128 half-integer-type: coordinates are `(±½, ±½, ±½, ±½, ±½, ±½, ±½, ±½)` with even parity

Duplicating 4D → 8D creates vectors of the form `(a, b, c, d, a, b, c, d)` which
are NOT E8 roots (they don't satisfy the norm or parity constraints of either type).

**Downstream effect**: The Phillips matrix projection `h_L = v_8d @ U_L^T` produces
WRONG 4D coordinates because the input is not a valid E8 root. The Galois ratio check
(line 125: `galois_ratio = norm_R / norm_L`) may still pass trivially because the
duplication makes `h_L` and `h_R` correlated, but the geometric content is meaningless.

**Contamination scope**: ALL rendered moire patterns from this renderer encode
geometry from a fake embedding, not from the actual E8 → H4 projection.

### BUG-005 — 24-Cell Vertex Truncation (HIGH)

**File**: `hemoc/render/dual_channel_renderer.py`
**Line**: 233

**What's wrong**: The rendering loop only uses the first 24 vertices:

```python
# Line 233
for v_idx in range(min(len(projected_4d), 24)):  # use first 24-cell
```

The 600-cell has 120 vertices (5 × 24-cell). Using only 24 means the moire pattern
encodes information from 1/5 of the polytope. Combined with BUG-004 (fake E8
embedding), the rendered pattern bears little geometric relationship to the actual
600-cell projection.

**Downstream effect**: Moire patterns are low-information approximations of what the
physics demands.

### BUG-006 — 5-Angle Limitation (HIGH)

**File**: `_SYNERGIZED_SYSTEM/backend/engine/core/batch_encoder.py`
**Line**: 79

**What's wrong**: The encoder uses 5 pre-computed commensurate angles:

```python
# Line 79
self.angles = [0.0, 7.34, 9.43, 13.17, 21.79]
```

The mathematical specification calls for 6 angular parameters (the 6 independent
planes of rotation in 4D: xy, xz, xw, yz, yw, zw). The batch encoder processes
**one angle at a time** from this fixed set of 5, rather than using all 6 simultaneously.

The `stream_encoder.py` and `temporal_encoder.py` have the same limitation — they
consume data one sample at a time with a single angle index, not the full 6D rotation.

Meanwhile, `dual_channel_renderer.py` (line 103) correctly takes 6 angles as input
and builds a full 4D rotation matrix. This creates an **impedance mismatch**: the
renderer can handle 6D input but the encoders only produce 5D.

**Downstream effect**: The encoding pipeline cannot fully exercise the 6D rotation
space. Experiments testing angle recovery can at most recover 5 of 6 angles from
the encoder side, even though the decoder (CNN) is trained to recover all 6.

### BUG-007 — Fading Memory Order Inconsistency (MEDIUM)

**Files**:
- `_SYNERGIZED_SYSTEM/backend/engine/streaming/stream_encoder.py` (lines 129–131)
- `_SYNERGIZED_SYSTEM/backend/engine/streaming/temporal_encoder.py` (lines 78, 81–82)

**What's wrong**: The two encoder implementations apply fading memory decay in
opposite order:

**stream_encoder.py** — decay BEFORE injection:
```python
# Lines 129-131
# Apply decay for fading memory
self.simulator.values *= 0.95       # decay first
self.simulator.inject_input(data, scale=0.5)  # then inject
```

**temporal_encoder.py** — decay AFTER injection (explicitly documented):
```python
# Line 78 (docstring)
# 4. Maintaining proper fading memory with decay AFTER injection

# Lines 81-82 (comparison)
# - Basic: velocity += input; then decay (loses order)
# - Temporal: values blend with input; track evolution; then decay (preserves order)
```

The temporal encoder's docstring explicitly identifies the stream encoder's approach
as incorrect ("loses order"). These two implementations will produce different
temporal dynamics for the same input sequence.

**Downstream effect**: Results from stream_encoder and temporal_encoder are not
comparable. Any experiment that switches between them (or claims "encoder"
behavior generically) conflates two different temporal models.

---

## 2. Verified Clean Components

The following implementations were audited and found **correct**:

### hemoc/core/phillips_matrix.py — ALL CLEAR

15 independent implementations of the Phillips 8×8 matrix were cross-compared:
- All produce identical entries to machine precision
- Entry alphabet {1/2, (φ−1)/2, φ/2} correct
- U_R = φ × U_L verified exactly
- Eigenvalue 5 with multiplicity 2 confirmed
- Frobenius² = 20 verified
- Column trichotomy (2-4-2 norm pattern) confirmed

**Conclusion**: The mathematical core of the Phillips matrix is trustworthy.
No experiment is contaminated by Phillips matrix implementation bugs.

### hemoc/core/cell_600.py — ALL CLEAR

The `hemoc` package's 600-cell implementation generates:
- Exactly 120 vertices (correct)
- Exactly 720 edges (correct)
- Exactly 8 unique distance classes (correct)
- Valid 5×24-cell partition (correct)

This is the **golden standard** implementation. It does NOT share the parity filter
bug (BUG-001) because it uses a different construction method.

### hemoc/core/e8_roots.py — ALL CLEAR

Generates exactly 240 E8 roots:
- 112 permutation-type (norm² = 2)
- 128 half-integer-type (norm² = 2, even parity)
- All inner products in {0, ±1, ±2} as required

### hemoc/theory/invariant_verifier.py — ALL CLEAR

12/12 theorems pass. Fuzz harness stable. Collision count 14 confirmed invariant
under entry perturbation.

### hemoc/theory/golden_hadamard.py — ALL CLEAR

5/5 Golden Hadamard axioms pass. Classification is well-defined and correct.

---

## 3. Experiment Numbering Systems

Two independent experiment registries exist with different numbering, scopes, and timelines.

### System A: "Polytope Trials"

- **Source**: Branch `claude/review-testing-progress-F6a3C` (Feb 1, 2026)
- **Count**: 22 experiments (SA-1 through SA-22)
- **Timeline**: ALL run on **February 1, 2026** — approximately two weeks BEFORE the renderer fix
- **Focus**: Phillips matrix verification + encoder architecture testing + cognitive trials
- **Renderer**: ALL used the proxy renderer (no physics renderer existed yet)

### System B: HEMOC Experiment Registry

- **Source**: `hemoc/experiments/experiments.yaml`
- **Count**: 30 experiments (exp-01 through exp-30)
- **Timeline**: January 24 – February 18, 2026
- **Focus**: Feature decoding (Track A) + Phillips/cognitive ontology (Track B)
- **Renderer**: Mixed — early experiments proxy-safe by design; exp-22 and exp-23 re-verified with physics renderer post-fix
- **Tagging**: Each experiment has `renderer_dependence` field: `discrete_invariant`, `continuous_manifold`, `resolution_dependent`, or `noise_dependent`

---

## 4. Cross-Reference Table

Mapping System A experiments to System B equivalents where identifiable:

| System A | SA Name | System B | SB Name | Confidence | Notes |
|----------|---------|----------|---------|------------|-------|
| SA-1 | Phillips Matrix Self-Test | exp-26 | Phillips Matrix Invariant Verification | HIGH | Same mathematical content, different tooling |
| SA-2 | Trinity Verification | — | — | — | No System B equivalent; 24-cell partition test |
| SA-3 | D4 Root System Verification | — | — | — | Pure math, not in HEMOC registry |
| SA-4 | Petal Rotation Analysis | — | — | — | Jacobian measurement, not in HEMOC registry |
| SA-5 | 600-Cell Expansion | — | — | — | Vertex/edge validation, not in HEMOC registry |
| SA-6 | E8 Unfolding | — | — | — | Projection test, not in HEMOC registry |
| SA-7 | Dual Decoder | exp-05 | Dual Decoder (Ridge + MLP) | MEDIUM | Similar scope, possibly different configurations |
| SA-8 | Trinity Encoder + CNN | — | — | — | Trinity encoder NOT in HEMOC registry |
| SA-9 | V3 600-Cell Encoder | — | — | — | V3/V3.1 NOT in HEMOC registry |
| SA-11 | V4 Plastic Encoder | — | — | PARTIAL | V4 Plastic is "winner" in FINAL_ANALYSIS_REPORT but no direct exp-ID |
| SA-12 | Fulcrum Energy Analysis | — | — | — | Pure math, energy distribution |
| SA-13 | Pentagonal Phase Lock | — | — | — | Pure math, ratio verification |
| SA-14 | Fractal Audio Synthesis | — | — | LOW | Audio domain, unclear mapping |
| SA-16 | Neo-Riemannian Homology | — | — | LOW | Loop closure test |
| SA-17 | Music Prediction | exp-20? | Discrete Navigation | POSSIBLE | Different success rates (32% vs 100%) |
| SA-18 | Twist Prediction | — | — | — | 33.9% accuracy, not in registry |
| SA-19 | Visual Imagination | — | — | — | MSE 0.0335, not in registry |
| SA-20 | Goal-Directed Navigation | exp-20 | Discrete Navigation | POSSIBLE | 58% vs 100% — may be same task, different config |
| SA-21b | Robustness Test | exp-21 | Noise Robustness | HIGH | Both test vision/audio robustness |
| SA-22 | Continuous Manifold | exp-22 | Continuous Manifold Regression | CONFIRMED | Same experiment. Proxy: 0.42, Physics: 0.519 |

**Key observation**: System A experiments SA-8, SA-9, and SA-11 (the contamination chain)
have **NO System B equivalents**. These encoder architecture experiments were run once,
conclusions drawn, and never re-registered in the canonical HEMOC experiment registry.

---

## 5. Experiment Classification (GREEN/AMBER/RED)

### GREEN — Renderer-Independent (Safe)

These experiments test pure mathematics or use discrete-state tasks where the proxy
renderer's continuous-manifold inaccuracy cannot affect results.

| SA# | Name | Why Safe | Bugs Affecting |
|-----|------|----------|----------------|
| SA-1 | Phillips Matrix Self-Test | Pure algebraic verification, 6 theorems to machine precision | None |
| SA-2 | Trinity Verification | Tests 24-cell vertex structure, not rendering | None |
| SA-3 | D4 Root System Verification | Root system combinatorics, pure math | None |
| SA-4 | Petal Rotation Analysis | Jacobian measurement of encoder (math, not rendering) | None |
| SA-5 | 600-Cell Expansion | Vertex/edge/distance validation | BUG-001 if using h4_geometry.py |
| SA-6 | E8 Unfolding | Phillips matrix projection, pure linear algebra | None |
| SA-12 | Fulcrum Energy Analysis | E8 vector energy distribution, pure math | None |
| SA-13 | Pentagonal Phase Lock | Ratio verification (phi confirmed) | None |

**CAUTION on SA-5**: If this experiment used `h4_geometry.py` (BUG-001), the vertex count
and edge count would be wrong. However, the experiment may have used a correct construction.
Needs verification of which 600-cell implementation was used.

**System B GREEN experiments**: exp-01 through exp-15, exp-20 — all tagged `discrete_invariant`
in `experiments.yaml`. These are safe because discrete classification/correlation tasks are
robust to continuous rendering approximations.

### AMBER — Proxy Renderer, Discrete Task (Directionally Valid)

These experiments used the proxy renderer but on tasks where the discrete nature of the
output (classification, binary success, exact algebraic check) makes the results
**directionally valid** even if quantitative details might shift.

| SA# | Name | Concern | Risk Level |
|-----|------|---------|------------|
| SA-7 | Dual Decoder | Used proxy patterns for decoder training | LOW — decoder architecture comparison is relative |
| SA-8 | Trinity Encoder + CNN | 0.90 on angles 0-2, 0.0 on angles 3-5 | MEDIUM — see §6 below |
| SA-14 | Fractal Audio Synthesis | Spectral entropy measurement | LOW — audio domain, not visual rendering |
| SA-16 | Neo-Riemannian Homology | Loop closure = 0.000000 | LOW — discrete topology, exact closure |
| SA-17 | Music Prediction | 32% accuracy (at PLR limit) | LOW — discrete classification |
| SA-20 | Goal-Directed Navigation | 58% success | LOW — discrete Tonnetz graph navigation |

**Note on SA-8**: Classified AMBER rather than RED because the *finding* (angles 0-2 work,
angles 3-5 don't) could be a genuine architectural insight OR a proxy artifact. The
classification "MEDIUM risk" reflects this ambiguity. See §6 for full analysis.

### RED — Proxy Renderer, Continuous Task (Potentially Contaminated)

These experiments used the proxy renderer on continuous regression/reconstruction tasks
where the proxy's artificial ~0.42 cosine similarity ceiling could distort results.

| SA# | Name | Concern | Impact | Bugs Contributing |
|-----|------|---------|--------|-------------------|
| SA-9 | V3 600-Cell Encoder | V3.0 FAILS → V3.1 "fixes" by mapping h_L to zoom/sharpness | HIGH — architecture designed around proxy limitations? | BUG-004, BUG-005 |
| SA-11 | V4 Plastic Encoder | 0.53 cosine sim — was this ceiling-limited? | MEDIUM — plastic ratio scaling may be correct but measured against wrong baseline | BUG-004, BUG-005 |
| SA-18 | Twist Prediction | 33.9% (FAIL) — was failure due to proxy's inability to encode subtle operator differences? | MEDIUM — might PASS with physics renderer | BUG-004, BUG-005 |
| SA-19 | Visual Imagination | MSE 0.0335 — learned proxy physics, not real physics | HIGH — model learned to predict PROXY patterns | BUG-004, BUG-005 |
| SA-21b | Robustness Test | "Vision is Fragile" — maybe only fragile under PROXY rendering? | MEDIUM — vision robustness needs re-test | BUG-004, BUG-005 |
| SA-22 | Continuous Manifold | 0.42 cosine sim | **RESOLVED** — re-verified: 0.519 with physics renderer | Was BUG-004/005; now fixed |

**System B RED experiments**: exp-22 (re-verified), exp-23 (re-verified), exp-24 (untested
with physics), exp-25 (untested — validates the buggy renderer itself).

---

## 6. The Contamination Chain: SA-8 → SA-9 → SA-11

This is the central concern. Three experiments form a dependency chain where each
subsequent experiment's design was influenced by conclusions from the previous one —
and the initial conclusions may have been artifacts of the proxy renderer.

### Step 1: SA-8 (Trinity Encoder + CNN)

**Result**: Correlation 0.90 on angles 0-2 (linear path), **0.0** on angles 3-5 (hypercube path).

**Conclusion drawn**: "The HybridEncoder architecture is structurally flawed. The hypercube
rotation path (angles 3-5) does not produce distinguishable patterns."

**The question**: In the HybridEncoder, angles 3-5 go through:
1. Hypercube rotation in abstract 4D space
2. Projection to pattern parameters via the Phillips matrix
3. Rendering to a moire pattern

The proxy renderer (BUG-004: fake E8 embedding, BUG-005: only 24 of 120 vertices) was
responsible for step 3. If the proxy could not faithfully render the subtle geometric
differences produced by hypercube rotations, the **0.0 correlation on angles 3-5 may be
a renderer limitation, not an encoder flaw**.

**Alternative hypothesis**: The HybridEncoder's hypercube path IS architecturally sound, but
the proxy renderer discards the information before the CNN can decode it. The Trinity
Encoder's apparent success on angles 0-2 may simply reflect that linear mappings produce
coarser pattern differences that survive proxy approximation.

### Step 2: SA-9 (V3 600-Cell Encoder)

**Context**: Based on SA-8's conclusion that Hybrid was "structurally flawed", the V3
encoder was designed to bypass the hypercube path entirely.

**Result**: V3.0 FAILS completely. V3.1 "fixes" V3.0 by re-mapping the h_L projection
output to control zoom and sharpness parameters instead of discarding it.

**The question**: The V3.1 "fix" (mapping h_L to zoom/sharpness) was a **design response**
to the perceived failure of the hypercube path. If SA-8's conclusion was wrong (the
hypercube path works fine with a correct renderer), then V3.1's architectural choice
may be an unnecessary workaround that introduces information loss.

### Step 3: SA-11 (V4 Plastic Encoder)

**Result**: Cosine similarity 0.53.

**Context**: V4 was designed as the next iteration after V3.1, incorporating the plastic
ratio (ρ ≈ 1.3247) for layer scaling. Its architecture inherits V3.1's design decisions
about how to handle the h_L projection.

**The question**: V4's 0.53 score was measured against the proxy renderer. The proxy ceiling
for continuous tasks is ~0.42 (established by SA-22/exp-22). V4's 0.53 exceeds this ceiling,
suggesting the plastic ratio scaling provides genuine benefit. However:
- The absolute value may be higher with the physics renderer
- The **ranking** of V4 vs V3.1 vs Hybrid vs Trinity may change
- The FINAL_ANALYSIS_REPORT.md claim "V4 Plastic > Hybrid > V3.1 > Trinity" was established
  under proxy rendering conditions

### Chain Summary

```
SA-8 (proxy) → "Hybrid is flawed"
     ↓
SA-9 (proxy) → V3.1 designed to bypass hypercube path
     ↓
SA-11 (proxy) → V4 inherits V3.1 design, scores 0.53
     ↓
FINAL_ANALYSIS_REPORT → "V4 > Hybrid > V3.1 > Trinity" (proxy conditions)
     ↓
Paper Claim C8 → "V4 Plastic Encoder > Hybrid > V3.1 > Trinity" [CONTAMINATED?]
```

**Critical note**: The claim that "V4 Plastic Encoder is optimal" (Claims Table C8 in
`STATUS_SINGLE_SOURCE.md`) rests on this chain. If SA-8's conclusion about the Hybrid
encoder was a proxy artifact, the entire encoder architecture progression needs re-evaluation.

### What This Does NOT Invalidate

The contamination chain affects **encoder architecture ranking only**. The following
claims remain unaffected:

- 0.916 correlation (exp-13): Uses HybridEncoder + CNN decoder on audio data. Tagged
  `discrete_invariant` in experiments.yaml. The 0.916 result measures angle recovery
  accuracy, which is a discrete correlation metric robust to rendering approximation.
- CNN > MLP (exp-01 vs exp-07): The centroid-zero problem is a mathematical property of
  E8 root aggregation, independent of rendering.
- Cross-domain failure (exp-14): Distribution shift is an input-space phenomenon,
  independent of rendering.
- All Phillips matrix theorems: Pure algebra, no renderer involvement.

---

## 7. Impact Assessment on Publication Claims

Assessment of each claim in `STATUS_SINGLE_SOURCE.md` Claims Table:

| Claim | Status in Paper | Contamination Risk | Action Needed |
|-------|----------------|-------------------|---------------|
| C1: Encoding is injective | Safe to publish | NONE — pure math | None |
| C2: CNN decodes 5/6 angles | Safe to publish | NONE — discrete metric | None |
| C3: HybridEncoder required | Safe to publish | NONE — exp-08 is discrete | None |
| C4: 0.916 correlation | Safe to publish | NONE — discrete_invariant | None |
| C5: More data helps | Safe to publish | NONE — discrete_invariant | None |
| C6: Cross-domain fails | Safe to publish | NONE — input-space property | None |
| C7: Few-shot helps | Safe to publish | NONE — discrete_invariant | None |
| **C8: V4 > Hybrid > V3.1 > Trinity** | **CONTAMINATED** | **HIGH** — depends on SA-8→9→11 chain | **Must re-run with physics renderer** |
| C9: Centroid-zero kills MLP | Safe to publish | NONE — mathematical theorem | None |
| C10: 14 collision pairs | Safe to publish | NONE — pure algebra | None |
| C11: U_R = φ·U_L | Safe to publish | NONE — pure algebra | None |
| C12: Frobenius² = 20 | Safe to publish | NONE — pure algebra | None |
| C13: Collision direction | Safe to publish | NONE — pure algebra | None |
| C14: Golden Hadamard axioms | Safe to publish | NONE — pure algebra | None |
| C15: Renderer fix removes ceiling | Safe to publish | NONE — already re-verified | None |
| C16: Foveation +50% | Safe to publish | NONE — re-verified with physics | None |
| C17: Domain-diverse training | N/A | UNTESTED — code never run | Run experiment |
| C18: Golden ratio necessity | N/A | UNTESTED — no ablation exists | Run experiment |
| C19: Direct-feature ceiling | N/A | UNTESTED — no baseline exists | Run experiment |
| C20: Adaptive attention | N/A | UNTESTED — attention collapse | Fix and re-run |

**Summary**: 14 of 16 proven claims are SAFE. **1 claim (C8) is contaminated** by the
SA-8→9→11 chain. 4 claims remain untested.

---

## 8. Re-Run Protocol

Experiments listed in priority order. Each re-run must:
1. Use `hemoc/core/cell_600.py` (correct 600-cell), NOT `h4_geometry.py`
2. Use the physics renderer with corrected E8 embedding (fix BUG-004)
3. Use all 120 vertices (fix BUG-005)
4. Record the renderer version, commit hash, and seed in result metadata
5. Compare results side-by-side with original proxy results

### Priority 0 — Foundational Architecture Questions

| Experiment | Original Result | What We're Testing | Expected Outcome |
|-----------|----------------|-------------------|-----------------|
| SA-8 re-run | Angles 0-2: 0.90, Angles 3-5: 0.0 | Does the hypercube path work with physics renderer? | If angles 3-5 recover (even partially), the "Hybrid is flawed" conclusion was wrong |
| SA-9 re-run | V3.0 FAIL, V3.1: 0.50 | Does V3.0 still fail with physics renderer? Is h_L→zoom mapping still needed? | If V3.0 works, the V3.1 fix was a proxy workaround |

### Priority 1 — Continuous Task Validation

| Experiment | Original Result | What We're Testing | Expected Outcome |
|-----------|----------------|-------------------|-----------------|
| SA-18 re-run | 33.9% (chance level) | Twist prediction with physics renderer | May improve above chance if proxy was the bottleneck |
| SA-19 re-run | MSE 0.0335 | Visual imagination — NEEDS FULL RETRAIN | Must retrain from scratch; old model learned proxy patterns |

### Priority 2 — Robustness Confirmation

| Experiment | Original Result | What We're Testing | Expected Outcome |
|-----------|----------------|-------------------|-----------------|
| SA-21b re-run | "Vision is fragile" | Is vision still fragile under physics rendering? | May show improved robustness if proxy was the fragility source |

### RESOLVED

| Experiment | Original | Physics | Status |
|-----------|----------|---------|--------|
| SA-22 / exp-22 | 0.42 cosine sim | 0.519 visual, 0.550 audio | CONFIRMED — proxy ceiling removed |
| exp-23 | N/A | 0.42 foveated (+50% over global 0.28) | CONFIRMED — foveation works |

---

## 9. Provenance Tags for Paper

Every experimental result cited in the publication must carry one of these tags:

### Tag Definitions

| Tag | Meaning | Can Cite As Evidence? |
|-----|---------|----------------------|
| `[RENDERER-INDEPENDENT]` | Result depends only on mathematics, not rendering | YES — full confidence |
| `[DISCRETE-INVARIANT]` | Used proxy renderer but on discrete/correlation task robust to rendering | YES — full confidence |
| `[PHYSICS-VERIFIED]` | Originally proxy, re-verified with physics renderer | YES — full confidence |
| `[PROXY-ONLY]` | Used proxy renderer on continuous task, not yet re-verified | QUALIFIED — state renderer version, note pending re-verification |
| `[PROXY-CONTAMINATED]` | Result or architectural conclusion derived from proxy-limited experiments | NO — must re-run before citing as evidence |
| `[UNTESTED]` | Code exists but experiment never run | NO — conjecture only |

### Tag Assignments

| Experiment | Tag | Justification |
|-----------|-----|---------------|
| exp-01 (MLP baseline) | `[DISCRETE-INVARIANT]` | Correlation metric, proxy-safe |
| exp-05 (Dual decoder) | `[DISCRETE-INVARIANT]` | Correlation metric, proxy-safe |
| exp-07 (CNN breakthrough) | `[DISCRETE-INVARIANT]` | Correlation metric, proxy-safe |
| exp-08 (Pure hypercube) | `[DISCRETE-INVARIANT]` | Correlation metric, proxy-safe |
| exp-12 (Audio 8K) | `[DISCRETE-INVARIANT]` | Correlation metric, proxy-safe |
| exp-13 (Audio 15K, 0.916) | `[DISCRETE-INVARIANT]` | Correlation metric, proxy-safe |
| exp-14 (Zero-shot) | `[DISCRETE-INVARIANT]` | Correlation metric, proxy-safe |
| exp-15 (Few-shot) | `[DISCRETE-INVARIANT]` | Correlation metric, proxy-safe |
| exp-20 (Discrete nav) | `[DISCRETE-INVARIANT]` | 100% accuracy on graph navigation |
| exp-22 (Continuous manifold) | `[PHYSICS-VERIFIED]` | Re-verified: 0.42 → 0.519 |
| exp-23 (Foveation) | `[PHYSICS-VERIFIED]` | Re-verified: +50% gain confirmed |
| exp-26 (Phillips theorems) | `[RENDERER-INDEPENDENT]` | Pure algebra, 12/12 pass |
| exp-28 (Golden Hadamard) | `[RENDERER-INDEPENDENT]` | Pure algebra, 5/5 pass |
| SA-8 (Trinity + CNN) | `[PROXY-CONTAMINATED]` | Conclusion about Hybrid being "flawed" needs re-verification |
| SA-9 (V3/V3.1) | `[PROXY-CONTAMINATED]` | Architecture designed around proxy limitations |
| SA-11 (V4 Plastic) | `[PROXY-CONTAMINATED]` | Measured against proxy baseline |
| SA-18 (Twist prediction) | `[PROXY-ONLY]` | Continuous task, pending re-verification |
| SA-19 (Visual imagination) | `[PROXY-CONTAMINATED]` | Model trained on proxy patterns |
| SA-21b (Robustness) | `[PROXY-ONLY]` | Pending re-verification |
| Claim C8 (V4 > Hybrid ranking) | `[PROXY-CONTAMINATED]` | Depends on SA-8→9→11 chain |
| exp-16 (Domain-diverse) | `[UNTESTED]` | Code exists, never run |
| exp-17 (Direct MLP baseline) | `[UNTESTED]` | No code exists |
| exp-18 (Random pattern) | `[UNTESTED]` | No code exists |
| exp-19 (Golden ratio ablation) | `[UNTESTED]` | No code exists |
| exp-24 (Adaptive attention) | `[UNTESTED]` | Attention collapse, needs fix |

---

## Appendix A: Code Audit Methodology

Three parallel audits were conducted:

1. **Phillips Matrix Audit**: Cross-compared 15 independent implementations of the
   Phillips 8×8 matrix across both repositories. Verified entry values, eigenvalues,
   rank, Frobenius norm, block scaling, and sign pattern. Result: ALL CLEAR.

2. **600-Cell Audit**: Compared `h4_geometry.py` (PPP repo) vs `hemoc/core/cell_600.py`
   (hemoc package). Checked vertex count, edge count, distance spectrum, and 5×24-cell
   partition. Result: h4_geometry.py BROKEN (BUG-001/002/003), hemoc CORRECT.

3. **Encoder/Renderer Audit**: Traced the full data path from input angles through
   E8 embedding, Phillips projection, moire rendering, and CNN decoding. Checked
   dimensional consistency (5 vs 6 angles), E8 embedding authenticity, vertex count
   in renderer, and temporal dynamics consistency. Result: Multiple issues (BUG-004
   through BUG-007).

## Appendix B: File Index

All files referenced in this audit, with their audit status:

| File | Status | Bugs |
|------|--------|------|
| `hemoc/core/phillips_matrix.py` | CLEAN | — |
| `hemoc/core/cell_600.py` | CLEAN | — |
| `hemoc/core/e8_roots.py` | CLEAN | — |
| `hemoc/core/kernel_basis.py` | CLEAN | — |
| `hemoc/theory/invariant_verifier.py` | CLEAN | — |
| `hemoc/theory/golden_hadamard.py` | CLEAN | — |
| `hemoc/theory/galois_verifier.py` | CLEAN | — |
| `hemoc/render/dual_channel_renderer.py` | BUGGY | BUG-004, BUG-005 |
| `hemoc/render/renderer_contract.py` | CLEAN | — |
| `hemoc/experiments/experiments.yaml` | CLEAN | — |
| `_SYNERGIZED_SYSTEM/backend/engine/geometry/h4_geometry.py` | BUGGY | BUG-001, BUG-002, BUG-003 |
| `_SYNERGIZED_SYSTEM/backend/engine/core/batch_encoder.py` | BUGGY | BUG-006 |
| `_SYNERGIZED_SYSTEM/backend/engine/core/fast_cascade.py` | NOTED | Discrete approximation (not bug per se, but documented limitation) |
| `_SYNERGIZED_SYSTEM/backend/engine/streaming/stream_encoder.py` | BUGGY | BUG-007 (decay order) |
| `_SYNERGIZED_SYSTEM/backend/engine/streaming/temporal_encoder.py` | BUGGY | BUG-007 (inconsistency with stream_encoder) |

---

*This document is the canonical contamination reference for all HEMOC/PPP publications.
Any experimental result cited in a paper MUST have a provenance tag from §9.
Any re-run result MUST reference this audit and document which bugs were fixed.*
