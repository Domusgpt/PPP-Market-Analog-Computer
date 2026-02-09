# Coding Agent Prompt: PPP Synergized System

> **For AI coding agents working on this repository.** This document describes the
> mathematical architecture, physical design, and implementation constraints that
> govern every module in the Synergized System. Read this fully before writing code.

---

## 0. Ontological Premise

This is **not** a standard web application. It is a computational physics system whose
architecture is dictated by the geometry of regular 4-polytopes. Every module, data
channel, layer pair, rotation plane, and scaling factor exists because a specific
mathematical structure requires it. If a design choice looks arbitrary, you have not
yet understood the geometry.

The system is an **Optical Kirigami Moiré Encoder** — a physical analog computer
whose computation emerges from:
1. The interference of light through layered kirigami sheets (moiré patterns)
2. The tristable mechanics of cut/fold lattice cells
3. The reservoir computing capacity of the coupled sheet dynamics
4. The 4D geometric algebra that governs rotation, projection, and decomposition

---

## 1. The 24-Cell: The Fundamental Module

### 1.1 Why the 24-Cell Controls Everything

The 24-cell (icositetrachoron) is the **unique** regular 4-polytope with no analogue
in any other dimension. It has:

| Property | Value | Consequence |
|----------|-------|-------------|
| Vertices | 24 | = 24 unit Hurwitz quaternions |
| Edges | 96 | Each vertex has exactly 8 neighbors |
| Faces | 96 triangles | |
| Cells | 24 octahedra | Self-dual: vertices ↔ cells |
| Symmetry group | F₄ (order 1152) | |
| Lattice | D₄ | Densest 4D sphere packing |

**Vertex coordinates:** All permutations of **(±1, ±1, 0, 0)** — that is, choose 2
of 4 axes, assign ±1 to each. This gives C(4,2) × 2² = 24 vertices.

**Self-duality** means the 24-cell is its own dual: swap vertices and cells and you
get the same polytope. This is the topological foundation of the system's symmetry.

### 1.2 The Trinity Decomposition (Three 16-Cells)

The 24 vertices partition into **exactly three disjoint 16-cells** (cross-polytopes,
the 4D octahedron). This is the **trilatic decomposition**, governed by the subgroup
inclusion W(D₄) ⊂ W(F₄) with index 3.

| Channel | Axis Pairs | Vertices | Color Code |
|---------|------------|----------|------------|
| **Alpha** | (X,Y) and (Z,W) | (±1,±1,0,0) and (0,0,±1,±1) | Red/Cyan |
| **Beta** | (X,Z) and (Y,W) | (±1,0,±1,0) and (0,±1,0,±1) | Green/Magenta |
| **Gamma** | (X,W) and (Y,Z) | (±1,0,0,±1) and (0,±1,±1,0) | Blue/Yellow |

Each 16-cell has 8 vertices. 3 × 8 = 24. ✓

**This decomposition is not optional.** It determines:
- The **3 data channels** (Alpha/Beta/Gamma) in the physics engine
- The **3 layer pairs** (6 physical kirigami layers = 3 pairs × 2 layers)
- The **3 rotation plane pairs** for quaternion control
- The **3 independent 16-cell computational cores**

The correspondence between layer pairs and rotation planes:

| Layer Pair | Planes | Channel | Kirigami Layers |
|------------|--------|---------|-----------------|
| Pair 1 | XY, ZW | Alpha | Layer 1 (Cyan), Layer 2 (Magenta) |
| Pair 2 | XZ, YW | Beta | Layer 3 (Cyan), Layer 4 (Magenta) |
| Pair 3 | XW, YZ | Gamma | Layer 5 (Cyan), Layer 6 (Magenta) |

### 1.3 The Six Rotation Planes

In 4D, rotation happens in a **plane**, not around an axis. There are exactly
**six orthogonal rotation planes** in ℝ⁴:

```
XY (e₁₂)   XZ (e₁₃)   XW (e₁₄)
YZ (e₂₃)   YW (e₂₄)   ZW (e₃₄)
```

These six planes group into the three complementary pairs above. Each pair
corresponds to one Trinity channel. A rotation in one plane of a pair is
coupled to the complementary plane — this is the origin of **isoclinic
(Clifford) rotation**, where both planes rotate by equal angles simultaneously.

### 1.4 Quaternion Control of Layers

Any 4D rotation decomposes as:

```
R(p) = q_L · p · q_R†
```

where q_L and q_R are the **left** and **right** rotation quaternions. This
decomposition maps directly to the "2 directions per layer" actuation
requirement of the kirigami stack. The left quaternion controls one rotation
plane in a pair; the right quaternion controls the complementary plane.

The 24 vertices of the 24-cell **are** the 24 unit Hurwitz quaternions:
```
±1, ±i, ±j, ±k                         (8 units)
(±1 ± i ± j ± k) / 2                   (16 half-integer quaternions)
```

This is not a metaphor. The vertices literally form a multiplicative group
under quaternion multiplication, and every rotation in the system is a
conjugation by one of these elements.

---

## 2. The 600-Cell: The Constellation Target

### 2.1 From 24-Cell to 600-Cell

The 600-cell (hexacosichoron) is the most complex regular 4-polytope:

| Property | Value |
|----------|-------|
| Vertices | 120 (= binary icosahedral group 2I) |
| Edges | 720 |
| Faces | 1200 triangles |
| Cells | 600 tetrahedra |
| Symmetry group | H₄ (order 14,400) |

The critical relationship: **120 = 5 × 24**. The 120 vertices of the 600-cell
can be partitioned into **five disjoint 24-cells**. This is the architectural
basis of the **H4 Constellation**.

### 2.2 Five Disjoint 24-Cells

The five 24-cells are related by **golden ratio rotations of 72° (= 360°/5)**
in specific 4D planes. The golden ratio φ = (1+√5)/2 is intrinsic to H₄
symmetry.

```
ConstellationNetwork: 5 nodes
├── CENTER (identity rotation)
├── NORTH  (72° golden isoclinic rotation)
├── EAST   (144° golden isoclinic rotation)
├── SOUTH  (216° golden isoclinic rotation)
└── WEST   (288° golden isoclinic rotation)
```

Each node is a physical **24-cell prototype unit** containing:
- A kirigami stack (6 layers = 3 pairs)
- A quaternion controller
- 6 vertex ports for inter-module communication

When five units are coupled, their combined 5 × 24 = 120 vertices form the
complete 600-cell. Each unit contributes one of the five disjoint 24-cells.

### 2.3 The 25 Inscribed 24-Cells

Beyond the 5-partition, the 600-cell contains exactly **25 inscribed 24-cells**
(Denney et al. 2020). These 25 cells arrange in a 5×5 array, and each vertex
of the 600-cell belongs to exactly 5 of them:

```
25 × 24 vertices / 5 membership per vertex = 120 unique vertices ✓
```

The 25 inscribed 24-cells admit **10 distinct partitions** into 5 disjoint
24-cells each. The constellation architecture uses one such partition, but
the full structure of 25 inscribed cells governs the complete vertex-sharing
topology.

### 2.4 Vertex Construction

The 120 vertices of the 600-cell fall into three classes:

| Class | Count | Coordinates |
|-------|-------|-------------|
| Axis vertices | 8 | Permutations of (±1, 0, 0, 0) |
| Half-integer | 16 | (±½, ±½, ±½, ±½) |
| Golden vertices | 96 | Even permutations of (0, ±½, ±φ/2, ±1/(2φ)) |

The first 24 (8 + 16) form a 24-cell. The remaining 96 golden-ratio vertices
complete the 600-cell — and this is why φ pervades every aspect of the system.

---

## 3. The 120-Cell: The Dual and the Palindrome

### 3.1 Duality

The 120-cell (hecatonicosachoron) is the **dual** of the 600-cell:

| Property | 600-cell | 120-cell |
|----------|----------|----------|
| Vertices | 120 | 600 |
| Edges | 720 | 1200 |
| Faces | 1200 triangles | 720 pentagons |
| Cells | 600 tetrahedra | 120 dodecahedra |

Duality swaps vertices ↔ cells and edges ↔ faces. The 120-cell's vertices are
the cell-centers of the 600-cell.

### 3.2 The H4 Palindrome

The system supports a palindromic transformation sequence:

```
24-cell → 600-cell → 120-cell (dual) → 24-cell
```

This is the **deployment cycle**:
1. **LOCKED (State 0):** Compact 24-cell. All kirigami layers flat.
2. **AUXETIC (State ½):** Bistable transition. Negative Poisson's ratio expansion.
3. **DEPLOYED (State 1):** Full 600-cell projection. Constellation active.

The return path (contraction) reverses through the dual 120-cell back to the
24-cell, completing the palindrome.

---

## 4. The E₈ → H₄ Dimensional Cascade

### 4.1 E₈ Root System

The E₈ lattice in 8 dimensions has **240 roots** (minimal nonzero vectors):
- **112 permutation roots:** All permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
- **128 half-integer roots:** (±½)⁸ with an even number of minus signs

### 4.2 Baez Projection

The 4×8 **Baez projection matrix** maps E₈ roots from 8D to 4D using the
icosian representation and the golden ratio:

```
P = (1/2) × [1   φ   0  -1   φ   0   0   0]
             [φ   0   1   φ   0  -1   0   0]
             [0   1   φ   0  -1   φ   0   0]
             [0   0   0   0   0   0   1   φ]
```

### 4.3 Galois Conjugation and the Moiré Layer

Applying **Galois conjugation** (φ ↔ -1/φ) to the projection matrix yields a
**conjugate projection**. The 240 E₈ roots project to two concentric 600-cells:

- **Outer 600-cell:** φ-scaled (120 vertices)
- **Inner 600-cell:** (1/φ)-scaled (120 vertices)

The interference between these two nested 600-cells at different φ-scales
creates the **moiré layer effect** — the same mathematical structure that
produces physical moiré fringes in the kirigami sheets.

```
E₈ (240 roots, 8D)
    │
    ├── Baez projection ──→ Outer 600-cell (scale φ)
    │                          └── 5 × 24-cell partition
    │                               └── 3 × 16-cell Trinity per 24-cell
    │
    └── Conjugate projection ──→ Inner 600-cell (scale 1/φ)
                                    └── Same nested structure
                                         └── Moiré = Outer ⊗ Inner
```

### 4.4 φ-Nested Scaling

The golden ratio generates a self-similar nesting:
- φ² = φ + 1
- φⁿ = F(n)φ + F(n-1) where F(n) is the nth Fibonacci number
- 1/φ = φ - 1

Polytopes at successive φ-power scales create fractal-like interference at
every level of the hierarchy.

---

## 5. The Physical Implementation

### 5.1 Kirigami Sheet

Each physical sheet is a hexagonal lattice of **tristable cells**. Each cell
has three stable states governed by a triple-well potential:

```
U(x) = k · [x(x - 0.5)(x - 1)]²
```

| State | Value | Mechanical | Optical |
|-------|-------|------------|---------|
| Closed | 0 | Flat, no fold | Fully transmitting |
| Intermediate | 0.5 | Half-fold, bistable | Partial blocking |
| Open | 1 | Full fold | Fully blocking |

The three states mirror the three vertices of the **deployment state machine**
(LOCKED → AUXETIC → DEPLOYED).

### 5.2 Six-Layer Kirigami Stack

The prototype uses **six physical kirigami layers** arranged as three
complementary pairs (Cyan/Magenta per pair):

```
Layer 1 (Cyan)    ─┐ Pair 1 → Alpha channel (XY/ZW planes)
Layer 2 (Magenta) ─┘
Layer 3 (Cyan)    ─┐ Pair 2 → Beta channel (XZ/YW planes)
Layer 4 (Magenta) ─┘
Layer 5 (Cyan)    ─┐ Pair 3 → Gamma channel (XW/YZ planes)
Layer 6 (Magenta) ─┘
```

Each layer pair is controlled by one of the three Trinity channels. The
relative displacement between Cyan and Magenta layers within a pair produces
the moiré interference pattern for that channel.

### 5.3 Moiré Interference

When two periodic lattices are superimposed at a twist angle θ, a **moiré
superlattice** emerges with period:

```
L_M = a / (2 · sin(θ/2))
```

where `a` is the lattice constant. The transmission function for the combined
bilayer grating is:

```
T(x,y) = (1/3) Σᵢ cos(Gᵢ · r)
```

where Gᵢ are the reciprocal lattice vectors rotated by θ. This is a
**multiplicative** combination — the moiré pattern is the product of the two
layer transmissions.

### 5.4 Talbot Resonator

The Talbot effect produces self-imaging at integer and half-integer multiples
of the Talbot distance:

```
z_T = 2a² / λ
```

| Position | Pattern | Logic Gate |
|----------|---------|-----------|
| z = n·z_T (integer) | Original image | AND / OR |
| z = (n+½)·z_T (half) | Phase-shifted | NAND / XOR |

This creates **optical logic gates** from interference alone, without
electronic components.

### 5.5 Reservoir Computing

The coupled kirigami sheet dynamics form an **Echo State Network (ESN)**
reservoir computer:

- **Input:** Moiré pattern from each layer pair (3 channels)
- **Reservoir:** The ~200+ tristable cells with nearest-neighbor coupling
- **Readout:** Linear combination of cell states

Key metrics:
- **Memory Capacity:** How many past inputs the reservoir retains
- **Lyapunov Exponent:** Edge-of-chaos dynamics (optimal at λ ≈ 1.0)
- **Shannon Entropy:** Information content of the reservoir state

### 5.6 Feature Extraction Pipeline

```
Raw Frame
    ├── LPQ (Local Phase Quantization)  → texture features
    ├── Gabor filters                    → orientation/frequency
    ├── HOG (Histogram of Oriented Gradients) → shape features
    ├── Hu Moments (7 invariants)        → geometric moments
    ├── Spectral (FFT + SVD)             → frequency decomposition
    └── Wavelet (multi-scale)            → scale-space features
```

All features are extracted per-frame from the moiré interference pattern
and fed to the reservoir.

---

## 6. The Enforcer: Three Rule Sets

The `enforcer.py` module validates every computation against three
constraint sets derived from the physics:

### Rule Set 1: Moiré Integrity
- Fringe contrast must be in [0, 1] (Michelson contrast)
- Moiré period must satisfy L_M = a / (2·sin(θ/2))
- Amplification factor must be > 1 for valid superlattice

### Rule Set 2: Kirigami Mechanics
- Cell states must be in {0, 0.5, 1} (tristable only)
- Layer pair coupling must respect Trinity channel assignment
- Deployment state transitions must follow LOCKED → AUXETIC → DEPLOYED

### Rule Set 3: Reservoir Dynamics
- Spectral radius must be near 1.0 (edge of chaos)
- Memory capacity must be positive
- Entropy must be within [0, log₂(N)] bounds

**Never bypass the enforcer.** If a computation violates these rules, the
physics is wrong, not the rules.

---

## 7. Data Flow Architecture

```
Raw Input (audio/market/sensor)
    │
    ▼
┌─────────────────────────────────────────┐
│  Backend: Physics Reactor (Python)      │
│                                         │
│  enforcer.validate() ← Three rule sets  │
│         │                               │
│  Kirigami Sheet (tristable cells)       │
│         │                               │
│  Moiré Interference (bilayer gratings)  │
│         │                               │
│  Feature Extraction (LPQ/Gabor/HOG/...) │
│         │                               │
│  Reservoir (ESN + criticality control)  │
│         │                               │
│  Telemetry: 12-channel JSON per frame   │
└────────┬────────────────────────────────┘
         │ WebSocket ws://localhost:8765
         ▼
┌─────────────────────────────────────────┐
│  HemocPythonBridge.ts                   │
│  Implements PPPAdapter interface        │
│  Maps physics → 12-channel RawApiTick   │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Frontend: PPP Control Room (TypeScript)│
│                                         │
│  TimeBinder → phase-locked sync         │
│  GeometricLerp → SLERP interpolation    │
│  StereoscopicFeed → left/right bifurc.  │
│  Renderer → Canvas/WebGL visualization  │
└─────────────────────────────────────────┘
```

### 7.1 The 12-Channel Layout

| Index | Signal | Source | Range |
|-------|--------|--------|-------|
| 0 | moire_contrast | Fringe visibility (Michelson) | 0.0–1.0 |
| 1 | moire_frequency | Dominant spatial frequency | 0.0–1.0 |
| 2 | lattice_stress | Frobenius norm of stress tensor | 0.0–1.0 |
| 3 | reservoir_entropy | Shannon entropy (normalized) | 0.0–1.0 |
| 4 | reservoir_lyapunov | Edge-of-chaos metric | 0.0–1.0 |
| 5 | talbot_gap | Gap distance (normalized) | 0.0–1.0 |
| 6 | petal_rotation_mean | Mean petal angular state | 0.0–1.0 |
| 7 | cell_flat_fraction | Fraction of cells in state 0 | 0.0–1.0 |
| 8 | cell_half_fraction | Fraction of cells in state 0.5 | 0.0–1.0 |
| 9 | cell_full_fraction | Fraction of cells in state 1 | 0.0–1.0 |
| 10 | memory_capacity | Reservoir memory metric | 0.0–1.0 |
| 11 | logic_polarity | 0 = positive, 1 = negative | 0 or 1 |

**Constraint:** channels 7 + 8 + 9 must sum to 1.0 (all cells accounted for).

---

## 8. TypeScript/Math Core Architecture

### 8.1 Geometric Algebra — Cl(4,0)

The math core implements Clifford algebra Cl(4,0) with:
- **Vectors:** 4 basis vectors e₁, e₂, e₃, e₄
- **Bivectors:** 6 basis bivectors e₁₂, e₁₃, e₁₄, e₂₃, e₂₄, e₃₄
- **Rotors:** Even-grade elements R = scalar + bivector (generalizes quaternions)
- **Sandwich product:** R · v · R̃ rotates vector v by rotor R

### 8.2 Key Classes and Their Polytope Basis

| Class | File | Polytope | Purpose |
|-------|------|----------|---------|
| `Lattice24` | `CPE_Lattice24.ts` | 24-cell | Trinity decomposition, coherence, phase shifts |
| `Cell600` | `Cell600.ts` | 600-cell | 25 inscribed 24-cells, edge computation |
| `E8ProjectionPipeline` | `E8Projection.ts` | E₈→H₄ | Baez matrix, Galois conjugation, nested 600-cells |
| `GoldenRatioScaler` | `GoldenRatioScaling.ts` | φ-nesting | Moiré pattern detection between φ-scaled layers |
| `TrinityEngine` | `TrinityEngine.ts` | 3×16-cell | State vector Ψ = [w_α, w_β, w_γ] superposition |
| `ChronomorphicEngine` | `ChronomorphicEngine.ts` | All | Master coordinator for Trinity + Causal engines |

### 8.3 Key Python Classes and Their Polytope Basis

| Class | File | Polytope | Purpose |
|-------|------|----------|---------|
| `Polytope24Cell` | `h4_geometry.py` | 24-cell | Base module, vertex generation, layer mapping |
| `Polytope16Cell` | `h4_geometry.py` | 16-cell | Single Trinity channel (8 vertices) |
| `TrilaticDecomposition` | `h4_geometry.py` | 3×16-cell | Three-color decomposition of 24-cell |
| `Polytope600Cell` | `h4_geometry.py` | 600-cell | Target structure, 5 embedded 24-cells |
| `Polytope120Cell` | `h4_geometry.py` | 120-cell | Dual of 600-cell |
| `ConstellationNetwork` | `h4_constellation.py` | 5×24-cell | Five-node distributed 600-cell assembly |
| `ConstellationNode` | `h4_constellation.py` | 24-cell | Single physical prototype unit |
| `PhasonPropagator` | `h4_constellation.py` | 600-cell | Strain wave propagation between nodes |
| `Quaternion4D` | `quaternion_4d.py` | S³ | Unit quaternion rotation control |
| `IsoclinicRotation` | `quaternion_4d.py` | Clifford | Equal-angle rotation in two planes |
| `H4Geometry` | `h4_geometry.py` | All | Main geometry interface |

---

## 9. Hierarchy of Decompositions (The Master Key)

This is the complete decomposition hierarchy. Every structure in the system
exists at one of these levels:

```
E₈ root system (240 roots in 8D)
│
├── Baez projection (φ-scaled)
│   └── Outer 600-cell (120 vertices, scale φ)
│       ├── Partition into 5 disjoint 24-cells [Constellation]
│       │   ├── 24-cell₁ (CENTER) → 3 × 16-cell [Trinity α,β,γ]
│       │   │   ├── 16-cell_α (8 vertices) → Layer Pair 1 (XY/ZW)
│       │   │   ├── 16-cell_β (8 vertices) → Layer Pair 2 (XZ/YW)
│       │   │   └── 16-cell_γ (8 vertices) → Layer Pair 3 (XW/YZ)
│       │   ├── 24-cell₂ (NORTH) → 3 × 16-cell [Trinity α,β,γ]
│       │   ├── 24-cell₃ (EAST)  → 3 × 16-cell [Trinity α,β,γ]
│       │   ├── 24-cell₄ (SOUTH) → 3 × 16-cell [Trinity α,β,γ]
│       │   └── 24-cell₅ (WEST)  → 3 × 16-cell [Trinity α,β,γ]
│       │
│       └── 25 inscribed 24-cells (10 partitions of 5)
│           └── Each vertex belongs to exactly 5 of the 25
│
├── Galois conjugation (1/φ-scaled)
│   └── Inner 600-cell (120 vertices, scale 1/φ)
│       └── Same nested structure as outer
│
└── Outer ⊗ Inner = Moiré interference layer
    └── Constructive/destructive interference at near-coincidence points
```

### 9.1 Counting Verification

| Level | Count | Formula |
|-------|-------|---------|
| E₈ roots | 240 | 112 perm + 128 half-int |
| 600-cell vertices | 120 | |2I| (binary icosahedral group) |
| Disjoint 24-cells per 600-cell | 5 | 120/24 |
| Inscribed 24-cells per 600-cell | 25 | 5×5 array |
| 16-cells per 24-cell | 3 | W(D₄) ⊂ W(F₄) index 3 |
| Vertices per 16-cell | 8 | 4D cross-polytope |
| Layer pairs per 24-cell | 3 | One per Trinity channel |
| Rotation planes per pair | 2 | Complementary planes |
| Total rotation planes | 6 | C(4,2) = 6 |
| Kirigami layers per unit | 6 | 3 pairs × 2 layers |
| Cell states | 3 | {0, 0.5, 1} tristable |

Every number in this table is mathematically determined, not a design choice.

---

## 10. Musical Mapping (Calibration Domain)

The 24-cell also serves as a **musical coordinate system**:

| Musical Element | 24-Cell Structure | Quantity |
|----------------|-------------------|----------|
| Major + minor keys | Vertices | 24 (12 major + 12 minor) |
| Diatonic modes | Octahedral cells | 24 |
| Interval relationships | Edges | 96 |
| Major/minor duality | Self-duality | Vertices ↔ Cells |
| Circle of fifths | 72° rotation in XY plane | 360°/5 steps |
| Diminished 7th chord | Regular tetrahedron | Td symmetry |

This mapping provides **human-perceptible validation** of the geometric
computation: if the geometry is correct, the music sounds consonant.

---

## 11. Implementation Constraints for Coding Agents

### 11.1 Structural Invariants (NEVER violate)

1. **Trinity channels are always three.** Alpha, Beta, Gamma. Never add a fourth.
2. **16-cells have exactly 8 vertices.** Never 7, never 9.
3. **24-cells have exactly 24 vertices.** 3 × 8 = 24.
4. **600-cells have exactly 120 vertices.** 5 × 24 = 120.
5. **Layer pairs are always three.** Six layers = three pairs.
6. **Cell states are tristable.** {0, 0.5, 1} only.
7. **Cell fractions sum to 1.** flat + half + full = 1.0.
8. **Golden ratio is exact.** φ = (1+√5)/2. Never approximate as 1.6 or 1.62.
9. **Edge length of 24-cell is √2.** Not 1, not 2.
10. **Moiré is multiplicative.** Product of layer transmissions, not sum.

### 11.2 Naming Conventions

- **Trinity channels:** `alpha`, `beta`, `gamma` (lowercase in code)
- **Deployment states:** `LOCKED`, `AUXETIC`, `DEPLOYED` (uppercase enum)
- **Rotation planes:** `XY`, `ZW`, `XZ`, `YW`, `XW`, `YZ` (uppercase pairs)
- **Layer planes:** `LayerPlane.XY` etc. (enum values)
- **Constellation positions:** `CENTER`, `NORTH`, `EAST`, `SOUTH`, `WEST`
- **Polytope types:** `Polytope24Cell`, `Polytope16Cell`, `Cell600`, etc.

### 11.3 Testing Invariants

When writing tests, verify:
- 24-cell vertex generation produces exactly 24 vertices
- Trinity decomposition yields three groups of exactly 8
- All vertices are equidistant from origin (lie on 3-sphere)
- Neighbor count per vertex is 8 for 24-cell, 12 for 600-cell
- φ² = φ + 1 (to machine precision)
- Quaternion norm is preserved after all rotations
- Moiré period matches L_M = a / (2·sin(θ/2))
- Enforcer rules pass for all valid configurations

### 11.4 File Location Rules

| Module Type | Location | Language |
|-------------|----------|----------|
| Polytope geometry | `_SYNERGIZED_SYSTEM/lib/math_core/topology/` | TypeScript |
| Geometric algebra | `_SYNERGIZED_SYSTEM/lib/math_core/geometric_algebra/` | TypeScript |
| Physics engine | `_SYNERGIZED_SYSTEM/backend/engine/` | Python |
| Kirigami mechanics | `_SYNERGIZED_SYSTEM/backend/engine/kirigami/` | Python |
| Reservoir computing | `_SYNERGIZED_SYSTEM/backend/engine/reservoir/` | Python |
| Feature extraction | `_SYNERGIZED_SYSTEM/backend/engine/features/` | Python |
| Frontend UI | `_SYNERGIZED_SYSTEM/frontend/platform/` | TypeScript/JS |
| Bridge adapter | `frontend/platform/src/lib/adapters/HemocPythonBridge.ts` | TypeScript |
| Demos/visualization | `_SYNERGIZED_SYSTEM/demos/` | Python + TypeScript |
| Standalone dashboard | `synergized-system.html` | HTML/JS |

### 11.5 Common Mistakes to Avoid

1. **Confusing the two 24-cell vertex sets.** The "standard" 24-cell uses
   permutations of (±1, ±1, 0, 0). The Hurwitz quaternion form uses
   (±1, 0, 0, 0) + (±½, ±½, ±½, ±½). These are **different** 24-cells
   (dual to each other). The codebase uses the (±1, ±1, 0, 0) form for
   the Trinity decomposition and the Hurwitz form for the 600-cell embedding.

2. **Treating 4D rotation as 3D rotation.** In 3D, rotation is around an
   axis. In 4D, rotation is **in a plane**. There are 6 independent rotation
   planes, not 3 axes. The 6 layers directly encode these 6 planes.

3. **Ignoring the golden ratio.** If φ doesn't appear in your 600-cell code,
   something is wrong. The golden ratio is not optional — it is the ratio
   between the 600-cell edge length and circumradius.

4. **Making the moiré additive.** Moiré interference is the **product** of
   transmission functions, not the sum. T_total = T₁ · T₂, not T₁ + T₂.

5. **Using random values for cell states.** Cell states are {0, 0.5, 1} only.
   They are tristable, not continuous. The potential U(x) has three minima.

6. **Breaking the channel↔layer pair correspondence.** Alpha is ALWAYS
   layers 1-2 (XY/ZW). Beta is ALWAYS layers 3-4 (XZ/YW). Gamma is ALWAYS
   layers 5-6 (XW/YZ). This mapping comes from the Trinity decomposition
   and cannot be permuted.

---

## 12. Repository Structure

```
PPP-Market-Analog-Computer/
├── _SYNERGIZED_SYSTEM/           # Grand unified system (5 repos merged)
│   ├── backend/engine/           # Python physics reactor
│   │   ├── geometry/             # H4 polytopes, quaternion 4D
│   │   ├── constellation/        # 5-node 600-cell assembly
│   │   ├── kirigami/             # Tristable cells, deployment states
│   │   ├── physics/              # Moiré, Talbot, trilatic lattice
│   │   ├── reservoir/            # ESN, criticality, readout
│   │   ├── features/             # LPQ, Gabor, HOG, spectral, wavelet
│   │   ├── control/              # Tripole actuator (tip/tilt/piston)
│   │   ├── streaming/            # Audio + temporal encoding
│   │   ├── hdc/                  # Hyperdimensional computing
│   │   ├── ml/                   # Auto-tune, PyTorch
│   │   ├── core/                 # Batch encoder, fast cascade
│   │   ├── telemetry/            # Logging, metrics, profiling
│   │   ├── pipeline.py           # OpticalKirigamiMoire pipeline
│   │   ├── enforcer.py           # Three rule-set enforcement
│   │   ├── main.py               # Entry point
│   │   └── websocket_server.py   # WebSocket bridge to frontend
│   ├── frontend/platform/        # TypeScript PPP spine
│   │   ├── src/lib/temporal/     # TimeBinder, GeometricLerp
│   │   ├── src/lib/fusion/       # StereoscopicFeed, DataPrism
│   │   ├── src/lib/adapters/     # HemocPythonBridge, MarketQuote
│   │   ├── src/lib/contracts/    # PPPAdapter, PPPCoreConfig
│   │   ├── scripts/              # JS runtime (app, Sonic, Spinor, etc.)
│   │   └── tests/                # Adapter, phase-lock, calibration tests
│   ├── lib/math_core/            # Pure math library
│   │   ├── geometric_algebra/    # Cl(4,0), Lattice24, CausalReasoning
│   │   ├── topology/             # Cell600, E8Projection, GoldenRatio
│   │   ├── engine/               # TrinityEngine, ChronomorphicEngine
│   │   └── tda/                  # PersistentHomology, GhostFrequency
│   ├── demos/                    # Visualization demos
│   │   ├── python_demos/         # Full pipeline demo
│   │   └── visualization/        # E8Renderer, MoireOverlay
│   ├── system_manifest.md        # Complete source mapping
│   └── docker-compose.yml        # Orchestration
├── src/lib/                      # Original PPP library
├── scripts/                      # Original PPP JS scripts
├── synergized-system.html        # Standalone interactive dashboard
├── index.html                    # Phase-lock visualization
├── MusicGeometryDomain-Design*.md # Musical mapping design doc
├── CODING_AGENT_PROMPT.md        # THIS FILE
└── DEV_TRACK.md                  # Development tracking
```

---

## 13. Quick Reference: The Numbers That Matter

| Constant | Value | Where It Appears |
|----------|-------|-----------------|
| φ (golden ratio) | (1+√5)/2 ≈ 1.618034 | 600-cell edges, E₈ projection, moiré scaling |
| 1/φ | φ-1 ≈ 0.618034 | Inner 600-cell scale, edge length |
| √2 | ≈ 1.414214 | 24-cell edge length, circumradius |
| 24 | — | Vertices of 24-cell, Hurwitz quaternions |
| 120 | — | Vertices of 600-cell, |2I| |
| 240 | — | E₈ roots |
| 3 | — | Trinity channels, layer pairs, 16-cells per 24-cell |
| 5 | — | Disjoint 24-cells per 600-cell, nodes per constellation |
| 6 | — | Rotation planes in 4D, kirigami layers |
| 8 | — | Vertices per 16-cell, neighbors per vertex in 24-cell |
| 25 | — | Inscribed 24-cells in 600-cell |
| 72° | 2π/5 | Golden rotation angle between constellation nodes |
| 14,400 | — | Order of H₄ symmetry group |
| 1,152 | — | Order of F₄ symmetry group (24-cell) |

---

*When in doubt, check the geometry. The math is always right.*
