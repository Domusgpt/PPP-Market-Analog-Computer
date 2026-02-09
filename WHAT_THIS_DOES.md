# What This System Actually Does

## A Complete Technical Explanation of the PPP Synergized System

---

## The One-Sentence Answer

This system is a **software simulation of a physical analog computer** that performs
computation using light interference through stacked, cut-and-folded paper sheets
(kirigami), where the geometry of the cutting patterns, the angles between layers,
and the gaps between sheets are all governed by the mathematics of four-dimensional
polytopes and the golden ratio.

---

## Part I: The Physical Device Being Simulated

### What is an Optical Kirigami Moiré Encoder?

Imagine you have two transparent sheets, each printed with a fine hexagonal grid
pattern. If you lay one on top of the other, perfectly aligned, you see just the
grid. But if you **twist** one sheet by a small angle relative to the other, a new,
much larger pattern appears — a **moiré pattern**. This pattern has properties
(period, contrast, orientation) that depend on the twist angle and the grid spacing
in a precisely predictable way.

Now imagine instead of printed grids, each sheet is a **kirigami** — paper with a
pattern of cuts that allows it to fold in specific ways. Each small hexagonal cell
in the lattice can be in one of exactly three mechanical states:

| State | Name | What the cell looks like | What light does |
|-------|------|--------------------------|-----------------|
| 0 | **Closed** | Flat, unfolded | Light passes through freely |
| 0.5 | **Intermediate** | Half-folded (bistable point) | Partially blocks light |
| 1 | **Open** | Fully folded open | Fully blocks light |

These three states are **tristable** — each is a local energy minimum. The cell
naturally rests in one of these three positions and requires a deliberate push to
transition between them. This is governed by a triple-well potential energy function:

```
U(x) = k · [x · (x - 0.5) · (x - 1)]²
```

This function has exactly three minima at x = 0, x = 0.5, and x = 1 — the three
stable states. The barriers between them determine how much force is needed to switch
a cell.

### The Six-Layer Stack

The physical device uses **six kirigami layers** stacked on top of each other, organized
as **three pairs**:

```
┌─────────────────────────────────────────────────┐
│  Layer 1 (Cyan filter)    ─┐                    │
│  Layer 2 (Magenta filter) ─┘ Pair 1 (Alpha)     │
│                                                  │
│  Layer 3 (Cyan filter)    ─┐                     │
│  Layer 4 (Magenta filter) ─┘ Pair 2 (Beta)      │
│                                                  │
│  Layer 5 (Cyan filter)    ─┐                     │
│  Layer 6 (Magenta filter) ─┘ Pair 3 (Gamma)     │
│                                                  │
│  ▼ Light passes through all 6 layers ▼          │
│  ▼ producing combined moiré pattern  ▼          │
└─────────────────────────────────────────────────┘
```

Within each pair, one layer has a Cyan color filter and the other has a Magenta
color filter. This is **bichromatic** (two-color) encoding: by using complementary
colors, the system can distinguish the contribution of each layer in the combined
output. When Cyan and Magenta overlap, they produce different results than either
alone — this is how the system implements optical logic.

Each pair of layers produces its own moiré interference pattern based on the relative
twist angle between the Cyan and Magenta layers in that pair. The three pairs
therefore produce **three independent moiré channels**.

### Why Three Pairs? The Trinity Decomposition

The three-pair structure is not arbitrary — it is dictated by the mathematics of the
**24-cell**, a four-dimensional geometric object (polytope). The 24-cell has 24
vertices which partition into exactly **three groups of 8**, called the **Trinity
decomposition**:

- **Alpha** (8 vertices): Corresponds to rotations in the XY and ZW planes → Layer Pair 1
- **Beta** (8 vertices): Corresponds to rotations in the XZ and YW planes → Layer Pair 2
- **Gamma** (8 vertices): Corresponds to rotations in the XW and YZ planes → Layer Pair 3

Each group of 8 vertices forms a **16-cell** (a regular 4D shape, the 4D equivalent
of an octahedron). The three 16-cells are completely separate — no vertex belongs to
more than one group. Together, 3 × 8 = 24 vertices account for the entire 24-cell.

This means each layer pair in the physical device corresponds to one "dimension" of
4D rotation. By controlling all three pairs independently, the device can represent
any rotation in four-dimensional space.

### The Twist Angle: Commensurate Locking

You cannot use just any twist angle between layers. The system enforces **Rule Set 1:
Angular Commensurability**. The angle must be a **Pythagorean commensurate angle** —
an angle where the twisted lattice aligns back with itself after a finite number of
steps. Mathematically, these are angles θ where:

```
tan(θ/2) = n / (m · √3)
```

for small integers m, n. Each valid (m, n) pair produces a specific "cognitive mode":

| (m, n) | Angle | Mode Name | What it does |
|--------|-------|-----------|--------------|
| (0, 0) | 0.00° | Transparent | No moiré — direct imaging |
| (5, 1) | 7.34° | Edge Detection | Highlights boundaries |
| (4, 1) | 9.43° | Fine Texture | Small-scale detail |
| (3, 1) | 13.17° | Intermediate | Balanced all-purpose |
| (2, 1) | 21.79° | Coarse Structure | Large-scale features |

The system snaps any requested angle to the nearest commensurate value. Non-commensurate
angles produce aperiodic moiré patterns that are useless for computation. There is also
a special "magic angle" (~1.1°) where the moiré period becomes extremely large.

### The Gap: Talbot Resonance

The vertical distance (gap) between the two layers in a pair is also locked to specific
values by **Rule Set 3: Talbot Distance**. When light passes through a periodic grating,
it recreates an exact copy of the grating pattern at specific distances downstream — this
is the **Talbot effect**. The Talbot distance is:

```
z_T = 2a² / λ
```

where `a` is the lattice spacing and `λ` is the light wavelength.

| Gap Distance | What You See | Logic Gate |
|-------------|--------------|-----------|
| z = 1·z_T (integer multiple) | Exact copy of grating | **AND / OR** (positive logic) |
| z = 1.5·z_T (half-integer) | Phase-shifted copy | **NAND / XOR** (negative logic) |

This is the critical insight: **by changing the gap between layers, the system switches
between different logic operations without any electronics**. An AND gate at one gap
becomes a NAND gate at a different gap. The system is literally computing with light.

### The Tilt: Attention Direction

**Rule Set 2: Trilatic Tilt Symmetry** governs the tilt of one layer relative to
another. The tilt axis must be a multiple of 60° to preserve the C₃ symmetry of the
hexagonal lattice. Tilting creates a gradient across the moiré pattern — brighter on
one side, darker on the other — which functions as a spatial **attention mechanism**.
The system can "focus" on different regions of the input by changing the tilt direction.

### The Tripole Actuator

All of this is controlled by a **tripole kinematic system**: three linear actuators
arranged in an equilateral triangle (at 0°, 120°, 240°) plus a rotary stage. This
maps directly to the four control parameters:

| Actuator Mode | What It Controls | Physics |
|--------------|-----------------|---------|
| **Piston** (all three move together) | Gap (Talbot logic) | z = (A+B+C)/3 |
| **Tip** (A vs average of B,C) | Horizontal attention | X-gradient |
| **Tilt** (B vs C) | Vertical attention | Y-gradient |
| **Rotate** (rotary stage) | Twist angle (moiré mode) | Commensurate snap |

---

## Part II: What Computation Looks Like

### The Encode-Compute-Readout Cycle

Here is what happens when data enters the system:

#### Step 1: Input Injection

Raw data (an image, an audio spectrogram, a market data frame, a sensor reading) is
resized to match the kirigami grid (e.g., 32×32 or 64×64 cells). Each value in the
input is mapped to a perturbation force on the corresponding kirigami cell. Cells that
receive a strong force may snap from one tristable state to another.

```
Input (32×32 matrix of values)
    ↓
Force applied to each kirigami cell
    ↓
Some cells snap: 0→0.5, 0.5→1, etc.
    ↓
New kirigami state = "encoded input"
```

#### Step 2: Cascading Dynamics (Reservoir Computing)

After the initial injection, the cells **interact with their neighbors**. A cell that
snaps can push adjacent cells over their threshold, causing them to snap too. This
creates a cascade that propagates through the lattice.

This is **reservoir computing**. The hexagonal lattice of ~200+ coupled tristable cells
functions as an **Echo State Network (ESN)** — a type of recurrent neural network where
the internal dynamics are rich and complex, but only the output layer (readout) needs to
be trained.

The key property is **fading memory**: the reservoir's state depends on recent inputs
more than distant ones. If you feed a sequence of inputs, the reservoir state at time t
is a nonlinear function of the entire history of inputs, with exponentially decaying
influence. This allows the system to perform **temporal computation** — pattern recognition
that depends on sequences, not just single frames.

The cascade runs for a configurable number of steps (typically 20-50). Two critical metrics
are monitored:

- **Lyapunov exponent** ≈ 1.0: The system operates at the "edge of chaos" — complex enough
  to compute, stable enough not to explode. Below 1.0, information decays too fast. Above
  1.0, the system becomes chaotic and unpredictable.
- **Memory capacity**: How many past inputs the reservoir effectively "remembers."

#### Step 3: Moiré Pattern Generation

After the cascade stabilizes, light is passed through the stack of kirigami layers. Each
pair (Cyan/Magenta) produces a moiré interference pattern. The combined transmission is:

```
T_pair(x,y) = T_cyan(x,y) × T_magenta(R_θ(x,y))
```

where R_θ is the rotation by the twist angle. This is **multiplicative** — the moiré
pattern is the product of the two transmission functions, not their sum. This is physically
correct: light must pass through *both* layers, so the combined transmission is the product
of individual transmissions.

The moiré pattern has spatial structure (fringes, defects, gradients) that encodes information
about the kirigami state, and therefore about the input data.

#### Step 4: Feature Extraction

The moiré pattern is analyzed using multiple computer vision techniques to extract a
feature vector:

| Technique | What It Extracts | Number of Features |
|-----------|-----------------|-------------------|
| **LPQ** (Local Phase Quantization) | Texture/phase patterns via short-time Fourier transform | Variable |
| **Gabor Filters** | Orientation and spatial frequency content (8 orientations × 4 scales) | ~192 |
| **HOG** (Histogram of Oriented Gradients) | Shape and edge structure | Variable |
| **Hu Moments** | 7 rotation-invariant geometric moments | 7 |
| **Spectral Analysis** (FFT + SVD) | Frequency decomposition, dominant frequencies | ~33 |
| **Wavelet Transform** | Multi-scale time-frequency analysis | Variable |

The LPQ decoder deserves special mention. It uses the Short-Time Fourier Transform to
extract **phase information** from the moiré pattern — and phase is what carries information
in interference patterns. The LPQ decoder can also detect **phason worm defects** —
topological defects in the quasi-crystalline moiré pattern that correspond to specific
features of the encoded data.

All features are concatenated into a single feature vector that represents the complete
computational output of one frame.

#### Step 5: Classification Readout

The feature vector is passed to a simple **linear readout layer** (ridge regression).
This is the only trained component in the system — everything else is fixed physics.
The readout maps the high-dimensional feature space to output classes or predictions.

```
Input → Kirigami injection → Cascade dynamics → Moiré pattern
    → Feature extraction → Linear readout → Classification/Prediction
```

This is the power of reservoir computing: the nonlinear dynamics of the kirigami cascade
do the heavy computational lifting. The readout just needs to find a linear boundary in
the rich feature space that the reservoir produces. Training only the readout is fast,
requires little data, and avoids the instabilities of training recurrent networks via
backpropagation.

---

## Part III: The 4D Geometric Control System

### Why 4D Geometry?

The physical device has six control degrees of freedom (6 layers, each with its twist
angle and gap), organized as three pairs, each controlling two complementary rotation
planes. This maps exactly to **rotation in four-dimensional space**, which has exactly
**six independent rotation planes**:

```
4D Rotation Planes:     Layer Pairs:
XY, ZW (complement)  →  Pair 1 (Alpha)
XZ, YW (complement)  →  Pair 2 (Beta)
XW, YZ (complement)  →  Pair 3 (Gamma)
```

In 3D, rotation happens around an **axis** (there are 3 independent axes).
In 4D, rotation happens in a **plane** (there are 6 independent planes, grouping into
3 complementary pairs). This is not an analogy — it is a direct mathematical correspondence.

The 4D rotation is represented as a **quaternion operation**:

```
R(p) = q_L · p · q_R†
```

The left quaternion q_L controls one rotation plane in a pair; the right quaternion q_R†
controls the complementary plane. This is exactly the "2 directions per layer" actuation
of the kirigami stack.

### The 24-Cell as the Fundamental Computational Unit

The 24-cell is the natural coordinate system for this device because:

1. Its 24 vertices are the 24 **unit Hurwitz quaternions** — every valid rotation state
   of the device corresponds to one of these quaternions or an interpolation between them.

2. Its **Trinity decomposition** into three 16-cells corresponds exactly to the three
   layer pairs and their rotation plane assignments.

3. Its **self-duality** (vertices ↔ cells) means the device has a natural symmetry that
   maps inputs to outputs without breaking structure.

4. Its **D₄ lattice** achieves the densest possible sphere packing in 4D — meaning the
   24-cell is the most efficient way to tile 4D space, which makes it the optimal
   discretization for the device's state space.

### The 600-Cell: Constellations of 24-Cells

A single physical device unit is a 24-cell. But five such units can be **connected**
(physically coupled via magnetic pogo-pin connectors at hexagonal frame vertices) to
form a **600-cell** — a far more complex 4D polytope with 120 vertices, 720 edges,
1200 faces, and 600 tetrahedral cells.

The five units are related by **golden ratio rotations of 72°** (= 360°/5). The golden
ratio φ = (1+√5)/2 ≈ 1.618 is intrinsic to the 600-cell's symmetry group H₄.

```
Constellation Layout:

         NORTH (72°)
           │
   WEST ───CENTER───EAST (144°)
 (288°)    │
         SOUTH (216°)

Each node = one 24-cell prototype unit (6 kirigami layers)
Combined = 5 × 24 = 120 vertices = one 600-cell
```

When the constellation is active:
- Each unit processes its own moiré patterns independently (3 channels each = 15 channels total)
- Units communicate quaternion state through vertex port connections
- **Phason strain waves** propagate between units: when one unit's state is perturbed,
  the strain travels to neighboring units at a configurable wave speed with damping
- The network **synchronizes** via iterative quaternion averaging until all units converge

This distributed system can process higher-dimensional data that a single 24-cell unit
cannot represent.

### The E₈ → H₄ Projection: Where the Geometry Comes From

The deepest mathematical layer is the connection to the **E₈ root system** — a structure
in 8-dimensional space with 240 minimal vectors (roots). E₈ is the largest exceptional
simple Lie group and appears in string theory, error-correcting codes, and sphere packing.

The E₈ roots can be **projected** from 8D down to 4D using the **Baez projection matrix**,
a 4×8 matrix whose entries involve the golden ratio φ. This projection maps 240 E₈ roots
to two concentric 600-cells in 4D:

- An **outer 600-cell** at scale φ (120 vertices)
- An **inner 600-cell** at scale 1/φ (120 vertices)

The two 600-cells are related by **Galois conjugation** — replacing φ with -1/φ throughout.
Their superposition creates a **moiré-like interference** in 4D, analogous to the physical
moiré patterns in the kirigami layers. This is the theoretical foundation that connects:

```
E₈ symmetry (8D, pure mathematics)
    ↓ Baez projection
Two nested 600-cells (4D, golden ratio scaled)
    ↓ 5-partition
Five 24-cells per 600-cell
    ↓ Trinity decomposition
Three 16-cells per 24-cell
    ↓ Physical mapping
Three layer pairs per device unit
    ↓ Moiré interference
Optical computation
```

Each level of this hierarchy is a specific mathematical decomposition, and each level
corresponds to a physical element of the device.

---

## Part IV: The Software Architecture

### Two Runtimes

The system is implemented in two languages:

| Layer | Language | Role |
|-------|----------|------|
| **Backend** (Physics Reactor) | Python | Kirigami mechanics, moiré physics, reservoir dynamics, feature extraction, rule enforcement |
| **Frontend** (Control Room) | TypeScript/JavaScript | 4D geometry, phase-locked display, stereoscopic rendering, user interaction |

They communicate via **WebSocket** (port 8765). The Python backend broadcasts a telemetry
frame at ~30 FPS as JSON. The TypeScript frontend receives these frames through a bridge
adapter.

### The Backend: What Python Computes

The Python backend (`_SYNERGIZED_SYSTEM/backend/engine/`) simulates the physical device:

#### 1. Rule Enforcement (`enforcer.py`)

Before any computation, the enforcer validates that all parameters are physically legal:

- **Rule Set 1 (Angular Commensurability):** Snaps the twist angle to the nearest
  Pythagorean commensurate value. Rejects non-commensurate angles.
- **Rule Set 2 (Trilatic Tilt Symmetry):** Locks the tilt axis to multiples of 60°
  to preserve hexagonal C₃ symmetry.
- **Rule Set 3 (Talbot Distance):** Locks the layer gap to integer or half-integer
  multiples of the Talbot distance z_T = 2a²/λ.

The enforcer is the gatekeeper. No computation proceeds with invalid parameters.

#### 2. Kirigami Mechanics (`kirigami/`)

- **`tristable_cell.py`**: Simulates individual cells. Each cell has a triple-well
  potential, a damped oscillator dynamic (Velocity Verlet integration), and coupling to
  its neighbors. Cells track their transmission (how much light passes through), their
  resonance frequency, and their optical chirality.

- **`h4_kirigami.py`**: Implements the full six-layer stack. Each layer is a 16×16 grid
  of cells. Each cell can be actuated with rotating-squares kirigami mechanics (dual-axis:
  longitudinal and transverse), with hierarchical cut patterns at three levels of stiffness.
  The `LayerPair` class computes moiré interference between its Cyan and Magenta layers.
  The `H4KirigamiStack` manages all three pairs and produces the combined spectral output.

#### 3. Moiré Physics (`physics/`)

- **`moire_interference.py`**: Computes moiré patterns analytically. Given a lattice
  constant `a` and twist angle θ, computes the moiré period L_M = a/(2·sin(θ/2)),
  generates 2D interference fields, evaluates bichromatic (Cyan/Red, 500nm/650nm)
  patterns, and implements spatial logic gates (AND, OR, XOR, NAND) based on the
  transmission product.

- **`talbot_resonator.py`**: Computes Talbot self-imaging distances for the hexagonal
  lattice. Given lattice constant and wavelength, finds integer and half-integer
  resonance gaps, generates the resonance ladder, and estimates diffraction blur.

- **`trilatic_lattice.py`**: Generates the hexagonal lattice geometry. Computes
  primitive and reciprocal lattice vectors, superlattice periods for commensurate
  angles, and trilatic wave vectors. Detects the magic angle (~1.1°).

#### 4. Reservoir Computing (`reservoir/`)

- **`esn.py`**: Implements the Echo State Network. Creates a random sparse reservoir
  with hexagonal-like connectivity, scales the spectral radius to just below 1.0 for
  edge-of-chaos dynamics, uses leaky integration for temporal processing, and trains
  a ridge regression readout layer.

- **`criticality.py`**: Analyzes the reservoir's dynamical regime. Estimates Lyapunov
  exponents from trajectory divergence, computes avalanche branching ratios, measures
  spatial correlation lengths, and classifies the regime as subcritical (ordered),
  critical (edge of chaos, optimal), or supercritical (chaotic).

#### 5. Feature Extraction (`features/`)

- **`lpq_decoder.py`**: Local Phase Quantization via STFT. Extracts phase from moiré
  patterns in multiple quantization modes (binary, ternary, multi-level). Includes a
  `PhaseShiftDetector` that identifies topological defects ("phason worms") in
  quasi-crystalline moiré patterns.

- **`gabor.py`**: Multi-scale, multi-orientation Gabor filter bank (default: 8
  orientations × 4 scales = 32 filters). Extracts texture features: mean/std per
  filter, cross-scale correlations.

- **`spectral.py`**: FFT-based spectral analysis. Computes radial power profile,
  angular distribution, spectral centroid/spread/flatness, and dominant frequency peaks.

- Additional extractors: HOG (shape), Hu moments (geometric invariants), wavelet
  (multi-scale).

#### 6. Hyperdimensional Computing (`hdc/`)

An alternative computational framework. Encodes patterns into very high-dimensional
binary vectors (e.g., 10,000 bits) using three operations:
- **Bind** (XOR): Associates two concepts
- **Bundle** (majority vote): Combines multiple vectors
- **Permute** (circular shift): Encodes sequence position

This enables one-shot learning and associative memory — recognizing patterns after
seeing them only once.

#### 7. Geometry (`geometry/`)

- **`h4_geometry.py`**: Implements 4D polytopes (16-cell, 24-cell, 600-cell, 120-cell)
  with vertex generation, edge computation, trilatic decomposition, stereographic
  projection to 3D, and deployment state management (LOCKED/AUXETIC/DEPLOYED).

- **`quaternion_4d.py`**: Unit quaternion algebra for 4D rotation. Implements quaternion
  multiplication, conjugation, axis-angle conversion, Euler angle conversion, SLERP
  interpolation, and — critically — **isoclinic (Clifford) rotation** where both planes
  of a complementary pair rotate by equal angles simultaneously.

#### 8. Constellation (`constellation/`)

- **`h4_constellation.py`**: Manages five 24-cell units forming a 600-cell. Each unit
  (ConstellationNode) has 6 hexagonal vertex ports for inter-module communication.
  The PhasonPropagator simulates elastic strain waves between coupled units. The
  H4ConstellationController provides the complete API for deployment, quaternion
  application, data stream processing, and the palindrome sequence (24-cell → 600-cell
  → 120-cell → 24-cell).

#### 9. Pipeline (`pipeline.py`)

The orchestrator that ties everything together:

```python
okm = OpticalKirigamiMoire(config)
okm.set_mode(ComputationMode.TEXTURE)  # Sets twist angle, gap, tilt
result = okm.encode(input_data)        # Full encode cycle:
                                       #   1. Inject into kirigami
                                       #   2. Run cascade dynamics
                                       #   3. Generate moiré pattern
                                       #   4. Extract features
                                       #   5. Return EncodingResult
```

#### 10. WebSocket Server (`websocket_server.py`)

Wraps the engine as an async WebSocket service. Broadcasts telemetry frames at 30 FPS.
Each frame contains all physics metrics as a JSON payload. Falls back to a `SyntheticEngine`
that generates plausible telemetry when the full physics engine is unavailable.

### The Frontend: What TypeScript Computes

The TypeScript frontend (`_SYNERGIZED_SYSTEM/frontend/platform/` and `lib/math_core/`)
handles 4D geometry, visualization, and phase-locked rendering.

#### 1. The PPP Adapter Contract (`AdapterContracts.ts`)

Every data source implements the `PPPAdapter` interface:
```typescript
interface PPPAdapter {
  connect(): Promise<void> | void;
  disconnect(): Promise<void> | void;
  onTick(callback: (tick: RawApiTick) => void): () => void;
  metrics?(): Record<string, number | string | boolean>;
}
```

A `RawApiTick` contains: price, volume, bid, ask, timestamp, and a `channels[]` array
of up to 12 normalized [0-1] values.

#### 2. The HemocPythonBridge (`HemocPythonBridge.ts`)

Connects to the Python WebSocket server and translates physics telemetry into the PPP
tick format:

| Channel | Signal | What It Means |
|---------|--------|---------------|
| 0 | moire_contrast | How visible the moiré fringes are (0=invisible, 1=maximum) |
| 1 | moire_frequency | Dominant spatial frequency of the pattern |
| 2 | lattice_stress | Internal stress in the kirigami lattice |
| 3 | reservoir_entropy | Information content of the reservoir state |
| 4 | reservoir_lyapunov | Distance from edge-of-chaos (optimal near 0.5) |
| 5 | talbot_gap | Normalized gap distance |
| 6 | petal_rotation_mean | Average angular state of kirigami petals |
| 7 | cell_flat_fraction | Fraction of cells in state 0 (closed) |
| 8 | cell_half_fraction | Fraction of cells in state 0.5 (intermediate) |
| 9 | cell_full_fraction | Fraction of cells in state 1 (open) |
| 10 | memory_capacity | Reservoir's fading memory metric |
| 11 | logic_polarity | 0 = positive logic (AND/OR), 1 = negative (NAND/XOR) |

Note: channels 7 + 8 + 9 always sum to 1.0 (every cell must be in exactly one state).

#### 3. The TimeBinder (`TimeBinder.ts`)

Solves the **phase-lock problem**: physics telemetry arrives at irregular intervals
(whenever the Python engine finishes a frame), but the display must render at a smooth
60+ FPS. The TimeBinder:

1. Stores incoming ticks in a **lock-free ring buffer** with O(1) insertion
2. Uses **binary search** (O(log n)) to find the two ticks that bracket the current
   display time
3. **Linearly interpolates** between them for smooth animation
4. Computes derived quantities: momentum, volatility, spread, bid/ask pressure
5. Maps market dynamics to **4D rotation angles** across the 6 rotation planes

This is the "temporal loom" — it weaves irregular data ticks into a smooth, continuous
4D rotation signal.

#### 4. The StereoscopicFeed (`StereoscopicFeed.ts`)

Splits the synchronized data into two visual streams:

- **Left Eye**: Standard 2D chart data (OHLC candlesticks, volume bars)
- **Right Eye**: 4D geometric projection data (polytope state, rotation, Trinity weights)

The `DataPrism` class manages this bifurcation and implements the **crosshair lock**:
when the user moves a crosshair on the 2D chart, the 4D geometry snaps to that historical
moment, allowing the user to "time-travel" through the data and see what the 4D state
looked like at any past point.

#### 5. The Math Core

The TypeScript math library (`lib/math_core/`) implements the 4D geometry natively:

- **GeometricAlgebra.ts**: Full Clifford algebra Cl(4,0) — vectors, bivectors, rotors,
  and the sandwich product for rotation.
- **Lattice24.ts / CPE_Lattice24.ts**: The 24-cell with Trinity decomposition, coherence
  measurement, convexity checking, phase shift detection, and inter-axis tension calculation.
- **Cell600.ts**: The 600-cell with its 25 inscribed 24-cells and E₈ projection.
- **E8Projection.ts**: The 4×8 Baez matrix, Galois conjugation, and nested 600-cell
  generation from the 240 E₈ roots.
- **GoldenRatioScaling.ts**: φ-nested polytope structures and moiré pattern detection
  between layers at different golden ratio scales.
- **TrinityEngine.ts**: Manages the trinity state vector Ψ = [w_α, w_β, w_γ] as a
  superposition of the three 16-cell channels. Detects phase shifts (when the dominant
  axis changes), calculates inter-axis tension (high tension = near a phase boundary),
  and predicts upcoming transitions.
- **ChronomorphicEngine.ts**: The master engine that unifies causal reasoning (force →
  torque → rotor), Trinity phase detection, musical mapping, hyperdimensional computing,
  topological data analysis, and 600-cell operations into a single update loop.

---

## Part V: The Musical Mapping

The 24-cell also functions as a **musical coordinate system**:

| 24-Cell Element | Musical Element |
|----------------|-----------------|
| 24 vertices | 24 major and minor keys (12 major + 12 minor) |
| 24 octahedral cells | 24 diatonic modes |
| 96 edges | Interval relationships between keys |
| Self-duality (vertices ↔ cells) | Major/minor duality |
| Neighbor vertices (distance √2) | Closely related keys (circle of fifths) |
| Trinity decomposition | Three octatonic collections |
| Phase shift (Alpha→Beta) | Musical modulation (key change) |

Movement along the **circle of fifths** corresponds to a 72° rotation in the XY plane
of the 24-cell. The diminished 7th chord (B-D-F-Ab) forms a regular **tetrahedron**
with tetrahedral symmetry T_d.

This mapping provides **human-perceptible validation**: if the 4D geometry is correctly
computed, the musical mapping should produce consonant harmonies. If the geometry is
wrong, the music sounds wrong. The ear is an extremely sensitive geometric validator.

---

## Part VI: End-to-End Data Flow

Here is the complete path of data through the system, from raw input to rendered output:

```
1. RAW INPUT arrives
   (audio waveform, market tick, sensor reading, image)
        │
        ▼
2. PYTHON BACKEND processes it:
   a. Enforcer validates parameters (3 rule sets)
   b. Input resized to kirigami grid (e.g., 32×32)
   c. Cells perturbed by input values (force injection)
   d. Cascade dynamics run (20-50 steps, ESN reservoir)
   e. Moiré pattern computed (per layer pair, multiplicative)
   f. Features extracted (LPQ + Gabor + HOG + spectral + wavelet)
   g. Optional: linear readout classifies pattern
   h. Telemetry assembled: 12 channels of normalized physics metrics
        │
        ▼
3. WEBSOCKET transmits telemetry as JSON at 30 FPS
        │
        ▼
4. HEMOC PYTHON BRIDGE receives frame:
   a. Parses HemocPhysicsPayload JSON
   b. Normalizes 12 channels to [0, 1]
   c. Converts to RawApiTick format
   d. Passes to PPP pipeline
        │
        ▼
5. TIMEBINDER phase-locks the data:
   a. Stores tick in ring buffer (O(1))
   b. Computes momentum, volatility, spread from tick history
   c. Maps dynamics to 6 rotation plane angles
   d. On each display frame (60 FPS):
      - Binary search for bracketing ticks (O(log n))
      - Interpolate price vector linearly
      - SLERP interpolate 4D rotation
      - Produce SyncedFrame
        │
        ▼
6. STEREOSCOPIC FEED bifurcates:
   a. Left Eye: 2D chart data (OHLC, volume)
   b. Right Eye: 4D geometric data (polytope, Trinity, rotation)
   c. Crosshair lock: chart interaction → 4D time-travel
        │
        ▼
7. MATH CORE computes geometry:
   a. TrinityEngine: Which channel (α/β/γ) dominates? Phase shift?
   b. Lattice24: Coherence? Inside convex hull? Nearest vertex?
   c. ChronomorphicEngine: Causal forces, musical key, topology
   d. Optional: Cell600 for 600-cell operations, E8 for projection
        │
        ▼
8. RENDERER displays:
   a. Canvas/WebGL visualization of moiré patterns
   b. 4D polytope projection (stereographic → 3D → 2D)
   c. Trinity channel indicators (α/β/γ weights)
   d. Reservoir state heatmap
   e. Talbot mode indicator
   f. Phase shift alerts
   g. Musical key/chord display
```

---

## Part VII: What Makes This Different

### It is not a neural network

The system does not use backpropagation. The kirigami reservoir provides nonlinear
computation through physics (mechanical cascade dynamics). Only the final readout
layer is trained, via simple ridge regression.

### It is not digital logic

The system computes with **light interference**, not transistor switches. Logic gates
(AND, OR, NAND, XOR) emerge from the physics of moiré patterns and Talbot self-imaging.
Switching between AND and NAND requires changing a physical gap, not rewiring circuits.

### It is not arbitrary

Every parameter in the system (the number of layers, the number of channels, the angles,
the gaps, the golden ratio scaling) is mathematically **determined** by the geometry of
4D polytopes. There are no hyperparameters chosen by trial and error. The 24-cell tells
you there must be 3 channels. The 600-cell tells you 5 units form a constellation. E₈
tells you the projection uses the golden ratio. The physics constrains the design
completely.

### It is an analog computer

The name "PPP Market Analog Computer" is literal. This is an **analog computer** — a
device that computes using continuous physical quantities (light intensity, mechanical
strain, interference fringes) rather than discrete bits. The software simulates this
analog device so it can process data (market data, audio, images) through the physics
pipeline without building the physical hardware.

---

## Appendix: Glossary

| Term | Definition |
|------|-----------|
| **24-cell** | Regular 4D polytope with 24 vertices, unique to 4D. The fundamental module. |
| **600-cell** | Regular 4D polytope with 120 vertices. Five 24-cells form one 600-cell. |
| **Trinity decomposition** | Partition of 24-cell into 3 disjoint 16-cells (Alpha/Beta/Gamma). |
| **Kirigami** | Paper with cuts that allow controlled folding. Not origami (no cuts). |
| **Moiré pattern** | Large-scale interference pattern from overlapping fine grids. |
| **Talbot effect** | Self-imaging of periodic gratings at specific distances. |
| **Commensurate angle** | Twist angle where the moiré superlattice is periodic. |
| **ESN** | Echo State Network — reservoir computer with fixed internal dynamics and trained readout. |
| **Tristable** | Having exactly three stable equilibrium states. |
| **Isoclinic rotation** | 4D rotation where both planes of a complementary pair rotate equally. |
| **Golden ratio (φ)** | (1+√5)/2 ≈ 1.618. Fundamental to H₄ symmetry and the 600-cell. |
| **E₈** | Exceptional Lie group. Its 240 roots project to two nested 600-cells in 4D. |
| **Phason** | Strain wave in a quasi-crystal or quasi-periodic structure. |
| **LPQ** | Local Phase Quantization — extracts phase information from images via STFT. |
| **Baez projection** | 4×8 matrix (involving φ) that projects E₈ roots from 8D to 4D. |
| **Galois conjugation** | Swapping φ ↔ -1/φ. Maps outer 600-cell to inner 600-cell. |
| **Clifford algebra Cl(4,0)** | The geometric algebra of 4D Euclidean space. |
| **PPPAdapter** | Interface contract for data sources feeding the visualization pipeline. |
| **Phase-lock** | Synchronizing irregular data ticks with smooth display frame rate. |
| **DataPrism** | Bifurcates synchronized data into 2D chart (Left Eye) and 4D geometry (Right Eye). |

---

*This document describes the system as implemented in the codebase. The physical device
it simulates has been designed but not yet fabricated. The software is a faithful
simulation of the physics that would occur in the physical device.*
