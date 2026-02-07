# Evolution Paths: PPP Synergized System

> Where the system is, where it goes, and the exact steps to get there.

---

## Current State Assessment

### What Works (Verified Complete)

| Subsystem | Lines | Status |
|-----------|-------|--------|
| Rule Enforcer (3 rule sets) | 415 | Real physics, all rules enforced |
| Tristable Cell Mechanics | 321 | Triple-well potential, Velocity Verlet integration, neighbor coupling |
| Kirigami Sheet | ~300 | Full hexagonal lattice, cascade dynamics, cut patterns |
| Moiré Interference | ~400 | Bichromatic logic, fringe contrast, spatial logic gates |
| Talbot Resonator | ~200 | Integer/half-integer modes, resonance ladder |
| Trilatic Lattice | ~250 | Commensurate angles, reciprocal vectors, magic angle |
| ESN Reservoir | ~300 | Leaky integration, spectral radius control, ridge readout |
| Criticality Analyzer | ~200 | Lyapunov exponent, branching ratio, correlation length |
| Feature Extractors (6 types) | ~1200 | LPQ, Gabor, HOG, Hu moments, spectral, wavelet |
| HDC Encoder | ~400 | Bind/bundle/permute/similarity, associative memory |
| Temporal Encoder | ~200 | Fading memory, position injection, logic mode switching |
| H4 Geometry (Python) | ~400 | 16-cell, 24-cell, 600-cell, 120-cell, trilatic decomposition |
| Quaternion 4D (Python) | ~300 | Isoclinic rotation, SLERP, left/right decomposition |
| H4 Constellation (Python) | 722 | 5-node network, phason propagation, palindrome sequence |
| WebSocket Server | 239 | 30 FPS telemetry, synthetic fallback engine |
| Tripole Actuator | ~300 | Piston/tip/tilt control, plane equation solver |
| PPP Adapters (TypeScript) | ~400 | HemocPythonBridge, MarketQuote, AdapterBridge |
| TimeBinder | ~300 | Phase-locked ring buffer, binary search interpolation |
| StereoscopicFeed | ~700 | DataPrism, left/right bifurcation, crosshair lock |
| GeometricAlgebra Cl(4,0) | ~500 | Full Clifford algebra with sandwich product |
| Lattice24 (TypeScript, 2 versions) | ~1700 | Trinity decomposition, coherence, Voronoi, phase shifts |
| Cell600 (TypeScript) | 577 | 120 vertices, 25 inscribed 24-cells, E₈ projection |
| E8Projection (TypeScript) | 519 | Baez matrix, Galois conjugation, nested 600-cells |
| GoldenRatioScaling (TypeScript) | 458 | φ-nesting, moiré detection, Fibonacci approximation |
| TrinityEngine | 512 | Phase superposition, tension, polytonal detection |
| ChronomorphicEngine | 759 | Unified orchestration of all subsystems |
| CausalReasoningEngine | ~400 | Force → torque → rotor physics |
| PersistentHomology | ~400 | Vietoris-Rips filtration, Betti numbers, persistence pairs |
| GhostFrequencyDetector | ~300 | Void detection, implied harmonic inference |
| Standalone Dashboard | 1407 | synergized-system.html with live visualization |

**Total: ~13,000+ lines of implemented logic across ~50 modules.**

### What's Broken (Blocking Issues)

| Issue | File | Problem | Impact |
|-------|------|---------|--------|
| Import path error | `backend/engine/pipeline.py:33` | `from .rules.enforcer` → `rules/` dir doesn't exist | Pipeline can't load — **entire Python backend blocked** |
| No package config | Repository root | No `setup.py` or `pyproject.toml` | Can't `pip install`, can't resolve imports across modules |
| Demo import error | `demos/python_demos/full_pipeline_demo.py:25` | `from optical_kirigami_moire import ...` | Demos can't run |
| Docker blocked | `docker-compose.yml` | Depends on pipeline.py loading | Both containers fail to start |

### What's Missing (Gaps)

| Gap | Current State | Impact |
|-----|---------------|--------|
| Zero Python tests | No `test_*.py` files anywhere | No verification of physics correctness |
| No integration test | Nothing exercises the full encode cycle | Can't prove end-to-end works |
| PyTorch encoder incomplete | `ml/torch_encoder.py` has partial forward() | Can't train differentiable moiré layer |
| No `__init__.py` for package | Missing in key directories | Python module resolution fragile |
| No type checking CI | No mypy/pyright/tsc in CI | Type errors may exist undetected |
| GitHub Actions = deploy only | `.github/workflows/deploy-pages.yml` | No test runs on push/PR |
| No WASM build | Math core is TypeScript, not compiled | Performance-limited in browser |

---

## Path 0: Triage (Do This First)

**Goal:** Make the system actually run.

### 0.1 Fix the Pipeline Import

```python
# backend/engine/pipeline.py line 33
# CURRENT (broken):
from .rules.enforcer import RuleEnforcer, AngleLock, TiltLock, GapLock, LogicPolarity

# FIXED:
from .enforcer import RuleEnforcer, AngleLock, TiltLock, GapLock, LogicPolarity
```

One line. Unblocks the entire Python backend.

### 0.2 Add Python Package Configuration

Create `_SYNERGIZED_SYSTEM/backend/pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "hemoc-sgf-engine"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "websockets>=12.0",
]

[project.optional-dependencies]
ml = ["torch>=2.0"]
dev = ["pytest>=7.0", "pytest-asyncio>=0.21"]

[tool.setuptools.packages.find]
where = ["."]
include = ["engine*"]
```

### 0.3 Add Missing `__init__.py` Files

Ensure every Python package directory has an `__init__.py` with proper exports.

### 0.4 Fix Demo Imports

```python
# demos/python_demos/full_pipeline_demo.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "backend"))
from engine.pipeline import OpticalKirigamiMoire, PipelineConfig, ComputationMode
from engine.enforcer import RuleEnforcer, LogicPolarity
```

### 0.5 Verify Docker Compose

After fixing imports, test:
```bash
cd _SYNERGIZED_SYSTEM
docker compose up --build
```

**Time estimate: One focused session.**

---

## Path 1: Testing Infrastructure

**Goal:** Prove the physics is correct. Build confidence for all future changes.

### 1.1 Python Unit Tests for Physics Core

Create `_SYNERGIZED_SYSTEM/backend/tests/`:

```
tests/
├── conftest.py                    # Shared fixtures
├── test_enforcer.py               # Rule enforcement (3 rule sets)
├── test_tristable_cell.py         # Cell mechanics and state transitions
├── test_kirigami_sheet.py         # Lattice dynamics and cascade
├── test_moire_interference.py     # Moiré pattern correctness
├── test_talbot_resonator.py       # Talbot distances and modes
├── test_trilatic_lattice.py       # Commensurate angles
├── test_esn_reservoir.py          # Echo state network dynamics
├── test_criticality.py            # Edge-of-chaos metrics
├── test_feature_extraction.py     # All 6 feature extractors
├── test_hdc_encoder.py            # Hyperdimensional operations
├── test_h4_geometry.py            # Polytope vertex counts and edges
├── test_quaternion_4d.py          # Rotation correctness
├── test_h4_constellation.py       # 5-node network assembly
└── test_pipeline_integration.py   # Full encode-compute-readout cycle
```

**Key test invariants** (from the geometry):

```python
# test_h4_geometry.py

def test_24cell_has_24_vertices():
    cell = Polytope24Cell()
    assert len(cell.vertices) == 24

def test_trinity_decomposition_is_partition():
    cell = Polytope24Cell()
    trilatic = TrilaticDecomposition(cell)
    alpha = set(trilatic.get_channel_vertices(TrilaticChannel.ALPHA))
    beta = set(trilatic.get_channel_vertices(TrilaticChannel.BETA))
    gamma = set(trilatic.get_channel_vertices(TrilaticChannel.GAMMA))
    assert len(alpha) == 8
    assert len(beta) == 8
    assert len(gamma) == 8
    assert alpha & beta == set()  # Disjoint
    assert alpha & gamma == set()
    assert beta & gamma == set()
    assert len(alpha | beta | gamma) == 24  # Covers all vertices

def test_24cell_vertices_equidistant_from_origin():
    cell = Polytope24Cell()
    radii = [v.distance_from_origin() for v in cell.vertices]
    assert all(abs(r - radii[0]) < 1e-10 for r in radii)
    assert abs(radii[0] - math.sqrt(2)) < 1e-10

def test_24cell_neighbor_count():
    cell = Polytope24Cell()
    for v in cell.vertices:
        neighbors = cell.get_neighbors(v)
        assert len(neighbors) == 8

def test_600cell_has_120_vertices():
    cell = Polytope600Cell()
    assert len(cell.vertices) == 120

def test_600cell_decomposes_into_5_24cells():
    cell = Polytope600Cell()
    partitions = cell.get_24cell_partition()
    assert len(partitions) == 5
    all_vertices = set()
    for part in partitions:
        assert len(part) == 24
        all_vertices.update(part)
    assert len(all_vertices) == 120

def test_golden_ratio_identity():
    phi = (1 + math.sqrt(5)) / 2
    assert abs(phi**2 - phi - 1) < 1e-14
    assert abs(1/phi - (phi - 1)) < 1e-14

def test_quaternion_norm_preservation():
    q1 = Quaternion4D(w=0.5, x=0.5, y=0.5, z=0.5)
    q2 = Quaternion4D.from_axis_angle(np.array([1,0,0]), math.pi/3)
    product = q1 * q2
    assert abs(product.norm() - 1.0) < 1e-10
```

```python
# test_moire_interference.py

def test_moire_period_formula():
    """L_M = a / (2 * sin(θ/2))"""
    mi = MoireInterference(lattice_constant=1.0, wavelength=550.0)
    for angle_deg in [7.34, 9.43, 13.17, 21.79]:
        theta = math.radians(angle_deg)
        expected = 1.0 / (2 * math.sin(theta / 2))
        computed = mi.compute_moire_period(angle_deg)
        assert abs(computed - expected) < 0.01

def test_cell_fractions_sum_to_one():
    sheet = KirigamiSheet(rows=16, cols=16)
    sheet.inject_input(np.random.randn(16, 16))
    sheet.run_cascade(steps=30)
    flat, half, full = sheet.get_state_fractions()
    assert abs(flat + half + full - 1.0) < 1e-10

def test_talbot_integer_vs_half():
    resonator = TalbotResonator(lattice_constant=1.0, wavelength=550e-3)
    z_t = resonator.talbot_length
    int_state = resonator.evaluate(z_t)       # Integer Talbot
    half_state = resonator.evaluate(z_t * 1.5) # Half-integer Talbot
    assert int_state.mode == TalbotMode.INTEGER
    assert half_state.mode == TalbotMode.HALF_INTEGER
```

### 1.2 Integration Test: Full Encode Cycle

```python
# test_pipeline_integration.py

def test_full_encode_cycle():
    """End-to-end: input → kirigami → moiré → features → classification."""
    config = PipelineConfig(grid_size=(16, 16), cascade_steps=20, n_outputs=3)
    okm = OpticalKirigamiMoire(config)
    okm.set_mode(ComputationMode.TEXTURE)

    # Create distinguishable patterns
    horizontal = np.sin(np.linspace(0, 4*np.pi, 16).reshape(1,-1).repeat(16, axis=0))
    vertical = horizontal.T
    checkerboard = np.kron(np.eye(4), np.ones((4,4)))

    # Train
    patterns = [horizontal, vertical, checkerboard] * 5
    labels = np.array([0, 1, 2] * 5)
    encoded = [okm.encode(p) for p in patterns]
    moire_patterns = [e.moire_pattern for e in encoded]
    accuracy = okm.train_readout(moire_patterns, labels)
    assert accuracy > 0.6  # Should distinguish these easily

    # Classify new samples
    pred, conf = okm.classify(okm.encode(horizontal + np.random.randn(16,16)*0.05).moire_pattern)
    assert pred == 0
```

### 1.3 TypeScript Test Expansion

The existing 8 test files cover adapters and phase-lock. Add:

```
tests/
├── lattice24.test.ts              # 24-cell geometry invariants
├── cell600.test.ts                # 600-cell decomposition
├── e8-projection.test.ts          # E₈→H₄ projection correctness
├── golden-ratio-scaling.test.ts   # φ-nesting and moiré detection
├── trinity-engine.test.ts         # Phase shift detection
├── geometric-algebra.test.ts      # Cl(4,0) operations
├── persistent-homology.test.ts    # Betti numbers
└── chronomorphic-engine.test.ts   # Unified orchestration
```

### 1.4 CI/CD Pipeline

Create `.github/workflows/test.yml`:

```yaml
name: Test Suite
on: [push, pull_request]
jobs:
  python-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -e ".[dev]"
        working-directory: _SYNERGIZED_SYSTEM/backend
      - run: pytest tests/ -v --tb=short
        working-directory: _SYNERGIZED_SYSTEM/backend

  typescript-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: '20' }
      - run: npm install
      - run: npm run test:all
```

**Milestone: Green CI on every push. Physics correctness verified by 50+ tests.**

---

## Path 2: Make It Run Live

**Goal:** Someone can clone the repo, run one command, and see the system working.

### 2.1 Single-Command Local Launch

Create `run.sh` at the repo root:

```bash
#!/bin/bash
# Start the Python physics engine in the background
cd _SYNERGIZED_SYSTEM/backend
python -m engine.websocket_server &
ENGINE_PID=$!

# Wait for WebSocket to be ready
sleep 2

# Open the dashboard
if command -v open &>/dev/null; then
  open ../synergized-system.html
elif command -v xdg-open &>/dev/null; then
  xdg-open ../synergized-system.html
fi

echo "Engine PID: $ENGINE_PID"
echo "Dashboard: synergized-system.html"
echo "WebSocket: ws://localhost:8765"
echo "Press Ctrl+C to stop"
wait $ENGINE_PID
```

### 2.2 Connect Dashboard to Live Engine

The `synergized-system.html` currently uses a synthetic engine built into the
JavaScript. Wire it to the real Python WebSocket server:

1. Add a toggle: "Synthetic / Live Engine"
2. When "Live" is selected, open `ws://localhost:8765` and use real physics telemetry
3. Map HemocPhysicsPayload fields to the dashboard widgets
4. Show connection status indicator (connected/disconnected/reconnecting)

### 2.3 Interactive Parameter Controls

The dashboard already has mode selectors and buttons. Extend to send commands
back to the Python engine via WebSocket:

```javascript
ws.send(JSON.stringify({
  command: "set_mode",
  params: { mode: "texture", angle: 13.17, gap: "integer" }
}));
```

Expose all control parameters:
- Twist angle (dropdown of commensurate modes + manual with auto-snap)
- Talbot mode (integer/half-integer toggle → switches AND↔NAND logic)
- Tilt direction (60° increments + magnitude)
- Cascade steps (slider: 10-100)
- Reservoir leak rate (slider: 0.1-0.9)
- Input injection: paste an image, upload a CSV, or use microphone audio

### 2.4 Docker Compose (Fixed)

After fixing the import error, verify:
```bash
cd _SYNERGIZED_SYSTEM
docker compose up --build
# → engine on :8765, frontend on :5173
# → Open http://localhost:5173
```

**Milestone: Clone → `./run.sh` → live moiré computation in the browser.**

---

## Path 3: Real Data Sources

**Goal:** Feed real-world data through the physics pipeline.

### 3.1 Audio Input

The `streaming/audio_stream.py` and `streaming/temporal_encoder.py` already exist.
Build the complete audio pipeline:

```
Microphone → Web Audio API (browser)
    → FFT spectrogram (256 frequency bins × time frames)
    → WebSocket to Python engine
    → Resize to kirigami grid (32×32)
    → Inject into kirigami reservoir
    → Cascade dynamics
    → Moiré pattern encodes spectral content
    → Feature extraction (LPQ phase = audio phase!)
    → Real-time classification (speech, music, noise, silence)
```

The LPQ decoder is particularly powerful for audio because it extracts **phase
information** — and in audio, phase carries the information that distinguishes
speech from noise. The moiré encoding preserves phase structure that standard
magnitude-only spectrograms discard.

### 3.2 Market Data Input

The `MarketQuoteAdapter` already normalizes bid/ask/spread/imbalance. Extend to:

```
Live market feed (WebSocket from exchange/broker API)
    → MarketQuoteAdapter normalizes to RawApiTick
    → TimeBinder phase-locks to 60 FPS
    → Map 4 market dimensions to 4D quaternion:
        price_direction → w component
        volume_impulse  → x component
        spread_change   → y component
        bid_ask_imbal   → z component
    → Inject quaternion into 24-cell
    → Trinity decomposition reveals which channels dominate:
        Alpha dominant → price-driven regime
        Beta dominant  → volume-driven regime
        Gamma dominant → spread/liquidity-driven regime
    → Phase shifts = regime changes (detected by TrinityEngine)
    → Moiré pattern encodes market microstructure
    → Reservoir processes temporal sequences
    → Readout predicts: regime classification, volatility forecast, anomaly detection
```

### 3.3 Image/Video Input

Use the existing feature extraction pipeline for computer vision:

```
Camera frame (640×480)
    → Resize to kirigami grid (64×64)
    → Inject pixel intensities into cells
    → Different moiré modes for different tasks:
        TRANSPARENT (0°)    → Pass-through (no encoding)
        EDGE_DETECT (7.34°) → Edge-enhanced features
        FINE (9.43°)        → Texture classification
        INTERMEDIATE (13°)  → General-purpose features
        COARSE (21.79°)     → Large-scale structure
    → Feature vector from moiré pattern
    → One-shot classification via HDC (no training needed!)
```

The HDC encoder enables **one-shot learning**: show the system one example of
"cat" and one example of "dog", and it can classify new images without training
a neural network. This is because hyperdimensional vectors preserve similarity
structure — similar inputs produce similar hypervectors.

### 3.4 Sensor Fusion

When multiple inputs are available simultaneously:

```
Audio channel → Alpha (Layer Pair 1)
Market channel → Beta (Layer Pair 2)
Image channel → Gamma (Layer Pair 3)
```

Each Trinity channel processes one modality independently. Cross-channel
interactions happen through the 24-cell geometry: vertices from different
16-cells share edges, so information flows between modalities via the
geometric structure.

**Milestone: Real audio/market/image data flowing through the physics pipeline
with live visualization.**

---

## Path 4: The 600-Cell Constellation (Distributed Computing)

**Goal:** Multiple 24-cell units working together.

### 4.1 Multi-Process Architecture

Each constellation node runs as an independent process:

```
Node 0 (CENTER):  python -m engine.node --id 0 --port 8760
Node 1 (NORTH):   python -m engine.node --id 1 --port 8761
Node 2 (EAST):    python -m engine.node --id 2 --port 8762
Node 3 (SOUTH):   python -m engine.node --id 3 --port 8763
Node 4 (WEST):    python -m engine.node --id 4 --port 8764
Orchestrator:     python -m engine.constellation_server --port 8765
```

Each node has its own kirigami stack, reservoir, and feature extractors.
The orchestrator manages:
- Inter-node quaternion state broadcast
- Phason strain propagation
- Global synchronization rounds
- Collective moiré output aggregation

### 4.2 Phason Strain as Information Channel

When one node's state is perturbed (e.g., by a new input), the strain
propagates to neighbors:

```
Node 0 receives input
    → Node 0 state changes
    → Strain wave propagates:
        t=1: Node 0 → Center ports → Nodes 1,2,3,4
        t=2: Nodes 1,2,3,4 adjust states
        t=3: Damped oscillation → convergence
    → Collective state encodes distributed information
```

The damping factor controls how much influence one node has on others.
Low damping = tight coupling = all nodes act as one.
High damping = loose coupling = nodes operate semi-independently.

### 4.3 Palindrome Deployment Sequence

Implement the full H4 palindrome as a runtime mode:

```
Phase 1: LOCKED → AUXETIC (24-cell → bistable transition)
    All 5 nodes expand through negative Poisson's ratio regime
    Kirigami cells open from state 0 → state 0.5

Phase 2: AUXETIC → DEPLOYED (bistable → 600-cell projection)
    Vertex ports extend, inter-node connections established
    5 × 24 = 120 active vertices
    Full H₄ symmetry available

Phase 3: DEPLOYED computation
    Global quaternion rotations, phason propagation
    600-cell operations: inscribed 24-cell detection, vertex classification

Phase 4: DEPLOYED → AUXETIC → LOCKED (palindrome return)
    Via 120-cell dual structure
    Graceful shutdown with state preservation
```

### 4.4 25 Inscribed 24-Cell Navigation

The 600-cell contains 25 inscribed 24-cells, not just the 5 disjoint ones.
Any vertex belongs to exactly 5 of the 25. Implement navigation:

```typescript
// Which 24-cells contain vertex #42?
const containing = cell600.getContaining24Cells(42);
// → [3, 7, 12, 18, 23]  (five 24-cells)

// Switch "perspective" to inscribed 24-cell #12
const cell12 = cell600.getInscribed24Cell(12);
const trinityOf12 = trinityDecompose(cell12);
// → Now the Trinity channels reflect a different 24-cell's structure
```

This allows the system to explore the same data from 25 different "viewpoints",
each corresponding to a different inscribed 24-cell.

**Milestone: Five independent processes forming a working constellation with
phason strain communication.**

---

## Path 5: Differentiable Physics (Machine Learning Bridge)

**Goal:** Make the moiré encoding learnable while preserving physical constraints.

### 5.1 Complete the PyTorch Encoder

The `ml/torch_encoder.py` has the structure but incomplete forward pass. Complete it:

```python
class TorchMoireLayer(nn.Module):
    """Differentiable moiré interference layer."""

    def __init__(self, grid_size=32, n_orientations=6):
        super().__init__()
        self.grid_size = grid_size
        # Learnable parameters (within physics constraints)
        self.twist_angles = nn.Parameter(torch.zeros(3))  # 3 Trinity channels
        self.gap_offsets = nn.Parameter(torch.zeros(3))    # Talbot perturbations
        self.tilt_vectors = nn.Parameter(torch.zeros(3, 2))  # 2D tilt per channel

    def forward(self, x):
        # x: (batch, 1, H, W) input image
        batch_size = x.shape[0]
        outputs = []

        for ch in range(3):  # Alpha, Beta, Gamma
            # Snap angle to nearest commensurate (differentiable relaxation)
            angle = self.soft_commensurate_snap(self.twist_angles[ch])

            # Generate moiré field (differentiable)
            moire = self.differentiable_moire(x, angle, self.gap_offsets[ch])

            # Apply tilt attention
            tilt = torch.sigmoid(self.tilt_vectors[ch])
            moire = moire * self.tilt_mask(tilt)

            outputs.append(moire)

        # Stack 3 Trinity channels
        return torch.stack(outputs, dim=1)  # (batch, 3, H, W)

    def soft_commensurate_snap(self, angle):
        """Differentiable approximation of commensurate angle snapping."""
        # Soft nearest-neighbor to the set of valid angles
        valid_angles = torch.tensor([0.0, 7.34, 9.43, 13.17, 21.79])
        distances = (angle - valid_angles).pow(2)
        weights = F.softmin(distances / self.temperature, dim=0)
        return (weights * valid_angles).sum()
```

### 5.2 Physics-Informed Loss Functions

The enforcer's three rule sets become **differentiable penalty terms**:

```python
def physics_loss(model, x, y_true):
    y_pred = model(x)
    task_loss = F.cross_entropy(y_pred, y_true)

    # Rule 1: Angles must be near commensurate values
    angle_penalty = commensurate_distance(model.twist_angles)

    # Rule 2: Tilt must respect C₃ symmetry
    tilt_penalty = hexagonal_symmetry_violation(model.tilt_vectors)

    # Rule 3: Gap must be near Talbot resonance
    gap_penalty = talbot_distance(model.gap_offsets, model.talbot_length)

    # Geometric: Trinity weights should sum to 1
    trinity_penalty = trinity_normalization_loss(model)

    return task_loss + 0.1 * (angle_penalty + tilt_penalty + gap_penalty + trinity_penalty)
```

### 5.3 Learned Feature Selection

Instead of using all 6 feature extractors, learn which features matter most:

```python
class LearnableFeatureSelector(nn.Module):
    def __init__(self, feature_dims):
        super().__init__()
        # feature_dims: dict of extractor_name → dimension
        self.attention = nn.ParameterDict({
            name: nn.Parameter(torch.ones(dim))
            for name, dim in feature_dims.items()
        })

    def forward(self, features):
        # features: dict of extractor_name → feature_tensor
        weighted = []
        for name, feat in features.items():
            weight = torch.sigmoid(self.attention[name])
            weighted.append(feat * weight)
        return torch.cat(weighted, dim=-1)
```

### 5.4 End-to-End Differentiable Pipeline

```
Input image
    → TorchMoireLayer (learnable angles, gaps, tilts)
    → TorchReservoir (fixed or slowly adapting ESN)
    → TorchFeatureExtractor (differentiable Gabor + spectral)
    → LearnableFeatureSelector (attention over extractors)
    → Linear readout
    → Loss = task_loss + physics_penalties
    → Backprop through moiré layer (learns optimal angles/gaps)
```

The key insight: the physical constraints (commensurate angles, Talbot gaps,
C₃ symmetry) are not abandoned — they become **soft constraints** via penalty
terms. The network learns the best parameters **within** the space of physically
valid configurations.

**Milestone: Differentiable moiré encoder that learns optimal physical
parameters for a given task while respecting physics constraints.**

---

## Path 6: Topological Data Analysis (TDA) Integration

**Goal:** Use persistent homology to detect structure invisible to other methods.

### 6.1 Moiré Pattern Topology

Apply PersistentHomology to moiré patterns:

```typescript
const pattern = getMoirePattern();  // 2D intensity field
const pointCloud = patternToPointCloud(pattern);  // Threshold → point set
const persistence = computePersistence(pointCloud);

// β₀ = connected components (= number of distinct moiré fringes)
// β₁ = loops (= enclosed regions in the moiré pattern)
// β₂ = voids (= 3D cavities if we stack multiple frames)

const bettiNumbers = persistence.getBettiNumbers();
```

Track Betti numbers over time:
- Sudden increase in β₁ → new loop structures appearing → regime change
- β₀ decreasing → fringes merging → approaching magic angle
- Persistence diagram shows which features are "real" (long bars) vs. noise (short bars)

### 6.2 Ghost Frequency Detection

The `GhostFrequencyDetector` finds **implied frequencies** — frequencies that
should exist based on the topological structure but are absent from the spectrum.
These "ghosts" indicate:

- Missing harmonics (the fundamental is present but overtones are suppressed)
- Interference cancellation (two signals destructively cancel at specific frequencies)
- Lattice defects (the quasi-crystal has vacancies where vertices should be)

```typescript
const ghosts = ghostDetector.analyze(moirePattern, trinityState);
for (const ghost of ghosts.missingVertices) {
    // ghost.expectedPosition: where the vertex should be in the 24-cell
    // ghost.evidenceStrength: how confident we are it's missing
    // ghost.impliedBy: which topological voids suggest it
}
```

### 6.3 Phason Worm Tracking

The LPQ decoder can detect "phason worms" — line defects in quasi-crystalline
patterns that propagate through the lattice. Track them:

```python
detector = PhaseShiftDetector(sensitivity=0.5)
worms = detector.detect_phason_worms(moire_pattern)
for worm in worms:
    # worm.path: sequence of cell positions the defect traverses
    # worm.velocity: propagation speed
    # worm.channel: which Trinity channel it's in
    # worm.crossing: does it cross from Alpha to Beta? (inter-axis event)
```

Phason worms that cross Trinity channels are **phase shift events** — they
correspond to modulations in the musical mapping and regime changes in the
market data mapping.

**Milestone: Real-time topological analysis of moiré patterns with ghost
frequency detection and phason worm tracking.**

---

## Path 7: Musical Mapping Implementation

**Goal:** Turn the geometry into sound for human-perceptible validation.

### 7.1 Implement MusicGeometryDomain

The design document (`MusicGeometryDomain-Design (1).md`) is complete. Build it:

```typescript
// lib/domains/MusicGeometryDomain.ts

export class MusicGeometryDomain {
    private lattice: Lattice24;
    private keyMap: Map<string, number>;  // key name → vertex id

    constructor(config: MusicGeometryConfig) {
        this.lattice = new Lattice24();
        this.keyMap = this.buildKeyMapping(config.pitchToXY);
    }

    noteToCoordinate(note: string): Vector4D {
        const vertexId = this.keyMap.get(note);
        return this.lattice.getVertex(vertexId).coordinates;
    }

    measureConsonance(note1: string, note2: string): number {
        const v1 = this.noteToCoordinate(note1);
        const v2 = this.noteToCoordinate(note2);
        const dist = distance(v1, v2);
        // Consonance inversely proportional to geometric distance
        return 1 / (1 + dist);
    }

    detectModulation(trinityState: TrinityState): string {
        // Phase shift = key change
        if (trinityState.lastPhaseShift) {
            const fromKey = this.axisToKey(trinityState.lastPhaseShift.from);
            const toKey = this.axisToKey(trinityState.lastPhaseShift.to);
            return `Modulation: ${fromKey} → ${toKey}`;
        }
        return "Stable";
    }
}
```

### 7.2 Audio Sonification

Convert the 24-cell state into sound in real time:

```
Trinity state [w_α, w_β, w_γ] → 3 oscillator volumes
Dominant vertex → root note (MIDI pitch)
Neighbor vertices → chord tones
Phase shift → key change (audible modulation)
Coherence → consonance/dissonance
Tension → filter cutoff (tense = bright, relaxed = mellow)
```

Use the Web Audio API (already partially implemented in SonicGeometryEngine.js):

```javascript
const musicMapper = new MusicGeometryDomain();
const audioCtx = new AudioContext();

function sonify(trinityState) {
    const key = musicMapper.stateToKey(trinityState);
    const chord = musicMapper.stateToChord(trinityState);
    const tension = musicMapper.stateTension(trinityState);

    // Set oscillator frequencies to chord tones
    for (let i = 0; i < chord.notes.length; i++) {
        oscillators[i].frequency.setTargetAtTime(
            noteToHz(chord.notes[i]),
            audioCtx.currentTime, 0.1
        );
    }

    // Tension controls filter
    filter.frequency.setTargetAtTime(
        200 + tension * 2000,  // 200 Hz (relaxed) to 2200 Hz (tense)
        audioCtx.currentTime, 0.05
    );
}
```

### 7.3 Neo-Riemannian Transformations

The 24-cell geometry naturally supports **Neo-Riemannian operations** (PLR
transformations used in music theory):

- **P (Parallel):** Major ↔ minor with same root. Corresponds to reflection
  across a Trinity channel boundary.
- **L (Leading-tone):** Move one note by half step. Corresponds to movement
  to a neighboring vertex.
- **R (Relative):** Major ↔ relative minor. Corresponds to diagonal movement
  in the 24-cell.

These are geometric operations on the polytope, not just musical abstractions.

**Milestone: Live sonification of 24-cell state. Phase shifts audible as
key changes. Consonance validates geometric coherence.**

---

## Path 8: WebAssembly Performance

**Goal:** Run the math core at native speed in the browser.

### 8.1 Compile Core Math to WASM

The TypeScript math core (Cl(4,0), Lattice24, Cell600, E8Projection) is
the performance bottleneck. Rewrite the hot paths in Rust and compile to WASM:

```rust
// src/lattice24.rs

#[wasm_bindgen]
pub struct Lattice24 {
    vertices: [[f64; 4]; 24],
    neighbors: [[u8; 8]; 24],
    trinity: [u8; 24],  // 0=Alpha, 1=Beta, 2=Gamma
}

#[wasm_bindgen]
impl Lattice24 {
    pub fn new() -> Self {
        let mut lattice = Lattice24 {
            vertices: [[0.0; 4]; 24],
            neighbors: [[0; 8]; 24],
            trinity: [0; 24],
        };
        lattice.generate_vertices();
        lattice.compute_neighbors();
        lattice.assign_trinity();
        lattice
    }

    pub fn find_nearest(&self, x: f64, y: f64, z: f64, w: f64) -> u8 {
        let point = [x, y, z, w];
        let mut min_dist = f64::MAX;
        let mut nearest = 0u8;
        for (i, v) in self.vertices.iter().enumerate() {
            let dist = self.distance_sq(&point, v);
            if dist < min_dist {
                min_dist = dist;
                nearest = i as u8;
            }
        }
        nearest
    }
}
```

### 8.2 SIMD Optimization

WASM SIMD (128-bit) can process 4D vectors in a single instruction:

```rust
use std::arch::wasm32::*;

fn dot_product_4d(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    unsafe {
        let va = f64x2_make(a[0], a[1]);
        let vb = f64x2_make(b[0], b[1]);
        let vc = f64x2_make(a[2], a[3]);
        let vd = f64x2_make(b[2], b[3]);
        let prod1 = f64x2_mul(va, vb);
        let prod2 = f64x2_mul(vc, vd);
        let sum = f64x2_add(prod1, prod2);
        f64x2_extract_lane::<0>(sum) + f64x2_extract_lane::<1>(sum)
    }
}
```

### 8.3 GPU Compute via WebGPU

For the moiré interference computation (the most expensive operation), use
WebGPU compute shaders:

```wgsl
@compute @workgroup_size(16, 16)
fn compute_moire(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let x = f32(id.x) / f32(grid_size);
    let y = f32(id.y) / f32(grid_size);

    // Layer 1 (Cyan) transmission
    var t1 = 0.0;
    for (var i = 0u; i < 3u; i++) {
        let g = reciprocal_vectors[i];
        t1 += cos(g.x * x + g.y * y);
    }
    t1 = t1 / 3.0;

    // Layer 2 (Magenta) - rotated by twist angle
    let rx = x * cos_theta - y * sin_theta;
    let ry = x * sin_theta + y * cos_theta;
    var t2 = 0.0;
    for (var i = 0u; i < 3u; i++) {
        let g = reciprocal_vectors[i];
        t2 += cos(g.x * rx + g.y * ry);
    }
    t2 = t2 / 3.0;

    // Moiré = multiplicative combination
    output[id.x + id.y * grid_size] = t1 * t2;
}
```

**Milestone: 60+ FPS for full 600-cell operations in the browser. Real-time
moiré computation at 256×256 resolution via GPU.**

---

## Path 9: Physical Prototype

**Goal:** Build the actual hardware.

### 9.1 Bill of Materials

| Component | Specification | Qty | Purpose |
|-----------|--------------|-----|---------|
| Kirigami sheets | 100μm Mylar, laser-cut hexagonal pattern | 6 | Optical layers |
| Color filters | Cyan (500nm) and Magenta (550nm) dichroic | 6 | Bichromatic encoding |
| Piezo actuators | PI P-841.60 (60μm range, 1nm resolution) | 3 | Tripole gap control |
| Rotation stage | Newport PR50CC (0.001° resolution) | 1 | Twist angle |
| LED source | Thorlabs M530L4 (Cyan) + M625L4 (Red) | 2 | Bichromatic illumination |
| Camera sensor | FLIR BFS-U3-51S5M-C (5MP, global shutter) | 1 | Moiré pattern capture |
| Raspberry Pi 5 | 8GB RAM | 1 | Embedded compute |
| Hexagonal frame | 3D printed, magnetic pogo-pin connectors | 5 | Constellation assembly |

### 9.2 Fabrication Sequence

1. **Laser-cut kirigami patterns** on Mylar sheets (hexagonal lattice, a = 100μm)
2. **Laminate color filters** onto sheets (Cyan on layers 1,3,5; Magenta on 2,4,6)
3. **Assemble tripole mount** (3 piezos at 120° on aluminum baseplate)
4. **Mount rotation stage** on tripole platform
5. **Stack layers** with spacers (gap = Talbot distance ≈ 36mm for a=100μm, λ=550nm)
6. **Align optics** (LED → collimator → stack → camera)
7. **Flash firmware** (Python engine → Raspberry Pi)
8. **Calibrate** (measure actual Talbot distances, adjust gap to integer/half-integer)

### 9.3 Constellation Assembly

Five prototype units arranged in a cross pattern:

```
     [NORTH]
       │
[WEST]─[CENTER]─[EAST]
       │
     [SOUTH]
```

Magnetic pogo-pin connectors at hexagonal frame vertices auto-align when
units are brought together. Data flows through the pins.

### 9.4 Software-Hardware Correspondence

| Software Module | Hardware Component |
|----------------|-------------------|
| `tristable_cell.py` | Individual kirigami cut cell |
| `kirigami_sheet.py` | One physical Mylar layer |
| `moire_interference.py` | Light passing through layer pair |
| `talbot_resonator.py` | Gap between layers |
| `tripole_actuator.py` | Three piezo actuators |
| `enforcer.py` | Calibration constraints |
| `esn.py` → reservoir | Coupled cell dynamics (physical) |
| Camera sensor | Feature extraction input |
| `pipeline.py` | Complete encode-compute-readout |

**Milestone: Working physical prototype that matches simulation output
within calibrated tolerance.**

---

## Path 10: Research Extensions

### 10.1 Quasicrystal Computing

The moiré pattern at non-commensurate angles forms a **quasicrystal** — an
aperiodic tiling with long-range order but no translational symmetry. This
is the Penrose tiling in 2D, and the 24-cell / 600-cell projections in 4D.

Research question: **Is quasicrystalline computation more powerful than
periodic computation?** The aperiodic structure may support richer dynamics
(more complex reservoir states) than periodic lattices.

### 10.2 Topological Quantum Error Correction Analogues

The 24-cell's self-duality and the E₈ lattice's sphere-packing optimality
are used in **quantum error-correcting codes** (the E₈ lattice code). The
moiré system may be able to implement **classical analogues** of topological
error correction:

- Errors = misclassified cell states (noise flips a cell from 0 to 0.5)
- Correction = the cascade dynamics restore the correct state via neighbor coupling
- The 24-cell geometry determines the minimum distance of the "code"

### 10.3 Higher-Dimensional Projections

Beyond E₈ → H₄, explore:

- **Leech lattice → 4D:** The Leech lattice (24D) is related to the 24-cell
  via the Golay code. Its 196,560 minimal vectors could project to even richer
  4D structures.

- **Monster group connections:** The Monster group (the largest sporadic
  simple group) has deep connections to the Leech lattice and modular forms.
  The 24-cell may be a "shadow" of these structures.

### 10.4 Biological Analogues

The stacked kirigami sheet architecture resembles:

- **Cortical columns** in the brain (layered, with lateral inhibition)
- **Retinal processing** (stacked layers of photoreceptors → bipolar → ganglion)
- **Crystallin proteins** in the eye lens (stacked, with graded refractive index)

Research question: **Do biological visual systems use moiré-like interference
for feature extraction?** The hexagonal packing of photoreceptors and the
layered structure of the retina suggest they might.

### 10.5 Market Microstructure

The system was originally designed for market data ("PPP Market Analog Computer").
The deep research path:

- **Order book as kirigami:** Bid/ask levels = cell states (0 = no orders,
  0.5 = partial fill, 1 = full level)
- **Price impact as moiré:** The interference between buyer and seller lattices
  creates a moiré pattern whose period corresponds to volatility
- **Market regimes as Trinity channels:** Trending (Alpha), mean-reverting (Beta),
  and structural-break (Gamma) regimes correspond to different dominant 16-cells
- **Flash crashes as phason worms:** Sudden topological defects propagating
  through the order book

---

## Summary: Recommended Execution Order

```
Phase 0 (Week 1):     Fix blocking bugs → system runs
Phase 1 (Weeks 2-3):  Testing infrastructure → 50+ tests, CI green
Phase 2 (Weeks 3-4):  Live dashboard → clone-and-run experience
Phase 3 (Weeks 4-8):  Real data sources → audio, market, image pipelines
Phase 4 (Weeks 6-10): 600-cell constellation → distributed computation
Phase 5 (Weeks 8-12): Differentiable physics → learnable parameters
Phase 6 (Weeks 8-12): TDA integration → topological analysis (parallel with 5)
Phase 7 (Weeks 10-14): Musical mapping → sonification and validation
Phase 8 (Weeks 12-16): WASM/WebGPU → browser-native performance
Phase 9 (Months 4-8): Physical prototype → hardware fabrication
Phase 10 (Ongoing):   Research extensions → new mathematics
```

Paths 3-7 can be pursued in parallel by different contributors.
Path 8 is a performance optimization that can happen whenever it becomes the bottleneck.
Path 9 requires physical fabrication resources.
Path 10 is open-ended research.

---

*The geometry tells you what to build. The physics tells you how to build it.
The music tells you if you built it right.*
