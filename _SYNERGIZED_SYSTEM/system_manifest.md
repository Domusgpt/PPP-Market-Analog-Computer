# Synergized System Manifest

> Grand Unified Repository: 5 repos consolidated into one layered architecture.

## Architecture Overview

```
_SYNERGIZED_SYSTEM/
├── backend/engine/        ← The Physics Reactor   (Python, HEMOC-SGF)
├── frontend/platform/     ← The Control Room      (TypeScript, PPP)
├── lib/math_core/         ← The Laws of Physics   (TypeScript, HEMAC + CPE)
├── demos/                 ← The Holographic Display (Python + TypeScript)
└── docker-compose.yml     ← Orchestration
```

## Data Flow

```
Raw Input (audio/market/sensor)
    │
    ▼
┌────────────────────────────────────┐
│  Backend: Physics Reactor          │
│  (Python HEMOC-SGF Engine)         │
│  Kirigami → Moiré → Reservoir     │
│  Telemetry extraction per frame    │
└──────────┬─────────────────────────┘
           │ WebSocket (JSON)
           ▼
┌────────────────────────────────────┐
│  HemocPythonBridge.ts              │
│  Normalizes physics → PPP channels │
│  Maps to RawApiTick contract       │
└──────────┬─────────────────────────┘
           │
           ▼
┌────────────────────────────────────┐
│  Frontend: PPP Control Room        │
│  TimeBinder → GeometricLerp        │
│  → StereoscopicFeed → UI          │
│  Record / Replay / Map / Monitor   │
└────────────────────────────────────┘
```

---

## Source Mapping: Where Every Module Now Lives

### 1. Backend — Physics Reactor

| New Location | Original Repo | Original Path | Purpose |
|---|---|---|---|
| `backend/engine/physics/` | HEMOC-SGF | `src/physics/` | Moiré interference, Talbot resonator, trilatic lattice |
| `backend/engine/kirigami/` | HEMOC-SGF | `src/kirigami/` | Tristable cells, kirigami sheet, H4 kirigami |
| `backend/engine/control/` | HEMOC-SGF | `src/control/` | Tripole actuator (tip/tilt/piston) |
| `backend/engine/reservoir/` | HEMOC-SGF | `src/reservoir/` | ESN, criticality, learnable, multiscale, readout |
| `backend/engine/telemetry/` | HEMOC-SGF | `src/telemetry/` | Logger, metrics, profiler, tracer |
| `backend/engine/streaming/` | HEMOC-SGF | `src/streaming/` | Audio stream, buffer, stream/temporal encoder |
| `backend/engine/features/` | HEMOC-SGF | `src/features/` | Gabor, HOG, LPQ, moments, spectral, wavelet |
| `backend/engine/geometry/` | HEMOC-SGF | `src/geometry/` | H4 geometry, quaternion 4D |
| `backend/engine/constellation/` | HEMOC-SGF | `src/constellation/` | H4 constellation |
| `backend/engine/hdc/` | HEMOC-SGF | `src/hdc/` | Hyperdimensional computing encoder/pipeline |
| `backend/engine/ml/` | HEMOC-SGF | `src/ml/` | Auto-tune, datasets, PyTorch encoder |
| `backend/engine/core/` | HEMOC-SGF | `src/core/` | Batch encoder, cache, fast cascade/moiré |
| `backend/engine/main.py` | Rotorary | `hemoc_main.py` | Main encoding loop entry point |
| `backend/engine/pipeline.py` | Rotorary | `optical_kirigami_moire/pipeline.py` | OpticalKirigamiMoire pipeline class |
| `backend/engine/enforcer.py` | Rotorary | `optical_kirigami_moire/rules/enforcer.py` | Three rule-set enforcement |
| `backend/engine/websocket_server.py` | **NEW** | — | WebSocket bridge server (streams telemetry) |

### 2. Frontend — Control Room

| New Location | Original Repo | Original Path | Purpose |
|---|---|---|---|
| `frontend/platform/src/lib/temporal/TimeBinder.ts` | PPP | `src/lib/temporal/TimeBinder.ts` | Phase-locked temporal sync with RingBuffer |
| `frontend/platform/src/lib/temporal/GeometricLerp.ts` | PPP | `src/lib/temporal/GeometricLerp.ts` | SLERP for 4D rotations, Quaternion class |
| `frontend/platform/src/lib/fusion/StereoscopicFeed.ts` | PPP | `src/lib/fusion/StereoscopicFeed.ts` | Left/Right eye bifurcation, DataPrism |
| `frontend/platform/src/lib/adapters/AdapterBridge.ts` | PPP | `src/lib/adapters/AdapterBridge.ts` | Generic adapter → feed connector |
| `frontend/platform/src/lib/adapters/HemocOddsAdapter.ts` | PPP | `src/lib/adapters/HemocOddsAdapter.ts` | Legacy odds normalization |
| `frontend/platform/src/lib/adapters/MarketQuoteAdapter.ts` | PPP | `src/lib/adapters/MarketQuoteAdapter.ts` | Market quote normalization |
| `frontend/platform/src/lib/adapters/HemocPythonBridge.ts` | **NEW** | — | HEMOC-SGF ↔ PPP WebSocket bridge adapter |
| `frontend/platform/src/lib/contracts/AdapterContracts.ts` | PPP | `src/lib/contracts/AdapterContracts.ts` | PPPAdapter + PPPCoreConfig interfaces |
| `frontend/platform/scripts/` | PPP | `scripts/` | Full JS runtime (app, PhaseLock, Sonic, Spinor, Calibration) |
| `frontend/platform/tests/` | PPP | `tests/` | Adapter, phase-lock, calibration, telemetry tests |

### 3. Math Core — Laws of Physics

| New Location | Original Repo | Original Path | Purpose |
|---|---|---|---|
| `lib/math_core/geometric_algebra/GeometricAlgebra.ts` | HEMAC | `src/physics/GeometricAlgebra.ts` | Cl(4,0) — vectors, bivectors, rotors, sandwich product |
| `lib/math_core/geometric_algebra/Lattice24.ts` | HEMAC | `src/physics/Lattice24.ts` | 24-cell with Trinity decomposition (Alpha/Beta/Gamma) |
| `lib/math_core/geometric_algebra/CausalReasoningEngine.ts` | HEMAC | `src/physics/CausalReasoningEngine.ts` | Force → Torque → Rotor physics |
| `lib/math_core/geometric_algebra/types.ts` | HEMAC | `src/physics/types.ts` | Vector4D, Bivector4D, Rotor, LatticeVertex types |
| `lib/math_core/topology/CPE_GeometricAlgebra.ts` | CPE | `lib/math/GeometricAlgebra.ts` | CPE's Cl(4,0) implementation |
| `lib/math_core/topology/CPE_Lattice24.ts` | CPE | `lib/topology/Lattice24.ts` | CPE's 24-cell implementation |
| `lib/math_core/topology/Cell600.ts` | CPE | `lib/topology/Cell600.ts` | 600-cell (120 vertices, H4 symmetry) |
| `lib/math_core/topology/E8Projection.ts` | CPE | `lib/topology/E8Projection.ts` | E8 → H4 projection via Baez matrix |
| `lib/math_core/topology/GoldenRatioScaling.ts` | CPE | `lib/topology/GoldenRatioScaling.ts` | Phi-nested polytope structures |
| `lib/math_core/engine/TrinityEngine.ts` | CPE | `lib/engine/TrinityEngine.ts` | Trinity state vector superposition engine |
| `lib/math_core/engine/ChronomorphicEngine.ts` | CPE | `lib/engine/ChronomorphicEngine.ts` | Master coordination of Trinity + Causal engines |
| `lib/math_core/engine/CausalReasoningEngine.ts` | CPE | `lib/engine/CausalReasoningEngine.ts` | CPE's causal reasoning loop |
| `lib/math_core/tda/PersistentHomology.ts` | CPE | `lib/tda/PersistentHomology.ts` | Betti numbers, persistence diagrams |
| `lib/math_core/tda/GhostFrequencyDetector.ts` | CPE | `lib/tda/GhostFrequencyDetector.ts` | Void detection, implied harmonics |

### 4. Demos — Holographic Display

| New Location | Original Repo | Original Path | Purpose |
|---|---|---|---|
| `demos/python_demos/full_pipeline_demo.py` | Rotorary | `optical_kirigami_moire/demos/full_pipeline_demo.py` | End-to-end: rules, moiré, features, reservoir, logic |
| `demos/visualization/E8Renderer.ts` | HEMAC | `src/visualization/E8Renderer.ts` | 240 E8 roots → 4D → stereographic 3D rendering |
| `demos/visualization/MoireOverlay.ts` | HEMAC | `src/visualization/MoireOverlay.ts` | Moiré interference pattern canvas/WebGL rendering |

---

## Unique Deltas Preserved

### From HEMOC-SGF (Backend)
- **Honest Analysis / SVD Compression**: `backend/engine/features/spectral.py`, `wavelet.py`
- **Reservoir Criticality**: `backend/engine/reservoir/criticality.py` — edge-of-chaos dynamics
- **Talbot Resonator**: `backend/engine/physics/talbot_resonator.py` — integer/half-integer gap logic
- **Kirigami Mechanics**: `backend/engine/kirigami/` — tristable cells + cascade dynamics
- **HDC Encoder**: `backend/engine/hdc/` — hyperdimensional computing pipeline

### From PPP (Frontend)
- **TimeBinder**: `frontend/platform/src/lib/temporal/TimeBinder.ts` — phase-locked RingBuffer, O(log n) seek
- **GeometricLerp**: `frontend/platform/src/lib/temporal/GeometricLerp.ts` — SLERP, Rotor4D, quaternion
- **DataRecorder/Player**: `frontend/platform/scripts/DataRecorder.js`, `DataPlayer.js` — time-travel recording
- **Spinor Algebra Suite**: `frontend/platform/scripts/Spinor*.js` — 8 spinor processing modules
- **Sonic Geometry Engine**: `frontend/platform/scripts/SonicGeometryEngine.js` — 74KB audio analysis

### From HEMAC (Math Core)
- **Cl(4,0) Geometric Algebra**: `lib/math_core/geometric_algebra/GeometricAlgebra.ts` — true 4D rotation
- **24-Cell Trinity**: `lib/math_core/geometric_algebra/Lattice24.ts` — 3×8 vertex decomposition
- **Moiré Overlay Renderer**: `demos/visualization/MoireOverlay.ts` — interference visualization + GLSL
- **E8 Root Renderer**: `demos/visualization/E8Renderer.ts` — Moxness folding, triadic coloring

### From CPE (Math Core)
- **600-Cell Polytope**: `lib/math_core/topology/Cell600.ts` — 120 vertices, 25 inscribed 24-cells
- **E8 → H4 Projection**: `lib/math_core/topology/E8Projection.ts` — Baez matrix, Galois conjugation
- **Trinity Engine**: `lib/math_core/engine/TrinityEngine.ts` — Psi = [w_alpha, w_beta, w_gamma]
- **Persistent Homology**: `lib/math_core/tda/PersistentHomology.ts` — Betti numbers, barcode analysis

---

## New Components Created

### HemocPythonBridge.ts
**Location**: `frontend/platform/src/lib/adapters/HemocPythonBridge.ts`

The critical translator between Python physics and TypeScript UI. Implements `PPPAdapter` interface.

**Responsibilities:**
1. Connects via WebSocket to the Python engine at `ws://localhost:8765`
2. Receives `HemocPhysicsPayload` JSON per frame
3. Normalizes all physics metrics to 0.0–1.0 PPP signals
4. Maps to 12-channel `RawApiTick.channels[]` layout
5. Routes through `StereoscopicFeed` for phase-locked rendering
6. Supports manual ingestion for testing/playback

**Channel Map:**
| Index | Signal | Source |
|-------|--------|--------|
| 0 | moire_contrast | Fringe visibility |
| 1 | moire_frequency | Dominant spatial frequency |
| 2 | lattice_stress | Frobenius norm of stress tensor |
| 3 | reservoir_entropy | Shannon entropy |
| 4 | reservoir_lyapunov | Edge-of-chaos metric |
| 5 | talbot_gap | Gap distance |
| 6 | petal_rotation_mean | Mean petal angular state |
| 7 | cell_flat_fraction | Flat-state cell ratio |
| 8 | cell_half_fraction | Half-fold cell ratio |
| 9 | cell_full_fraction | Full-fold cell ratio |
| 10 | memory_capacity | Reservoir memory |
| 11 | logic_polarity | 0=positive, 1=negative |

### WebSocket Telemetry Server
**Location**: `backend/engine/websocket_server.py`

Python WebSocket service that wraps the HEMOC-SGF engine:
- Broadcasts physics telemetry at configurable FPS
- Accepts commands: `set_mode`, `pause`, `resume`, `reset`
- Falls back to synthetic telemetry when engine modules unavailable
- Pairs with `HemocPythonBridge.ts` on the frontend

---

## Docker Compose Services

| Service | Container | Port | Purpose |
|---------|-----------|------|---------|
| `engine` | hemoc-sgf-engine | 8765 | Python physics + WebSocket telemetry |
| `frontend` | ppp-frontend | 5173 | Vite dev server + PPP UI |

```bash
# Start everything
docker compose up --build

# Access the UI
open http://localhost:5173

# WebSocket direct access (for debugging)
wscat -c ws://localhost:8765
```

---

## Source Repository References

| Repo | Branch Used | Role |
|------|-------------|------|
| PPP Market Analog Computer | `codex/2026-02-06/analyze-recent-branches-for-refactor-plan` | Frontend spine (telemetry, adapters, UI) |
| HEMOC-SGF (via Rotorary) | `claude/analyze-repos-1fUCE` | Backend physics engine (Python) |
| HEMAC Holographic | `claude/integrate-chronomorphic-e8-engines-P9F3Q` | Geometric Algebra, E8, visualization |
| Chronomorphic Polytopal Engine | `claude/integrate-cpe-engine-CdsEN` | Trinity engine, 600-cell, E8 projection, TDA |
| HEMOC-Rotorary | `claude/analyze-repos-1fUCE` | Demo wrapper, consolidated pipeline |
