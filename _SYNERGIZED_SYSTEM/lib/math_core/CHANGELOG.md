# Math Core — Change Log

## 2026-02-15: CPE Integration + Import Fix + Module Consolidation

### Context

The TypeScript math core had 5 broken imports in `ChronomorphicEngine.ts`,
two missing modules (MusicGeometryDomain, HDCEncoder) that were only stubs,
and 3 duplicated files across directories. The real implementations exist in
the **Chronomorphic Polytopal Engine (CPE)** repo at
`Domusgpt/-Chronomorphic-Polytopal-Engine` (branch
`security/fix-hdc-memory-leak-*`), which has 21 TypeScript files totaling
12,347 lines of code.

This change ports the real CPE implementations into PPP's `math_core/`
directory, adapting import paths to match PPP's layout where types,
GeometricAlgebra, and Lattice24 live under `geometric_algebra/` instead of
CPE's `types/`, `math/`, and `topology/Lattice24`.

### Import Path Mapping (CPE → PPP)

| CPE path | PPP path | Reason |
|---|---|---|
| `../types/index.js` | `../geometric_algebra/types.js` | Types are in `geometric_algebra/` |
| `../math/GeometricAlgebra.js` | `../geometric_algebra/GeometricAlgebra.js` | GA is in `geometric_algebra/` |
| `../topology/Lattice24.js` | `../geometric_algebra/Lattice24.js` | Lattice24 is in `geometric_algebra/` |
| `../topology/Cell600.js` | `../topology/Cell600.js` | Same path |
| `../topology/GoldenRatioScaling.js` | `../topology/GoldenRatioScaling.js` | Same path |
| `../tda/*` | `../tda/*` | Same path |
| `../domains/*` | `../domains/*` | Same path |
| `../encoding/*` | `../encoding/*` | Same path |
| `../metacognition/*` | `../metacognition/*` | Same path (new dir) |
| `../applications/*` | `../applications/*` | Same path (new dir) |

### What Changed

#### 1. Archived Duplicate Files → `_archived/`

Three files existed in two locations with conflicting import paths.
Canonical versions are in `geometric_algebra/`. Duplicates moved to
`_archived/` (not deleted, marked as superseded).

| File | From | Canonical |
|---|---|---|
| `CPE_GeometricAlgebra.ts` | `topology/` | `geometric_algebra/GeometricAlgebra.ts` |
| `CPE_Lattice24.ts` | `topology/` | `geometric_algebra/Lattice24.ts` |
| `CausalReasoningEngine.ts` | `engine/` | `geometric_algebra/CausalReasoningEngine.ts` |

#### 2. Fixed ChronomorphicEngine.ts Imports (5 changes)

The unified engine (758 lines) referenced directories that don't exist in
PPP. All 5 broken imports were remapped:

```
'../types/index.js'            → '../geometric_algebra/types.js'
'../topology/Lattice24.js'     → '../geometric_algebra/Lattice24.js'
'./CausalReasoningEngine.js'   → '../geometric_algebra/CausalReasoningEngine.js'
'../math/GeometricAlgebra.js'  → '../geometric_algebra/GeometricAlgebra.js'
```

#### 3. Replaced MusicGeometryDomain Stub (130 → 628 lines)

The stub returned zeros and had TODO comments. Replaced with real CPE
implementation featuring:
- Key-to-vertex mapping (24 vertices → 24 keys via circle of fifths)
- Chord encoding with Trinity weight computation
- Neo-Riemannian PLR transformations
- Circle of fifths navigation
- Key detection from pitch class sets
- Position-to-key mapping via lattice proximity

#### 4. Replaced HDCEncoder Stub (185 → 552 lines)

The stub used `Math.random()` with no seeding and had placeholder encoding.
Replaced with real CPE implementation featuring:
- Seeded RNG for reproducibility
- LRU memory cache (configurable limit, default 10,000)
- Lattice-integrated vertex hypervectors
- Thermometer encoding for numeric values
- 4D vector encoding via bind/bundle
- Sequence encoding via permutation
- Similarity search (query top-K)

#### 5. Added 3 New Module Directories (7 files, ~5,400 lines)

**`metacognition/`** (2 files):
- `StateClassifier.ts` — Rule-based state classification (COHERENT, TRANSITIONING, AMBIGUOUS, POLYTONAL, STUCK, INVALID) with confidence scoring, action suggestions, and transition tracking
- `EmbeddingClassifier.ts` — Optional embedding-based classification using external APIs (OpenAI, Cohere) or custom embedders, with few-shot examples

**`applications/`** (3 files):
- `MusicGenerationMode.ts` — Algorithmic music generation: FFT→chroma→key detection, Trinity axis modulation, ghost frequency resolution as notes
- `AnomalyDetectionMode.ts` — Multivariate anomaly detection: distance/statistical/contextual/topological scoring, CUSUM drift detection, baseline profiling
- `RoboticsControlAdapter.ts` — IMU→4D polytope mapping: complementary filter, cascade PID control, motor mixing, trajectory following, quadcopter presets

**`shaders/`** (2 files):
- `TrinityShaders.ts` — WebGL2 vertex/fragment shaders: 6D rotation, stereographic projection, Trinity axis coloring, phase shift glow, edge rendering
- `MultiLayerRenderer.ts` — Multi-layer rendering: φ-scaled layers, cross-section slicing, depth sorting, layer presets (trinity, nested, e8)

#### 6. Added tsconfig.json

New `tsconfig.json` for type checking the math core:
- Target: ES2022, Module: ES2022, moduleResolution: node16
- Strict mode enabled
- Includes all 10 source directories, excludes `_archived/`

#### 7. Added index.ts Barrel Export

Unified entry point re-exporting all modules with properly adapted paths.

#### 8. Fixed WebSocket Server

- Changed bare imports to relative (`from physics.` → `from .physics.`)
- Added `RealEngine` class wrapping actual physics modules
- Conditional engine selection: `RealEngine` when physics available, `SyntheticEngine` fallback

### Files Added/Modified

| File | Action | Lines |
|---|---|---|
| `domains/MusicGeometryDomain.ts` | Replaced stub | 628 |
| `encoding/HDCEncoder.ts` | Replaced stub | 552 |
| `metacognition/StateClassifier.ts` | New | ~530 |
| `metacognition/EmbeddingClassifier.ts` | New | ~380 |
| `applications/MusicGenerationMode.ts` | New | ~400 |
| `applications/AnomalyDetectionMode.ts` | New | ~400 |
| `applications/RoboticsControlAdapter.ts` | New | ~650 |
| `shaders/TrinityShaders.ts` | New | ~350 |
| `shaders/MultiLayerRenderer.ts` | New | ~580 |
| `index.ts` | New | ~210 |
| `tsconfig.json` | Modified | +3 lines |
| `engine/ChronomorphicEngine.ts` | Modified | 5 import changes |
| `_archived/README.md` | New | ~20 |
| `_archived/` (3 files) | Moved | — |
| `../../backend/engine/websocket_server.py` | Modified | ~70 lines added |
