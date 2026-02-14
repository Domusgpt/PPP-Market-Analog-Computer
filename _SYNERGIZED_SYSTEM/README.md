# Synergized System: Optical Kirigami Moire Analog Computer

A physics-based analog computing engine that uses light interference through stacked kirigami sheets to perform computation. The system encodes arbitrary data (audio, images, sensor readings) into structured moire interference patterns.

## Architecture

```
_SYNERGIZED_SYSTEM/
├── backend/engine/          # Python physics engine (core computation)
│   ├── kirigami/            # Tristable cell lattice + reservoir dynamics
│   ├── physics/             # Moire interference, Talbot resonance, trilatic lattice
│   ├── geometry/            # H4 polytope geometry (24-cell, 600-cell, quaternions)
│   ├── constellation/       # 5-node 600-cell distributed network
│   ├── features/            # Gabor, spectral, HOG feature extraction
│   ├── hdc/                 # Hyperdimensional computing (bind/bundle/permute)
│   ├── reservoir/           # Echo State Network + criticality analysis
│   ├── control/             # Tripole actuator control
│   ├── enforcer.py          # Three rule sets enforcement
│   ├── pipeline.py          # Main OpticalKirigamiMoire pipeline
│   └── main.py              # CLI encoder entry point
├── backend/tests/           # 143 pytest tests
├── frontend/platform/       # TypeScript visualization spine
│   └── src/lib/             # PPP adapter contracts, fusion, temporal
├── lib/math_core/           # HEMAC geometric algebra + CPE topology
├── demos/                   # Python pipeline demos
└── bridge/                  # HemocPythonBridge (Python WebSocket → TypeScript)
```

## Quick Start

```bash
# From repository root
./run.sh --install    # Install Python dependencies
./run.sh --test       # Run 143 tests (all pass)
./run.sh              # Run full pipeline demo
```

Or manually:

```bash
cd _SYNERGIZED_SYSTEM/backend
pip install numpy scipy websockets pytest
python -m pytest tests/ -v          # Run tests
python -m engine.main --mode demo   # Run encoder demo
cd .. && python demos/python_demos/full_pipeline_demo.py  # Full demo
```

## Three Rule Sets

The system is governed by three physical rule sets enforced by `engine/enforcer.py`:

1. **Angular Commensurability** (Rule Set 1): Twist angles are quantized to commensurate values where `cos(theta) = (n^2 + 4mn + m^2) / (2(n^2 + mn + m^2))` for coprime integers (m,n). This ensures periodic moire superlattices.

2. **Trilatic Tilt Symmetry** (Rule Set 2): Tilt axes are constrained to hexagonal symmetry directions (k x 60 degrees). This preserves the trilatic (3-fold) symmetry of the lattice.

3. **Talbot Distance** (Rule Set 3): Layer gaps snap to Talbot resonances `z = n * z_T` (integer = AND/OR logic) or `z = (n + 1/2) * z_T` (half-integer = NAND/XOR logic), where `z_T = 2a^2 / lambda`.

## Core Concepts

**Tristable Cells**: Each cell in the kirigami sheet has three stable states {0, 0.5, 1} defined by a triple-well potential `U(x) = k[x(x-0.5)(x-1)]^2`.

**24-Cell Decomposition**: The system's geometry is governed by the 24-cell polytope (24 vertices in 4D), which decomposes into three mutually inscribed 16-cells (Alpha/Beta/Gamma channels). This Trinity decomposition maps to 3 cyan/magenta layer pairs in the physical device.

**600-Cell Constellation**: Five 24-cell nodes assemble into a 600-cell (120 vertices), forming the H4 constellation network for distributed computing.

**Reservoir Computing**: The kirigami lattice functions as an echo state network operating at the edge of chaos, providing fading memory for temporal data processing.

## Data Pipeline

```
Input Data → Normalize → Inject into KirigamiSheet
  → Cascade Dynamics (reservoir) → State Field
  → Moire Interference (layer1 x layer2) → Pattern
  → Feature Extraction (Gabor/Spectral/HOG/HDC)
  → Readout Layer → Classification
```

## Tests

143 tests across 11 modules covering:
- Rule enforcement (enforcer)
- Tristable cell mechanics
- Kirigami sheet dynamics
- Moire interference physics
- Talbot resonance
- H4 polytope geometry (24-cell, 16-cell, 600-cell)
- Trinity decomposition
- Quaternion 4D rotations
- Echo State Network reservoir
- Feature extraction (Gabor, spectral, HOG)
- Hyperdimensional computing
- Constellation network

## Related Documentation

- [CODING_AGENT_PROMPT.md](../CODING_AGENT_PROMPT.md) - Detailed architecture for coding agents
- [WHAT_THIS_DOES.md](../WHAT_THIS_DOES.md) - Comprehensive system explanation
- [EVOLUTION_PATHS.md](../EVOLUTION_PATHS.md) - Development roadmap
