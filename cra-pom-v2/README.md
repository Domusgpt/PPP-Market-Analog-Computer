# CRA-POM v2 - Geometric Cognition Kernel

A fully functional, non-simulated implementation of the Geometric Cognition Kernel where "reasoning" is a real geometric trajectory and "safety" is a calculated mathematical constraint.

## Architecture

### The Mathematical Core (No Simulations)

- **Quaternion Class**: Full Hamilton algebra implementation including product, conjugation, normalization, SLERP interpolation
- **24-Cell Lattice**: The actual D4 lattice vertices (permutations of ±1, ±1, 0, 0) - not a hypercube or 600-cell
- **Isoclinic Rotations**: Clifford translations using double quaternion multiplication (q_L × P × q_R)
- **Golden Ratio Ergodicity**: Non-repeating trajectories through the use of φ in rotation axes
- **Convexity Constraint**: Real L1-norm distance check for safety (|w| + |x| + |y| + |z| ≤ 2)

### The Audit Layer (TRACE)

- **Hash-Chained Telemetry**: SHA-256 using Web Crypto API
- **Immutable Ledger**: Each entry contains `hash_i = SHA256(state || previousHash || metadata)`
- **Real Geometric State**: Actual 4D coordinates, not placeholder strings
- **Chain Validation**: Full integrity verification of the audit trail

### The Visualization Layer

- **Stereographic Projection**: 4D → 3D → 2D pipeline
- **Perspective Rendering**: Camera-based 3D to 2D with depth sorting
- **Real-Time Updates**: requestAnimationFrame-driven at 60fps
- **Visual Feedback**: Color-coded safety status (SAFE/WARNING/VIOLATION)

## File Structure

```
cra-pom-v2/
├── package.json           # Dependencies and scripts
├── vite.config.ts         # Vite configuration
├── tsconfig.json          # TypeScript configuration
├── index.html             # Entry HTML
├── setup.sh               # Termux setup script
│
└── src/
    ├── main.tsx           # React entry point
    ├── App.tsx            # Main application orchestrator
    ├── vite-env.d.ts      # Vite types
    │
    ├── core/
    │   ├── index.ts       # Core exports
    │   ├── geometry.ts    # Quaternion, Lattice24, CognitiveManifold
    │   ├── trace.ts       # AuditChain with SHA-256 hash chain
    │   └── projection.ts  # 4D→3D→2D stereographic projection
    │
    └── components/
        ├── index.ts           # Component exports
        ├── PolytopeCanvas.tsx # 24-cell visualization
        ├── ControlPanel.tsx   # System controls
        └── AuditLog.tsx       # Live hash chain display
```

## Installation

### Termux (Android)

```bash
# Clone or copy the project
cd cra-pom-v2

# Run the setup script
./setup.sh

# Start the dev server
npm run dev
```

### Standard Node.js

```bash
cd cra-pom-v2
npm install
npm run dev
```

## Usage

### Controls

- **Run/Pause**: Toggle continuous inference steps
- **Single Step**: Advance one inference step
- **Auto-Rotate**: Spin the 3D view
- **Inject Entropy**: Perturb the thought vector with random values
- **Reset**: Return to initial state

### Understanding the Display

- **Left Panel**: System status, geometry metrics, chain statistics
- **Center**: 3D visualization of the 24-cell with thought vector
- **Right Panel**: Live TRACE audit log with SHA-256 hashes

### Safety States

- **SAFE** (Green): Thought vector well inside convex hull
- **WARNING** (Yellow): Approaching boundary
- **VIOLATION** (Red): Outside convex hull (auto-constrained)

## Mathematical Details

### 24-Cell Vertices

24 vertices at all permutations of (±1, ±1, 0, 0):
- 6 position combinations × 4 sign combinations = 24 vertices
- All at distance √2 from origin
- Edges connect vertices at distance √2

### Isoclinic Rotation

The double quaternion operation P' = q_L × P × q_R represents:
- Simultaneous rotation in two orthogonal planes
- A property unique to 4D (Clifford translation)
- Preserves the norm of P

### Golden Ratio Ergodicity

Using φ = (1 + √5)/2 in rotation axes ensures:
- Irrational rotation angles
- Non-periodic trajectories
- Dense coverage of the manifold

### Convexity Constraint

For the 24-cell with these vertices, interior is defined by:
- L1-norm constraint: |w| + |x| + |y| + |z| ≤ 2
- This is the mathematical definition of "safe" reasoning

## Performance

- Optimized for mobile (Termux)
- Uses requestAnimationFrame for smooth animation
- Canvas 2D rendering (no WebGL required)
- Hash computation is async to avoid blocking

## License

Part of the PPP (Polytopal Projection Processing) project.
