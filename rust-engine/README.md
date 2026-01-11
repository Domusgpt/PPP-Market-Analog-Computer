# Geometric Cognition Engine

A GPU-accelerated engine for analog cognitive computation using high-dimensional geometry and visual processing. This engine encodes data as configurations of 4D polytopes (24-cell, 600-cell, 120-cell) and projects them to 2D images, effectively using rendering as computation.

## Architecture

```
Input Data → 4D Geometric Encoding → 3D Projection → 2D Rendering → Output Analysis
```

### Modules

- **geometry/** - Core 4D polytope implementations, quaternion rotations, projections
- **cognitive/** - Trinity logic, dialectic reasoning, rule engine
- **rendering/** - WebGPU pipeline, WGSL shaders, pixel-level computation
- **pipeline/** - Data ingestion, mapping, output channels
- **analysis/** - Homological signal extraction, pattern detection

## Key Features

### Mathematically-Rigorous Geometry
- Accurate 24-cell, 600-cell, and 120-cell polytope models
- Quaternion-based 4D rotations (SO(4) decomposition)
- E₈ lattice projection with golden ratio (φ) scaling
- Multiple projection modes: orthographic, perspective, stereographic

### Triadic Dialectic Reasoning
- 24-cell Trinity decomposition into three orthogonal 16-cells (Alpha/Beta/Gamma)
- Thesis-antithesis-synthesis logic through geometric overlap
- Phillips Synthesis: visual interference as computational result

### Pixel-Rule Analog Processing
- Alpha blending for implicit state combination
- Custom WGSL shaders for cellular automata, diffusion, edge detection
- GPU compute shaders for parallel pixel-rule execution

### Homological Signal Extraction
- Betti number computation (β₀ clusters, β₁ loops, β₂ voids)
- Topological signal detection
- Pattern recognition for meaningful configurations

## Building

```bash
# Desktop application (with windowed display)
cargo build --release --features desktop

# Library only
cargo build --release

# WebAssembly target
cargo build --release --target wasm32-unknown-unknown --features web
```

## Usage

```bash
# Run with display
cargo run --release --features desktop

# Headless mode (testing/benchmarking)
cargo run --release -- --headless

# Performance benchmark
cargo run --release -- --benchmark

# Expanded 600-cell mode
cargo run --release --features desktop -- --expanded
```

## API Example

```rust
use geometric_cognition::{GeometricCognitionEngine, EngineConfig};

// Create engine with default configuration
let config = EngineConfig {
    target_fps: 60,
    data_channels: 64,
    expanded_mode: false,
    e8_layer_enabled: true,
    ..Default::default()
};

let mut engine = GeometricCognitionEngine::new(config);

// Inject external data
engine.data_pipeline_mut().inject(0, 0.75); // Channel 0
engine.data_pipeline_mut().inject_all(&[0.5, 0.3, 0.8]); // Multiple channels

// Update simulation
engine.update(1.0 / 60.0);

// Analyze state
let analysis = engine.analyze();
println!("Betti numbers: β₀={}, β₁={}", analysis.betti.b0, analysis.betti.b1);
println!("Detected patterns: {:?}", analysis.patterns);

// Access geometry directly
let state = engine.geometry().state();
println!("Synthesis detected: {}", state.synthesis_detected);
```

## Geometry Module

### 24-Cell (Icositetrachoron)
- 24 vertices, 96 edges, 96 triangular faces, 24 octahedral cells
- Coordinates: permutations of (±1, ±1, 0, 0)
- Trinity decomposition: 3 × 16-cell (Alpha, Beta, Gamma)

### 600-Cell (Hexacosichoron)
- 120 vertices, 720 edges, 1200 faces, 600 tetrahedral cells
- Vertices correspond to 120 icosian quaternions
- Contains ~25 embedded 24-cells

### 120-Cell (Hecatonicosachoron)
- 600 vertices, 1200 edges, 720 faces, 120 dodecahedral cells
- Dual to the 600-cell

## Cognitive Layer

### Trinity State
```rust
// Access Trinity components
let trinity = engine.cognitive_layer().trinity_state();
println!("Tension: {}", trinity.dialectic_tension());
println!("Coherence: {}", trinity.synthesis_coherence());
println!("Resonant: {}", trinity.is_resonant());
```

### Cognitive Rules
```rust
use geometric_cognition::cognitive::{CognitiveRule, RuleCondition, RuleAction};

// Add custom rule
let rule = CognitiveRule::new(
    "high_tension_expand",
    RuleCondition::TensionAbove(0.8),
    RuleAction::SetMode(GeometryMode::Expanded600),
);
engine.cognitive_layer_mut().add_rule(rule);
```

## Data Pipeline

### Input Mapping
```rust
use geometric_cognition::pipeline::{MappingConfig, MappingTarget, MappingFunction};

// Map channel 0 to XY rotation with custom function
let mapping = MappingConfig::new(0, MappingTarget::RotationXY)
    .with_range(0.0, std::f64::consts::TAU)
    .with_function(MappingFunction::SmoothStep);

engine.data_pipeline_mut().mapper_mut().set_mapping(mapping);
```

### Output Signals
```rust
let output = engine.data_pipeline().output();
if let Some(tension) = output.get_scalar("tension") {
    println!("Current tension: {}", tension);
}

// Export history to JSON
let json = output.export_json()?;
```

## Performance

- Target: 60 FPS with 64 data channels
- 24-cell mode: ~0.1ms per frame (CPU only)
- 600-cell mode: ~0.5ms per frame (CPU only)
- GPU rendering adds minimal overhead with modern hardware

## License

MIT

## References

1. Regular Polytopes (H.S.M. Coxeter)
2. Quaternions and Rotation Sequences (J.B. Kuipers)
3. Computational Topology (H. Edelsbrunner, J.L. Harer)
4. E₈ Root System and 4D Polytopes (J.H. Conway)
