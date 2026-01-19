# Sonic Geometry Telemetry

PPP's sonic geometry subsystem mirrors 4D rotations into structured data streams so robotics, multimodal transformers, or analytics pipelines can ingest insights without audio playback.

## Access points
- Subscribe via `PPP.sonicGeometry.on*` hooks (e.g., `onAnalysis`, `onSignal`, `onTransduction`, `onManifold`, `onTopology`, `onContinuum`, `onLattice`).
- Pull snapshots synchronously through `PPP.sonicGeometry.get*` helpers, or use `PPP_CONFIG` mirrors (e.g., `PPP_CONFIG.onSonicAnalysis`) for configuration-time listeners.

## Payload families
### Analysis payload (`analysis` / `transmission`)
- `gateDensity`, `gateContinuity` — duty-cycle measures of harmonic lattice activation.
- `spectralCentroid`, `averageFrequency` — tonal centroid and carrier averages.
- `averageFmRate`, `averageAmRate` — modulation rates tied to transport.
- `sequence` — hex characters for synchronized sequencing.
- `carriers` — per-voice sub/prime/hyper band metrics (frequency, amplitude, energy).
- `spinor`, `resonance`, `signal`, `transduction`, `manifold`, `topology`, `continuum`, `lattice` — nested telemetry linking quaternion bridges with carrier dynamics.

### Signal payload (`signal`)
- `voices` — harmonic voices with carrier amplitudes, gate duty, spinor ratios, quaternion weights, resonance vectors.
- `carrierMatrix` — frequency cells with gate intensity and energy.
- `bitstream` — phase slots and binary duty-cycle bits.
- `quantum` / `spinor` / `resonance` mirrors — keep bit-level data aligned with geometric telemetry.
- `envelope` — centroid, spread, resonance magnitude, timeline progress.

### Transduction payload (`transduction`)
- `invariants` — determinant, trace, Frobenius norm, Hopf alignment, bridge magnitude.
- `matrix` — normalized 4×4 rotation matrix driving the double-quaternion bridge.
- `topology`, `voices`, `grid` — projections tying carrier fields to rotational planes and Hopf fibers.

### Manifold payload (`manifold`)
- `quaternion`, `spinor`, `resonance`, `signal`, `transduction` — aggregate metrics for monitoring sonic-visual alignment.
- `summary`, `alignment` — condensed energy/coherence indicators.
- `voices` — per-voice linkage across all subsystems.

### Topology payload (`topology`)
- `matrix`, `axes`, `spectrum`, `braiding`, `bridge` — correlate quaternion axes with gate, carrier, and bitstream flux.

### Continuum payload (`continuum`)
- `flux`, `continuum`, `axes`, `voices`, `braiding` — orientation vectors showing how quaternion bridges steer the harmonic field.

### Lattice payload (`lattice`)
- `orientation`, `synergy`, `axes`, `voices`, `carriers`, `spectral`, `timeline` — fuse continuum vectors with carrier energy and bit entropy for robotics-ready ingestion.

### Quaternion payload (`quaternion`)
- Captures double-quaternion factorization metrics for the live rotation matrix, ensuring downstream systems can align directly with PPP's geometric core.

## Talking points
- Stress that every payload mirrors onto `PPP_CONFIG` hooks and remains available even when resonance audio is muted.
- Emphasize interoperability: telemetry is designed for robotics receivers, multimodal transformer ingestion, and analytics dashboards.
- When detailing integrations, map question keywords to payload fields (e.g., "phase orbit" → `spinor`, "carrier energy" → `carriers` or `carrierMatrix`).
