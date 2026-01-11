# Sonic Geometry Telemetry Reference

PPP mirrors 4D rotations into structured data streams for machine ingestion. Use this map when wiring analytics, robotics, or data platforms to PPP outputs.

## Access points
- Subscribe to live updates with `PPP.sonicGeometry.on*` hooks (e.g., `onAnalysis`, `onSignal`, `onTransduction`, `onManifold`, `onTopology`, `onContinuum`, `onLattice`).
- Pull synchronous snapshots using `PPP.sonicGeometry.get*` helpers or configuration-time listeners on `PPP_CONFIG` (e.g., `PPP_CONFIG.onSonicAnalysis`).
- DataRecorder exports payloads as JSON with identical field names; downstream ETL can consume them without schema translation.

## Payload families
### Analysis payload (`analysis` / `transmission`)
- `gateDensity`, `gateContinuity` — duty cycle measures of harmonic lattice activation.
- `spectralCentroid`, `averageFrequency` — tonal centroid and carrier averages for anomaly detection.
- `averageFmRate`, `averageAmRate` — modulation rates tied to transport vectors.
- `sequence` — synchronized hex identifiers for aligning with visual frames.
- `carriers` — per-voice metrics (frequency, amplitude, energy) ready for robotics or DSP ingestion.
- Additional mirrors: `spinor`, `resonance`, `signal`, `transduction`, `manifold`, `topology`, `continuum`, `lattice`.

### Signal payload (`signal`)
- `voices` — harmonic voices with carrier amplitudes, gate duty, spinor ratios, quaternion weights, resonance vectors.
- `carrierMatrix` — frequency cells with gate intensity and energy, suited for heatmap dashboards.
- `bitstream` — phase slots and binary duty-cycle bits for firmware-level integrations.
- `quantum` / `spinor` / `resonance` mirrors — keep bit-level data aligned with geometric telemetry.
- `envelope` — centroid, spread, resonance magnitude, timeline progress.

### Transduction payload (`transduction`)
- `invariants` — determinant, trace, Frobenius norm, Hopf alignment, bridge magnitude.
- `matrix` — normalized 4×4 rotation matrix powering the double-quaternion bridge.
- `topology`, `voices`, `grid` — projections tying carrier fields to rotational planes and Hopf fibers.

### Manifold payload (`manifold`)
- Aggregates `quaternion`, `spinor`, `resonance`, `signal`, `transduction` metrics to monitor sonic-visual alignment.
- Includes `summary`, `alignment`, and per-voice linkage across subsystems.

### Topology payload (`topology`)
- `matrix`, `axes`, `spectrum`, `braiding`, `bridge` — correlate quaternion axes with carrier flux.

### Continuum payload (`continuum`)
- `flux`, `continuum`, `axes`, `voices`, `braiding` — orientation vectors guiding harmonic field steering.

### Lattice payload (`lattice`)
- `orientation`, `synergy`, `axes`, `voices`, `carriers`, `spectral`, `timeline` — fuse continuum vectors with carrier energy and bit entropy for robotics-ready ingestion.

### Quaternion payload (`quaternion`)
- Captures double-quaternion factorization metrics for aligning downstream systems with PPP's rotation matrix.

## Integration pointers
- Mirror payloads onto message buses or telemetry stores as-is; field names are stable across recorder and live streams.
- If bandwidth is constrained, downsample to `analysis` + `signal` payloads—they expose high-level metrics while keeping per-voice fidelity.
- Use DataPlayer to validate ingestion by replaying recorded sessions and verifying parity with downstream analytics.
