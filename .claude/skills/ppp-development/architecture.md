# PPP Architecture & Runtime Notes

## Core paradigm
- **Polytopal Projection Processing (PPP)** encodes state as unified 4D polytopes (tesseracts, 600-cells) evolved through six concurrent rotation planes (XY, XZ, YZ, XW, YW, ZW).
- Sensor fusion inputs feed directly into the rotation manifold, enabling GPS-denied navigation and multimodal analytics without collapsing dimensionality.
- The projection pipeline prioritizes deterministic transforms so downstream subsystems (visual, sonic, analytics) can replay identical manifolds across runs.

## Visualization and engine composition
- WebGPU/WebGL2 renderer sustains 60fps while staying under a 4GB GPU budget; channels scale to 64 synchronized feeds with shadow projections for machine vision.
- `HypercubeCore` orchestrates render loops, preset loading, and uniform updates, with helper modules under `scripts/` exposing CLI hooks for automation.
- Recorder/Player suites support JSON export, looped playback, timeline scrubbing, uniform snapshots, and shortcut bindings (Space, ←/→, Home/End) for QA sessions.

## Sonic geometry coupling
- The **SonicGeometryEngine** maps rotation matrices into harmonic fields using double-quaternion bridges and Hopf fiber alignment.
- Sonic telemetry remains accessible when audio is muted; receivers can subscribe via `PPP.sonicGeometry.*` APIs or configuration mirrors in `PPP_CONFIG`.
- Spinor subsystems (ResonanceAtlas, SignalFabric, TransductionGrid, MetricManifold, TopologyWeave, FluxContinuum, ContinuumLattice) expose progressive analytics for robotics and multimodal ingestion.

## Developer touchpoints
- Core access points live on the global `PPP` object (`PPP.init`, `PPP.render`, `PPP.sonicGeometry.getResonance()`), with mirrors in `PPP_CONFIG` for build-time wiring.
- `DEV_TRACK.md` captures sequencing decisions and migration notes—consult before altering channel limits, preset schemas, or telemetry contracts.
- The repository's `scripts/` helpers automate packaging, preset validation, and playback/record pipelines; extend them instead of reimplementing bespoke tooling.

## Design ethos to preserve
- Maintain the "Revolutionary Computational Paradigm" narrative: PPP replaces sequential processing with geometric reasoning that unifies visualization, telemetry, and harmonics.
- Keep developer ergonomics centered on rapid experimentation—configuration-first controls, JSON payloads, and restart-free tuning.
- Guard interoperability: every runtime enhancement should maintain compatibility with robotics receivers, analytics dashboards, and multimodal transformers relying on PPP payloads.
