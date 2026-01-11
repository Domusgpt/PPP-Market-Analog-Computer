# PPP Data Ingestion Guide

## Preparation checklist
- Confirm presets in `assets/presets/` match the sensor payloads you plan to stream; adjust normalization curves before runtime.
- Decide whether you need live streaming (`PPP.sonicGeometry.on*`) or offline review (DataRecorder JSON + DataPlayer replay).
- Allocate storage/bandwidth budgets—64 channels at 60fps generate sizable telemetry; plan compression or sampling strategies.

## Live integration steps
1. **Initialize PPP**: Load the web runtime or embed the core bundle; call `PPP.init()` with the desired preset.
2. **Attach listeners**: Register to the relevant sonic geometry hooks (e.g., `PPP.sonicGeometry.onAnalysis`) and visual callbacks if you need synchronized frames.
3. **Normalize inputs**: Feed sensor data through PPP's expected range (see preset metadata); avoid saturating rotation planes by clamping to documented bounds.
4. **Stream to downstream systems**:
   - For robotics: forward `lattice`/`continuum` payloads over your control bus.
   - For analytics: publish `analysis`/`signal` payloads to telemetry stores (Kafka, Pulsar, etc.).
   - For dashboards: convert `carrierMatrix` and `topology` matrices into heatmaps.
5. **Monitor health**: Watch `gateContinuity`, `resonance`, and `bridge` metrics for drift; trigger alerts when thresholds breach domain-specific limits.

## Recorded workflow
1. **Capture**: Use `scripts/data-recorder.js` or in-app recorder controls to save sessions as JSON, keeping notation consistent with live payloads.
2. **Store**: Index recordings with metadata (preset version, sensor firmware, timestamps) so they can be correlated with downstream analyses.
3. **Replay**: Run `scripts/data-player.js` or the UI player to validate ingestion pipelines—compare downstream metrics to PPP visual/sonic output.
4. **Annotate**: Document notable frames or anomalies for future training datasets.

## Interop best practices
- Treat PPP JSON as the source of truth; avoid renaming fields so Skills and downstream tools remain aligned.
- Version presets and ingestion schemas together—when presets change, update ETL configs and notify stakeholders.
- Use silent mode when audio output is unnecessary; telemetry fidelity is unaffected and reduces operational noise.
- Reference the development Skill before modifying payload structures to ensure contract stability.
