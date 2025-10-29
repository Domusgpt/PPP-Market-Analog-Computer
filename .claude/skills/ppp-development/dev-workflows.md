# PPP Development Workflows

## Baseline workflow
1. **Review docs**: Skim `README.md` for architecture context and `DEV_TRACK.md` for recent shifts.
2. **Define channel/preset needs**: Use the preset JSON schemas under `assets/presets/` as templates; keep channel counts within documented GPU budgets.
3. **Prototype in scripts/**: Leverage helpers like `scripts/data-recorder.js` or `scripts/data-player.js` before touching core runtime loops.
4. **Validate telemetry contracts**: Ensure new fields propagate through `PPP.sonicGeometry` hooks and their `PPP_CONFIG` mirrors.
5. **Document changes**: Append concise notes to `DEV_TRACK.md` so future sessions understand migration implications.

## Adding a new sensor stream
- Extend preset descriptors to include the sensor label and normalization curves.
- Update channel bindings in `assets/hypercube-core-config.json` (or equivalent) while honoring 64-channel ceiling.
- Mirror data into sonic telemetry by mapping to appropriate payload families (`analysis`, `signal`, `manifold`, etc.).
- Run DataRecorder to capture a validation session; confirm playback parity with DataPlayer.

## Evolving sonic geometry analytics
- Start from existing spinor modules (ResonanceAtlas → ContinuumLattice) and ensure continuity across the progression.
- When adding metrics, update both live hooks (e.g., `PPP.sonicGeometry.onTopology`) and snapshot getters.
- Keep payload names descriptive and consistent—downstream robotics and analytics clients rely on stable keys.
- Provide example interpretations in `README.md` if the metric requires domain-specific explanation.

## Hardening for release
- Exercise keyboard shortcuts, timeline scrubbing, and preset switching to confirm UX polish.
- Profile GPU and CPU utilization during a 64-channel run; keep GPU memory under 4GB.
- Export telemetry JSON and verify schema compatibility with ingestion pipelines described in the operations Skill.
- Summarize the release in `DEV_TRACK.md`, noting any API deltas or migration steps consumers must follow.
