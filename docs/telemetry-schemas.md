# PPP Telemetry Schemas

The Sonic Geometry pipeline emits immutable JSON snapshots every frame so robotics and multimodal research stacks can ingest the
same data that drives the in-browser visualization. This guide documents the schema for each channel and links to canonical samp
les under [`samples/telemetry/`](../samples/telemetry/).

All payloads are JSON-serializable and safe to persist verbatim. Numeric values are normalized between 0–1 unless otherwise noted.

## Live Telemetry Status (`PPP.live.getStatus()`)

Hardware harnesses frequently poll `PPP.live.getStatus()` to track ingest health. The payload mirrors the control-panel telemetry banner (`liveStreamTelemetry`) and includes:

| Field | Type | Description |
| --- | --- | --- |
| `mode` | string | `idle`, `websocket`, `serial`, or the custom source passed to `PPP.live.ingest()`. |
| `connected` | boolean | Adapter connection state. |
| `frames` | number | Frames successfully applied since the last reset. |
| `lastSource` | string | The most recent frame source identifier. |
| `lastTimestamp` | number | Timestamp from the latest frame (falls back to `Date.now()` when absent). |
| `lastLatency` / `avgLatency` / `minLatency` / `maxLatency` | number | Millisecond latency metrics derived from frame timestamps vs. wall-clock arrival. |
| `interFrameGap` / `peakInterFrameGap` | number | Milliseconds between the last two frames and the peak gap observed this session. |
| `channelCount` / `channelLimit` | number | Last frame’s mapped channel count and the enforced ceiling. |
| `channelSaturation.latest` / `.peak` | number | Utilization ratio (0–1) for the most recent frame and the observed max. |
| `drops` / `parseErrors` | number | Aggregate malformed-frame drops and JSON parse failures (across adapters + API ingestion). |
| `connectAttempts` / `reconnectAttempts` | number | Total connection attempts and reconnect cycles for the active adapter. |
| `checksum.status` | string | `valid`, `mismatch`, `skipped`, or `absent` for the most recent frame. |
| `checksum.validated` / `checksum.failures` | number | Counts of checksum successes vs. mismatches during the session. |
| `checksum.lastReported` / `checksum.lastComputed` | string \| null | Reported checksum value and the runtime’s computed hash when available. |
| `lastStatus` / `lastStatusLevel` / `lastStatusAt` | string, string, number | Most recent status message, severity (`info`, `warning`, `error`), and timestamp emitted by the adapters or API. |
| `lastErrorCode` | string \| null | Adapter-supplied error code (`websocket-error`, `serial-parse-error`, etc.). |
| `telemetrySummary` | string | Human-readable rollup matching the UI banner (`frames 128 • latency 34ms (avg 32ms) • …`). |
| `statusLog` | object[] | Rolling window (last 10) of `{ message, level, code, mode, connected, metrics, at }` entries. |

Automation clients can use `PPP.live.onFrame()` or adapter `onStatus` callbacks to receive the same `metrics` object in real time. Each metrics payload matches `snapshotAdapterMetrics()` in [`scripts/LiveQuaternionAdapters.js`](../scripts/LiveQuaternionAdapters.js) and includes counters (`drops`, `parseErrors`, `checksum`, `latency`, `interFrameGap`, `channelSaturation`) for fine-grained monitoring.

## Frame Envelope

Every telemetry frame exported by PPP bundles the following common envelope:

| Field | Type | Description |
| --- | --- | --- |
| `timestamp` | number | Engine-relative milliseconds when the frame was processed. |
| `outputMode` | string | Either `analysis` (silent) or `hybrid` (audio + data). |
| `voiceCount` | number | Harmonic voices active in the current lattice. |
| `transport` | object | Playback/live transport metadata (`playing`, `progress`, `mode`, `frameIndex`, `frameCount`, `loop`). |
| `timelineProgress` | number | Normalized position in the current recording or calibration sequence. |
| `summary` | string | Human-readable harmonic synopsis for logs/diagnostics. |

The canonical frame example lives in [`samples/telemetry/analysis.json`](../samples/telemetry/analysis.json).

## Quaternion Bridge (`analysis.quaternion`)

Derived from the active rotation uniforms. Mirrors the output of `computeQuaternionBridge`.

| Field | Type | Notes |
| --- | --- | --- |
| `left` / `right` | number[4] | Double-quaternion decomposition. |
| `bridgeVector` / `normalizedBridge` | number[4] | Spinor bridge vector and normalized copy. |
| `hopfFiber` | number[4] | Hopf fiber aligned with the visual rotation core. |
| `leftAngle` / `rightAngle` | number | Radians. |
| `dot` | number | Alignment between left/right quaternions. |
| `bridgeMagnitude` | number | Norm of the bridge vector. |

## Spinor Harmonic Coupler (`analysis.spinor`)

See [`samples/telemetry/analysis.json`](../samples/telemetry/analysis.json) → `spinor`.

| Field | Type | Notes |
| --- | --- | --- |
| `ratios` | number[] | Frequency ratios per voice. |
| `panOrbit` / `phaseOrbit` | number[] | Hopf-aligned pan/phase sequences. |
| `axis.left` / `axis.right` | number[3] | Double-quaternion axes. |
| `coherence` / `braidDensity` | number | Spinor health metrics. |
| `pitchLattice` | object[] | Index, ratio, cents, pan. |

## Resonance Atlas (`analysis.resonance`)

Serialized in [`samples/telemetry/resonance.json`](../samples/telemetry/resonance.json).

| Field | Type | Notes |
| --- | --- | --- |
| `timeline` | number | Normalized transport progress. |
| `matrix` | number[4][4] | Rotation matrix driving the atlas (mirrors quaternion bridge). |
| `axes` | number[4][4] | Normalized resonance axes. |
| `bridge` | object | Bridge vector, normalized copy, magnitude, coherence, braidDensity. |
| `hopf` | number[4] | Hopf fiber coordinates. |
| `voices` | object[] | Per-voice resonance vectors, carrier centroids, gate metrics. |
| `aggregate` | object | Centroid, magnitudes, projections, gate/phase statistics. |

## Signal Fabric (`analysis.signal`)

Full signal payload: [`samples/telemetry/signal.json`](../samples/telemetry/signal.json).

| Field | Type | Notes |
| --- | --- | --- |
| `progress` | number | Normalized timeline position. |
| `transport` | object | Minimal transport state for robotics receivers. |
| `quantum` | object | Hopf fiber snapshot. |
| `spinor` | object \| null | Spinor lattice metrics (omitted in sanitized exports). |
| `resonance` | object | Resonance atlas aggregate mirrored from `analysis.resonance`. |
| `voices` | object[] | Per-voice carrier amplitudes, gate duty, spinor ratios. |
| `carrierMatrix` | object[] | Flattened carrier grid with frequency, energy, duty cycle. |
| `bitstream` | object | Hex sequence + binary segments for synchronized demodulators. |
| `envelope` | object | Spectral centroid/spread, resonance magnitude, continuum progress. |

The sanitized telemetry samples in `samples/telemetry/*.json` omit deep voice arrays for lightweight sharing; the full payload is
available through the PPP runtime and recorder exports.

## Transduction Grid (`analysis.transduction`)

[`samples/telemetry/transduction.json`](../samples/telemetry/transduction.json)

| Field | Type | Notes |
| --- | --- | --- |
| `matrix` | number[4][4] | Quaternion-aligned rotation matrix. |
| `invariants` | object | Determinant, trace, Frobenius, bridge magnitude, hopf alignment. |
| `topology` | object | Bridge + hopf vectors with braid/coherence references. |
| `voices` | object[] | Per-voice projections, gate stats, carrier energy, hopf samples. |
| `grid` | object[] | Matrix-flattened carrier entries for deterministic remapping. |

## Metric Manifold (`analysis.manifold`)

[`samples/telemetry/manifold.json`](../samples/telemetry/manifold.json)

| Field | Type | Notes |
| --- | --- | --- |
| `quaternion` | object | Bridge magnitude, hopf alignment, trace, determinant, Frobenius energy. |
| `spinor` | object | Coherence, braid density, ratio entropy, pan/phase variance. |
| `resonance` | object | Aggregate centroid and carrier statistics. |
| `signal` | object | Bitstream density, spectral centroid/spread, resonance envelope. |
| `transduction` | object | Grid energy, gate coherence, projection variance, invariants. |
| `alignment` | object | Correlations between bridge, resonance centroid, signal/manifold energy. |
| `voices` | object[] | Per-voice summary linking gate, spinor, resonance, signal, transduction data. |

## Topology Weave (`analysis.topology`)

[`samples/telemetry/topology.json`](../samples/telemetry/topology.json)

| Field | Type | Notes |
| --- | --- | --- |
| `matrix` | number[4][4] | Resonance atlas basis for axis analytics. |
| `axes` | object[] | Bridge/hopf coupling, magnitude, gate/ratio/carrier/bit flux per axis. |
| `spectrum` | object | Spinor ratio stats, gate mean/variance, carrier magnitude snapshot. |
| `braiding` | object | Correlations linking bridge, hopf, gate, grid, bit flux. |
| `bridge` | object | Normalized bridge + hopf fiber with live magnitude. |

## Flux Continuum (`analysis.continuum`)

[`samples/telemetry/continuum.json`](../samples/telemetry/continuum.json)

| Field | Type | Notes |
| --- | --- | --- |
| `flux` | object | Density, variance, bridge/hopf alignment scores, aggregated gate/coherence/bit metrics. |
| `continuum` | object | Quaternion-locked orientation vectors, magnitudes, sequencing strings. |
| `axes` | object[] | Topology axis intensities, correlations, couplings. |
| `voices` | object[] | Manifold voice metrics aligned with continuum orientation + energy. |
| `braiding` | object | Flux relationships between bridge, hopf, gate, grid, bit channels. |

## Continuum Lattice (`analysis.lattice`)

[`samples/telemetry/lattice.json`](../samples/telemetry/lattice.json)

| Field | Type | Notes |
| --- | --- | --- |
| `orientation` | object | Continuum, voice, bridge, and hopf vectors plus residuals. |
| `synergy` | object | Coherence-, braid-, and carrier-weighted means, gate averages, ratio variance, sequence density. |
| `axes` | object[] | Per-axis continuum/bridge/hopf projections, flux totals. |
| `voices` | object[] | Voice-level continuum alignment, bit entropy, carrier energy, dominant frequency. |
| `carriers` | object[] | Carrier matrix with continuum weights and gate duty. |
| `spectral` | object | Carrier span, energy, bit density. |
| `timeline` | object | Per-voice weight traces. |

## Continuum Constellation (`analysis.constellation`)

[`samples/telemetry/constellation.json`](../samples/telemetry/constellation.json)

| Field | Type | Notes |
| --- | --- | --- |
| `nodes` | object[] | Centroid-focused constellation nodes with dispersion, gating, sequence metrics. |
| `alignment` | object | Bridge/Hopf projections, continuum alignment, carrier summaries. |
| `energy` | object | Aggregated carrier energy and spinor variance. |
| `timeline` | object | Progression of constellation dispersion + gate ratios. |

## Calibration Dataset Manifest

`CalibrationDatasetBuilder` exports a parity manifest after running the canonical calibration plan. The manifest is JSON serializable and safe to publish alongside recorder exports.

| Field | Type | Notes |
| --- | --- | --- |
| `generatedAt` | string | ISO timestamp when the manifest was built. |
| `metadata` | object | Arbitrary metadata (builder version, author, dataset tags). |
| `totals.sequenceCount` | number | Number of calibration sequences executed. |
| `totals.sampleCount` | number | Frames captured across all sequences. |
| `score` | number \| null | Aggregate 0–1 fidelity score (higher is better). |
| `sampleScoreAverage` | number \| null | Mean of per-frame parity scores. |
| `parity` | object | Aggregated metrics keyed by parity channel (see below). |
| `sequences` | object[] | `id`, `label`, `frames`, `status`, `sampleRate`, `durationSeconds`, `completed`. |
| `samples` | object[] | Optional array of captured frames with telemetry clones and parity metrics. |

`parity` aggregates numeric metrics derived from each sample:

- `visualMean`, `visualStd` – Channel averages/std deviation after mapping.
- `continuumGridMean`, `continuumCarrierMean`, `visualContinuumDelta` – Visual vs. continuum energy comparison.
- `carrierEnergyMean`, `carrierGateRatio`, `bitDensity`, `envelopeResonance` – Sonic carrier statistics and gate density.
- `continuumAlignmentMean`, `bitEntropyMean`, `uniformVectorMagnitude`, `resonanceAggregateEnergy`, `spinorCoherence` – Spinor alignment, entropy, and resonance metrics.

Every entry in `samples` includes the original calibration payload (`values`, `uniforms`, `transport`, `analysis` clones) plus:

- `parity` – Per-frame metrics mirroring the aggregated keys above.
- `score` – 0–1 fidelity score computed from delta, gate ratio, resonance, and spinor coherence.
- `screenshot` – Optional base64 renderer capture when enabled.

Use `PPP.calibration.dataset.runPlan()` (browser) or instantiate `CalibrationDatasetBuilder` directly (headless) to generate manifests for downstream robotics and multimodal research pipelines.

### Reference Release: `ppp-calibration-reference-1`

- Generated via `node scripts/buildCalibrationDataset.js` (headless Sonic Geometry engine + toolkit).
- Captures three default sequences (`hopf-orbit`, `flux-ramp`, `spinor-coherence`) for a total of 584 frames.
- Aggregated fidelity metrics: dataset score 0.6109, mean per-frame score 0.6413, visual↔continuum delta average 0.4485, spinor coherence average 0.5752, carrier gate ratio locked at 1.0.
- Summary payload: [`samples/calibration/ppp-calibration-dataset-summary.json`](../samples/calibration/ppp-calibration-dataset-summary.json) (no samples) lists the manifest artifact hash/size. Generate the full manifest locally with `node scripts/buildCalibrationDataset.js --out dist/calibration/ppp-calibration-dataset.json` (paths resolve from the repo root unless absolute, append `.json.gz` for compressed output as needed).
- Insight narrative + outlier trace: [`samples/calibration/ppp-calibration-insights.json`](../samples/calibration/ppp-calibration-insights.json) for QA dashboards, including a reference to the manifest digest.
- Regenerate on demand with `node scripts/buildCalibrationDataset.js --out dist/calibration/ppp-calibration-dataset.json` (use `--no-samples` for parity-only summaries and append `.json.gz` to produce gzipped artifacts when desired).

## Sanitized Robotics Payloads

The `samples/telemetry/*.json` files drop deep per-voice arrays to keep distribution payloads light. For full fidelity (matching
 test fixtures), consume recorder exports, the headless runner output, or live PPP callbacks (`PPP.sonicGeometry.on*`).

## Validation

Automated regression coverage lives in [`tests/sonicGeometryEngine.test.js`](../tests/sonicGeometryEngine.test.js). The test execu
tes the Sonic Geometry engine against the canonical fixture and asserts bridge → constellation parity with tight numerical toler
ance, guaranteeing schema stability across releases.

