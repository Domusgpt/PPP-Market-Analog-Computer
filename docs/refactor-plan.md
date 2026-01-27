# PPP Core Integration & Refactor Plan

## Purpose
Unify recent Phase-Lock work with external dashboards (e.g., HEMOC live SPX) into a single, durable foundation while keeping the PPP core stable, testable, and extensible. This plan organizes the refactor into phases that preserve current functionality while enabling modular integration of new visualizations and data sources.

## Goals
- Establish **PPP Core** as the canonical synchronization + telemetry foundation.
- Decouple ingestion from rendering so multiple dashboards can attach cleanly.
- Standardize telemetry contracts to eliminate re-implementation and data drift.
- Provide a UI integration layer that can host multiple dashboards without core changes.
- Enable reliable testing and quality gates for every phase.

## Non-Goals (Initial Phase)
- Rewriting all existing UI layouts.
- Replacing Phase-Lock math or 4D geometry logic without evidence of defects.
- Building bespoke integrations for every dashboard without a generic adapter pattern.

---

## Phase 0 — Baseline Inventory & Alignment (1–2 days)
**Outcome:** Confirm what exists, create a shared vocabulary, and prevent duplicate effort.

1) **Catalog current modules & flows**
   - Phase-Locked Stereoscopy pipeline (TimeBinder → GeometricLerp → StereoscopicFeed). This is the canonical time-sync layer.  
   - Telemetry schemas (analysis, signal, transduction, manifold, topology, continuum, lattice). This is the canonical API contract for downstream systems.
   - Existing Phase-Lock Live UI for layout references and design heuristics.

2) **Identify external dashboards/components to merge**
   - For HEMOC (and other dashboards), capture the data sources, UI widgets, and data flow expectations.
   - Identify which portions are visual layer vs. data layer.

3) **Define the integration contract**
   - All dashboard data should flow through TimeBinder/StereoscopicFeed.
   - All downstream consumers should use Telemetry schema objects.

---

## Phase 1 — PPP Core Boundary (2–4 days)
**Outcome:** Establish a stable, versioned core boundary.

### 1.1 Core Modules
- **Temporal sync**: TimeBinder (buffered phase-lock), GeometricLerp (SLERP smoothing), StereoscopicFeed (left/right eye bifurcation).  
- **Telemetry**: Sonic Geometry analysis payloads (quaternion/spinor/resonance/signal/transduction/manifold/topology/continuum/lattice).

### 1.2 Interfaces
- **Input adapter contract** (RawApiTick → MarketTick) with normalized timestamps and channel mappings.
- **Output contract**: telemetry snapshots as authoritative system state.

### 1.3 Stability commitments
- Public API signature is versioned and tested (semver-style if needed).
- Add regression tests for TimeBinder + StereoscopicFeed + Telemetry contract.

---

## Phase 2 — Ingress Adapter Layer (2–5 days)
**Outcome:** Multiple data sources plug into a uniform format.

### 2.1 Adapter Design
- **Adapter interface**: `connect()`, `disconnect()`, `onTick(rawTick)`, `metrics()`.
- **Mapping policy**: price/volume/bid/ask + multi-channel mapping into PPP `MarketTick`.

### 2.2 Target adapters (initial)
- Market APIs (REST/WebSocket).
- File-based playback or recorder exports.
- External dashboards (HEMOC) that emit or expect custom signals.

### 2.3 Validation
- Each adapter emits the same telemetry summary for identical input sequences.

---

## Phase 3 — UI Integration Layer (2–6 days)
**Outcome:** Dashboards consume PPP Core without rewriting core logic.

### 3.1 Unified UI Contract
- **Shared event streams**: `frame`, `chart`, `cpe`, `seek`, `sync-error`.
- **Unified telemetry UI**: Live status, latency, channel saturation, etc.

### 3.2 Integration patterns
- **Host shell**: A top-level UI container that can swap dashboards as routes or tabs.
- **Widget bundles**: Each dashboard UI becomes a bundle with a single `connect(feed)` entry point.

### 3.3 Visual cohesion
- Adopt a shared design system token set (colors, spacing, typography) so dashboards feel coherent while preserving unique visuals.

---

## Phase 4 — Cross-Dashboard Refactor (2–4 weeks, staged)
**Outcome:** Migrate external dashboards without breaking functionality.

### 4.1 Migration steps per dashboard
1. Extract data layer → Adapter.
2. Wire to PPP Core (TimeBinder + StereoscopicFeed).
3. Replace internal state with telemetry-based rendering.
4. Validate against canonical tests & telemetry snapshots.

### 4.2 Stability gates
- Phase-lock alignment tests (e.g., crosshair seek yields same concept step).
- Snapshot testing of telemetry outputs.
- Visual regression checks if applicable.

---

## Phase 5 — QA & Release Hardening (1–2 weeks)
**Outcome:** Reliable deployments with clear failure modes and monitoring.

### 5.1 Test Matrix
- **Core math & time sync**: unit tests + regression fixtures.
- **Telemetry schema**: strict JSON schema validation (if added).
- **UI regression**: screenshot baselines for important dashboards.

### 5.2 Monitoring
- Telemetry health metrics surfaced in UI (frames, latency, drops, checksum).
- Adapter reconnect / error state tracking.

---

## Dependencies & Tooling
- **TypeScript toolchain** for tests + linting.
- **Test harness** for phase-lock and telemetry schema validation.
- **Optional**: Snapshot-based golden files for telemetry, plus UI screenshot testing.

---

## Risks & Mitigations
- **Risk:** Dashboards depend on hidden coupling to old data flows.  
  **Mitigation:** Provide adapter shims and compatibility layers.

- **Risk:** Telemetry schema changes break downstream consumers.  
  **Mitigation:** Versioned schema + compatibility adapters.

- **Risk:** Performance regressions under heavy input.  
  **Mitigation:** Add perf baselines for ingestion + frame sync.

---

## Next Steps (Immediate)
1. Confirm which external dashboards we’re merging (HEMOC + others).
2. Build adapter prototypes for each dashboard’s data feed.
3. Freeze PPP Core API boundaries and add tests.
4. Create UI integration shell for multi-dashboard hosting.

---

## Open Questions (for you to decide)
- Should we version the PPP Core as a separate package (npm/workspace) or keep it internal?
- What’s the desired primary UI shell (single dashboard w/ plugins vs. multi-route host)?
- Which telemetry streams are mandatory for first-wave integrations?
- Should we prioritize performance telemetry dashboards early or after integration?
