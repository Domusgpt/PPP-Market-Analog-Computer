# Phase 0 Inventory & Alignment (Completed)

## Scope
This document finalizes Phase 0 by cataloging the current PPP Core modules, defining integration contracts, and listing the canonical documentation that governs telemetry and synchronization behavior.

---

## 1) Module Map (Current State)

### Temporal Synchronization (PPP Core)
- **TimeBinder**: Phase-locked buffer + interpolation guardrails (TimeBinder.ts).
- **GeometricLerp**: SLERP interpolation for 4D rotations (GeometricLerp.ts).

### Stereoscopic Fusion (PPP Core)
- **StereoscopicFeed**: Primary façade for ingest, phase-locked frames, and crosshair synchronization (StereoscopicFeed.ts).

---

## 2) Canonical Contracts

### Input Contract (RawApiTick → MarketTick)
**RawApiTick** (ingress shape):
- `price`, `volume`, `bid`, `ask`, `timestamp`, `sequence`, `channels[]`

**IndexedTick** (local stamp):
- `localTimestamp`, `apiLatency`, `sequenceId`

**MarketTick** (PPP Core tick):
- `timestamp`, `sequence`, `priceVector`, `rotation`, `latency`

### Output Contract (Synchronized Frames)
- **SyncedFrame**: unified render state (`priceVector`, `rotation`, `interpolationFactor`, `isExact`).
- **StereoscopicFrame**: `leftEye` (chart) + `rightEye` (CPE render data) + `phaseOffset`.

### Telemetry Contract (Downstream Consumers)
- Canonical telemetry schema is documented in `docs/telemetry-schemas.md`.
- Core channels: `analysis`, `signal`, `transduction`, `manifold`, `topology`, `continuum`, `lattice`.

---

## 3) File Inventory (Authoritative Sources)

| Area | File | Purpose |
| --- | --- | --- |
| Temporal sync | `src/lib/temporal/TimeBinder.ts` | Phase-locked buffering, interpolation constraints, metrics |
| SLERP | `src/lib/temporal/GeometricLerp.ts` | Quaternion + rotor interpolation for 4D rotations |
| Stereoscopic feed | `src/lib/fusion/StereoscopicFeed.ts` | Ingest, chart + CPE frames, crosshair seek |
| Telemetry schema | `docs/telemetry-schemas.md` | Canonical downstream payload contract |
| Phase-lock overview | `docs/PHASE-LOCKED-STEREOSCOPY.md` | Architectural description & tests |

---

## 4) Integration Contract (Phase 0 Outcome)

### Required behavior
- **All ingestion flows** normalize to `RawApiTick` and then to `MarketTick` through TimeBinder.
- **All dashboards** consume `StereoscopicFeed.frame()` or `StereoscopicFeed.seek()` to ensure phase-lock alignment.
- **All downstream systems** adhere to telemetry schemas for stable data interchange.

### Minimum telemetry streams for Phase 1
- `analysis` (always-on)
- `signal` (deterministic transport payload)

---

## 5) Phase 0 Acceptance Criteria (Now Met)
- [x] PPP Core module map documented.
- [x] Input/Output contracts enumerated.
- [x] Telemetry contract references identified.
- [x] Authoritative files enumerated for Phase 1 work.

---

## 6) Next Hand-off (Phase 1 Ready)
- Confirm external dashboards (HEMOC + others).
- Create adapter skeletons based on `RawApiTick` contract.
- Lock PPP Core API boundary and tests before adding integrations.
