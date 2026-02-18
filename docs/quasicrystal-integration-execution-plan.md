# Quasicrystal Integration Execution Plan

**Status:** Active  
**Execution model:** Sequential-first (single integration spine), with controlled parallel test hardening later  
**Started (UTC):** 2026-02-18T02:03:39Z

## Why Sequential First

Given the current codebase shape, these workstreams are not independent enough for safe immediate parallelization:

1. **Architecture mode integration** (legacy vs quasicrystal) is a foundation change that impacts encoder and streaming pipelines.
2. **Golden MRA feature wiring** depends on stable architecture and feature API contracts.
3. **Galois dual-invariant verification** changes data contracts and telemetry semantics.
4. **Public API exports + docs/tests** should follow finalized interfaces to avoid churn.

Parallelizing all tracks immediately would likely create repeated conflict cycles in:
- `backend/engine/main.py`
- `backend/engine/streaming/stream_encoder.py`
- `backend/engine/features/extractor.py`
- `backend/engine/geometry/__init__.py`

## Execution Strategy

### Phase A — Sequential Integration Spine (start now)

1. Introduce `architecture_mode` and wire quasicrystal path in offline encoder.
2. Mirror architecture mode into streaming encoder.
3. Add runtime compatibility checks and baseline tests for both modes.

### Phase B — Feature & Verification Formalization

4. Add GoldenMRA adapter into unified feature extractor.
5. Expand GaloisVerifier to ratio + product invariant outputs.
6. Update tests and telemetry payload assertions.

### Phase C — API Surface & Documentation Stabilization

7. Export quasicrystal classes via geometry package API.
8. Synchronize docs/spec constants with implemented defaults.
9. Record final dev-track summary + integration notes.

## Controlled Parallelism (later)

After Phase A lands, we can safely parallelize:
- **Track P1:** GoldenMRA feature adapter + extractor tests.
- **Track P2:** GaloisVerifier invariant extension + batch statistics.
- **Track P3:** Documentation/ledger updates + import/export smoke tests.

These can run in parallel because Phase A will have stabilized interfaces and reduced merge hotspots.

## Merge & Quality Gates

Every sub-phase must pass:
- targeted unit tests for changed modules,
- import smoke tests for API surfaces,
- no regression in existing architecture tests,
- documented ledger entry in `DEV_TRACK.md`.

## Progress Ledger

- [x] 2026-02-18T02:03:39Z — Execution model chosen: sequential-first.
- [x] 2026-02-18T02:03:39Z — Plan document created.
- [x] Phase A complete (2026-02-18T02:11:20Z).
- [ ] Phase B complete.
- [ ] Phase C complete.

