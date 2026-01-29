# Phase 2 – Ingress Adapter Layer (In Progress)

## Scope
Introduce initial adapter implementations that normalize external sources into `RawApiTick` for PPP Core.

---

## Deliverables (Completed)

### HemocOddsAdapter
- Adapter that ingests sportsbook odds and emits a normalized `RawApiTick` with consensus and vig metrics.
- Exposes channels for consensus home/away, max deviation, average vig, edge score, and per-book vig values.

---

## Next Steps
- Add additional adapters (REST/WebSocket, recorder playback).
- Define validation fixtures that compare telemetry summaries across adapters for identical inputs.
