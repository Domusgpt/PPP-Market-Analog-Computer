# Phase 2 – Ingress Adapter Layer (In Progress)

## Scope
Introduce initial adapter implementations that normalize external sources into `RawApiTick` for PPP Core.

---

## Deliverables (Completed)

### MarketQuoteAdapter (Primary)
- Adapter that ingests market quotes (bid/ask/last/volume) and emits a normalized `RawApiTick`.
- Exposes channels for mid, spread, imbalance, sizes, last, and volume to keep focus on asset pricing.

### HemocOddsAdapter (Secondary/Legacy)
- Adapter that ingests sportsbook odds and emits a normalized `RawApiTick` with consensus/vig metrics.
- Retained for backward compatibility with prior HEMOC demos.

---

## Next Steps
- Add additional adapters (REST/WebSocket, recorder playback).
- Define validation fixtures that compare telemetry summaries across adapters for identical inputs.
