# Metamorphic Topology Telemetry Schema

**Document ID:** MCE-TELEMETRY-2026-01
**Purpose:** Extend PPP telemetry and audit payloads with topology stage metadata.

## 1. Topology Stage Fields

Every frame or batch summary SHOULD include:

```json
{
  "topologyStage": "HYPERCUBE",
  "tensionScore": 0.62
}
```

These fields provide explainable markers for when cognition inflates or deflates the active manifold, aligning with the existing telemetry schema guarantees.【F:docs/telemetry-schemas.md†L1-L49】

## 2. Transition Event Payload

Topology transitions are emitted as explicit events:

```json
{
  "eventType": "TOPOLOGY_TRANSITION",
  "timestamp": 1735832664000,
  "previousHash": "2f2c...",
  "stage": "CELL24",
  "transition": {
    "from": "HYPERCUBE",
    "to": "CELL24",
    "reason": "tension>0.8",
    "tensionScore": 0.84
  },
  "eventHash": "6fe1..."
}
```

This mirrors the existing evidence event schema while adding stage metadata for audit traceability.【F:docs/geometric-audit-schema.md†L1-L79】

## 3. Batch Metric Summary

Batch summaries SHOULD include:

```json
{
  "batchMetrics": {
    "coherenceMean": 0.71,
    "coherenceVariance": 0.08,
    "boundaryRiskP95": 0.63
  }
}
```

These metrics are derived from Epistaorthognition’s coherence and boundary proximity definitions, aggregated across batch trajectories for HPC execution.【F:DEV_TRACK.md†L192-L213】

---

**Status:** Schema ready for streaming + audit integration.
