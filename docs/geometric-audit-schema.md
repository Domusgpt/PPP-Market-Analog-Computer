# Geometric Audit Telemetry Schema

This document captures the canonical payload shapes emitted by the geometric audit pipeline. These shapes are designed to be stable, hash-linked, and safe for governance exports.

## Evidence Event (`CONSTELLATION_SNAPSHOT`, etc.)

```json
{
  "eventType": "CONSTELLATION_SNAPSHOT",
  "timestamp": 1735832664000,
  "previousHash": "2f2c...",
  "polytopalFingerprint": {
    "constellation": "8b2a...",
    "topology": "430f...",
    "quaternion": "92c0..."
  },
  "payload": {
    "vertices": [[0.1, 0.2, 0.3, 0.4]],
    "topologySignatures": [[0.3, 1.2]],
    "quaternion": [0.5, 0.6, 0.7, 0.8],
    "metadata": {
      "transport": { "playing": true, "mode": "live" },
      "progress": 0.42,
      "energy": { "carrier": 0.8, "grid": 0.6 },
      "spinor": { "coherence": 0.9, "braid": 0.4 }
    },
    "mode": "constellation"
  },
  "metadata": {},
  "eventHash": "6fe1..."
}
```

### Key fields
- `previousHash`: hash of the previous evidence event (or chain head hash if history was trimmed).
- `polytopalFingerprint`: fixed-length hashes of the constellation vertices, topology signatures, and quaternion commitment.
- `payload`: canonicalized geometric state suitable for audit and replay.
- `eventHash`: SHA-256 hash of the evidence object (excluding `eventHash` itself).

## Batch

```json
{
  "index": 12,
  "root": "8cd1...",
  "events": [/* evidence items */],
  "proofs": [
    { "hash": "6fe1...", "proof": [{ "position": "left", "hash": "aa3b..." }], "root": "8cd1..." }
  ],
  "summary": {
    "count": 10,
    "eventTypes": { "CONSTELLATION_SNAPSHOT": 10 },
    "window": { "start": 1735832664000, "end": 1735832667000 },
    "missingFingerprints": 0
  },
  "anchored": {
    "hash": "trace-anchor",
    "anchoredAt": 1735832668000
  }
}
```

### Anchoring
- When a batch is sealed, it can be anchored via `anchorBatch`.
- Anchors may be external ledger hashes or TRACE IDs; store them under `anchored.hash`.

## Session State

```json
{
  "chain": [/* newest evidence subset */],
  "pending": [/* pending events */],
  "batches": [/* sealed batches */],
  "batchSize": 10,
  "maxChainLength": 500,
  "maxBatches": 20,
  "chainHeadHash": "2f2c...",
  "batchOffset": 5
}
```

### Notes
- `maxChainLength` / `maxBatches` bound in-memory retention for long-running sessions.
- `chainHeadHash` captures the hash immediately preceding the retained chain segment.
