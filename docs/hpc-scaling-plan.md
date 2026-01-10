# HPC Scaling Plan

**Document ID:** MCE-HPC-2026-01
**Purpose:** Define GPU sizing, batch geometry, and audit throughput targets.

## 1. Compute Model

- **State Tensor:** `S ∈ ℝ^(B × T × 4)`
- **Precision:** Float64 for geometric algebra fidelity.
- **Topology:** Metamorphic selection (Simplex5 → Hypercube8 → 24-Cell).

This plan preserves the CPE invariant that reasoning is rotation in 4D while scaling to HPC batch execution.【F:docs/CPE-STATUS-REPORT-2026-01-09.md†L43-L88】

## 2. Recommended Profiles

| Profile | Batch (B) | History (T) | Use Case |
| --- | --- | --- | --- |
| Edge Test | 128 | 60 | Local simulation + dev |
| Standard | 1024 | 120 | Production streaming |
| Deep Audit | 2048 | 240 | Forensic replay |

## 3. Bandwidth Considerations

The Chronomorphic expansion highlights memory bandwidth as the limiting factor for large polytopal transforms. Prioritize GPU interconnects (NVLink/InfiniBand) over raw FLOPS to keep state tensors resident and avoid CPU round-trips.【F:Chronomorphic Polytopal Engine Expansion.md†L107-L116】

## 4. Audit Throughput

Each batch step emits a root hash and appends to a Merkle Mountain Range (MMR) to avoid linear chain bottlenecks. This is compatible with the existing geometric audit schema while scaling to GPU throughput demands.【F:docs/geometric-audit-schema.md†L40-L79】

---

**Status:** Ready for cluster sizing and deployment planning.
