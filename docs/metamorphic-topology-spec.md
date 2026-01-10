# Metamorphic Topology Specification

**Document ID:** MCE-TOPOLOGY-2026-01
**Purpose:** Define a developmental topology controller that inflates from Simplex → Hypercube → 24-Cell.

## 1. Overview

The Chronomorphic Polytopal Engine (CPE) models reasoning as 4D rotation constrained by convex topology. This specification introduces a **metamorphic topology controller** that dynamically selects the active polytope (Simplex5 → Hypercube8 → 24-Cell) based on ambiguity/tension metrics while preserving the core “reasoning is rotation” invariant and convexity constraints.【F:docs/CPE-STATUS-REPORT-2026-01-09.md†L14-L88】

## 2. Topology Stages

| Stage | Polytope | Cognitive Role | Operational Role |
| --- | --- | --- | --- |
| SIMPLEX | 5-Cell | Association / axioms | Maximal connectivity, fastest validity checks |
| HYPERCUBE | 8-Cell | Discrimination / opposites | Orthogonal separation, binary logic |
| CELL24 | 24-Cell | Synthesis / contextual truth | Full nuance, self-dual constraints |

The Chronomorphic expansion already frames polytopes as a vocabulary of computational substrates and an optimization strategy; metamorphosis makes that choice dynamic rather than static.【F:Chronomorphic Polytopal Engine Expansion.md†L12-L52】

## 3. Topology Controller API

```ts
interface TopologyController {
  stage: 'SIMPLEX' | 'HYPERCUBE' | 'CELL24';
  activeLattice: TopologyProvider;
  updateTension(tensionScore: number): TransitionEvent | null;
  evaluate(position: Vector4D): ConvexityResult;
}
```

## 4. Ambiguity/Tension Metrics

The controller is driven by batch-level Epistaorthognition metrics (coherence variance, boundary risk percentiles). These metrics are derived from the same convexity logic already used for validation, now aggregated over a batch of hypotheses for HPC execution.【F:DEV_TRACK.md†L192-L213】

### Recommended thresholds (default)

| Threshold | Meaning | Value |
| --- | --- | --- |
| `deflateThreshold` | return to SIMPLEX | 0.20 |
| `hypercubeThreshold` | promote to HYPERCUBE | 0.50 |
| `cell24Threshold` | promote to CELL24 | 0.80 |

## 5. Transition Rules

1. SIMPLEX → HYPERCUBE when `tensionScore > hypercubeThreshold`.
2. HYPERCUBE → CELL24 when `tensionScore > cell24Threshold`.
3. Any stage → SIMPLEX when `tensionScore < deflateThreshold`.
4. Enforce `minStageDuration` to avoid oscillation.

## 6. Validation Guarantees

Each topology provider must expose:
- `checkConvexity(position)` for validity checks.
- `computeCoherence(position)` for alignment metrics.
- `vertices`, `neighbors` for lattice context.

These guarantees mirror the existing Lattice24 interface while making it interchangeable with additional polytopes.

---

**Status:** Implemented in `lib/topology/TopologyController.ts`, `Simplex5.ts`, and `Hypercube8.ts`.
