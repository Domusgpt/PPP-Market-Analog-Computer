# Ambiguity/Tension Metrics Concept Note

## Purpose
This note defines ambiguity/tension metrics that describe how concept representations evolve across the **Simplex → Hypercube → 24-Cell** progression. The metrics quantify when a conceptual space needs to expand its dimensionality or symmetry so it can preserve **convex, coherent regions** while accommodating more complex distinctions. The framing builds on existing convexity and conceptual-space grounding (e.g., Gärdenfors-style constraints and convex-hull validity checks) already used in the system.

## Metric Definitions

### 1) Ambiguity (Variance) Score
**Goal:** capture how dispersed or uncertain a concept is within its current representational polytope.

**Definition (concept-level):**
- Let \(x_i\) be samples or states representing a concept cluster in the current space.
- Let \(\mu\) be the centroid of the cluster.
- Define **ambiguity variance** as:

\[
\text{Ambiguity}(C) = \frac{1}{n}\sum_{i=1}^{n} \|x_i - \mu\|^2
\]

**Interpretation:** low variance indicates a crisp, stable concept; high variance indicates the representation is “blurry,” suggesting insufficient structure in the current polytope.

### 2) Tension (Conflict) Score
**Goal:** capture contradictory pulls between competing concept prototypes within the same region.

**Definition (concept-set level):**
- Let \(p_k\) be prototype anchors for competing concepts (or antipodal archetypes).
- Define **tension** as the average pairwise conflict energy:

\[
\text{Tension}(P) = \frac{2}{m(m-1)}\sum_{i<j} \big(1 - \cos(\theta_{ij})\big)
\]

Where \(\theta_{ij}\) is the angle between prototypes \(p_i\) and \(p_j\).

**Interpretation:** higher tension indicates conflicting concept pulls that cannot be resolved without adding structure (extra axes or symmetry) to keep regions convex and separable.

### 3) Convexity Stress (Optional Composite)
**Goal:** determine whether the current polytope can maintain convex concept regions while ambiguity/tension increases.

\[
\text{ConvexityStress} = \alpha\,\text{Ambiguity} + \beta\,\text{Tension}
\]

This composite increases when ambiguous clusters and conflicting prototypes push the representation toward the boundaries of the convex hull.

## Transition Mapping: Simplex → Hypercube → 24-Cell

### Simplex Stage (Association)
- **Metrics:** low ambiguity, low tension.
- **Geometry:** minimal vertices and simplex-like structure capture “association-first” organization.
- **Transition trigger:** rising ambiguity (variance) indicates the need for **orthogonal axes** to separate emerging distinctions.

### Hypercube Stage (Discrimination)
- **Metrics:** moderate ambiguity, rising tension as binary contrasts appear.
- **Geometry:** orthogonal axes support **discrete, axis-aligned discriminations** (e.g., true/false, on/off, present/absent).
- **Transition trigger:** tension across multiple axes indicates conflicts that are not well represented by axis-aligned corners alone.

### 24-Cell Stage (Synthesis)
- **Metrics:** high ambiguity and high multi-axis tension.
- **Geometry:** 24-Cell symmetry provides **balanced, self-dual structure** to integrate many conflicting pulls while keeping the space convex.
- **Result:** synthesis of concepts becomes possible without collapsing convexity, because the lattice offers more symmetrical “resolution directions.”

## Justification via Convexity & Conceptual-Space Foundations
Conceptual spaces depend on **convex regions** to keep category membership coherent; when ambiguity and tension grow, a higher-symmetry polytope is required to preserve convexity. The stepwise expansion from simplex to hypercube to 24-Cell mirrors how convex-hull validity checks and Gärdenfors-style constraints already guide the system: as convexity stress increases, the geometry must expand to keep concept regions convex and separable.

## Cognitive Development Parallel
- **Association (Simplex):** early cognition groups stimuli via proximity and co-occurrence.
- **Discrimination (Hypercube):** growing cognitive capacity introduces axes of contrast, enabling finer categorical distinctions.
- **Synthesis (24-Cell):** mature cognition integrates multiple conflicting criteria into a stable, symmetric concept lattice that supports nuanced synthesis rather than simple binary splits.

## Summary
The ambiguity/tension metrics provide **quantitative signals** for when the system should move from **Simplex → Hypercube → 24-Cell**, aligning geometric growth with conceptual-space convexity requirements and the progression of cognitive development.
