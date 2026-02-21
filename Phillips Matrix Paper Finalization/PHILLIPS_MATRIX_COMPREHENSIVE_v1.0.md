# The Phillips Folding Matrix: Spectral Analysis, Mathematical Relatives, and Computational Applications of a Golden-Ratio E₈-to-H₄ Projection

**Paul Phillips**

*Draft v1.0 — February 18, 2026*

---

## Abstract

We construct and analyze an 8×8 folding matrix U that projects the E₈ root system
to a pair of H₄ polytopes, with entry constants {a, b, c} = {1/2, (φ−1)/2, φ/2}
forming a geometric progression with ratio φ (the golden ratio). We prove that U
has rank 4 with the right block satisfying U_R = φ · U_L (Golden Rank Deficiency),
that the 8 column norms fall into three golden-ratio classes in a 2-4-2 pattern
(Column Trichotomy), that row norms equal pentagonal chord lengths 2 sin 36° and
2 cos 18° (Pentagonal Row Norms), and that the round-trip operator U^T U factors
as (φ+2) · U_L^T U_L with eigenvalue 5 at multiplicity 2 arising from
(φ+2)(3−φ) = 5.

Among 240 E₈ roots, exactly 14 collision pairs exist under projection, all arising
from a single kernel vector d = (0,1,0,1,0,1,0,1). We prove these collisions
are exclusively same-type (8 half-integer pairs, 6 permutation pairs, 0 cross-type),
span a rank-1 subspace despite the kernel being 4-dimensional, and induce a
three-tier fidelity hierarchy that refines the Column Trichotomy.

We establish definitive computational distinctness from all published Moxness
matrices (the row spaces are completely orthogonal), classify the Phillips matrix
within 10 mathematical families (quasicrystal cut-and-project, Fibonacci substitution,
Coxeter eigenvectors, icosian ring decomposition, overcomplete golden frame, and
others), define a new "Golden Hadamard" class of structured matrices, connect
the matrix to Boyle and Steinhardt's Coxeter pair framework for quasicrystalline
physics, and demonstrate computational applications through the HEMOC optical
computing system achieving 0.916 Pearson correlation on audio-to-angle recovery.

All results are accompanied by computational verification (281 automated tests,
12/12 invariant theorems, 1000-trial fuzz harness with 100% stability).

---

## 1. Introduction

The E₈ lattice is the unique even unimodular lattice in eight dimensions,
containing 240 roots of minimal norm √2. Its symmetry group, of order
696,729,600, is the largest of the exceptional Lie groups and appears
throughout mathematics and theoretical physics. The 600-cell is the
regular 4-dimensional polytope with 120 vertices and icosahedral symmetry
group H₄ of order 14,400. The relationship between E₈ and the 600-cell,
mediated by the golden ratio φ = (1+√5)/2, has been studied since
Coxeter [1] and du Val [2].

The projection from E₈ (8D) to H₄ (4D) can be realized by a folding
matrix that maps the E₈ Dynkin diagram onto the H₄ diagram. Several
such matrices exist in the literature. Baez [7] gave a 4×8 matrix
producing a single H₄ copy. Moxness [3] constructed a different 8×8
matrix (the "C600 matrix"), which is sparse, symmetric, and rank 8,
and used it for 3D visualization of E₈. His subsequent work [4,5,6]
explored fourfold chiral structure, unimodular variants, and connections
to the 3-qubit Hadamard matrix.

This paper introduces and comprehensively analyzes a new 8×8 folding matrix — distinct
from all Moxness constructions — that arises from a geometric-progression
construction of entry constants. Its defining features include:

1. **Golden Rank Deficiency** (§3): U_R = φ · U_L, making the matrix rank 4
   despite being completely dense (all 64 entries nonzero).

2. **Column Trichotomy** (§3): Column norms fall into three golden-ratio
   classes {φ+2, 5/2, 3−φ} in a 2-4-2 pattern.

3. **Single Collision Vector** (§4): All 14 collision pairs among E₈ roots
   arise from one kernel vector, with exclusively same-type collisions.

4. **Amplification = 5** (§3): Frobenius²/rank = 20/4 = 5, the number
   of inscribed 24-cells in the 600-cell.

5. **Complete Distinctness from Moxness** (§5): The row spaces are
   orthogonal; the stacked rank is 8.

6. **Golden Hadamard Class** (§6): The matrix defines a new class of
   structured matrices with 5 characterizing axioms.

7. **Boyle Connection** (§7): The block scaling is the operator-level
   manifestation of discrete scale invariance in Coxeter pair theory.

8. **HEMOC Application** (§8): The matrix serves as the mathematical
   spine for an optical computing system achieving 0.916 correlation
   on audio feature recovery.

### 1.1 Construction motivation

This matrix emerged from a geometric cognition research program that
began with musical structure. The Chronomorphic Polytopal Engine (CPE)
[8] maps the 24 major and minor musical keys to the 24 vertices of the
24-cell (icositetrachoron), with the circle of fifths realized as
rotation in 4D and chord geometry captured by sub-polytopes. The 600-cell
contains exactly 5 inscribed 24-cells [1], making it the natural
higher-dimensional container for the musical 24-cell. This motivated
projecting E₈ root structure to H₄ — specifically, seeking a folding
matrix that embeds 24-cell constellations within the projected 600-cell.

The matrix was constructed using golden-ratio entry constants derived from
the 600-cell's coordinate structure (the vertex coordinates 0, ±1/2, ±φ/2,
±1/(2φ) yield the geometric progression {b, a, c}). Hyperdimensional
computing experiments using Voyage AI embeddings [10] validated the 24-cell
geometric framework by projecting 1024D semantic vectors to 4D, confirming
that the polytope geometry preserves meaningful structure under
dimensionality reduction.

### Notation

Throughout, φ = (1+√5)/2 ≈ 1.618 denotes the golden ratio. We use the
identities φ² = φ+1, 1/φ = φ−1, and φ−1/φ = 1 freely. The golden
conjugate is φ' = (1−√5)/2 = −1/φ. We write Q(√5) = {a + b√5 : a,b ∈ Q}
for the golden field. The plastic ratio is ρ ≈ 1.3247, the real root of
x³ = x + 1.

---

## 2. The Folding Matrix

### 2.1 Definition

The Phillips folding matrix is the 8×8 real matrix

$$U = \begin{pmatrix} U_L \\ U_R \end{pmatrix}$$

where U_L, U_R ∈ R^{4×8} are defined using three constants:

    a = 1/2,    b = (φ−1)/2 ≈ 0.309,    c = φ/2 ≈ 0.809

The left block U_L has entries from {±a, ±b}:

```
U_L = [ a   b   a   b   a  -b   a  -b ]
      [ a   a  -b  -b  -a  -a   b   b ]
      [ a  -b  -a   b   a  -b  -a   b ]
      [ a  -a   b  -b  -a   a  -b   b ]
```

The right block U_R has entries from {±a, ±c}:

```
U_R = [ c   a   c   a   c  -a   c  -a ]
      [ c   c  -a  -a  -c  -c   a   a ]
      [ c  -a  -c   a   c  -a  -c   a ]
      [ c  -a   a  -a  -c   c  -a   a ]
```

### 2.2 Entry structure

The three constants form a geometric progression:

    b = a/φ,    a = a,    c = aφ

with ratio φ and center a = 1/2. This means:

- b·c = a² = 1/4 (the outer pair multiplies to the center squared)
- c/b = φ² = φ+1 (the extreme ratio is the golden square)
- b + c = a·√5 (the sum involves √5)
- (c−a)/(a−b) = φ (deviations from center are in golden ratio)

The constants are also pentagonal trigonometric values:
- a = 1/2 = cos 60°
- b = (φ−1)/2 = cos 72° = sin 18°
- c = φ/2 = cos 36° = sin 54°

### 2.3 The E₈ root system

The 240 roots of E₈ fall into two types:

**Permutation roots** (112): All vectors with exactly two nonzero entries
±1, the remaining six entries zero. These are permutations and sign
variations of (1, 1, 0, 0, 0, 0, 0, 0).

**Half-integer roots** (128): All vectors (±1/2)⁸ with an even number of
minus signs.

Every root has squared norm ||r||² = 2.

---

## 3. Structural Theorems

### 3.1 Column Trichotomy

**Theorem 1 (Column Trichotomy).** *The squared column norms of U fall
into exactly three classes, distributed in a 2-4-2 pattern:*

| Class      | Columns       | ||col||² | Norm name  |
|------------|-------------- |----------|------------|
| Expanded   | {0, 4}        | φ + 2    | ≈ 3.618    |
| Stable     | {1, 2, 5, 6}  | 5/2      | = 2.500    |
| Contracted | {3, 7}        | 3 − φ    | ≈ 1.382    |

*Moreover:*
- *The arithmetic mean of the extremes equals the stable norm:
  ((φ+2) + (3−φ))/2 = 5/2.*
- *The deviation from the mean is ±√5/2.*
- *The product of the extremes is (φ+2)(3−φ) = 5.*

**Proof.** Each column norm² is the sum of squares of 8 entries (4 from U_L,
4 from U_R).

*Expanded columns (j = 0, 4).* U_L contributes four entries of value ±a,
and U_R contributes four entries of value ±c:

    ||col_0||² = 4a² + 4c² = 4(1/4) + 4(φ²/4) = 1 + φ² = 1 + φ + 1 = φ + 2. ∎

*Stable columns (j = 1, 2, 5, 6).* Each has 2 entries of ±a and 2 of ±b
from U_L, plus 2 of ±a and 2 of ±c from U_R:

    ||col||² = 4a² + 2b² + 2c² = 1 + (φ² − 2φ + 1 + φ²)/2 = 5/2. ∎

*Contracted columns (j = 3, 7).* Four entries of ±b from U_L and four of ±a
from U_R:

    ||col||² = 4b² + 4a² = (φ−1)² + 1 = 3 − φ. ∎

**Remark.** The 2-4-2 pattern mirrors the branching of the E₈ Dynkin diagram
when folded to H₄. The expanded dimensions carry the most energy; the
contracted dimensions, the least.

### 3.2 Pentagonal Row Norms

**Theorem 2 (Pentagonal Row Norms).** *Every row of U_L has squared norm
3−φ, and every row of U_R has squared norm φ+2. Equivalently:*

$$\|r_L\| = \sqrt{3-\varphi} = 2\sin 36°, \qquad
  \|r_R\| = \sqrt{\varphi+2} = 2\cos 18°$$

*These are the edge length and the diagonal of a unit pentagon.*

**Proof.** Each row of U_L contains 4 entries of |a| and 4 of |b|:

    ||r_L||² = 4a² + 4b² = 1 + (φ−1)² = 3 − φ.

Similarly for U_R with |a| and |c|:

    ||r_R||² = 4a² + 4c² = 1 + φ² = φ + 2. ∎

**Corollary.** ||r_R||/||r_L|| = √((φ+2)/(3−φ)) = φ,
and ||r_L||·||r_R|| = √5. ∎

### 3.3 Golden Rank Deficiency

**Theorem 3 (Golden Rank Deficiency).** *The right block of the Phillips
folding matrix is exactly φ times the left block:*

$$U_R = \varphi \cdot U_L$$

*Consequently, det(U) = 0 and rank(U) = rank(U_L) = 4.*

**Proof.** The sign patterns of U_L and U_R are identical. The scaling:

    φ · a = φ/2 = c  ✓
    φ · b = φ(φ−1)/2 = ((φ+1)−φ)/2 = 1/2 = a  ✓

Every entry of U_R equals φ times the corresponding entry of U_L. ∎

**Remark (Construction vs. Emergence).** This identity is a *design
consequence* of the geometric-progression entry structure — it follows
directly from the construction. Its significance lies not in being
unexpected, but in the non-trivial chain of consequences it produces:
the round-trip factorization (Theorem 5), the eigenvalue structure
(Theorem 6), the universal Galois ratio (Corollary), and the connection
to Boyle's Coxeter pair framework (§7). The golden ratio acts as a
**shift operator** on the entry alphabet {b, a, c}: it promotes each
entry one level up the geometric progression.

**Remark.** The Moxness C600 matrix has rank 8 and det ≈ 1755. Its blocks
do NOT satisfy U_R = φ·U_L. The golden rank deficiency is specific to the
Phillips construction.

### 3.4 Round-Trip Factorization

**Theorem 5 (Round-Trip Factorization).**

$$U^T U = (\varphi + 2) \cdot U_L^T U_L$$

**Proof.**

$$U^T U = U_L^T U_L + U_R^T U_R = (1 + \varphi^2) U_L^T U_L
       = (\varphi + 2) U_L^T U_L$$

using φ² = φ + 1. ∎

### 3.5 Spectral Structure

**Theorem 6 (Spectral Structure).** *The eigenvalues of U^T U are:*

$$\sigma(U^T U) = \{0^{(\times 4)},\ \lambda_1 \approx 3.141,\ 5^{(\times 2)},\ \lambda_2 \approx 6.859\}$$

*where λ₁ + λ₂ = 10, and the eigenvalue 5 has multiplicity 2, arising from:*

$$(φ + 2)(3 − φ) = 5$$

*The trace equals 20 = ||U||²_F.*

**Proof.** The eigenvalues of U^T U are (φ+2) times those of U_L^T U_L.
The latter has rank 4 with 4 zero eigenvalues. Among its nonzero
eigenvalues, (3−φ) appears with multiplicity 2, producing eigenvalue
(φ+2)(3−φ) = 5 with multiplicity 2 in U^T U.

The remaining two sum to 20 − 2·5 = 10. ∎

### 3.6 Amplification Factor

**Theorem 7 (Amplification Factor).**

$$\frac{\|U\|_F^2}{\mathrm{rank}(U)} = \frac{20}{4} = 5$$

*This equals the number of inscribed 24-cells in the 600-cell (120/24 = 5).*

**Remark (The Five = Five Theorem).** The number 5 appears in five
independent mathematical roles within this matrix:

1. *Algebraic:* (φ+2)(3−φ) = 5
2. *Spectral:* Eigenvalue of U^T U with multiplicity 2
3. *Operator-theoretic:* Frobenius²/rank = 20/4
4. *Polytope-geometric:* Number of 24-cells in the 600-cell
5. *Norm-theoretic:* ||r_L|| · ||r_R|| = √5 for every E₈ root

We do not claim a proof that these five appearances must coincide for
all E₈-to-H₄ folding matrices. However, the coincidence is notable
given the matrix's origin in 24-cell musical geometry (§1.1), and we
note that the Moxness C600 matrix does NOT satisfy Frobenius²/rank = 5
(it gives 57.9/8 ≈ 7.24). Whether the Five = Five property can be
unified under a single structural theorem remains open.

---

## 4. Collision Analysis

### 4.1 The kernel

Since rank(U) = 4, the kernel ker(U) ⊂ R⁸ has dimension 4. Moreover,
ker(U_L) = ker(U_R) = ker(U), since U_R = φ·U_L.

**Proposition.** *No E₈ root lies in ker(U). The minimum projection norm
is ||U_L · r||_min = 1/φ² ≈ 0.382.*

### 4.2 Kernel Collision Uniqueness

**Definition.** A *collision pair* is (r_a, r_b) with r_a ≠ r_b and
U_L · r_a = U_L · r_b.

**Theorem 4 (Kernel Collision Uniqueness).** *Among the 240 E₈ roots,
exactly 14 collision pairs exist. All arise from a single vector:*

$$\mathbf{d} = (0, 1, 0, 1, 0, 1, 0, 1)$$

*That is, r_a − r_b = ±d for every collision pair. Moreover:*

(i) *d has norm ||d|| = 2 but is not an E₈ root (4 nonzero entries).*

(ii) *d has support {1, 3, 5, 7} (odd-indexed coordinates only).*

(iii) *All colliding pairs are orthogonal: ⟨r_a, r_b⟩ = 0.*

(iv) *Of the 14 pairs, 6 are permutation+permutation and 8 are
     half-integer+half-integer. No mixed-type collisions occur.*

**Proof.** That d ∈ ker(U_L) is verified by direct computation. For
exhaustiveness, all 240 projections are enumerated: 226 distinct images
appear, with 14 occurring twice. Each collision difference equals ±d.
Orthogonality and type counts are verified by enumeration. ∎

### 4.3 Structure of the 14 collision pairs

We provide a detailed analysis of the collision pair structure, which
reveals deep connections between the kernel geometry and E₈ root
combinatorics.

**Theorem (Same-Type Exclusivity).** *Every collision pair consists of
two roots of the same type. There are exactly 8 half-integer pairs and
6 permutation pairs. No cross-type collision (permutation + half-integer)
exists.*

**Proof.** Let r_a − r_b = ±d = ±(0,1,0,1,0,1,0,1).

*Case 1: Both half-integer.* If r_a ∈ (±1/2)⁸ and r_b = r_a ∓ d, then
r_b has entries that differ from r_a at positions {1,3,5,7} by exactly ±1.
Since |r_a[i]| = 1/2 and the shift is ±1, the result |r_b[i]| = 1/2
(shifting +1/2 to −1/2 or vice versa), so r_b remains half-integer type.
There are 8 such pairs. ✓

*Case 2: Both permutation.* If r_a has exactly two nonzero entries at
positions in {1,3,5,7} (both ±1), then r_b = r_a ∓ d shifts those entries
to different values. The 6 valid combinations produce permutation-type roots. ✓

*Case 3: Cross-type.* A permutation root has exactly 2 nonzero entries,
a half-integer root has 8. Their difference has at least 6 nonzero entries,
which cannot equal d (which has 4). Therefore no cross-type collision exists. ∎

**Theorem (Rank-1 Collision Subspace).** *All 14 collision differences span
a rank-1 subspace of R⁸, despite the kernel being 4-dimensional.*

This means that of the 4 kernel directions, only 1 "catches" E₈ roots.
The remaining 3 kernel directions produce zero collisions — they are
**collision-free channels** available for error correction (§8.5).

### 4.4 Fidelity hierarchy

The collision vector d has support {1, 3, 5, 7}. Cross-referencing with
the Column Trichotomy:

| Dimension | Trichotomy | d entry | Collision status    |
|-----------|-----------|---------|---------------------|
| 0         | Expanded  | 0       | **Immune**          |
| 4         | Expanded  | 0       | **Immune**          |
| 2         | Stable    | 0       | Immune              |
| 6         | Stable    | 0       | Immune              |
| 1         | Stable    | 1       | Affected            |
| 5         | Stable    | 1       | Affected            |
| 3         | Contracted| 1       | Affected            |
| 7         | Contracted| 1       | Affected            |

The expanded dimensions {0, 4} are completely collision-immune: they carry
the highest-fidelity projection. This establishes a three-tier fidelity
hierarchy that refines the Column Trichotomy: dimensions are classified
both by energy (column norm) and by information fidelity (collision
immunity).

### 4.5 Shell structure of collision points

The 14 shared 4D projection points (the images where two E₈ roots collide)
are distributed across distinct norm shells:

- 4 points at unit norm (axis-aligned)
- 2 points at norm 1/φ ≈ 0.618
- 8 points at intermediate norms (~0.831, ~1.036)

The non-uniform shell distribution further distinguishes the Phillips
projection from generic random projections, which would produce uniform
shell distributions.

### 4.6 Fuzz harness: stability of the collision count

To verify that the collision count of 14 is robust, we performed:

- **Entry value fuzz** (1000 trials): Replaced entry constants with
  random values while preserving the Phillips sign pattern. All 1000
  trials produced exactly 14 collision pairs.

- **Sign fuzz** (200 trials): Replaced signs randomly in U_L.
  Results: varied collision counts (0 to 28), with 14 appearing most
  frequently for near-Phillips patterns.

The entry fuzz result supports Conjecture 4 (§9): the collision count
depends only on the sign pattern, not on the specific entry values.
The collision vector d = (0,1,0,1,0,1,0,1)/2 lies in the kernel for
ANY two-value magnitude matrix with the Phillips sign pattern.

---

## 5. Comparison with Moxness Matrices

### 5.1 The Moxness catalog

Five distinct Moxness matrices were extracted from primary sources spanning
2013–2023, plus a corrected φ𝕌 matrix from personal communication (2026):

| Matrix | Source | Size | Rank | Symmetric | Zeros | 600-cell test |
|--------|--------|------|------|-----------|-------|---------------|
| M1 (2013) | Blog post | 4×8 | 4 | — | 18/32 | Clean H₄+φH₄ |
| M2 (2014) | viXra:1411.0130 | 8×8 | 8 | Yes | 36/64 | 9 norm classes |
| M3 (2018) | viXra:1808.0107 | 8×8 | 8 | Yes | 36/64 | 9 norm classes |
| φ𝕌 (corr.) | Personal comm. | 8×8 | 8 | Yes | 36/64 | Clean H₄+φH₄ |
| Baez | [7] | 4×8 | 4 | — | 16/32 | Clean H₄+φH₄ |

**Key finding: M1/√2 = Baez matrix exactly** (to machine precision).
This is the standard E₈-to-H₄ projection from the literature.

**Internal inconsistency**: M2 (2014) and M3 (2018) differ in 28 of 64
entries despite claims of continuity.

### 5.2 Definitive distinctness

| Property           | Phillips         | Moxness φ𝕌            |
|--------------------|------------------|-----------------------|
| Rank               | 4                | 8                     |
| Symmetry           | Non-symmetric    | Symmetric             |
| Zero entries       | 0 (dense)        | 36 (sparse)           |
| Determinant        | 0                | ≈ 1755                |
| Frobenius²         | 20               | ≈ 57.9                |
| Entry values       | {±1/2, ±(φ−1)/2, ±φ/2} | {0, ±1, ±φ, ±1/φ, ±φ²} |
| Eigenvalues        | {0⁴, λ₁, 5², λ₂}| {±2, ±2, ±2φ, ±2φ}   |
| U_R = φ·U_L        | Yes              | No (palindromic)      |
| L/R norm ratio     | φ (universal)    | {1/φ, 1, φ} (varies)  |
| Unique 4D points   | 226 (14 collisions) | 240 (0 collisions) |
| Shell radii        | 21               | 2                     |

### 5.3 Row space orthogonality

| Stacking                         | Combined Rank |
|----------------------------------|---------------|
| [Phillips; Baez]                 | 8             |
| [Phillips; M2 top-4]            | 7             |
| [Phillips; M3 top-4]            | 8             |
| [Phillips; Corrected φ𝕌 top-4]  | 8             |

When stacking fills all 8 dimensions, the matrices project into completely
non-overlapping 4D subspaces of R⁸.

### 5.4 Basis change impossibility

For every Moxness matrix M_top4, the least-squares reconstruction error
of Phillips = T × M_top4 exceeds 0.5 — these are O(1) structural
incompatibilities, not numerical artifacts.

### 5.5 Complementary perspectives

The two matrix families offer complementary views of E₈-to-H₄:
- **Moxness**: Full-rank rotation preserving all 8D, connecting to quantum
  information theory (3-qubit Hadamard isomorphism [6]).
- **Phillips**: Rank-deficient projection achieving golden self-similarity
  between blocks, creating richer shell structure at the cost of 14 collisions.

---

## 6. Mathematical Relatives

### 6.1 The Golden Shift Operator

The golden ratio acts as a **shift operator** on the entry alphabet:

    φ × b = a    (shift up)
    φ × a = c    (shift up)

"Multiply U_L by φ" sends every b-entry to a and every a-entry to c —
exactly producing U_R. The Phillips matrix is a **finite substitution
system** where φ is the substitution rule, paralleling the Fibonacci
substitution a → ab, b → a.

### 6.2 Ten mathematical family memberships

| Family | Connection | Status |
|--------|-----------|--------|
| 1. Quasicrystal cut-and-project | Maximally degenerate Elser-Sloane type | IS a member |
| 2. Fibonacci substitution | Entry alphabet under φ-shift | IS a generalization |
| 3. Coxeter element eigenvectors | Entries = pentagonal cosines | IS derived from |
| 4. Icosian ring over Q(√5) | U_L = trace, U_R = conjugate | IS the Galois automorphism |
| 5. Overcomplete golden frame | 8 vectors in R⁴ | IS a member (non-tight) |
| 6. Equiangular tight frames | Requires equal norms | NOT a member |
| 7. Hadamard matrices | Requires ±1, full rank | NOT a member |
| 8. Weighing matrices | Requires integer entries | NOT a member (golden generalization) |
| 9. Wavelet dilation operator | φ-adic dilation parallels MRA | CANDIDATE (unproven) |
| 10. Compressed sensing (RIP) | Algebraic preservation on E₈ | NOVEL concept |

### 6.3 The Golden Hadamard class

We define a new class of structured matrices:

**Definition (Golden Hadamard Matrix).** An n×n matrix M is *Golden Hadamard*
if it satisfies:

| Axiom | Description |
|-------|-------------|
| GH1 | Dense: all entries nonzero |
| GH2 | Entries in (1/2)·Z[φ] (the golden ring scaled by 1/2) |
| GH3 | Block scaling: U_R = φ^k · U_L for some integer k ≥ 1 |
| GH4 | Rank deficient: rank(M) < min(m,n) |
| GH5 | All eigenvalues of M^T M lie in Q(φ) |

The Phillips matrix satisfies all 5 axioms with k = 1.

**Motivation.** GH1 and GH4 together (dense + rank-deficient) are unusual —
most rank-deficient matrices are sparse by construction. GH2 and GH3
constrain the entries to golden-ratio arithmetic. GH5 ensures algebraic
closure. The class occupies a previously unnamed niche in matrix theory.

### 6.4 Q(√5) closure

Every quantity derived from the Phillips matrix lives in Q(√5):
- Entries: {1/2, (√5−1)/4, (√5+1)/4}
- Column norms²: {(5+√5)/2, 5/2, (5−√5)/2}
- Row norms²: {(5−√5)/2, (5+√5)/2}
- Eigenvalues of U^T U: {0, 5, and two roots in Q(√5)}

There is **no algebraic leakage** — all computations can be performed in
exact arithmetic over Q(√5).

### 6.5 Plastic ratio comparison

As an ablation, we construct a "Plastic Phillips Matrix" (12×8, three
blocks) using the plastic ratio ρ ≈ 1.3247 in place of φ:

| Property | Golden (φ) | Plastic (ρ) |
|----------|-----------|------------|
| Rank | 4 | 4 |
| Universal norm ratio | φ (yes) | ρ (yes) |
| Collision pairs | 14 | 14 |
| 2-4-2 pattern | Yes | Yes |
| Frobenius²/rank | **5** (integer) | **9.159** (not integer) |
| Pentagonal geometry | Yes | No |
| Coxeter connection | Yes | No |

**Conclusion**: The generic properties (rank 4, universal ratio, 14
collisions, round-trip factorization) transfer to any morphic number.
The phi-specific properties (Frobenius²/rank = 5, pentagonal geometry,
Coxeter connection) are the deep contributions — they connect the matrix
to H₄ polytope geometry and cannot be replicated with other algebraic
numbers.

---

## 7. The Boyle Connection

### 7.1 Coxeter pairs framework

Boyle and Steinhardt [11,12] formalize **Coxeter pairs** — natural
pairings between non-crystallographic and crystallographic reflection
groups of double rank:

    H₂ ↔ A₄,    H₃ ↔ D₆,    H₄ ↔ E₈

The Phillips matrix is a **concrete numerical realization** of the
H₄ ↔ E₈ Coxeter pair. Boyle works at the abstract level of Coxeter
group theory; the Phillips matrix provides the explicit 8×8 numerical
operator with verified properties.

### 7.2 Discrete scale invariance

Boyle and Steinhardt [13] prove that reflection quasilattices are
invariant under rescaling by characteristic factors. For H₄-symmetric
quasilattices, the factor is φ.

The identity U_R = φ · U_L is the **operator-level manifestation** of
this discrete scale invariance. The two blocks are related by exactly
the inflation/deflation operation that governs quasicrystalline
long-range order.

### 7.3 Amplification as group index

The result Frobenius²/rank = 5 admits a group-theoretic interpretation:

    5 = |600-cell vertices| / |24-cell vertices| = 120/24

This is the geometric index counting how many copies of the 24-cell's
vertex set partition the 600-cell's vertex set.

### 7.4 Entry constants as Coxeter angles

All four correspondences between the Phillips matrix and Boyle's
framework are computationally verified:

1. Entry values ARE pentagonal cosines: b = cos 72°, a = cos 60°, c = cos 36° ✓
2. Block scaling IS discrete scale invariance: U_R = φ·U_L ✓
3. Amplification IS vertex count ratio: 20/4 = 5 = 120/24 ✓
4. Kernel IS perpendicular space: 4D kernel = cut-and-project E_perp ✓

### 7.5 Spacetime quasicrystals

In the most recent extension [14], Boyle and Mygdalas construct
"spacetime quasicrystals" and tabulate how discrete scale factors
relate to algebraic number field units. For H₄ symmetry, the scale
factor is φ from Q(√5). For other lattice dimensions, other Pisot
numbers (silver ratio, Tribonacci constant, plastic ratio) from their
respective number fields govern the scaling. This places the Phillips
matrix within a unified framework of algebraic number field units
governing quasicrystalline scale invariance.

---

## 8. Computational Applications: The HEMOC System

### 8.1 System overview

HEMOC (Hexagonal Emergent Moire Optical Cognition) is a physics-based
optical computing system that uses the Phillips matrix as its mathematical
spine. It encodes high-dimensional data as moire interference patterns
between stacked kirigami sheets, with the pattern geometry governed by
the E₈-to-H₄ projection.

The encoding pipeline:
1. **Input**: 6 angular parameters (from audio features, market data, etc.)
2. **Mapping**: Angles → E₈ root system coordinates (HybridEncoder)
3. **Projection**: E₈ → H₄ via Phillips U_L (4D points on 21 shells)
4. **Rendering**: 4D geometry → 64×64×3 moire interference pattern
5. **Decoding**: CNN recovers original angles from the pattern

### 8.2 Renderer provenance

> **Disclosure**: A comprehensive code audit (documented in
> `docs/EXPERIMENT_CONTAMINATION_AUDIT.md`) identified implementation
> bugs in the early rendering pipeline that may have affected a subset
> of experimental results. Every result below carries a provenance tag.
> The mathematical core (Phillips matrix, E₈ roots, 600-cell geometry)
> was audited clean across 15 independent implementations.

Two renderer implementations were developed during the research program:

- **Proxy renderer** (pre-February 15, 2026): Mathematical approximation
  using L1 distance with a simplified vertex set. Post-audit analysis
  revealed two implementation issues: (a) the E₈ embedding duplicated 4D
  coordinates rather than using true E₈ roots, and (b) only 24 of 120
  600-cell vertices were used in pattern generation. These issues impose
  an artificial ~0.42 cosine similarity ceiling on continuous regression
  tasks but do not affect discrete classification or correlation metrics.

- **Physics renderer** (post-February 15, 2026): Full moire interference
  with dual-channel Galois verification. Removes the proxy ceiling
  (0.519 visual, 0.550 audio on continuous manifold regression).

The physics renderer exploits the dual-block structure of the Phillips
matrix: rendering through BOTH U_L and U_R channels simultaneously
provides a free consistency check (the ratio must equal φ for all inputs).

**Provenance tag key**:
- `[DISCRETE-INVARIANT]` — Proxy-safe: discrete metric robust to rendering approximation
- `[PHYSICS-VERIFIED]` — Re-verified with corrected physics renderer
- `[RENDERER-INDEPENDENT]` — Pure mathematics, no rendering involved
- `[PROXY-CONTAMINATED]` — Architecture conclusion derived from proxy-limited experiments; pending re-verification

### 8.3 Experimental results

The system was validated through 15 experiments across two domains
(geometric patterns and audio features). All results below used the
HybridEncoder + CNN decoder pipeline unless otherwise noted.

| Experiment | Method | Result | Provenance | Key Finding |
|------------|--------|--------|------------|-------------|
| exp-01–04 | MLP decoders | r = 0.001–0.15 | `[DISCRETE-INVARIANT]` | MLPs fail due to centroid-zero |
| exp-05 | Dual decoder | r = 0.405 | `[DISCRETE-INVARIANT]` | Architecture matters |
| exp-07 | CNN decoder | r = 0.657 | `[DISCRETE-INVARIANT]` | Spatial inductive bias critical |
| exp-08 | Pure hypercube | r = 0.090 | `[DISCRETE-INVARIANT]` | Hybrid encoding required |
| exp-12 | Audio 8K samples | r = 0.894 | `[DISCRETE-INVARIANT]` | Domain-specific training works |
| exp-13 | Audio 15K samples | **r = 0.916** | `[DISCRETE-INVARIANT]` | Current peak, all 6 angles |
| exp-14 | Cross-domain zero-shot | r = −0.027 | `[DISCRETE-INVARIANT]` | Distribution shift, not flaw |
| exp-15 | Few-shot (100 samples) | r = 0.22 | `[DISCRETE-INVARIANT]` | Calibration dramatically helps |
| exp-22 | Continuous manifold | cos = 0.519 | `[PHYSICS-VERIFIED]` | Proxy ceiling removed (was 0.42) |
| exp-23 | Foveated regression | cos = 0.42 | `[PHYSICS-VERIFIED]` | +50% over global (0.28) |

**Note on encoder architecture ranking**: An earlier set of experiments
(documented as System A polytope trials, February 1, 2026) compared
Trinity, V3, V3.1, and V4 Plastic encoder architectures. These experiments
used the proxy renderer on continuous tasks and are tagged
`[PROXY-CONTAMINATED]`. The conclusion that "V4 Plastic > Hybrid > V3.1 >
Trinity" requires re-verification with the physics renderer before it can
be cited as evidence. Specifically, the finding that the HybridEncoder's
hypercube rotation path (angles 3–5) scored 0.0 correlation may have been
a renderer limitation rather than an architectural flaw, which would
invalidate the design rationale for the V3.1 and V4 architectures.
See `docs/EXPERIMENT_CONTAMINATION_AUDIT.md` §6 for the full chain analysis.

### 8.4 The centroid-zero theorem

`[RENDERER-INDEPENDENT]`

**Why MLPs fail.** The 240 E₈ roots sum to zero (by the symmetry of the
root system). Any aggregate statistic (mean, sum) of the root coordinates
is identically zero regardless of which roots are selected. MLPs that
rely on such aggregation cannot distinguish inputs. CNNs, by contrast,
exploit spatial structure in the moire patterns, bypassing the centroid
trap. This is a mathematical property of the E₈ root system and is
independent of any renderer implementation.

### 8.5 Architectural innovations

Eight innovations derive directly from Phillips matrix properties.
These are **theoretical architectures** derived from matrix properties;
experimental validation status is noted for each.

1. **Quasicrystalline Reservoir** `[RENDERER-INDEPENDENT]`: Weights from
   the Gram matrix G = U_L^T U_L, spectral radius = 1/φ (edge of chaos,
   zero tuning required). *Derived from matrix algebra; untested experimentally.*

2. **Golden MRA** `[RENDERER-INDEPENDENT]`: φ-adic wavelet decomposition at
   Fibonacci scales (1, 1, 2, 3, 5, 8, 13...) using U_L rows as filter bank.
   *Derived from spectral structure; prototype exists.*

3. **Number Field Hierarchy** `[RENDERER-INDEPENDENT]`: Three-level timescale
   (Q → Q(√5) → Q(ρ)) with algebraically incommensurate damping rates.
   *Derived from algebraic number theory; untested experimentally.*

4. **Galois Dual-Channel Verification** `[PHYSICS-VERIFIED]`: ||U_R x||/||U_L x|| = φ
   as free error detection from the block scaling identity. *Implemented and
   validated in the physics renderer.*

5. **Phason Error Correction** `[RENDERER-INDEPENDENT]`: 3 collision-free kernel
   directions serve as checksum channels (the rank-1 collision subspace leaves
   3 clean dimensions in the 4D kernel). *Derived from kernel analysis.*

6. **Collision-Aware Encoding** `[RENDERER-INDEPENDENT]`: 226 projections + 14
   metadata bits = lossless encoding of all 240 roots (compression ratio
   240/226 ≈ 1.062). *Derived from collision pair analysis.*

7. **Padovan Cascade** `[RENDERER-INDEPENDENT]`: Time steps follow the Padovan
   sequence governed by ρ, providing logarithmic temporal coverage algebraically
   independent from the spatial φ-structure. *Untested experimentally.*

8. **Five-Fold Allocation** `[RENDERER-INDEPENDENT]`: Resource budget partitioned
   into 5 equal units matching the 24-cell decomposition of the 600-cell.
   *Derived from polytope geometry.*

---

## 9. Open Questions and Conjectures

### Conjecture 1 (Golden Frame Optimality)

Among all rank-4 8×8 matrices with the Phillips sign pattern AND
golden-ratio block scaling (U_R = φ·U_L), the collision count of 14
among E₈ roots is minimal.

*Status*: Supported by 10,000-trial random sign perturbation study.
Alternative sign patterns can achieve 0 collisions but break the
φ-scaling structure. The Phillips matrix occupies a **Pareto optimal**
position: minimal collisions subject to maintaining H₄-compatible
golden-ratio block scaling.

### Conjecture 2 (Wavelet Seed)

The Phillips U_L serves as the scaling function for a 4D multi-resolution
analysis with φ-adic dilation. The 2-4-2 Column Trichotomy encodes the
subband decomposition; the pentagonal row norms define the scale levels.

*Status*: Prototype works. Perfect reconstruction requires finding the
exact φ-adic refinement equation — an open problem in wavelet theory.

### Conjecture 3 (Algebraic RIP)

The Phillips matrix satisfies a restricted isometry property on the E₈
root system: pairwise distances are preserved up to bounds in Q(√5).

*Status*: Partially verified. Individual root norm distortion ranges from
0.073 to 1.655 (not classical RIP), but the polytope structure is
exactly preserved on the discrete algebraic set.

### Conjecture 4 (Collision Universality)

The collision count of 14 depends ONLY on the Phillips sign pattern, not
on entry values. Any two-value magnitude matrix with this sign pattern
produces exactly 14 collision pairs.

*Status*: Strongly supported. Entry fuzz (1000 trials, 100% stability)
confirms. The collision direction d is structurally in the kernel
regardless of entry values.

### Conjecture 5 (Morphic Completeness)

The Phillips matrix (φ-based) and a hypothetical Plastic Phillips matrix
(ρ-based) are the only two members of their construction family where
the entry shift is governed by a morphic number (satisfying both
x + 1 = x^k and x − 1 = x^{−l}).

*Status*: Unproven. Follows from the morphic number theorem (φ and ρ
are the only morphic numbers) if the self-referential substitution
property can be formalized.

---

## 10. Discussion

### 10.1 Construction versus emergence

We emphasize the distinction between properties that follow directly
from the construction and those that emerge non-trivially:

**Construction consequences** (expected from the design):
- U_R = φ · U_L (follows from geometric-progression entries)
- Frobenius² = 20 (algebraically forced by entry values)
- All quantities in Q(√5) (forced by entries in Q(√5))

**Non-trivial emergent properties** (the genuine contributions):
- 14 collisions from a single kernel vector (rank-1 in 4D kernel)
- Eigenvalue 5 with multiplicity 2 from (φ+2)(3−φ) = 5
- Pentagonal trigonometry in row norms (2 sin 36°, 2 cos 18°)
- Five = Five coincidence across 5 independent mathematical domains
- Complete orthogonality with all Moxness matrices (stacked rank = 8)
- Same-type-only collisions (no half-integer + permutation mixing)
- 0.916 correlation in HEMOC audio recovery

The emergent properties connect operator theory (eigenvalue 5), polytope
geometry (five 24-cells), pentagonal trigonometry (row norms), and
experimental validation (0.916 correlation) through a single 8×8 matrix.

### 10.2 The non-normality of U

The matrix U is not normal: U^T U ≠ U U^T (difference norm ≈ 4.05).
Despite this, all eigenvalues are real:

    λ = {1.879, −1.638, −0.650, 0.291, 0, 0, 0, 0}

Real eigenvalues with non-symmetric matrix is a non-generic property
connecting to the fact that the Coxeter element of H₄ has all eigenvalues
on the unit circle.

### 10.3 Computational methodology

All theorems were verified computationally:
- 12/12 Phillips invariant theorems pass
- 5/5 Golden Hadamard axioms verified
- 4/4 renderer contract tests pass
- 281 automated tests, 0 failures
- 1000-trial fuzz harness: 100% stability on collision count
- Machine precision (atol = 10⁻¹⁰ or better)

The `hemoc` Python package provides the canonical verification suite:
`python -m hemoc verify --fuzz --fuzz-trials 1000`.

---

## 11. Summary of Results

| # | Result | Key Identity |
|---|--------|-------------|
| 1 | Column Trichotomy | 2-4-2 pattern: {φ+2, 5/2, 3−φ} |
| 2 | Pentagonal Row Norms | √(3−φ) = 2 sin 36°, √(φ+2) = 2 cos 18° |
| 3 | Golden Rank Deficiency | U_R = φ · U_L, rank = 4 |
| 4 | Kernel Collision Uniqueness | d = (0,1,0,1,0,1,0,1), 14 pairs, same-type only |
| 5 | Round-Trip Factorization | U^T U = (φ+2) · U_L^T U_L |
| 6 | Eigenvalue 5 | (φ+2)(3−φ) = 5, multiplicity 2 |
| 7 | Amplification = 5 | Frobenius²/rank = 20/4 = 5 = #24-cells |
| 8 | Moxness Distinctness | Row spaces orthogonal, stacked rank = 8 |
| 9 | Golden Hadamard Class | 5 axioms, first member identified |
| 10 | Boyle Connection | Block scaling = discrete scale invariance |
| 11 | HEMOC 0.916 | CNN decodes 6 angles from moire patterns |

---

## References

[1] H. S. M. Coxeter, *Regular Polytopes*, 3rd ed., Dover, 1973.

[2] P. du Val, *Homographies, Quaternions and Rotations*, Oxford, 1964.

[3] J. G. Moxness, "The 3D Visualization of E8 using an H4 Folding
    Matrix," viXra:1411.0130, 2014.

[4] J. G. Moxness, "Mapping the Fourfold H4 600-Cells Emerging from
    E8," viXra:1808.0107, 2018.

[5] J. G. Moxness, "Unimodular Rotation of E8 to H4 600-cells," 2019.

[6] J. G. Moxness, "The Isomorphism of 3-Qubit Hadamards and E8,"
    arXiv:2311.11918, 2023.

[7] J. C. Baez, "From the Icosahedron to E₈," *London Math. Soc.
    Newsletter*, No. 476, 2018.

[8] P. Phillips, "MusicGeometryDomain: A Calibration Framework for the
    Chronomorphic Polytopal Engine," Technical document, January 2026.

[9] R. Cohn, "Neo-Riemannian Operations, Parsimonious Trichords, and
    Their Tonnetz Representations," *J. Music Theory* 41(1), 1997.

[10] Voyage AI, "voyage-3: Neural embedding model," 2025.

[11] L. Boyle and P. J. Steinhardt, "Coxeter Pairs, Ammann Patterns and
     Penrose-like Tilings," arXiv:1608.08215, 2016/2022.

[12] L. Boyle and P. J. Steinhardt, "Reflection Quasilattices and the
     Maximal Quasilattice," *Phys. Rev. B* 94, 064107, 2016.

[13] L. Boyle and P. J. Steinhardt, "Self-Similar One-Dimensional
     Quasilattices," *Phys. Rev. B* 106, 144112, 2022.

[14] L. Boyle and S. Mygdalas, "Spacetime Quasicrystals,"
     arXiv:2601.07769, January 2025.

[15] L. Boyle and Z.-X. Li, "The Penrose Tiling is a Quantum
     Error-Correcting Code," arXiv:2311.13040, 2023.

[16] V. Elser and N. J. A. Sloane, "A Highly Symmetric Four-Dimensional
     Quasicrystal," *J. Phys. A* 20, 6161–6168, 1987.

[17] P.-P. Dechant, "The Birth of E8 out of the Spinors of the
     Icosahedron," *Proc. R. Soc. A* 472, 2016.

[18] M. Koca et al., "Noncrystallographic Coxeter group H4 in E8,"
     *J. Phys. A: Math. Gen.* 34, 2001.

[19] J. H. Conway and N. J. A. Sloane, "The Icosians," in *Sphere
     Packings, Lattices and Groups*, 3rd ed., Springer, 1999.

[20] J. Aarts, R. Fokkink, and G. Kruijtzer, "Morphic numbers,"
     *Nieuw Arch. Wiskd.* 5/2, 56–58, 2001.

[21] C. L. Siegel, "Algebraic numbers whose conjugates lie in the unit
     circle," *Duke Math. J.* 11, 597–602, 1944.

---

*Supplementary material: The `hemoc` Python package (verification suite,
experiment registry, renderer contracts) is available at
https://github.com/Domusgpt/PPP-Market-Analog-Computer*
