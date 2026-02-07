# Letter to J. Gregory Moxness

**From:** Paul Phillips
**Date:** February 7, 2026
**Re:** New spectral results on your H4 folding matrix; planned publication

---

Dear Gregory,

Thank you for your earlier correspondence. I'm writing with some results
I've found while working with your H4 folding matrix that I think you'll
find interesting, and to discuss a paper I'd like to publish that builds
on your work.

## What I found

I've been running your 8x8 E8-to-H4 folding matrix through a
computational analysis pipeline -- projecting all 240 E8 roots and
studying the operator structure of the matrix itself. The starting point
was your matrix exactly as published, with the entry constants a = 1/2,
b = (phi-1)/2, c = phi/2.

The central discovery is this: **the bottom four rows (U_R) are exactly
phi times the top four rows (U_L).**

That is:

    U_R = phi * U_L

verified to machine precision (error < 10^-15). This means your 8x8
matrix has rank 4, not 8. The right H4 block carries no independent
information beyond the left -- it's the same projection scaled by the
golden ratio, with no rotation.

I want to be clear: this is not a defect. It turns out to be a remarkably
clean structural property that leads to a chain of results I haven't seen
stated anywhere in the literature.

## The results

Starting from U_R = phi * U_L, I've been able to prove (both algebraically
and computationally, with 281 automated tests):

**1. Column Trichotomy.** The squared column norms of the 8x8 matrix fall
into exactly three classes in a 2-4-2 pattern:

| Class       | Columns  | Norm^2  |
|-------------|----------|---------|
| Expanded    | {0, 4}   | phi + 2 |
| Stable      | {1,2,5,6}| 2.5     |
| Contracted  | {3, 7}   | 3 - phi |

The mean of the extremes equals the stable value (2.5), and the deviation
is exactly sqrt(5)/2. This establishes a dimensional fidelity hierarchy
in the projection.

**2. Pentagonal Row Norms.** All U_L rows have norm sqrt(3-phi) = 2*sin(36 degrees),
and all U_R rows have norm sqrt(phi+2) = 2*cos(18 degrees). The pentagon
geometry of H4 is encoded directly in the row structure.

**3. Round-Trip Factorization.** Since U_R = phi * U_L:

    U^T U = (1 + phi^2) * U_L^T U_L = (phi + 2) * U_L^T U_L

The round-trip operator factors through a single block, scaled by phi + 2.

**4. The Eigenvalue 5.** The round-trip operator U^T U has eigenvalues
{0, 0, 0, 0, 3.14, 5, 5, 6.86}, where the eigenvalue **5** appears with
multiplicity 2. It arises from (phi+2)(3-phi) = 5 -- the same sqrt(5)-coupling
that binds the two blocks. The sum of all eigenvalues is 20 (the Frobenius
norm squared).

**5. Amplification = 5 = number of 24-cells.** The ratio Frobenius^2 / rank
= 20/4 = 5, which equals the number of inscribed 24-cells in the 600-cell.
The sqrt(5)-coupling, the spectral structure, and the polytope geometry all
converge on the same number.

**6. Single Collision Vector.** The rank-4 kernel causes 14 pairs of E8 roots
(out of 240) to collide in the 4D projection. Every collision arises from a
single vector d = (0,1,0,1,0,1,0,1) in the kernel. This vector lives at
odd-indexed dimensions only, and all colliding pairs are orthogonal in E8.
The expanded columns {0, 4} are completely collision-immune.

**7. Frobenius norm^2 = 20**, matching the vertex valence of the 600-cell
(20 tetrahedra per vertex). Block decomposition: ||U_L||^2 = 4(3-phi),
||U_R||^2 = 4(phi+2), and ||U_R||^2 = phi^2 * ||U_L||^2 exactly.

## How I found this

I'm building a system called PPP (Phase-locked Price Projection) that uses
geometric algebra for market data visualization. As part of that work, I
implemented your folding matrix in Python to project E8 roots to 4D and
began writing automated verification tests. When I checked how many unique
4D points emerged from the 240 roots, I expected 240 and got 226. That
discrepancy led me to investigate the kernel, which led to the rank-4
discovery, which led to everything else.

The entire analysis is backed by a test suite (281 tests, 0 failures)
that verifies each claim to machine precision. I'm happy to share the
full codebase.

## The paper I'd like to write

I'm planning a paper tentatively titled something like:

*"Spectral Analysis of the Moxness H4 Folding Matrix: Golden Rank
Deficiency and the Origin of 5 in E8-to-H4 Projection"*

The paper would:

- Present your matrix with full attribution as the object of study
- Cite your publications (the 2014 visualization paper, the 2018 fourfold
  mapping paper, the 2019 unimodular version, and the 2023 Hadamard
  isomorphism paper on arXiv)
- State and prove the results above as new theorems about your matrix
- Note that your unimodular version (Det=1) necessarily has rank 8,
  so the rank-4 property is specific to the original form -- and argue
  that this is structurally meaningful rather than a limitation
- Include computational verification details and the test suite as
  supplementary material

I want to be straightforward: the matrix is yours. The analysis is mine.
I think the results are strong enough to publish, but I'd value your
perspective on them. In particular:

1. Were you aware that U_R = phi * U_L? I haven't found this stated in
   your publications, but you know the matrix far better than I do.
2. Do any of these results appear in work I may have missed?
3. Would you be open to reviewing a draft, or potentially collaborating?

I'm also curious whether the rank-4 property connects to your Hadamard
results from the 2023 paper. Your finding that U*U - (U*U)^{-1} = J
(the reverse identity) analyzes the full 8x8 operator, while my results
focus on the sub-block structure. There may be a deeper connection.

I'd be grateful for any feedback, and I'm happy to send along the code,
the test output, or a preliminary draft at your convenience.

Best regards,
Paul Phillips

---

**Enclosures available on request:**
- Python implementation of both Baez 4x8 and Phillips/Moxness 8x8 pipelines
- Full test suite (281 tests) with verification of all stated results
- Exploration script with raw numerical output
- Preliminary documentation of all theorems with proofs
