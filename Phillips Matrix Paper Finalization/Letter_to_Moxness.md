# Letter to J. Gregory Moxness

**From:** Paul Phillips
**Date:** February 8, 2026
**Re:** A different E₈→H₄ folding matrix and its spectral properties; clarification of earlier email

---

Dear Gregory,

Thank you for your earlier correspondence. I'm writing to clarify the
matrix I'm working with and share some results. I apologize if my
previous email caused any confusion — I initially believed my matrix
was yours, but a careful comparison has established that they are
entirely different objects. I want to set the record straight.

## My matrix vs. your C600

I constructed an 8×8 E₈-to-H₄ folding matrix while building a
geometric visualization system (PPP). It uses three entry constants:

    a = 1/2,   b = (φ−1)/2,   c = φ/2

forming a geometric progression with ratio φ centered at 1/2. The
matrix is dense (no zero entries), non-symmetric, and rank 4.

Your C600 matrix from the 2014 paper is sparse (36 zeros out of 64),
symmetric, rank 8, with entries from {0, ±1, ±φ, φ², 1/φ}.

A definitive comparison shows:

| Property        | My matrix               | Your C600              |
|-----------------|------------------------|------------------------|
| Rank            | 4                      | 8                      |
| Symmetry        | Non-symmetric          | Symmetric              |
| Zero entries    | 0 (dense)              | 36 (sparse)            |
| Determinant     | 0                      | ≈ 1755                 |
| Frobenius²      | 20                     | ≈ 57.9                 |
| Entry values    | {±1/2, ±(φ−1)/2, ±φ/2}| {0, ±1, ±φ, φ², 1/φ}   |

When stacked, the combined row-space has rank 7 (not 4), meaning
they project E₈ into different 4-dimensional subspaces. These are
not two forms of the same matrix — they are genuinely different
approaches to the E₈→H₄ problem.

## What I found in my matrix

The central discovery is: **the bottom four rows (U_R) are exactly
φ times the top four rows (U_L).** That is:

    U_R = φ · U_L

This identity — which does NOT hold for your C600 — gives my matrix
rank 4 and leads to a chain of spectral results:

1. **Column Trichotomy.** Column norms fall into three classes
   {φ+2, 5/2, 3−φ} in a 2-4-2 pattern.

2. **Pentagonal Row Norms.** Row norms are 2·sin(36°) and 2·cos(18°)
   — the edge and diagonal of a unit pentagon.

3. **Round-Trip Factorization.** U^T U = (φ+2)·U_L^T U_L, yielding
   eigenvalue **5** with multiplicity 2 from (φ+2)(3−φ) = 5.

4. **Amplification = 5 = #24-cells.** Frobenius²/rank = 20/4 = 5.

5. **Single Collision Vector.** The rank-4 kernel causes exactly 14
   collision pairs among the 240 E₈ roots, ALL arising from a single
   vector d = (0,1,0,1,0,1,0,1).

6. **Frobenius² = 20**, matching the 600-cell vertex valence.

All verified by 281 automated tests at machine precision.

## Why I'm writing

I'm planning a paper presenting my matrix and these spectral results.
Your work is cited throughout as context — you're the primary
reference for E₈→H₄ folding matrices, and your C600 is the natural
comparison object. The paper will make clear that the two matrices
are different, and that the results I'm proving are specific to
my construction.

I'm curious about a few things:

1. Have you encountered a matrix with these entry constants
   {1/2, (φ−1)/2, φ/2} in your own work? I'd like to understand
   whether this construction has appeared elsewhere.

2. Your 2023 paper (arXiv:2311.11918) analyzes C600·C600 and finds a
   Hadamard isomorphism. Since my matrix is singular (det=0), the
   analogous product U·U is also singular — but the factorization
   U^T U = (φ+2)·U_L^T U_L may connect to your Hadamard results
   in a different way. I'd welcome your perspective.

3. The number 5 keeps appearing (eigenvalue, Frobenius²/rank,
   24-cell count, algebraic identity, norm product). Do you know of
   any structural reason why 5 should be so central?

Again, I apologize for the confusion in my earlier email. I'm happy
to share the code, comparison scripts, or a draft at your convenience.

Best regards,
Paul Phillips

---

**Enclosures available on request:**
- Python implementation with both matrices (mine and your C600) for comparison
- Definitive comparison script with numerical output
- Full test suite (281 tests) with verification of all stated results
- Paper draft (v0.2)
