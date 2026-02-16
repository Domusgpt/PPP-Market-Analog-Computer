# Archived Modules

These TypeScript modules are **superseded** by their canonical versions in `../geometric_algebra/`.

They were archived because they contained conflicting import paths (referencing non-existent
`../math/` and `../types/` directories) while the `geometric_algebra/` versions use correct
relative imports.

| Archived File | Canonical Location | Reason |
|---|---|---|
| `CPE_GeometricAlgebra.ts` | `../geometric_algebra/GeometricAlgebra.ts` | Identical content, broken import paths |
| `CPE_Lattice24.ts` | `../geometric_algebra/Lattice24.ts` | Identical content, broken import paths |
| `CausalReasoningEngine.ts` | `../geometric_algebra/CausalReasoningEngine.ts` | Identical content, broken `../math/` imports |

**Do not import from this directory.** Use `geometric_algebra/` instead.
