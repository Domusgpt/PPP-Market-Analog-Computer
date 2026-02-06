# Phase 1 – PPP Core Boundary (Completed)

## Scope
Formalize the PPP Core boundary by defining adapter contracts, consolidating core configuration, and ensuring phase-lock tests are runnable in the standard toolchain.

---

## Deliverables

### 1) Adapter Contract (Code)
- Added a formal adapter interface (`PPPAdapter`) that normalizes external sources into `RawApiTick`.
- Added a `PPPCoreConfig` type to encapsulate TimeBinder + smoothing flags for integration shells.

### 2) Core Exports
- Exposed contracts via `src/lib/index.ts` so consumers can import from a single root.

### 3) Test Tooling Alignment
- Added `test:phase-lock` (tsx) and `test:all` scripts so the Phase‑Locked Stereoscopy test suite can run alongside existing JS tests.

---

## Acceptance Criteria (Met)
- [x] Core adapter contract available in codebase.
- [x] Core config boundary available in codebase.
- [x] Phase‑lock tests runnable via package scripts.

---

## Notes
This boundary does **not** change runtime behavior. It formalizes integration contracts so Phase 2 adapter work can proceed without ambiguity.
