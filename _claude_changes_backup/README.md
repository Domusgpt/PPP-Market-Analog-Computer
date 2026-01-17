# Claude's Changes Backup

These files contain modifications made during the simulation module work session.

## What's here:

- `Lattice600.ts` - Modified: inlined vector functions, changed imports
- `ThreeBodyPhaseSpace.ts` - Modified: fixed Figure-8 velocity signs, changed imports
- `SimulationRunner.ts` - New: RK4 + PPP comparison (has energy drift issues)
- `run_test.ts` - New: quick test runner
- `run_simulation.ts` - New: simulation script
- `visualize_orbit.ts` - New: ASCII orbit visualization
- `AUDIT_REPORT.md` - New: incomplete audit report

## Issues:

The pppStep implementation causes ~28% energy drift because the encode/blend/decode cycle doesn't preserve energy. The proper approach should use the existing CausalReasoningEngine.ts.

## Original files restored:

The 3-body files (E8H4Folding.ts, Lattice600.ts, ThreeBodyPhaseSpace.ts, TrinityDecomposition.ts) have been restored to their original state at the root level from origin/3-body-proof branch.
