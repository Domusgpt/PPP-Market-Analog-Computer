/**
 * HEMAC Physics Module
 *
 * Integrates the Chronomorphic Polytopal Engine (CPE) physics system.
 *
 * Core Principles (from PPP White Paper):
 * - "Reasoning is Rotation": Logical inference = applying rotor R to state S
 * - "Force ∧ State = Torque": Input generates rotation via wedge product
 * - "Unitary Update: R·S·R̃": Transformations preserve norm (truth value)
 *
 * The Three Causal Constraints (Gärdenfors):
 * 1. MONOTONICITY: Larger forces → larger results
 * 2. CONTINUITY: Small force changes → small result changes
 * 3. CONVEXITY: Intermediate forces → intermediate results
 *
 * Source: cloned_repos/chronomorphic-engine/lib/
 * - engine/CausalReasoningEngine.ts (641 lines)
 * - math/GeometricAlgebra.ts
 * - topology/Lattice24.ts
 * - types/index.ts (529 lines)
 */

// Re-export types
export * from './types.js';

// Re-export geometric algebra functions
export * from './GeometricAlgebra.js';

// Re-export lattice
export * from './Lattice24.js';

// Re-export engine
export {
  CausalReasoningEngine,
  createCausalReasoningEngine,
  createCausalReasoningEngineAt
} from './CausalReasoningEngine.js';
