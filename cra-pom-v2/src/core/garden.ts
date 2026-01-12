// File: src/core/garden.ts
// Garden of Forking Paths - Multi-future trajectory prediction
// Implements the PPP principle: "the future is a polytope of possibilities"

import { PhasorVector } from './fhrr';
import { RuleLibrary } from './rotor';
import { ConvexPolytope } from './polytope';

/**
 * A single possible future trajectory
 */
export interface FuturePath {
  id: string;
  probability: number;
  states: PhasorVector[];
  rules: string[]; // Rules applied at each step
  terminated: boolean;
  terminationReason?: string;
}

/**
 * A branching point where the future splits
 */
export interface Fork {
  step: number;
  state: PhasorVector;
  branches: FuturePath[];
  decisionBasis: string; // What caused the fork
}

/**
 * Result of multi-future prediction
 */
export interface FutureBundle {
  currentState: PhasorVector;
  paths: FuturePath[];
  forks: Fork[];
  totalProbability: number;
  convergencePoint?: PhasorVector; // Where paths re-merge, if any
}

/**
 * Constraint that futures must respect
 */
export interface FutureConstraint {
  name: string;
  type: 'polytope' | 'similarity' | 'energy';
  polytope?: ConvexPolytope;
  targetVector?: PhasorVector;
  threshold?: number;
  penalty: number; // Probability reduction for violation
}

/**
 * Garden of Forking Paths
 *
 * From the PPP paper:
 * "The future is not a single vector, but a Polytope of Possibility.
 *  As time progresses, uncertainty grows and the polytope expands.
 *  Decision points are 'forks' where trajectories split via orthogonal rotations."
 *
 * This class manages multiple simultaneous possible futures, each represented
 * as a trajectory through the semantic space.
 */
export class GardenOfForkingPaths {
  private ruleLibrary: RuleLibrary;
  private constraints: FutureConstraint[] = [];
  private maxPaths: number;
  private probabilityThreshold: number;
  private maxDepth: number;

  constructor(
    ruleLibrary: RuleLibrary,
    options: {
      dimension?: number;
      maxPaths?: number;
      probabilityThreshold?: number;
      maxDepth?: number;
    } = {}
  ) {
    this.ruleLibrary = ruleLibrary;
    this.maxPaths = options.maxPaths ?? 8;
    this.probabilityThreshold = options.probabilityThreshold ?? 0.01;
    this.maxDepth = options.maxDepth ?? 10;
  }

  /**
   * Add a constraint that futures must respect
   */
  addConstraint(constraint: FutureConstraint): void {
    this.constraints.push(constraint);
  }

  /**
   * Predict multiple possible futures from a starting state
   *
   * The core algorithm:
   * 1. Start with current state and probability 1
   * 2. At each step, find applicable rules
   * 3. If multiple rules apply, FORK the future
   * 4. Apply each rule along its branch
   * 5. Prune low-probability branches
   * 6. Continue until max depth or all paths terminate
   */
  predictFutures(
    initialState: PhasorVector,
    steps: number
  ): FutureBundle {
    const paths: FuturePath[] = [
      {
        id: 'root',
        probability: 1.0,
        states: [initialState.clone()],
        rules: [],
        terminated: false,
      },
    ];

    const forks: Fork[] = [];
    let nextPathId = 1;

    // Evolve for specified number of steps
    for (let step = 0; step < Math.min(steps, this.maxDepth); step++) {
      const newPaths: FuturePath[] = [];

      for (const path of paths) {
        if (path.terminated) {
          newPaths.push(path);
          continue;
        }

        const currentState = path.states[path.states.length - 1];

        // Find applicable rules
        const applicableRules = this.ruleLibrary.findApplicableRules(currentState);

        if (applicableRules.length === 0) {
          // No rules apply - path terminates
          path.terminated = true;
          path.terminationReason = 'no_applicable_rules';
          newPaths.push(path);
          continue;
        }

        // Check if we should fork (multiple high-applicability rules)
        const highApplicability = applicableRules.filter((r) => r.applicability > 0.3);

        if (highApplicability.length > 1 && newPaths.length + paths.length < this.maxPaths) {
          // FORK: Create a branch for each applicable rule
          const fork: Fork = {
            step,
            state: currentState.clone(),
            branches: [],
            decisionBasis: `multiple_rules:${highApplicability.map((r) => r.rule.name).join(',')}`,
          };

          // Calculate probabilities for each branch
          const totalApplicability = highApplicability.reduce((s, r) => s + r.applicability, 0);

          for (const { rule, applicability } of highApplicability) {
            const newState = rule.rotor.apply(currentState);
            const branchProbability = (path.probability * applicability) / totalApplicability;

            // Check constraints
            const constraintPenalty = this.checkConstraints(newState);

            const newPath: FuturePath = {
              id: `path_${nextPathId++}`,
              probability: branchProbability * (1 - constraintPenalty),
              states: [...path.states, newState],
              rules: [...path.rules, rule.name],
              terminated: false,
            };

            // Terminate if probability too low
            if (newPath.probability < this.probabilityThreshold) {
              newPath.terminated = true;
              newPath.terminationReason = 'low_probability';
            }

            fork.branches.push(newPath);
            newPaths.push(newPath);
          }

          forks.push(fork);
        } else {
          // No fork - apply the best rule
          const bestRule = applicableRules[0].rule;
          const newState = bestRule.rotor.apply(currentState);

          // Check constraints
          const constraintPenalty = this.checkConstraints(newState);

          path.states.push(newState);
          path.rules.push(bestRule.name);
          path.probability *= 1 - constraintPenalty;

          if (path.probability < this.probabilityThreshold) {
            path.terminated = true;
            path.terminationReason = 'low_probability';
          }

          newPaths.push(path);
        }
      }

      paths.length = 0;
      paths.push(...newPaths);

      // Prune to max paths (keep highest probability)
      if (paths.length > this.maxPaths) {
        paths.sort((a, b) => b.probability - a.probability);
        paths.length = this.maxPaths;
      }
    }

    // Normalize probabilities
    const totalProb = paths.reduce((s, p) => s + p.probability, 0);
    for (const path of paths) {
      path.probability /= totalProb;
    }

    // Check for convergence
    const convergencePoint = this.findConvergence(paths);

    return {
      currentState: initialState,
      paths,
      forks,
      totalProbability: totalProb,
      convergencePoint,
    };
  }

  /**
   * Check constraints and return total penalty (0-1)
   */
  private checkConstraints(state: PhasorVector): number {
    let totalPenalty = 0;

    for (const constraint of this.constraints) {
      switch (constraint.type) {
        case 'polytope':
          if (constraint.polytope) {
            // Convert phasor to hypervector-like for polytope check
            const hvLike = new Float32Array(state.dimension);
            for (let i = 0; i < state.dimension; i++) {
              hvLike[i] = Math.cos(state.phases[i]);
            }
            const result = constraint.polytope.containsPoint(hvLike);
            if (!result.isInside) {
              totalPenalty += constraint.penalty;
            }
          }
          break;

        case 'similarity':
          if (constraint.targetVector && constraint.threshold !== undefined) {
            const sim = state.similarity(constraint.targetVector);
            if (sim < constraint.threshold) {
              totalPenalty += constraint.penalty * (1 - sim / constraint.threshold);
            }
          }
          break;

        case 'energy':
          // Check coherence as energy proxy
          const coherence = state.coherence();
          if (constraint.threshold !== undefined && coherence < constraint.threshold) {
            totalPenalty += constraint.penalty;
          }
          break;
      }
    }

    return Math.min(totalPenalty, 1);
  }

  /**
   * Find if paths converge to a similar point
   */
  private findConvergence(paths: FuturePath[]): PhasorVector | undefined {
    if (paths.length < 2) return undefined;

    // Get final states
    const finalStates = paths
      .filter((p) => p.states.length > 0)
      .map((p) => p.states[p.states.length - 1]);

    if (finalStates.length < 2) return undefined;

    // Check pairwise similarity
    let totalSim = 0;
    let count = 0;

    for (let i = 0; i < finalStates.length; i++) {
      for (let j = i + 1; j < finalStates.length; j++) {
        totalSim += finalStates[i].similarity(finalStates[j]);
        count++;
      }
    }

    const avgSim = totalSim / count;

    // If average similarity > 0.8, consider converged
    if (avgSim > 0.8) {
      // Return weighted average as convergence point
      return PhasorVector.bundle(finalStates);
    }

    return undefined;
  }

  /**
   * Get the most probable future
   */
  getMostProbableFuture(bundle: FutureBundle): FuturePath | undefined {
    if (bundle.paths.length === 0) return undefined;
    return bundle.paths.reduce((best, path) =>
      path.probability > best.probability ? path : best
    );
  }

  /**
   * Get the "expected" future (probability-weighted average of final states)
   */
  getExpectedFuture(bundle: FutureBundle): PhasorVector | undefined {
    const activePathsWithFinals = bundle.paths
      .filter((p) => p.states.length > 0)
      .map((p) => ({
        state: p.states[p.states.length - 1],
        weight: p.probability,
      }));

    if (activePathsWithFinals.length === 0) return undefined;

    // Weighted bundle
    const dimension = activePathsWithFinals[0].state.dimension;
    const phases = new Float32Array(dimension);

    for (let i = 0; i < dimension; i++) {
      let sumRe = 0;
      let sumIm = 0;
      for (const { state, weight } of activePathsWithFinals) {
        sumRe += weight * Math.cos(state.phases[i]);
        sumIm += weight * Math.sin(state.phases[i]);
      }
      phases[i] = Math.atan2(sumIm, sumRe);
      if (phases[i] < 0) phases[i] += 2 * Math.PI;
    }

    return new PhasorVector(dimension, phases);
  }

  /**
   * Check if a "goal" polytope is reachable
   */
  checkReachability(
    initialState: PhasorVector,
    goalPolytope: ConvexPolytope,
    maxSteps: number = 20
  ): {
    reachable: boolean;
    probability: number;
    shortestPath?: FuturePath;
  } {
    const bundle = this.predictFutures(initialState, maxSteps);

    let bestPath: FuturePath | undefined;
    let bestProbability = 0;

    for (const path of bundle.paths) {
      for (const state of path.states) {
        // Convert to hypervector-like for polytope check
        const hvLike = new Float32Array(state.dimension);
        for (let i = 0; i < state.dimension; i++) {
          hvLike[i] = Math.cos(state.phases[i]);
        }

        if (goalPolytope.containsPoint(hvLike).isInside) {
          if (path.probability > bestProbability) {
            bestProbability = path.probability;
            bestPath = path;
          }
          break;
        }
      }
    }

    return {
      reachable: bestProbability > 0,
      probability: bestProbability,
      shortestPath: bestPath,
    };
  }

  /**
   * Compute the "Future Polytope" - the convex hull of all possible futures
   */
  computeFuturePolytope(bundle: FutureBundle): {
    center: PhasorVector;
    spread: number;
    volume: number;
  } {
    const finalStates = bundle.paths
      .filter((p) => p.states.length > 0)
      .map((p) => p.states[p.states.length - 1]);

    if (finalStates.length === 0) {
      return {
        center: bundle.currentState.clone(),
        spread: 0,
        volume: 0,
      };
    }

    // Center is the average
    const center = PhasorVector.bundle(finalStates);

    // Spread is the average distance from center
    let totalDist = 0;
    for (const state of finalStates) {
      totalDist += 1 - state.similarity(center);
    }
    const spread = totalDist / finalStates.length;

    // Volume is approximated by spread^dimension
    const volume = Math.pow(spread, Math.min(finalStates.length, 10));

    return { center, spread, volume };
  }

  /**
   * Get a summary of the future prediction
   */
  summarize(bundle: FutureBundle): string {
    const mostProbable = this.getMostProbableFuture(bundle);
    const futurePolytope = this.computeFuturePolytope(bundle);

    const lines: string[] = [
      `Garden of Forking Paths Summary:`,
      `- Active paths: ${bundle.paths.length}`,
      `- Forks encountered: ${bundle.forks.length}`,
      `- Future spread: ${(futurePolytope.spread * 100).toFixed(1)}%`,
    ];

    if (mostProbable) {
      lines.push(`- Most probable path: ${mostProbable.id} (${(mostProbable.probability * 100).toFixed(1)}%)`);
      lines.push(`  Rules: ${mostProbable.rules.join(' → ')}`);
    }

    if (bundle.convergencePoint) {
      lines.push(`- Paths converge (futures re-merge)`);
    }

    return lines.join('\n');
  }
}

/**
 * TimeHorizon - Models how uncertainty grows over time
 */
export class TimeHorizon {
  private uncertaintyGrowthRate: number;
  private maxUncertainty: number;

  constructor(growthRate: number = 0.1, maxUncertainty: number = 1.0) {
    this.uncertaintyGrowthRate = growthRate;
    this.maxUncertainty = maxUncertainty;
  }

  /**
   * Get uncertainty at a given time step
   */
  uncertaintyAt(steps: number): number {
    return Math.min(
      this.maxUncertainty,
      1 - Math.exp(-this.uncertaintyGrowthRate * steps)
    );
  }

  /**
   * Get the "confidence horizon" - steps until uncertainty exceeds threshold
   */
  confidenceHorizon(threshold: number = 0.5): number {
    if (threshold >= this.maxUncertainty) return Infinity;
    return Math.ceil(-Math.log(1 - threshold) / this.uncertaintyGrowthRate);
  }
}

export default { GardenOfForkingPaths, TimeHorizon };
