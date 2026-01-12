// File: src/core/analysis.ts
// Geometric Analysis Tools - Coherence metrics and trajectory analysis

import { Quaternion, PHI } from './geometry';
import type { TrajectoryHistory, TrajectoryStats } from './trajectory';

/**
 * Coherence metrics for the cognitive manifold
 */
export interface CoherenceMetrics {
  /** Spinor alignment (left · right quaternion) */
  spinorAlignment: number;
  /** Isoclinic symmetry measure */
  isoclinicSymmetry: number;
  /** Golden ratio resonance (how close trajectory is to φ-based patterns) */
  goldenResonance: number;
  /** Ergodicity estimate (coverage uniformity) */
  ergodicityScore: number;
  /** Stability measure (low = chaotic, high = stable) */
  stabilityIndex: number;
  /** Overall coherence (0-1) */
  overallCoherence: number;
}

/**
 * Geometric invariants
 */
export interface GeometricInvariants {
  /** Distance from origin */
  radialDistance: number;
  /** L1 norm (|w| + |x| + |y| + |z|) */
  l1Norm: number;
  /** L2 norm (Euclidean) */
  l2Norm: number;
  /** LInf norm (max absolute component) */
  lInfNorm: number;
  /** Quaternion angle (2 * arccos(|w|)) */
  quaternionAngle: number;
  /** Component variance */
  componentVariance: number;
  /** Axis of rotation (for non-scalar quaternions) */
  rotationAxis: [number, number, number] | null;
}

/**
 * Lattice analysis results
 */
export interface LatticeAnalysis {
  /** Index of nearest vertex */
  nearestVertexIndex: number;
  /** Distance to nearest vertex */
  nearestVertexDistance: number;
  /** Indices of nearest 3 vertices */
  nearestTriad: [number, number, number];
  /** Barycentric coordinates within nearest face */
  barycentricCoords: [number, number, number];
  /** Which octant of the 24-cell */
  octant: string;
  /** Penetration depth (negative = inside) */
  penetrationDepth: number;
}

/**
 * Compute coherence metrics for the current state
 */
export function computeCoherenceMetrics(
  position: Quaternion,
  leftRotor: Quaternion,
  rightRotor: Quaternion,
  trajectory?: TrajectoryHistory
): CoherenceMetrics {
  // Spinor alignment: dot product of left and right rotors
  const spinorAlignment = Math.abs(leftRotor.dot(rightRotor));

  // Isoclinic symmetry: how balanced the rotors are
  const leftNorm = leftRotor.norm();
  const rightNorm = rightRotor.norm();
  const isoclinicSymmetry = 1 - Math.abs(leftNorm - rightNorm) / Math.max(leftNorm, rightNorm, 0.001);

  // Golden resonance: check for φ-based patterns in position
  const posArray = position.toArray();
  let goldenMatch = 0;
  for (let i = 0; i < 4; i++) {
    const ratio = Math.abs(posArray[i]) / Math.max(...posArray.map(Math.abs), 0.001);
    const phiDist = Math.min(Math.abs(ratio - PHI), Math.abs(ratio - 1 / PHI), Math.abs(ratio - 1));
    goldenMatch += 1 - Math.min(1, phiDist);
  }
  const goldenResonance = goldenMatch / 4;

  // Ergodicity: based on trajectory spread and coverage
  let ergodicityScore = 0.5; // Default
  if (trajectory && trajectory.length > 10) {
    const spread = trajectory.getSpread();
    const expectedSpread = 0.5; // Expected spread in unit 24-cell
    ergodicityScore = Math.min(1, spread / expectedSpread);

    // Adjust for oscillation (oscillating = low ergodicity)
    if (trajectory.isOscillating()) {
      ergodicityScore *= 0.5;
    }
  }

  // Stability: based on trajectory curvature
  let stabilityIndex = 0.5;
  if (trajectory && trajectory.length > 10) {
    const stats = trajectory.getStats();
    // Low curvature = high stability
    stabilityIndex = Math.max(0, 1 - stats.averageCurvature / Math.PI);
  }

  // Overall coherence: weighted average
  const overallCoherence =
    spinorAlignment * 0.25 +
    isoclinicSymmetry * 0.2 +
    goldenResonance * 0.15 +
    ergodicityScore * 0.2 +
    stabilityIndex * 0.2;

  return {
    spinorAlignment,
    isoclinicSymmetry,
    goldenResonance,
    ergodicityScore,
    stabilityIndex,
    overallCoherence,
  };
}

/**
 * Compute geometric invariants for a quaternion
 */
export function computeGeometricInvariants(q: Quaternion): GeometricInvariants {
  const arr = q.toArray();

  // Various norms
  const radialDistance = q.norm();
  const l1Norm = arr.reduce((sum, v) => sum + Math.abs(v), 0);
  const l2Norm = radialDistance;
  const lInfNorm = Math.max(...arr.map(Math.abs));

  // Quaternion angle
  const quaternionAngle = 2 * Math.acos(Math.min(1, Math.abs(q.w)));

  // Component variance
  const mean = arr.reduce((sum, v) => sum + v, 0) / 4;
  const componentVariance = arr.reduce((sum, v) => sum + (v - mean) ** 2, 0) / 4;

  // Rotation axis (imaginary part normalized)
  let rotationAxis: [number, number, number] | null = null;
  const imagNorm = Math.sqrt(q.x ** 2 + q.y ** 2 + q.z ** 2);
  if (imagNorm > 1e-6) {
    rotationAxis = [q.x / imagNorm, q.y / imagNorm, q.z / imagNorm];
  }

  return {
    radialDistance,
    l1Norm,
    l2Norm,
    lInfNorm,
    quaternionAngle,
    componentVariance,
    rotationAxis,
  };
}

/**
 * Analyze position relative to the 24-cell lattice
 */
export function analyzeLatticePosition(
  position: Quaternion,
  vertices: readonly Quaternion[]
): LatticeAnalysis {
  // Find distances to all vertices
  const distances = vertices.map((v, i) => ({
    index: i,
    distance: position.distanceTo(v),
  }));

  // Sort by distance
  distances.sort((a, b) => a.distance - b.distance);

  const nearestVertexIndex = distances[0].index;
  const nearestVertexDistance = distances[0].distance;
  const nearestTriad: [number, number, number] = [
    distances[0].index,
    distances[1].index,
    distances[2].index,
  ];

  // Compute barycentric coordinates (simplified - based on inverse distances)
  // Note: Full barycentric would use v0, v1, v2 vertices but we use distance approximation
  const d0 = distances[0].distance;
  const d1 = distances[1].distance;
  const d2 = distances[2].distance;
  const dSum = 1 / d0 + 1 / d1 + 1 / d2;

  const barycentricCoords: [number, number, number] = [
    1 / d0 / dSum,
    1 / d1 / dSum,
    1 / d2 / dSum,
  ];

  // Determine octant based on signs
  const signs = position.toArray().map((v) => (v >= 0 ? '+' : '-'));
  const octant = signs.join('');

  // Penetration depth (L1 norm - 2)
  const penetrationDepth =
    Math.abs(position.w) +
    Math.abs(position.x) +
    Math.abs(position.y) +
    Math.abs(position.z) -
    2;

  return {
    nearestVertexIndex,
    nearestVertexDistance,
    nearestTriad,
    barycentricCoords,
    octant,
    penetrationDepth,
  };
}

/**
 * Compute trajectory quality metrics
 */
export interface TrajectoryQuality {
  /** How smooth the trajectory is (0-1) */
  smoothness: number;
  /** Coverage of the manifold (0-1) */
  coverage: number;
  /** Time spent in safe region (0-1) */
  safetyRatio: number;
  /** Exploration efficiency */
  explorationEfficiency: number;
  /** Overall quality score */
  qualityScore: number;
}

export function computeTrajectoryQuality(
  stats: TrajectoryStats,
  isOscillating: boolean
): TrajectoryQuality {
  const { totalPoints, pathLength, statusTime, averageCurvature } = stats;

  // Smoothness: inverse of average curvature
  const smoothness = Math.max(0, 1 - averageCurvature / Math.PI);

  // Coverage: based on spread and path length
  const expectedPathLength = totalPoints * 0.1; // Expected step size
  const coverage = Math.min(1, pathLength / Math.max(expectedPathLength, 1));

  // Safety ratio
  const totalTime = statusTime.SAFE + statusTime.WARNING + statusTime.VIOLATION;
  const safetyRatio = totalTime > 0 ? statusTime.SAFE / totalTime : 1;

  // Exploration efficiency: coverage per path length, penalized for oscillation
  let explorationEfficiency = totalPoints > 0 ? coverage / Math.max(1, pathLength / totalPoints) : 0;
  if (isOscillating) {
    explorationEfficiency *= 0.5;
  }

  // Overall quality
  const qualityScore =
    smoothness * 0.2 + coverage * 0.25 + safetyRatio * 0.3 + explorationEfficiency * 0.25;

  return {
    smoothness,
    coverage,
    safetyRatio,
    explorationEfficiency,
    qualityScore,
  };
}

/**
 * Format metrics for display
 */
export function formatMetricValue(value: number, decimals = 3): string {
  return value.toFixed(decimals);
}

/**
 * Get a qualitative assessment of a metric
 */
export function getMetricAssessment(
  value: number,
  thresholds = { low: 0.3, medium: 0.6, high: 0.8 }
): 'critical' | 'low' | 'medium' | 'high' | 'excellent' {
  if (value < thresholds.low) return 'critical';
  if (value < thresholds.medium) return 'low';
  if (value < thresholds.high) return 'medium';
  if (value < 0.95) return 'high';
  return 'excellent';
}
