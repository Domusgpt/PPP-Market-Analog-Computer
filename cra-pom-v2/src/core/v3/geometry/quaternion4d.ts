/**
 * Quaternion-Based 4D Rotations
 *
 * Rotations in 4D are more complex than 3D - they require TWO quaternions
 * (left and right) to fully describe all possible rotations in SO(4).
 *
 * MATHEMATICAL BASIS:
 * - SO(4) ≅ (SU(2) × SU(2)) / Z₂
 * - Any 4D rotation can be expressed as: v' = q_L * v * q_R^(-1)
 * - Special case: isoclinic rotation uses q_L = q_R
 *
 * ISOCLINIC ROTATIONS:
 * These are unique to 4D - they rotate ALL points by the same angle.
 * Key property: continuously rotating a 24-cell isoclinically traces
 * out the 120 vertices of a 600-cell!
 *
 * MUSICAL APPLICATION:
 * - Rotation = key modulation
 * - Isoclinic rotation = smooth chromatic ascent/descent
 * - Specific rotations map to musical transformations
 *
 * REFERENCES:
 * - https://en.wikipedia.org/wiki/Rotations_in_4-dimensional_Euclidean_space
 * - https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
 */

import type { Vector4D } from '../music/music-geometry-domain';

// ============================================================================
// Types
// ============================================================================

export interface Quaternion {
  w: number;  // Real part
  x: number;  // i component
  y: number;  // j component
  z: number;  // k component
}

export type RotationPlane =
  | 'WX' | 'WY' | 'WZ'  // Rotations involving W axis
  | 'XY' | 'XZ' | 'YZ'; // Standard 3D-like rotations

export interface Rotation4D {
  left: Quaternion;   // Left quaternion
  right: Quaternion;  // Right quaternion
}

// ============================================================================
// Quaternion Operations
// ============================================================================

/**
 * Create a quaternion from components
 */
export function createQuaternion(w: number, x: number, y: number, z: number): Quaternion {
  return { w, x, y, z };
}

/**
 * Create identity quaternion (no rotation)
 */
export function identityQuaternion(): Quaternion {
  return { w: 1, x: 0, y: 0, z: 0 };
}

/**
 * Quaternion magnitude (norm)
 */
export function quaternionNorm(q: Quaternion): number {
  return Math.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
}

/**
 * Normalize quaternion to unit length
 */
export function normalizeQuaternion(q: Quaternion): Quaternion {
  const norm = quaternionNorm(q);
  if (norm === 0) return identityQuaternion();
  return {
    w: q.w / norm,
    x: q.x / norm,
    y: q.y / norm,
    z: q.z / norm,
  };
}

/**
 * Quaternion conjugate (q* = w - xi - yj - zk)
 */
export function quaternionConjugate(q: Quaternion): Quaternion {
  return { w: q.w, x: -q.x, y: -q.y, z: -q.z };
}

/**
 * Quaternion inverse (q^-1 = q* / |q|²)
 */
export function quaternionInverse(q: Quaternion): Quaternion {
  const normSq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
  if (normSq === 0) return identityQuaternion();
  return {
    w: q.w / normSq,
    x: -q.x / normSq,
    y: -q.y / normSq,
    z: -q.z / normSq,
  };
}

/**
 * Quaternion multiplication (Hamilton product)
 * q1 * q2 = (w1w2 - x1x2 - y1y2 - z1z2) +
 *           (w1x2 + x1w2 + y1z2 - z1y2)i +
 *           (w1y2 - x1z2 + y1w2 + z1x2)j +
 *           (w1z2 + x1y2 - y1x2 + z1w2)k
 */
export function quaternionMultiply(q1: Quaternion, q2: Quaternion): Quaternion {
  return {
    w: q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
    x: q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
    y: q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
    z: q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
  };
}

/**
 * Create quaternion from axis-angle representation
 */
export function quaternionFromAxisAngle(
  axis: { x: number; y: number; z: number },
  angle: number
): Quaternion {
  const halfAngle = angle / 2;
  const sinHalf = Math.sin(halfAngle);
  const norm = Math.sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);

  if (norm === 0) return identityQuaternion();

  return normalizeQuaternion({
    w: Math.cos(halfAngle),
    x: (axis.x / norm) * sinHalf,
    y: (axis.y / norm) * sinHalf,
    z: (axis.z / norm) * sinHalf,
  });
}

/**
 * Spherical linear interpolation between quaternions
 */
export function quaternionSlerp(q1: Quaternion, q2: Quaternion, t: number): Quaternion {
  // Compute dot product
  let dot = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z;

  // If dot is negative, negate one quaternion to take shorter path
  let q2Adj = q2;
  if (dot < 0) {
    q2Adj = { w: -q2.w, x: -q2.x, y: -q2.y, z: -q2.z };
    dot = -dot;
  }

  // If quaternions are very close, use linear interpolation
  if (dot > 0.9995) {
    return normalizeQuaternion({
      w: q1.w + t * (q2Adj.w - q1.w),
      x: q1.x + t * (q2Adj.x - q1.x),
      y: q1.y + t * (q2Adj.y - q1.y),
      z: q1.z + t * (q2Adj.z - q1.z),
    });
  }

  const theta = Math.acos(dot);
  const sinTheta = Math.sin(theta);
  const w1 = Math.sin((1 - t) * theta) / sinTheta;
  const w2 = Math.sin(t * theta) / sinTheta;

  return {
    w: w1 * q1.w + w2 * q2Adj.w,
    x: w1 * q1.x + w2 * q2Adj.x,
    y: w1 * q1.y + w2 * q2Adj.y,
    z: w1 * q1.z + w2 * q2Adj.z,
  };
}

// ============================================================================
// 4D Rotation Operations
// ============================================================================

/**
 * Apply a general 4D rotation to a point
 * Formula: v' = q_L * v * q_R^(-1)
 * where v is treated as a quaternion (w, x, y, z)
 */
export function rotate4D(point: Vector4D, rotation: Rotation4D): Vector4D {
  // Treat point as quaternion
  const v: Quaternion = {
    w: point.w,
    x: point.x,
    y: point.y,
    z: point.z,
  };

  // Compute q_L * v * q_R^(-1)
  const rightInv = quaternionInverse(rotation.right);
  const temp = quaternionMultiply(rotation.left, v);
  const result = quaternionMultiply(temp, rightInv);

  return {
    w: result.w,
    x: result.x,
    y: result.y,
    z: result.z,
  };
}

/**
 * Create an isoclinic rotation (q_L = q_R)
 * All points rotate by the same angle
 */
export function createIsoclinicRotation(angle: number, plane: RotationPlane): Rotation4D {
  const halfAngle = angle / 2;
  const cos = Math.cos(halfAngle);
  const sin = Math.sin(halfAngle);

  let q: Quaternion;

  switch (plane) {
    case 'WX':
      q = { w: cos, x: sin, y: 0, z: 0 };
      break;
    case 'WY':
      q = { w: cos, x: 0, y: sin, z: 0 };
      break;
    case 'WZ':
      q = { w: cos, x: 0, y: 0, z: sin };
      break;
    case 'XY':
      q = { w: cos, x: 0, y: 0, z: sin }; // XY rotation uses Z axis
      break;
    case 'XZ':
      q = { w: cos, x: 0, y: -sin, z: 0 }; // XZ rotation uses Y axis
      break;
    case 'YZ':
      q = { w: cos, x: sin, y: 0, z: 0 }; // YZ rotation uses X axis
      break;
    default:
      q = identityQuaternion();
  }

  return { left: q, right: q };
}

/**
 * Create a simple rotation (one quaternion is identity)
 * This is a "single" rotation in a 2D plane
 */
export function createSimpleRotation(angle: number, plane: RotationPlane): Rotation4D {
  const rotation = createIsoclinicRotation(angle, plane);
  return {
    left: rotation.left,
    right: identityQuaternion(),
  };
}

/**
 * Create a double rotation (different angles in two planes)
 * This is the most general 4D rotation
 */
export function createDoubleRotation(
  angle1: number,
  plane1: RotationPlane,
  angle2: number,
  plane2: RotationPlane
): Rotation4D {
  const rot1 = createIsoclinicRotation(angle1, plane1);
  const rot2 = createIsoclinicRotation(angle2, plane2);

  return {
    left: quaternionMultiply(rot1.left, rot2.left),
    right: quaternionMultiply(rot1.right, rot2.right),
  };
}

/**
 * Compose two 4D rotations
 */
export function composeRotations(r1: Rotation4D, r2: Rotation4D): Rotation4D {
  return {
    left: quaternionMultiply(r1.left, r2.left),
    right: quaternionMultiply(r1.right, r2.right),
  };
}

/**
 * Create identity rotation (no change)
 */
export function identityRotation(): Rotation4D {
  return {
    left: identityQuaternion(),
    right: identityQuaternion(),
  };
}

/**
 * Inverse of a 4D rotation
 */
export function inverseRotation(r: Rotation4D): Rotation4D {
  return {
    left: quaternionInverse(r.left),
    right: quaternionInverse(r.right),
  };
}

// ============================================================================
// Batch Operations
// ============================================================================

/**
 * Rotate multiple points by the same rotation
 */
export function rotatePoints4D(points: Vector4D[], rotation: Rotation4D): Vector4D[] {
  return points.map(p => rotate4D(p, rotation));
}

/**
 * Animate rotation over multiple steps
 */
export function animateRotation(
  points: Vector4D[],
  targetRotation: Rotation4D,
  steps: number
): Vector4D[][] {
  const frames: Vector4D[][] = [points];
  const identity = identityRotation();

  for (let i = 1; i <= steps; i++) {
    const t = i / steps;
    const interpolated: Rotation4D = {
      left: quaternionSlerp(identity.left, targetRotation.left, t),
      right: quaternionSlerp(identity.right, targetRotation.right, t),
    };
    frames.push(rotatePoints4D(points, interpolated));
  }

  return frames;
}

// ============================================================================
// Special Rotations
// ============================================================================

/**
 * Create rotation that maps one point to another
 * (on the unit hypersphere)
 */
export function rotationBetweenPoints(from: Vector4D, to: Vector4D): Rotation4D {
  // Normalize both points
  const normFrom = Math.sqrt(from.w ** 2 + from.x ** 2 + from.y ** 2 + from.z ** 2);
  const normTo = Math.sqrt(to.w ** 2 + to.x ** 2 + to.y ** 2 + to.z ** 2);

  const fromN: Quaternion = {
    w: from.w / normFrom,
    x: from.x / normFrom,
    y: from.y / normFrom,
    z: from.z / normFrom,
  };

  const toN: Quaternion = {
    w: to.w / normTo,
    x: to.x / normTo,
    y: to.y / normTo,
    z: to.z / normTo,
  };

  // Rotation quaternion that takes fromN to toN
  // q = toN * fromN^(-1)
  const q = quaternionMultiply(toN, quaternionInverse(fromN));

  // Use as left rotation only (simple rotation)
  return {
    left: normalizeQuaternion(q),
    right: identityQuaternion(),
  };
}

/**
 * Rotation for musical key modulation
 * Maps one vertex of the 24-cell to another
 */
export function musicalModulationRotation(
  fromKeyIndex: number,
  toKeyIndex: number,
  totalKeys: number = 24
): Rotation4D {
  // Calculate angle based on key difference
  const keyDiff = (toKeyIndex - fromKeyIndex + totalKeys) % totalKeys;
  const angle = (keyDiff / totalKeys) * 2 * Math.PI;

  // Use WX plane for primary modulation
  return createIsoclinicRotation(angle, 'WX');
}

// ============================================================================
// Exports
// ============================================================================

export const Quaternion4DModule = {
  // Quaternion operations
  createQuaternion,
  identityQuaternion,
  quaternionNorm,
  normalizeQuaternion,
  quaternionConjugate,
  quaternionInverse,
  quaternionMultiply,
  quaternionFromAxisAngle,
  quaternionSlerp,

  // 4D rotation operations
  rotate4D,
  createIsoclinicRotation,
  createSimpleRotation,
  createDoubleRotation,
  composeRotations,
  identityRotation,
  inverseRotation,

  // Batch operations
  rotatePoints4D,
  animateRotation,

  // Special rotations
  rotationBetweenPoints,
  musicalModulationRotation,
};

export default Quaternion4DModule;
