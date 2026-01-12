/**
 * Projection Pipeline: 4D → 3D → 2D
 *
 * Transforms high-dimensional polytope data into visualizable 2D coordinates.
 *
 * PROJECTION METHODS:
 * - Orthographic: Simple dimensional drop (preserves parallel lines)
 * - Stereographic: Angle-preserving projection from pole (conformal)
 * - Perspective: Distance-based scaling (depth perception)
 *
 * MATHEMATICAL BASIS:
 * 4D → 3D stereographic:
 *   Given point P = (w, x, y, z) on unit 4-sphere,
 *   project from north pole (1, 0, 0, 0) to hyperplane w = 0:
 *   (x', y', z') = (x, y, z) / (1 - w)
 *
 * 3D → 2D perspective:
 *   Given point P = (x, y, z) and camera at distance d:
 *   (x', y') = (x * d / z, y * d / z)
 *
 * COGNITIVE SIGNIFICANCE:
 * - Multiple simultaneous projections reveal different aspects of the same structure
 * - Animation through rotation reveals 4D structure through parallax
 * - Stereographic projection preserves circular structures (important for musical keys)
 *
 * REFERENCES:
 * - https://en.wikipedia.org/wiki/Stereographic_projection
 * - https://en.wikipedia.org/wiki/4-polytope#Visualization
 */

import type { Vector4D } from '../music/music-geometry-domain';
import type { Rotation4D } from './quaternion4d';
import { rotate4D, identityRotation } from './quaternion4d';

// ============================================================================
// Types
// ============================================================================

export interface Vector3D {
  x: number;
  y: number;
  z: number;
}

export interface Vector2D {
  x: number;
  y: number;
}

export type ProjectionMethod4D = 'orthographic' | 'stereographic' | 'perspective';
export type ProjectionMethod3D = 'orthographic' | 'perspective';

export interface Camera3D {
  position: Vector3D;
  target: Vector3D;
  up: Vector3D;
  fov: number;       // Field of view in radians
  near: number;      // Near clipping plane
  far: number;       // Far clipping plane
}

export interface ProjectionOptions4D {
  method: ProjectionMethod4D;
  rotation?: Rotation4D;        // Pre-rotation before projection
  perspectiveDistance?: number; // For perspective projection
  stereoPole?: 'positive' | 'negative'; // Which pole to project from
}

export interface ProjectionOptions3D {
  method: ProjectionMethod3D;
  camera?: Camera3D;
  scale?: number;
}

export interface ProjectedPoint {
  original4D: Vector4D;
  projected3D: Vector3D;
  projected2D: Vector2D;
  depth: number;  // For sorting/transparency
}

// ============================================================================
// 4D → 3D Projections
// ============================================================================

/**
 * Orthographic projection: drop W coordinate
 */
function orthographic4DTo3D(point: Vector4D): Vector3D {
  return {
    x: point.x,
    y: point.y,
    z: point.z,
  };
}

/**
 * Stereographic projection: angle-preserving
 * Projects from pole (±1, 0, 0, 0) to hyperplane W = 0
 */
function stereographic4DTo3D(point: Vector4D, pole: 'positive' | 'negative' = 'positive'): Vector3D {
  const poleW = pole === 'positive' ? 1 : -1;
  const denom = 1 - poleW * point.w;

  // Avoid division by zero at the pole
  if (Math.abs(denom) < 0.0001) {
    // Point is near the pole - project to infinity
    const scale = 1000;
    return {
      x: point.x * scale,
      y: point.y * scale,
      z: point.z * scale,
    };
  }

  return {
    x: point.x / denom,
    y: point.y / denom,
    z: point.z / denom,
  };
}

/**
 * Perspective projection: distance-based scaling in W
 */
function perspective4DTo3D(point: Vector4D, distance: number = 3): Vector3D {
  const scale = distance / (distance - point.w);

  return {
    x: point.x * scale,
    y: point.y * scale,
    z: point.z * scale,
  };
}

/**
 * Project a 4D point to 3D using specified method
 */
export function project4DTo3D(
  point: Vector4D,
  options: ProjectionOptions4D = { method: 'orthographic' }
): Vector3D {
  // Apply optional rotation first
  let p = point;
  if (options.rotation) {
    p = rotate4D(point, options.rotation);
  }

  switch (options.method) {
    case 'orthographic':
      return orthographic4DTo3D(p);

    case 'stereographic':
      return stereographic4DTo3D(p, options.stereoPole || 'positive');

    case 'perspective':
      return perspective4DTo3D(p, options.perspectiveDistance || 3);

    default:
      return orthographic4DTo3D(p);
  }
}

/**
 * Project multiple 4D points to 3D
 */
export function project4DTo3DBatch(
  points: Vector4D[],
  options: ProjectionOptions4D = { method: 'orthographic' }
): Vector3D[] {
  return points.map(p => project4DTo3D(p, options));
}

// ============================================================================
// 3D → 2D Projections
// ============================================================================

/**
 * Default camera looking down Z axis
 */
export function defaultCamera(): Camera3D {
  return {
    position: { x: 0, y: 0, z: 5 },
    target: { x: 0, y: 0, z: 0 },
    up: { x: 0, y: 1, z: 0 },
    fov: Math.PI / 4,
    near: 0.1,
    far: 100,
  };
}

/**
 * Orthographic 3D → 2D: drop Z coordinate
 */
function orthographic3DTo2D(point: Vector3D, scale: number = 1): Vector2D {
  return {
    x: point.x * scale,
    y: point.y * scale,
  };
}

/**
 * Perspective 3D → 2D with camera
 */
function perspective3DTo2D(point: Vector3D, camera: Camera3D): Vector2D {
  // Compute view direction
  const viewDir = normalize3D(subtract3D(camera.target, camera.position));
  const right = normalize3D(cross3D(viewDir, camera.up));
  const up = cross3D(right, viewDir);

  // Transform point to camera space
  const relative = subtract3D(point, camera.position);
  const cameraSpace: Vector3D = {
    x: dot3D(relative, right),
    y: dot3D(relative, up),
    z: dot3D(relative, viewDir),
  };

  // Perspective divide
  if (cameraSpace.z <= camera.near) {
    // Behind camera or too close
    return { x: 0, y: 0 };
  }

  const scale = 1 / Math.tan(camera.fov / 2);
  return {
    x: (cameraSpace.x / cameraSpace.z) * scale,
    y: (cameraSpace.y / cameraSpace.z) * scale,
  };
}

/**
 * Project a 3D point to 2D using specified method
 */
export function project3DTo2D(
  point: Vector3D,
  options: ProjectionOptions3D = { method: 'orthographic' }
): Vector2D {
  switch (options.method) {
    case 'orthographic':
      return orthographic3DTo2D(point, options.scale || 1);

    case 'perspective':
      return perspective3DTo2D(point, options.camera || defaultCamera());

    default:
      return orthographic3DTo2D(point, options.scale || 1);
  }
}

/**
 * Project multiple 3D points to 2D
 */
export function project3DTo2DBatch(
  points: Vector3D[],
  options: ProjectionOptions3D = { method: 'orthographic' }
): Vector2D[] {
  return points.map(p => project3DTo2D(p, options));
}

// ============================================================================
// Full Pipeline: 4D → 3D → 2D
// ============================================================================

/**
 * Project 4D point all the way to 2D with depth information
 */
export function projectFull(
  point: Vector4D,
  options4D: ProjectionOptions4D = { method: 'orthographic' },
  options3D: ProjectionOptions3D = { method: 'orthographic' }
): ProjectedPoint {
  const projected3D = project4DTo3D(point, options4D);
  const projected2D = project3DTo2D(projected3D, options3D);

  // Compute depth for sorting/transparency
  // Use both W and Z for depth
  const depth = point.w + projected3D.z * 0.5;

  return {
    original4D: point,
    projected3D,
    projected2D,
    depth,
  };
}

/**
 * Project multiple 4D points through full pipeline
 */
export function projectFullBatch(
  points: Vector4D[],
  options4D: ProjectionOptions4D = { method: 'orthographic' },
  options3D: ProjectionOptions3D = { method: 'orthographic' }
): ProjectedPoint[] {
  return points.map(p => projectFull(p, options4D, options3D));
}

/**
 * Project and sort by depth (back-to-front for correct rendering)
 */
export function projectFullSorted(
  points: Vector4D[],
  options4D: ProjectionOptions4D = { method: 'orthographic' },
  options3D: ProjectionOptions3D = { method: 'orthographic' }
): ProjectedPoint[] {
  const projected = projectFullBatch(points, options4D, options3D);
  return projected.sort((a, b) => a.depth - b.depth);
}

// ============================================================================
// Multi-View Projection
// ============================================================================

export interface MultiViewResult {
  front: Vector2D[];
  side: Vector2D[];
  top: Vector2D[];
  perspective: Vector2D[];
}

/**
 * Create multiple simultaneous projections from different viewpoints
 */
export function multiViewProject(points: Vector4D[]): MultiViewResult {
  const options3D: ProjectionOptions3D = { method: 'orthographic', scale: 1 };

  // Front view: XY plane (W,Z dropped to get to 3D, then Z dropped)
  const front = project4DTo3DBatch(points, { method: 'orthographic' })
    .map(p => ({ x: p.x, y: p.y }));

  // Side view: ZY plane (rotate 90° around Y axis first)
  const side = project4DTo3DBatch(points, { method: 'orthographic' })
    .map(p => ({ x: p.z, y: p.y }));

  // Top view: XZ plane (rotate 90° around X axis first)
  const top = project4DTo3DBatch(points, { method: 'orthographic' })
    .map(p => ({ x: p.x, y: p.z }));

  // Perspective view
  const perspCamera: Camera3D = {
    position: { x: 3, y: 2, z: 4 },
    target: { x: 0, y: 0, z: 0 },
    up: { x: 0, y: 1, z: 0 },
    fov: Math.PI / 4,
    near: 0.1,
    far: 100,
  };

  const perspective = project4DTo3DBatch(points, {
    method: 'perspective',
    perspectiveDistance: 3,
  }).map(p => project3DTo2D(p, { method: 'perspective', camera: perspCamera }));

  return { front, side, top, perspective };
}

// ============================================================================
// Animation Support
// ============================================================================

export interface AnimationFrame {
  points2D: Vector2D[];
  points3D: Vector3D[];
  rotation: Rotation4D;
  time: number;
}

/**
 * Generate animation frames for a rotation sequence
 */
export function animateProjection(
  points: Vector4D[],
  targetRotation: Rotation4D,
  frames: number,
  options4D: Omit<ProjectionOptions4D, 'rotation'> = { method: 'stereographic' },
  options3D: ProjectionOptions3D = { method: 'perspective', camera: defaultCamera() }
): AnimationFrame[] {
  const { quaternionSlerp, identityQuaternion } = require('./quaternion4d');
  const identity = identityRotation();
  const result: AnimationFrame[] = [];

  for (let i = 0; i <= frames; i++) {
    const t = i / frames;
    const rotation: Rotation4D = {
      left: quaternionSlerp(identity.left, targetRotation.left, t),
      right: quaternionSlerp(identity.right, targetRotation.right, t),
    };

    const points3D = project4DTo3DBatch(points, { ...options4D, rotation });
    const points2D = project3DTo2DBatch(points3D, options3D);

    result.push({
      points2D,
      points3D,
      rotation,
      time: t,
    });
  }

  return result;
}

// ============================================================================
// Viewport Transformation
// ============================================================================

export interface Viewport {
  x: number;       // Left edge
  y: number;       // Top edge
  width: number;
  height: number;
}

/**
 * Transform normalized coordinates to viewport coordinates
 */
export function toViewport(point: Vector2D, viewport: Viewport): Vector2D {
  return {
    x: viewport.x + (point.x + 1) * viewport.width / 2,
    y: viewport.y + (1 - point.y) * viewport.height / 2,  // Flip Y for screen coords
  };
}

/**
 * Transform batch of points to viewport coordinates
 */
export function toViewportBatch(points: Vector2D[], viewport: Viewport): Vector2D[] {
  return points.map(p => toViewport(p, viewport));
}

// ============================================================================
// 3D Vector Utilities
// ============================================================================

function subtract3D(a: Vector3D, b: Vector3D): Vector3D {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

function dot3D(a: Vector3D, b: Vector3D): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

function cross3D(a: Vector3D, b: Vector3D): Vector3D {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}

function normalize3D(v: Vector3D): Vector3D {
  const len = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  if (len === 0) return { x: 0, y: 0, z: 1 };
  return { x: v.x / len, y: v.y / len, z: v.z / len };
}

export function distance3D(a: Vector3D, b: Vector3D): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

export function distance2D(a: Vector2D, b: Vector2D): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

// ============================================================================
// ProjectionPipeline Class
// ============================================================================

export class ProjectionPipeline {
  private options4D: ProjectionOptions4D;
  private options3D: ProjectionOptions3D;
  private viewport: Viewport;

  constructor(
    options4D: ProjectionOptions4D = { method: 'stereographic' },
    options3D: ProjectionOptions3D = { method: 'perspective', camera: defaultCamera() },
    viewport: Viewport = { x: 0, y: 0, width: 800, height: 600 }
  ) {
    this.options4D = options4D;
    this.options3D = options3D;
    this.viewport = viewport;
  }

  /**
   * Set 4D projection options
   */
  set4DOptions(options: Partial<ProjectionOptions4D>): void {
    this.options4D = { ...this.options4D, ...options };
  }

  /**
   * Set 3D projection options
   */
  set3DOptions(options: Partial<ProjectionOptions3D>): void {
    this.options3D = { ...this.options3D, ...options };
  }

  /**
   * Set viewport
   */
  setViewport(viewport: Viewport): void {
    this.viewport = viewport;
  }

  /**
   * Set 4D rotation
   */
  setRotation(rotation: Rotation4D): void {
    this.options4D.rotation = rotation;
  }

  /**
   * Project single point to viewport coordinates
   */
  project(point: Vector4D): ProjectedPoint & { viewport2D: Vector2D } {
    const result = projectFull(point, this.options4D, this.options3D);
    return {
      ...result,
      viewport2D: toViewport(result.projected2D, this.viewport),
    };
  }

  /**
   * Project multiple points to viewport coordinates
   */
  projectBatch(points: Vector4D[]): (ProjectedPoint & { viewport2D: Vector2D })[] {
    return points.map(p => this.project(p));
  }

  /**
   * Project and sort by depth
   */
  projectSorted(points: Vector4D[]): (ProjectedPoint & { viewport2D: Vector2D })[] {
    const projected = this.projectBatch(points);
    return projected.sort((a, b) => a.depth - b.depth);
  }

  /**
   * Project edges (pairs of vertex indices)
   */
  projectEdges(
    vertices: Vector4D[],
    edges: [number, number][]
  ): { start: Vector2D; end: Vector2D; depth: number }[] {
    const projected = this.projectBatch(vertices);

    return edges.map(([i, j]) => ({
      start: projected[i].viewport2D,
      end: projected[j].viewport2D,
      depth: (projected[i].depth + projected[j].depth) / 2,
    }));
  }

  /**
   * Get current 4D options
   */
  get4DOptions(): ProjectionOptions4D {
    return { ...this.options4D };
  }

  /**
   * Get current 3D options
   */
  get3DOptions(): ProjectionOptions3D {
    return { ...this.options3D };
  }

  /**
   * Get current viewport
   */
  getViewport(): Viewport {
    return { ...this.viewport };
  }
}

// ============================================================================
// Exports
// ============================================================================

export const ProjectionModule = {
  // 4D → 3D
  project4DTo3D,
  project4DTo3DBatch,

  // 3D → 2D
  project3DTo2D,
  project3DTo2DBatch,
  defaultCamera,

  // Full pipeline
  projectFull,
  projectFullBatch,
  projectFullSorted,

  // Multi-view
  multiViewProject,

  // Animation
  animateProjection,

  // Viewport
  toViewport,
  toViewportBatch,

  // Utilities
  distance3D,
  distance2D,

  // Class
  ProjectionPipeline,
};

export default ProjectionModule;
