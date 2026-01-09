// File: src/core/projection.ts
// Stereographic Projection Pipeline: 4D → 3D → 2D
// Provides visualization coordinates for the geometric cognition engine

import { Quaternion, type Lattice24 } from './geometry';

/**
 * 3D point representation
 */
export interface Point3D {
  x: number;
  y: number;
  z: number;
}

/**
 * 2D point representation for canvas rendering
 */
export interface Point2D {
  x: number;
  y: number;
}

/**
 * Projected vertex with metadata
 */
export interface ProjectedVertex {
  /** Original 4D quaternion */
  original: Quaternion;
  /** 3D stereographic projection */
  point3D: Point3D;
  /** 2D perspective projection for canvas */
  point2D: Point2D;
  /** Depth value for z-ordering */
  depth: number;
  /** Scale factor from projection (for point sizing) */
  scale: number;
  /** Index in the original lattice */
  index: number;
}

/**
 * Projected edge for rendering
 */
export interface ProjectedEdge {
  /** Start vertex */
  start: ProjectedVertex;
  /** End vertex */
  end: ProjectedVertex;
  /** Average depth for z-ordering */
  avgDepth: number;
  /** Original edge indices */
  indices: [number, number];
}

/**
 * Complete projection result for a frame
 */
export interface ProjectionResult {
  /** Projected vertices */
  vertices: ProjectedVertex[];
  /** Projected edges */
  edges: ProjectedEdge[];
  /** Current thought vector projection */
  thoughtVector: ProjectedVertex | null;
  /** Canvas dimensions used */
  canvasWidth: number;
  canvasHeight: number;
  /** Projection parameters */
  params: ProjectionParams;
}

/**
 * Projection parameters
 */
export interface ProjectionParams {
  /** Projection center w-coordinate */
  projectionW: number;
  /** Camera distance for 3D→2D */
  cameraDistance: number;
  /** Field of view multiplier */
  fov: number;
  /** Rotation angle for 3D view */
  rotationX: number;
  rotationY: number;
  rotationZ: number;
}

/**
 * Default projection parameters
 */
export const DEFAULT_PROJECTION_PARAMS: ProjectionParams = {
  projectionW: 2.5,
  cameraDistance: 4,
  fov: 300,
  rotationX: 0.4,
  rotationY: 0.3,
  rotationZ: 0,
};

/**
 * Stereographic projection from 4D to 3D
 *
 * Projects from S³ (3-sphere in 4D) to R³ using stereographic projection.
 * The projection point is at (0, 0, 0, projectionW).
 *
 * Formula: (x, y, z, w) → (x/(projectionW-w), y/(projectionW-w), z/(projectionW-w))
 *
 * @param q - 4D quaternion point
 * @param projectionW - w-coordinate of projection point
 */
export function stereographicProject4Dto3D(
  q: Quaternion,
  projectionW: number = DEFAULT_PROJECTION_PARAMS.projectionW
): Point3D {
  const denominator = projectionW - q.w;

  // Handle singularity (point at projection center)
  if (Math.abs(denominator) < 1e-6) {
    return { x: 0, y: 0, z: 1000 }; // Project to "infinity"
  }

  return {
    x: q.x / denominator,
    y: q.y / denominator,
    z: q.z / denominator,
  };
}

/**
 * Apply 3D rotation matrix
 */
function rotate3D(
  point: Point3D,
  rotX: number,
  rotY: number,
  rotZ: number
): Point3D {
  let { x, y, z } = point;

  // Rotate around X axis
  const cosX = Math.cos(rotX);
  const sinX = Math.sin(rotX);
  const y1 = y * cosX - z * sinX;
  const z1 = y * sinX + z * cosX;
  y = y1;
  z = z1;

  // Rotate around Y axis
  const cosY = Math.cos(rotY);
  const sinY = Math.sin(rotY);
  const x2 = x * cosY + z * sinY;
  const z2 = -x * sinY + z * cosY;
  x = x2;
  z = z2;

  // Rotate around Z axis
  const cosZ = Math.cos(rotZ);
  const sinZ = Math.sin(rotZ);
  const x3 = x * cosZ - y * sinZ;
  const y3 = x * sinZ + y * cosZ;

  return { x: x3, y: y3, z: z2 };
}

/**
 * Perspective projection from 3D to 2D
 *
 * @param point3D - 3D point
 * @param params - Projection parameters
 * @param canvasWidth - Canvas width in pixels
 * @param canvasHeight - Canvas height in pixels
 */
export function perspectiveProject3Dto2D(
  point3D: Point3D,
  params: ProjectionParams,
  canvasWidth: number,
  canvasHeight: number
): { point2D: Point2D; scale: number; depth: number } {
  // Apply 3D rotation
  const rotated = rotate3D(
    point3D,
    params.rotationX,
    params.rotationY,
    params.rotationZ
  );

  // Perspective division
  const zOffset = rotated.z + params.cameraDistance;
  const scale = params.fov / Math.max(zOffset, 0.1);

  // Project to 2D canvas coordinates (centered)
  const centerX = canvasWidth / 2;
  const centerY = canvasHeight / 2;

  return {
    point2D: {
      x: centerX + rotated.x * scale,
      y: centerY - rotated.y * scale, // Flip Y for canvas coordinates
    },
    scale: Math.max(0.1, scale / params.fov),
    depth: rotated.z,
  };
}

/**
 * Complete 4D → 2D projection pipeline
 */
export function projectQuaternionTo2D(
  q: Quaternion,
  params: ProjectionParams,
  canvasWidth: number,
  canvasHeight: number,
  index: number
): ProjectedVertex {
  const point3D = stereographicProject4Dto3D(q, params.projectionW);
  const { point2D, scale, depth } = perspectiveProject3Dto2D(
    point3D,
    params,
    canvasWidth,
    canvasHeight
  );

  return {
    original: q,
    point3D,
    point2D,
    depth,
    scale,
    index,
  };
}

/**
 * Project an entire 24-cell lattice with the current thought vector
 */
export function projectLattice(
  lattice: Lattice24,
  thoughtVector: Quaternion | null,
  canvasWidth: number,
  canvasHeight: number,
  params: ProjectionParams = DEFAULT_PROJECTION_PARAMS
): ProjectionResult {
  // Project all vertices
  const vertices: ProjectedVertex[] = lattice.vertices.map((v, i) =>
    projectQuaternionTo2D(v, params, canvasWidth, canvasHeight, i)
  );

  // Project edges
  const edges: ProjectedEdge[] = lattice.edges.map(([i, j]) => {
    const start = vertices[i];
    const end = vertices[j];
    return {
      start,
      end,
      avgDepth: (start.depth + end.depth) / 2,
      indices: [i, j],
    };
  });

  // Sort edges by depth (back to front)
  edges.sort((a, b) => a.avgDepth - b.avgDepth);

  // Project thought vector if present
  let projectedThoughtVector: ProjectedVertex | null = null;
  if (thoughtVector) {
    projectedThoughtVector = projectQuaternionTo2D(
      thoughtVector,
      params,
      canvasWidth,
      canvasHeight,
      -1
    );
  }

  return {
    vertices,
    edges,
    thoughtVector: projectedThoughtVector,
    canvasWidth,
    canvasHeight,
    params,
  };
}

/**
 * Projection parameters animator for smooth camera motion
 */
export class ProjectionAnimator {
  private _params: ProjectionParams;
  private _targetParams: ProjectionParams;
  private _animationSpeed: number;

  constructor(
    initialParams: ProjectionParams = DEFAULT_PROJECTION_PARAMS,
    animationSpeed = 0.05
  ) {
    this._params = { ...initialParams };
    this._targetParams = { ...initialParams };
    this._animationSpeed = animationSpeed;
  }

  get params(): ProjectionParams {
    return { ...this._params };
  }

  setTarget(target: Partial<ProjectionParams>): void {
    this._targetParams = { ...this._targetParams, ...target };
  }

  update(): boolean {
    let changed = false;
    const speed = this._animationSpeed;

    const lerp = (current: number, target: number): number => {
      const diff = target - current;
      if (Math.abs(diff) < 0.0001) return target;
      changed = true;
      return current + diff * speed;
    };

    this._params = {
      projectionW: lerp(this._params.projectionW, this._targetParams.projectionW),
      cameraDistance: lerp(
        this._params.cameraDistance,
        this._targetParams.cameraDistance
      ),
      fov: lerp(this._params.fov, this._targetParams.fov),
      rotationX: lerp(this._params.rotationX, this._targetParams.rotationX),
      rotationY: lerp(this._params.rotationY, this._targetParams.rotationY),
      rotationZ: lerp(this._params.rotationZ, this._targetParams.rotationZ),
    };

    return changed;
  }

  /**
   * Auto-rotate the view
   */
  autoRotate(deltaTime: number, speed = 0.2): void {
    this._params.rotationY += speed * deltaTime;
    this._targetParams.rotationY = this._params.rotationY;
  }

  reset(): void {
    this._params = { ...DEFAULT_PROJECTION_PARAMS };
    this._targetParams = { ...DEFAULT_PROJECTION_PARAMS };
  }
}

/**
 * State for projection - for TRACE logging
 */
export interface ProjectionState {
  params: ProjectionParams;
  thoughtVector2D: Point2D | null;
  thoughtVector3D: Point3D | null;
}

/**
 * Get projection state for logging
 */
export function getProjectionState(
  result: ProjectionResult
): ProjectionState {
  return {
    params: result.params,
    thoughtVector2D: result.thoughtVector?.point2D ?? null,
    thoughtVector3D: result.thoughtVector?.point3D ?? null,
  };
}
