// File: src/core/trajectory.ts
// Trajectory History - Tracks the cognitive path through 4D space

import { Quaternion, type ConvexityStatus } from './geometry';

/**
 * A single point in the trajectory history
 */
export interface TrajectoryPoint {
  /** Position in 4D space */
  position: Quaternion;
  /** Step index when this point was recorded */
  step: number;
  /** Timestamp */
  timestamp: number;
  /** Convexity status at this point */
  status: ConvexityStatus;
  /** Distance from origin */
  distanceFromOrigin: number;
  /** Whether entropy was injected at this step */
  entropyInjected: boolean;
}

/**
 * Trajectory statistics
 */
export interface TrajectoryStats {
  /** Total points in history */
  totalPoints: number;
  /** Total path length (sum of distances between consecutive points) */
  pathLength: number;
  /** Average speed (distance per step) */
  averageSpeed: number;
  /** Maximum distance from origin reached */
  maxDistance: number;
  /** Minimum distance from origin */
  minDistance: number;
  /** Time in each status */
  statusTime: Record<ConvexityStatus, number>;
  /** Number of boundary crossings */
  boundaryCrossings: number;
  /** Curvature estimate (average angular change) */
  averageCurvature: number;
}

/**
 * TrajectoryHistory - Maintains a record of the cognitive path
 */
export class TrajectoryHistory {
  private _points: TrajectoryPoint[] = [];
  private _maxPoints: number;
  private _pathLength = 0;
  private _boundaryCrossings = 0;

  constructor(maxPoints = 500) {
    this._maxPoints = maxPoints;
  }

  /**
   * Get all points in the trajectory
   */
  get points(): readonly TrajectoryPoint[] {
    return this._points;
  }

  /**
   * Get the number of points
   */
  get length(): number {
    return this._points.length;
  }

  /**
   * Get the maximum number of points to retain
   */
  get maxPoints(): number {
    return this._maxPoints;
  }

  /**
   * Set max points (will trim if needed)
   */
  set maxPoints(value: number) {
    this._maxPoints = Math.max(10, value);
    this.trim();
  }

  /**
   * Add a new point to the trajectory
   */
  addPoint(
    position: Quaternion,
    step: number,
    status: ConvexityStatus,
    entropyInjected = false
  ): void {
    const distanceFromOrigin = position.norm();

    // Calculate path length increment
    if (this._points.length > 0) {
      const lastPoint = this._points[this._points.length - 1];
      const distance = position.distanceTo(lastPoint.position);
      this._pathLength += distance;

      // Detect boundary crossing
      if (lastPoint.status !== status) {
        if (
          (lastPoint.status === 'SAFE' && status !== 'SAFE') ||
          (lastPoint.status !== 'SAFE' && status === 'SAFE')
        ) {
          this._boundaryCrossings++;
        }
      }
    }

    const point: TrajectoryPoint = {
      position,
      step,
      timestamp: Date.now(),
      status,
      distanceFromOrigin,
      entropyInjected,
    };

    this._points.push(point);
    this.trim();
  }

  /**
   * Trim the history to max points
   */
  private trim(): void {
    if (this._points.length > this._maxPoints) {
      const toRemove = this._points.length - this._maxPoints;
      this._points.splice(0, toRemove);
    }
  }

  /**
   * Get recent points (for visualization)
   */
  getRecent(count: number): TrajectoryPoint[] {
    const start = Math.max(0, this._points.length - count);
    return this._points.slice(start);
  }

  /**
   * Get trajectory statistics
   */
  getStats(): TrajectoryStats {
    const statusTime: Record<ConvexityStatus, number> = {
      SAFE: 0,
      WARNING: 0,
      VIOLATION: 0,
    };

    let maxDistance = 0;
    let minDistance = Infinity;
    let totalCurvature = 0;
    let curvatureCount = 0;

    for (let i = 0; i < this._points.length; i++) {
      const point = this._points[i];
      statusTime[point.status]++;

      if (point.distanceFromOrigin > maxDistance) {
        maxDistance = point.distanceFromOrigin;
      }
      if (point.distanceFromOrigin < minDistance) {
        minDistance = point.distanceFromOrigin;
      }

      // Calculate curvature (requires 3 points)
      if (i >= 2) {
        const p0 = this._points[i - 2].position;
        const p1 = this._points[i - 1].position;
        const p2 = point.position;

        const v1 = p1.subtract(p0);
        const v2 = p2.subtract(p1);

        const dot = v1.dot(v2);
        const mag1 = v1.norm();
        const mag2 = v2.norm();

        if (mag1 > 1e-10 && mag2 > 1e-10) {
          const cosAngle = Math.max(-1, Math.min(1, dot / (mag1 * mag2)));
          const angle = Math.acos(cosAngle);
          totalCurvature += angle;
          curvatureCount++;
        }
      }
    }

    const totalPoints = this._points.length;
    const averageSpeed = totalPoints > 1 ? this._pathLength / (totalPoints - 1) : 0;
    const averageCurvature = curvatureCount > 0 ? totalCurvature / curvatureCount : 0;

    return {
      totalPoints,
      pathLength: this._pathLength,
      averageSpeed,
      maxDistance,
      minDistance: minDistance === Infinity ? 0 : minDistance,
      statusTime,
      boundaryCrossings: this._boundaryCrossings,
      averageCurvature,
    };
  }

  /**
   * Get the center of mass of the trajectory
   */
  getCenterOfMass(): Quaternion {
    if (this._points.length === 0) {
      return new Quaternion(0, 0, 0, 0);
    }

    let sumW = 0, sumX = 0, sumY = 0, sumZ = 0;
    for (const point of this._points) {
      sumW += point.position.w;
      sumX += point.position.x;
      sumY += point.position.y;
      sumZ += point.position.z;
    }

    const n = this._points.length;
    return new Quaternion(sumW / n, sumX / n, sumY / n, sumZ / n);
  }

  /**
   * Get the "spread" of the trajectory (standard deviation)
   */
  getSpread(): number {
    if (this._points.length < 2) {
      return 0;
    }

    const center = this.getCenterOfMass();
    let sumSqDist = 0;

    for (const point of this._points) {
      const dist = point.position.distanceTo(center);
      sumSqDist += dist * dist;
    }

    return Math.sqrt(sumSqDist / this._points.length);
  }

  /**
   * Check if trajectory is oscillating (going back and forth)
   */
  isOscillating(windowSize = 20): boolean {
    if (this._points.length < windowSize * 2) {
      return false;
    }

    const recent = this.getRecent(windowSize * 2);
    let directionChanges = 0;

    for (let i = 2; i < recent.length; i++) {
      const p0 = recent[i - 2].position;
      const p1 = recent[i - 1].position;
      const p2 = recent[i].position;

      const v1 = p1.subtract(p0);
      const v2 = p2.subtract(p1);
      const dot = v1.dot(v2);

      if (dot < 0) {
        directionChanges++;
      }
    }

    // If more than 40% of moves are direction changes, it's oscillating
    return directionChanges / (recent.length - 2) > 0.4;
  }

  /**
   * Clear the trajectory history
   */
  clear(): void {
    this._points = [];
    this._pathLength = 0;
    this._boundaryCrossings = 0;
  }

  /**
   * Export trajectory data
   */
  export(): TrajectoryExport {
    return {
      points: this._points.map((p) => ({
        w: p.position.w,
        x: p.position.x,
        y: p.position.y,
        z: p.position.z,
        step: p.step,
        timestamp: p.timestamp,
        status: p.status,
        entropyInjected: p.entropyInjected,
      })),
      stats: this.getStats(),
      exportedAt: new Date().toISOString(),
    };
  }
}

/**
 * Export format for trajectory data
 */
export interface TrajectoryExport {
  points: Array<{
    w: number;
    x: number;
    y: number;
    z: number;
    step: number;
    timestamp: number;
    status: ConvexityStatus;
    entropyInjected: boolean;
  }>;
  stats: TrajectoryStats;
  exportedAt: string;
}
