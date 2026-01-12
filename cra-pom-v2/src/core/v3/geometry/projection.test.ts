/**
 * Projection Pipeline Tests
 *
 * Verifies 4D → 3D → 2D projection transformations:
 * - Orthographic projection (dimension dropping)
 * - Stereographic projection (conformal)
 * - Perspective projection (depth perception)
 * - Multi-view projections
 * - Animation support
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  project4DTo3D,
  project4DTo3DBatch,
  project3DTo2D,
  project3DTo2DBatch,
  projectFull,
  projectFullBatch,
  projectFullSorted,
  multiViewProject,
  toViewport,
  toViewportBatch,
  distance3D,
  distance2D,
  defaultCamera,
  ProjectionPipeline,
  type Vector3D,
  type Vector2D,
} from './projection';
import type { Vector4D } from '../music/music-geometry-domain';
import { createIsoclinicRotation } from './quaternion4d';

describe('4D → 3D Projections', () => {
  const testPoint: Vector4D = { w: 0.5, x: 1, y: 2, z: 3 };
  const originPoint: Vector4D = { w: 0, x: 0, y: 0, z: 0 };
  const unitWPoint: Vector4D = { w: 1, x: 0, y: 0, z: 0 };

  describe('Orthographic Projection', () => {
    it('should drop W coordinate', () => {
      const result = project4DTo3D(testPoint, { method: 'orthographic' });

      expect(result.x).toBe(testPoint.x);
      expect(result.y).toBe(testPoint.y);
      expect(result.z).toBe(testPoint.z);
    });

    it('should preserve origin', () => {
      const result = project4DTo3D(originPoint, { method: 'orthographic' });

      expect(result.x).toBe(0);
      expect(result.y).toBe(0);
      expect(result.z).toBe(0);
    });

    it('should project batch correctly', () => {
      const points: Vector4D[] = [
        { w: 0, x: 1, y: 0, z: 0 },
        { w: 0.5, x: 0, y: 1, z: 0 },
        { w: 1, x: 0, y: 0, z: 1 },
      ];

      const results = project4DTo3DBatch(points, { method: 'orthographic' });

      expect(results.length).toBe(3);
      expect(results[0].x).toBe(1);
      expect(results[1].y).toBe(1);
      expect(results[2].z).toBe(1);
    });
  });

  describe('Stereographic Projection', () => {
    it('should preserve origin', () => {
      const result = project4DTo3D(originPoint, { method: 'stereographic' });

      expect(result.x).toBe(0);
      expect(result.y).toBe(0);
      expect(result.z).toBe(0);
    });

    it('should project opposite pole to origin', () => {
      const southPole: Vector4D = { w: -1, x: 0, y: 0, z: 0 };
      const result = project4DTo3D(southPole, { method: 'stereographic', stereoPole: 'positive' });

      // South pole should project near origin
      expect(result.x).toBeCloseTo(0, 5);
      expect(result.y).toBeCloseTo(0, 5);
      expect(result.z).toBeCloseTo(0, 5);
    });

    it('should expand points far from projection pole', () => {
      const farPoint: Vector4D = { w: 0.9, x: 0.1, y: 0, z: 0 };
      const result = project4DTo3D(farPoint, { method: 'stereographic', stereoPole: 'positive' });

      // Points near pole should be expanded
      expect(Math.abs(result.x)).toBeGreaterThan(0.1);
    });

    it('should handle projection from negative pole', () => {
      const point: Vector4D = { w: 0.5, x: 0.5, y: 0, z: 0 };
      const resultPos = project4DTo3D(point, { method: 'stereographic', stereoPole: 'positive' });
      const resultNeg = project4DTo3D(point, { method: 'stereographic', stereoPole: 'negative' });

      // Different poles should give different results
      expect(resultPos.x).not.toBeCloseTo(resultNeg.x, 2);
    });
  });

  describe('Perspective Projection', () => {
    it('should scale based on W distance', () => {
      // Points closer to the view plane (higher W) get more scaling
      const closeToPlane: Vector4D = { w: 2, x: 1, y: 0, z: 0 };
      const farFromPlane: Vector4D = { w: 0, x: 1, y: 0, z: 0 };

      const closeResult = project4DTo3D(closeToPlane, { method: 'perspective', perspectiveDistance: 3 });
      const farResult = project4DTo3D(farFromPlane, { method: 'perspective', perspectiveDistance: 3 });

      // Points closer to view plane (higher W) should appear larger (more scaling)
      expect(Math.abs(closeResult.x)).toBeGreaterThan(Math.abs(farResult.x));
    });

    it('should preserve origin', () => {
      const result = project4DTo3D(originPoint, { method: 'perspective' });

      expect(result.x).toBe(0);
      expect(result.y).toBe(0);
      expect(result.z).toBe(0);
    });

    it('should scale with perspective distance', () => {
      const point: Vector4D = { w: 1, x: 1, y: 0, z: 0 };

      const closeCamera = project4DTo3D(point, { method: 'perspective', perspectiveDistance: 2 });
      const farCamera = project4DTo3D(point, { method: 'perspective', perspectiveDistance: 5 });

      // Different distances should give different scales
      expect(Math.abs(closeCamera.x)).not.toBeCloseTo(Math.abs(farCamera.x), 2);
    });
  });

  describe('With Rotation', () => {
    it('should apply rotation before projection', () => {
      // Use a point with non-zero X so rotation effect is visible in 3D projection
      const point: Vector4D = { w: 0.5, x: 0.5, y: 0.5, z: 0 };
      const rotation = createIsoclinicRotation(Math.PI / 2, 'XY');

      const withoutRotation = project4DTo3D(point, { method: 'orthographic' });
      const withRotation = project4DTo3D(point, { method: 'orthographic', rotation });

      // Rotation in XY plane should swap/change X and Y components
      // Results should differ after rotation
      expect(withoutRotation.x).toBe(0.5);
      expect(withoutRotation.y).toBe(0.5);
      // After XY rotation by 90°, coordinates should be different
      const changed = Math.abs(withRotation.x - 0.5) > 0.01 ||
                      Math.abs(withRotation.y - 0.5) > 0.01;
      expect(changed).toBe(true);
    });
  });
});

describe('3D → 2D Projections', () => {
  const testPoint3D: Vector3D = { x: 1, y: 2, z: 3 };
  const originPoint3D: Vector3D = { x: 0, y: 0, z: 0 };

  describe('Orthographic Projection', () => {
    it('should drop Z coordinate', () => {
      const result = project3DTo2D(testPoint3D, { method: 'orthographic' });

      expect(result.x).toBe(testPoint3D.x);
      expect(result.y).toBe(testPoint3D.y);
    });

    it('should apply scale', () => {
      const result = project3DTo2D(testPoint3D, { method: 'orthographic', scale: 2 });

      expect(result.x).toBe(testPoint3D.x * 2);
      expect(result.y).toBe(testPoint3D.y * 2);
    });

    it('should project batch correctly', () => {
      const points: Vector3D[] = [
        { x: 1, y: 0, z: 0 },
        { x: 0, y: 1, z: 5 },
        { x: 2, y: 3, z: -1 },
      ];

      const results = project3DTo2DBatch(points, { method: 'orthographic' });

      expect(results.length).toBe(3);
      expect(results[0].x).toBe(1);
      expect(results[1].y).toBe(1);
    });
  });

  describe('Perspective Projection', () => {
    it('should project with default camera', () => {
      const result = project3DTo2D(testPoint3D, {
        method: 'perspective',
        camera: defaultCamera(),
      });

      expect(typeof result.x).toBe('number');
      expect(typeof result.y).toBe('number');
      expect(Number.isFinite(result.x)).toBe(true);
    });

    it('should handle points at origin', () => {
      const camera = defaultCamera();
      const result = project3DTo2D(originPoint3D, { method: 'perspective', camera });

      expect(Number.isFinite(result.x)).toBe(true);
      expect(Number.isFinite(result.y)).toBe(true);
    });

    it('should return zero for points behind camera', () => {
      const behindPoint: Vector3D = { x: 0, y: 0, z: 10 }; // Camera at z=5
      const camera = defaultCamera();
      const result = project3DTo2D(behindPoint, { method: 'perspective', camera });

      // Point behind camera
      expect(result.x).toBe(0);
      expect(result.y).toBe(0);
    });
  });
});

describe('Full Pipeline', () => {
  const testPoint: Vector4D = { w: 0.5, x: 1, y: 2, z: 3 };

  it('should project through full pipeline', () => {
    const result = projectFull(testPoint);

    expect(result.original4D).toEqual(testPoint);
    expect(typeof result.projected3D.x).toBe('number');
    expect(typeof result.projected2D.x).toBe('number');
    expect(typeof result.depth).toBe('number');
  });

  it('should compute depth from W and Z', () => {
    const highW: Vector4D = { w: 1, x: 0, y: 0, z: 0 };
    const lowW: Vector4D = { w: -1, x: 0, y: 0, z: 0 };

    const highResult = projectFull(highW);
    const lowResult = projectFull(lowW);

    expect(highResult.depth).toBeGreaterThan(lowResult.depth);
  });

  it('should batch project correctly', () => {
    const points: Vector4D[] = [
      { w: 0, x: 1, y: 0, z: 0 },
      { w: 0.5, x: 0, y: 1, z: 0 },
    ];

    const results = projectFullBatch(points);

    expect(results.length).toBe(2);
    expect(results[0].original4D).toEqual(points[0]);
    expect(results[1].original4D).toEqual(points[1]);
  });

  it('should sort by depth', () => {
    const points: Vector4D[] = [
      { w: 1, x: 0, y: 0, z: 0 },   // High depth
      { w: -1, x: 0, y: 0, z: 0 },  // Low depth
      { w: 0, x: 0, y: 0, z: 0 },   // Middle depth
    ];

    const sorted = projectFullSorted(points);

    expect(sorted[0].depth).toBeLessThanOrEqual(sorted[1].depth);
    expect(sorted[1].depth).toBeLessThanOrEqual(sorted[2].depth);
  });
});

describe('Multi-View Projection', () => {
  it('should create all four views', () => {
    const points: Vector4D[] = [
      { w: 0, x: 1, y: 2, z: 3 },
      { w: 0.5, x: 2, y: 1, z: 0 },
    ];

    const result = multiViewProject(points);

    expect(result.front.length).toBe(2);
    expect(result.side.length).toBe(2);
    expect(result.top.length).toBe(2);
    expect(result.perspective.length).toBe(2);
  });

  it('should project different coordinates for different views', () => {
    const point: Vector4D = { w: 0, x: 1, y: 2, z: 3 };
    const result = multiViewProject([point]);

    // Front view: XY
    expect(result.front[0].x).toBe(1);
    expect(result.front[0].y).toBe(2);

    // Side view: ZY
    expect(result.side[0].x).toBe(3);
    expect(result.side[0].y).toBe(2);

    // Top view: XZ
    expect(result.top[0].x).toBe(1);
    expect(result.top[0].y).toBe(3);
  });
});

describe('Viewport Transformation', () => {
  const viewport = { x: 100, y: 50, width: 800, height: 600 };

  it('should transform center to viewport center', () => {
    const center: Vector2D = { x: 0, y: 0 };
    const result = toViewport(center, viewport);

    expect(result.x).toBe(100 + 800 / 2);  // 500
    expect(result.y).toBe(50 + 600 / 2);   // 350
  });

  it('should transform corners correctly', () => {
    const topLeft: Vector2D = { x: -1, y: 1 };
    const bottomRight: Vector2D = { x: 1, y: -1 };

    const tlResult = toViewport(topLeft, viewport);
    const brResult = toViewport(bottomRight, viewport);

    expect(tlResult.x).toBe(100);  // Left edge
    expect(tlResult.y).toBe(50);   // Top edge
    expect(brResult.x).toBe(900);  // Right edge
    expect(brResult.y).toBe(650);  // Bottom edge
  });

  it('should transform batch correctly', () => {
    const points: Vector2D[] = [
      { x: 0, y: 0 },
      { x: 1, y: 1 },
      { x: -1, y: -1 },
    ];

    const results = toViewportBatch(points, viewport);

    expect(results.length).toBe(3);
    expect(results[0].x).toBe(500);
  });
});

describe('Distance Functions', () => {
  describe('distance3D', () => {
    it('should compute zero for same points', () => {
      const p: Vector3D = { x: 1, y: 2, z: 3 };
      expect(distance3D(p, p)).toBe(0);
    });

    it('should compute unit distance correctly', () => {
      const a: Vector3D = { x: 0, y: 0, z: 0 };
      const b: Vector3D = { x: 1, y: 0, z: 0 };
      expect(distance3D(a, b)).toBe(1);
    });

    it('should compute diagonal distance', () => {
      const a: Vector3D = { x: 0, y: 0, z: 0 };
      const b: Vector3D = { x: 1, y: 1, z: 1 };
      expect(distance3D(a, b)).toBeCloseTo(Math.sqrt(3), 10);
    });
  });

  describe('distance2D', () => {
    it('should compute zero for same points', () => {
      const p: Vector2D = { x: 1, y: 2 };
      expect(distance2D(p, p)).toBe(0);
    });

    it('should compute unit distance correctly', () => {
      const a: Vector2D = { x: 0, y: 0 };
      const b: Vector2D = { x: 1, y: 0 };
      expect(distance2D(a, b)).toBe(1);
    });

    it('should compute diagonal distance', () => {
      const a: Vector2D = { x: 0, y: 0 };
      const b: Vector2D = { x: 3, y: 4 };
      expect(distance2D(a, b)).toBe(5);  // 3-4-5 triangle
    });
  });
});

describe('Default Camera', () => {
  it('should return valid camera configuration', () => {
    const camera = defaultCamera();

    expect(camera.position).toBeDefined();
    expect(camera.target).toBeDefined();
    expect(camera.up).toBeDefined();
    expect(camera.fov).toBeGreaterThan(0);
    expect(camera.near).toBeGreaterThan(0);
    expect(camera.far).toBeGreaterThan(camera.near);
  });

  it('should look down Z axis', () => {
    const camera = defaultCamera();

    expect(camera.position.z).toBeGreaterThan(camera.target.z);
  });
});

describe('ProjectionPipeline Class', () => {
  let pipeline: ProjectionPipeline;

  beforeEach(() => {
    pipeline = new ProjectionPipeline();
  });

  describe('Construction', () => {
    it('should create with default options', () => {
      expect(pipeline.get4DOptions().method).toBe('stereographic');
      expect(pipeline.get3DOptions().method).toBe('perspective');
    });

    it('should create with custom options', () => {
      const custom = new ProjectionPipeline(
        { method: 'orthographic' },
        { method: 'orthographic' }
      );

      expect(custom.get4DOptions().method).toBe('orthographic');
      expect(custom.get3DOptions().method).toBe('orthographic');
    });
  });

  describe('Options', () => {
    it('should update 4D options', () => {
      pipeline.set4DOptions({ method: 'perspective', perspectiveDistance: 5 });

      const options = pipeline.get4DOptions();
      expect(options.method).toBe('perspective');
      expect(options.perspectiveDistance).toBe(5);
    });

    it('should update 3D options', () => {
      pipeline.set3DOptions({ method: 'orthographic', scale: 2 });

      const options = pipeline.get3DOptions();
      expect(options.method).toBe('orthographic');
      expect(options.scale).toBe(2);
    });

    it('should update viewport', () => {
      pipeline.setViewport({ x: 0, y: 0, width: 1920, height: 1080 });

      const viewport = pipeline.getViewport();
      expect(viewport.width).toBe(1920);
      expect(viewport.height).toBe(1080);
    });

    it('should set rotation', () => {
      const rotation = createIsoclinicRotation(Math.PI / 6, 'WY');
      pipeline.setRotation(rotation);

      const options = pipeline.get4DOptions();
      expect(options.rotation).toBeDefined();
    });
  });

  describe('Projection', () => {
    it('should project single point', () => {
      const point: Vector4D = { w: 0, x: 1, y: 0, z: 0 };
      const result = pipeline.project(point);

      expect(result.original4D).toEqual(point);
      expect(result.projected3D).toBeDefined();
      expect(result.projected2D).toBeDefined();
      expect(result.viewport2D).toBeDefined();
    });

    it('should project batch', () => {
      const points: Vector4D[] = [
        { w: 0, x: 1, y: 0, z: 0 },
        { w: 0.5, x: 0, y: 1, z: 0 },
      ];

      const results = pipeline.projectBatch(points);

      expect(results.length).toBe(2);
      expect(results[0].viewport2D).toBeDefined();
    });

    it('should sort by depth', () => {
      const points: Vector4D[] = [
        { w: 1, x: 0, y: 0, z: 0 },
        { w: -1, x: 0, y: 0, z: 0 },
        { w: 0, x: 0, y: 0, z: 0 },
      ];

      const sorted = pipeline.projectSorted(points);

      expect(sorted[0].depth).toBeLessThanOrEqual(sorted[1].depth);
    });

    it('should project edges', () => {
      const vertices: Vector4D[] = [
        { w: 0, x: 0, y: 0, z: 0 },
        { w: 0, x: 1, y: 0, z: 0 },
        { w: 0, x: 0, y: 1, z: 0 },
      ];
      const edges: [number, number][] = [[0, 1], [1, 2], [2, 0]];

      const result = pipeline.projectEdges(vertices, edges);

      expect(result.length).toBe(3);
      expect(result[0].start).toBeDefined();
      expect(result[0].end).toBeDefined();
      expect(typeof result[0].depth).toBe('number');
    });
  });
});
