// File: src/components/PolytopeCanvas.tsx
// HTML5 Canvas renderer for the 24-cell with thought vector visualization

import { useRef, useEffect, useCallback, useState } from 'react';
import type { ProjectionResult, ProjectedVertex, ProjectedEdge } from '../core/projection';
import type { ConvexityStatus } from '../core/geometry';

/**
 * Color scheme for different states
 */
const COLORS = {
  background: '#0a0a0f',
  latticeVertex: {
    safe: '#00ff88',
    warning: '#ffaa00',
    violation: '#ff3366',
  },
  latticeEdge: {
    safe: 'rgba(0, 255, 136, 0.3)',
    warning: 'rgba(255, 170, 0, 0.3)',
    violation: 'rgba(255, 51, 102, 0.3)',
  },
  thoughtVector: {
    safe: '#00ffff',
    warning: '#ffff00',
    violation: '#ff0000',
  },
  glow: {
    safe: 'rgba(0, 255, 255, 0.6)',
    warning: 'rgba(255, 255, 0, 0.6)',
    violation: 'rgba(255, 0, 0, 0.6)',
  },
  grid: 'rgba(255, 255, 255, 0.05)',
};

export interface PolytopeCanvasProps {
  /** Projection result to render */
  projection: ProjectionResult | null;
  /** Current convexity status */
  status: ConvexityStatus;
  /** Width of the canvas */
  width: number;
  /** Height of the canvas */
  height: number;
  /** Show grid overlay */
  showGrid?: boolean;
  /** Show vertex indices */
  showLabels?: boolean;
  /** Enable auto-rotation visual feedback */
  isAnimating?: boolean;
}

/**
 * PolytopeCanvas - Renders the 24-cell and thought vector
 */
export function PolytopeCanvas({
  projection,
  status,
  width,
  height,
  showGrid = true,
  showLabels = false,
  isAnimating = false,
}: PolytopeCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [fps, setFps] = useState(0);
  const frameCountRef = useRef(0);
  const lastTimeRef = useRef(performance.now());

  /**
   * Get color based on status
   */
  const getStatusColor = useCallback(
    (type: 'vertex' | 'edge' | 'thought' | 'glow') => {
      const colorMap = {
        vertex: COLORS.latticeVertex,
        edge: COLORS.latticeEdge,
        thought: COLORS.thoughtVector,
        glow: COLORS.glow,
      };
      return colorMap[type][status.toLowerCase() as Lowercase<ConvexityStatus>];
    },
    [status]
  );

  /**
   * Draw the grid overlay
   */
  const drawGrid = useCallback(
    (ctx: CanvasRenderingContext2D) => {
      ctx.strokeStyle = COLORS.grid;
      ctx.lineWidth = 1;

      const gridSize = 50;
      const centerX = width / 2;
      const centerY = height / 2;

      // Vertical lines
      for (let x = centerX % gridSize; x < width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }

      // Horizontal lines
      for (let y = centerY % gridSize; y < height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // Center crosshair
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
      ctx.beginPath();
      ctx.moveTo(centerX, 0);
      ctx.lineTo(centerX, height);
      ctx.moveTo(0, centerY);
      ctx.lineTo(width, centerY);
      ctx.stroke();
    },
    [width, height]
  );

  /**
   * Draw an edge between two vertices
   */
  const drawEdge = useCallback(
    (ctx: CanvasRenderingContext2D, edge: ProjectedEdge) => {
      const { start, end } = edge;

      // Calculate opacity based on depth
      const avgDepth = edge.avgDepth;
      const depthFactor = Math.max(0.2, Math.min(1, 1 - avgDepth / 5));

      ctx.beginPath();
      ctx.moveTo(start.point2D.x, start.point2D.y);
      ctx.lineTo(end.point2D.x, end.point2D.y);

      ctx.strokeStyle = getStatusColor('edge');
      ctx.lineWidth = 1.5 * depthFactor;
      ctx.globalAlpha = depthFactor;
      ctx.stroke();
      ctx.globalAlpha = 1;
    },
    [getStatusColor]
  );

  /**
   * Draw a vertex point
   */
  const drawVertex = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      vertex: ProjectedVertex,
      highlight = false
    ) => {
      const { point2D, scale, depth, index } = vertex;

      // Size based on perspective and depth
      const baseSize = highlight ? 6 : 3;
      const size = baseSize * Math.max(0.5, Math.min(2, scale * 1.5));

      // Opacity based on depth
      const depthFactor = Math.max(0.3, Math.min(1, 1 - depth / 5));

      ctx.beginPath();
      ctx.arc(point2D.x, point2D.y, size, 0, Math.PI * 2);
      ctx.fillStyle = getStatusColor('vertex');
      ctx.globalAlpha = depthFactor;
      ctx.fill();
      ctx.globalAlpha = 1;

      // Draw label if enabled
      if (showLabels) {
        ctx.font = '10px monospace';
        ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.fillText(String(index), point2D.x + size + 2, point2D.y + 3);
      }
    },
    [getStatusColor, showLabels]
  );

  /**
   * Draw the thought vector with glow effect
   */
  const drawThoughtVector = useCallback(
    (ctx: CanvasRenderingContext2D, vertex: ProjectedVertex) => {
      const { point2D, scale } = vertex;

      const size = 8 * Math.max(0.5, Math.min(2, scale * 1.5));
      const glowSize = size * 3;

      // Outer glow
      const gradient = ctx.createRadialGradient(
        point2D.x,
        point2D.y,
        0,
        point2D.x,
        point2D.y,
        glowSize
      );
      gradient.addColorStop(0, getStatusColor('glow'));
      gradient.addColorStop(0.5, getStatusColor('glow').replace('0.6', '0.2'));
      gradient.addColorStop(1, 'transparent');

      ctx.beginPath();
      ctx.arc(point2D.x, point2D.y, glowSize, 0, Math.PI * 2);
      ctx.fillStyle = gradient;
      ctx.fill();

      // Inner circle
      ctx.beginPath();
      ctx.arc(point2D.x, point2D.y, size, 0, Math.PI * 2);
      ctx.fillStyle = getStatusColor('thought');
      ctx.fill();

      // Core dot
      ctx.beginPath();
      ctx.arc(point2D.x, point2D.y, size * 0.4, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
    },
    [getStatusColor]
  );

  /**
   * Draw status indicator
   */
  const drawStatusIndicator = useCallback(
    (ctx: CanvasRenderingContext2D) => {
      const statusText = status;
      const padding = 10;

      ctx.font = 'bold 12px monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';

      // Background
      const textWidth = ctx.measureText(statusText).width;
      ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
      ctx.fillRect(padding - 4, padding - 4, textWidth + 16, 20);

      // Status indicator circle
      ctx.beginPath();
      ctx.arc(padding + 4, padding + 6, 4, 0, Math.PI * 2);
      ctx.fillStyle = getStatusColor('thought');
      ctx.fill();

      // Text
      ctx.fillStyle = '#ffffff';
      ctx.fillText(statusText, padding + 14, padding);

      // FPS counter
      if (isAnimating) {
        const fpsText = `${fps} FPS`;
        ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.textAlign = 'right';
        ctx.fillText(fpsText, width - padding, padding);
      }
    },
    [status, fps, isAnimating, width, getStatusColor]
  );

  /**
   * Main render function
   */
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    if (showGrid) {
      drawGrid(ctx);
    }

    // If no projection, just show empty state
    if (!projection) {
      ctx.font = '14px monospace';
      ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.textAlign = 'center';
      ctx.fillText('Initializing...', width / 2, height / 2);
      return;
    }

    // Draw edges (back to front, already sorted by depth)
    for (const edge of projection.edges) {
      drawEdge(ctx, edge);
    }

    // Sort vertices by depth for proper occlusion
    const sortedVertices = [...projection.vertices].sort(
      (a, b) => a.depth - b.depth
    );

    // Draw vertices
    for (const vertex of sortedVertices) {
      drawVertex(ctx, vertex);
    }

    // Draw thought vector
    if (projection.thoughtVector) {
      drawThoughtVector(ctx, projection.thoughtVector);
    }

    // Draw status
    drawStatusIndicator(ctx);

    // FPS calculation
    frameCountRef.current++;
    const now = performance.now();
    if (now - lastTimeRef.current >= 1000) {
      setFps(frameCountRef.current);
      frameCountRef.current = 0;
      lastTimeRef.current = now;
    }
  }, [
    projection,
    width,
    height,
    showGrid,
    drawGrid,
    drawEdge,
    drawVertex,
    drawThoughtVector,
    drawStatusIndicator,
  ]);

  // Render on projection change
  useEffect(() => {
    render();
  }, [render]);

  // Handle canvas resize
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Set actual size in memory
    canvas.width = width;
    canvas.height = height;

    // Ensure crisp rendering
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
    }
  }, [width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{
        width: '100%',
        height: '100%',
        display: 'block',
        backgroundColor: COLORS.background,
      }}
    />
  );
}

export default PolytopeCanvas;
