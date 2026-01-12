// File: src/components/Sparkline.tsx
// Sparkline chart for real-time coherence visualization

import { useRef, useEffect, useMemo } from 'react';

export interface SparklineDataPoint {
  value: number;
  timestamp?: number;
}

export interface SparklineProps {
  /** Data points to display */
  data: SparklineDataPoint[];
  /** Width in pixels */
  width: number;
  /** Height in pixels */
  height: number;
  /** Line color */
  color?: string;
  /** Fill area under line */
  fill?: boolean;
  /** Show min/max indicators */
  showRange?: boolean;
  /** Show current value */
  showValue?: boolean;
  /** Label */
  label?: string;
  /** Min value for scale */
  minValue?: number;
  /** Max value for scale */
  maxValue?: number;
}

/**
 * Sparkline - Compact line chart for metrics visualization
 */
export function Sparkline({
  data,
  width,
  height,
  color = '#00ffff',
  fill = true,
  showRange = false,
  showValue = true,
  label,
  minValue = 0,
  maxValue = 1,
}: SparklineProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Calculate statistics
  const stats = useMemo(() => {
    if (data.length === 0) {
      return { min: 0, max: 1, current: 0, avg: 0 };
    }
    const values = data.map((d) => d.value);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const current = values[values.length - 1];
    const avg = values.reduce((a, b) => a + b, 0) / values.length;
    return { min, max, current, avg };
  }, [data]);

  // Render the sparkline
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    if (data.length < 2) {
      // Not enough data
      ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
      ctx.fillRect(0, height / 2 - 1, width, 2);
      return;
    }

    const padding = 2;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;

    // Calculate scale
    const valueRange = maxValue - minValue;
    const xStep = chartWidth / (data.length - 1);

    // Build path
    ctx.beginPath();
    data.forEach((point, i) => {
      const x = padding + i * xStep;
      const normalizedValue = (point.value - minValue) / valueRange;
      const y = padding + chartHeight - normalizedValue * chartHeight;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    // Draw fill
    if (fill) {
      ctx.lineTo(padding + chartWidth, padding + chartHeight);
      ctx.lineTo(padding, padding + chartHeight);
      ctx.closePath();

      const gradient = ctx.createLinearGradient(0, 0, 0, height);
      gradient.addColorStop(0, color.replace(')', ', 0.3)').replace('rgb', 'rgba'));
      gradient.addColorStop(1, 'transparent');
      ctx.fillStyle = gradient;
      ctx.fill();
    }

    // Draw line
    ctx.beginPath();
    data.forEach((point, i) => {
      const x = padding + i * xStep;
      const normalizedValue = (point.value - minValue) / valueRange;
      const y = padding + chartHeight - normalizedValue * chartHeight;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.stroke();

    // Draw current value dot
    if (data.length > 0) {
      const lastPoint = data[data.length - 1];
      const x = padding + (data.length - 1) * xStep;
      const normalizedValue = (lastPoint.value - minValue) / valueRange;
      const y = padding + chartHeight - normalizedValue * chartHeight;

      // Glow
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = color.replace(')', ', 0.4)').replace('rgb', 'rgba');
      ctx.fill();

      // Dot
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }
  }, [data, width, height, color, fill, minValue, maxValue]);

  const valueColor =
    stats.current >= 0.7 ? '#00ff88' : stats.current >= 0.4 ? '#ffaa00' : '#ff3366';

  return (
    <div style={{ marginBottom: '12px' }}>
      {(label || showValue) && (
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '4px',
          }}
        >
          {label && (
            <span style={{ color: 'rgba(255, 255, 255, 0.6)', fontSize: '10px' }}>
              {label}
            </span>
          )}
          {showValue && (
            <span style={{ color: valueColor, fontSize: '10px', fontFamily: 'monospace' }}>
              {(stats.current * 100).toFixed(0)}%
            </span>
          )}
        </div>
      )}
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{
          width: `${width}px`,
          height: `${height}px`,
          display: 'block',
          borderRadius: '4px',
          backgroundColor: 'rgba(255, 255, 255, 0.03)',
        }}
      />
      {showRange && data.length > 0 && (
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            marginTop: '4px',
            fontSize: '9px',
            color: 'rgba(255, 255, 255, 0.4)',
            fontFamily: 'monospace',
          }}
        >
          <span>min: {(stats.min * 100).toFixed(0)}%</span>
          <span>avg: {(stats.avg * 100).toFixed(0)}%</span>
          <span>max: {(stats.max * 100).toFixed(0)}%</span>
        </div>
      )}
    </div>
  );
}

/**
 * Multi-line sparkline for comparing multiple metrics
 */
export interface MultiSparklineProps {
  data: Array<{
    label: string;
    values: number[];
    color: string;
  }>;
  width: number;
  height: number;
}

export function MultiSparkline({ data, width, height }: MultiSparklineProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    const padding = 2;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;

    // Draw each series
    data.forEach(({ values, color }) => {
      if (values.length < 2) return;

      const xStep = chartWidth / (values.length - 1);

      ctx.beginPath();
      values.forEach((value, i) => {
        const x = padding + i * xStep;
        const y = padding + chartHeight - value * chartHeight;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });

      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.8;
      ctx.stroke();
      ctx.globalAlpha = 1;
    });
  }, [data, width, height]);

  return (
    <div>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{
          width: `${width}px`,
          height: `${height}px`,
          display: 'block',
          borderRadius: '4px',
          backgroundColor: 'rgba(255, 255, 255, 0.03)',
        }}
      />
      <div
        style={{
          display: 'flex',
          gap: '12px',
          marginTop: '6px',
          flexWrap: 'wrap',
        }}
      >
        {data.map(({ label, color, values }) => (
          <div
            key={label}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              fontSize: '9px',
            }}
          >
            <div
              style={{
                width: '8px',
                height: '2px',
                backgroundColor: color,
                borderRadius: '1px',
              }}
            />
            <span style={{ color: 'rgba(255, 255, 255, 0.5)' }}>{label}</span>
            <span style={{ color, fontFamily: 'monospace' }}>
              {values.length > 0 ? (values[values.length - 1] * 100).toFixed(0) : 0}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default Sparkline;
