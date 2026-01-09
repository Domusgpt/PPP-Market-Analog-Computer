// File: src/components/ControlPanel.tsx
// Left panel with system status and controls

import type { ConvexityResult } from '../core/geometry';
import type { ChainStatistics } from '../core/trace';

export interface ControlPanelProps {
  /** Current convexity result */
  convexity: ConvexityResult | null;
  /** Is the simulation running */
  isRunning: boolean;
  /** Auto-rotate enabled */
  autoRotate: boolean;
  /** Current step count */
  stepCount: number;
  /** Chain statistics */
  chainStats: ChainStatistics | null;
  /** Callbacks */
  onToggleRun: () => void;
  onToggleRotate: () => void;
  onStep: () => void;
  onInjectEntropy: () => void;
  onReset: () => void;
}

/**
 * Status indicator component
 */
function StatusIndicator({
  status,
  label,
}: {
  status: 'SAFE' | 'WARNING' | 'VIOLATION' | 'unknown';
  label: string;
}) {
  const colors = {
    SAFE: '#00ff88',
    WARNING: '#ffaa00',
    VIOLATION: '#ff3366',
    unknown: '#666666',
  };

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        marginBottom: '4px',
      }}
    >
      <div
        style={{
          width: '10px',
          height: '10px',
          borderRadius: '50%',
          backgroundColor: colors[status],
          boxShadow: `0 0 8px ${colors[status]}`,
        }}
      />
      <span style={{ color: '#ffffff', fontSize: '12px' }}>{label}</span>
    </div>
  );
}

/**
 * Metric display component
 */
function Metric({
  label,
  value,
  unit,
}: {
  label: string;
  value: string | number;
  unit?: string;
}) {
  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '4px 0',
        borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
      }}
    >
      <span style={{ color: 'rgba(255, 255, 255, 0.6)', fontSize: '11px' }}>
        {label}
      </span>
      <span style={{ color: '#00ffff', fontSize: '12px', fontFamily: 'monospace' }}>
        {typeof value === 'number' ? value.toFixed(4) : value}
        {unit && (
          <span style={{ color: 'rgba(255, 255, 255, 0.4)', marginLeft: '4px' }}>
            {unit}
          </span>
        )}
      </span>
    </div>
  );
}

/**
 * Button component
 */
function Button({
  onClick,
  children,
  active,
  variant = 'default',
}: {
  onClick: () => void;
  children: React.ReactNode;
  active?: boolean;
  variant?: 'default' | 'danger' | 'success';
}) {
  const colors = {
    default: { bg: '#1a1a2e', hover: '#2a2a4e', border: '#333366' },
    danger: { bg: '#2e1a1a', hover: '#4e2a2a', border: '#663333' },
    success: { bg: '#1a2e1a', hover: '#2a4e2a', border: '#336633' },
  };

  const c = colors[variant];

  return (
    <button
      onClick={onClick}
      style={{
        width: '100%',
        padding: '8px 12px',
        marginBottom: '6px',
        backgroundColor: active ? c.hover : c.bg,
        border: `1px solid ${c.border}`,
        borderRadius: '4px',
        color: '#ffffff',
        fontSize: '11px',
        fontFamily: 'monospace',
        cursor: 'pointer',
        transition: 'all 0.2s',
        textTransform: 'uppercase',
        letterSpacing: '0.5px',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.backgroundColor = c.hover;
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.backgroundColor = active ? c.hover : c.bg;
      }}
    >
      {children}
    </button>
  );
}

/**
 * ControlPanel - System status and controls
 */
export function ControlPanel({
  convexity,
  isRunning,
  autoRotate,
  stepCount,
  chainStats,
  onToggleRun,
  onToggleRotate,
  onStep,
  onInjectEntropy,
  onReset,
}: ControlPanelProps) {
  const status = convexity?.status ?? 'unknown';

  return (
    <div
      style={{
        width: '220px',
        height: '100%',
        backgroundColor: '#0d0d15',
        borderRight: '1px solid #1a1a2e',
        padding: '16px',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'auto',
      }}
    >
      {/* Header */}
      <div
        style={{
          marginBottom: '20px',
          paddingBottom: '12px',
          borderBottom: '1px solid #1a1a2e',
        }}
      >
        <h1
          style={{
            color: '#ffffff',
            fontSize: '14px',
            fontWeight: 'bold',
            margin: 0,
            letterSpacing: '1px',
          }}
        >
          CRA-POM v2
        </h1>
        <p
          style={{
            color: 'rgba(255, 255, 255, 0.5)',
            fontSize: '10px',
            margin: '4px 0 0 0',
          }}
        >
          Geometric Cognition Kernel
        </p>
      </div>

      {/* Convexity Status */}
      <div style={{ marginBottom: '20px' }}>
        <h2
          style={{
            color: 'rgba(255, 255, 255, 0.7)',
            fontSize: '10px',
            textTransform: 'uppercase',
            letterSpacing: '1px',
            marginBottom: '12px',
          }}
        >
          Convexity Status
        </h2>
        <StatusIndicator status={status as 'SAFE' | 'WARNING' | 'VIOLATION'} label={status} />
      </div>

      {/* Metrics */}
      <div style={{ marginBottom: '20px' }}>
        <h2
          style={{
            color: 'rgba(255, 255, 255, 0.7)',
            fontSize: '10px',
            textTransform: 'uppercase',
            letterSpacing: '1px',
            marginBottom: '12px',
          }}
        >
          Geometry Metrics
        </h2>
        <Metric label="Step Count" value={stepCount} />
        <Metric
          label="Distance (Origin)"
          value={convexity?.distanceFromOrigin ?? 0}
        />
        <Metric
          label="Penetration"
          value={convexity?.penetrationDepth ?? 0}
        />
        <Metric
          label="Nearest Vertex"
          value={`#${convexity?.nearestVertex.index ?? '-'}`}
        />
        <Metric
          label="Vertex Distance"
          value={convexity?.nearestVertex.distance ?? 0}
        />
      </div>

      {/* Chain Statistics */}
      {chainStats && (
        <div style={{ marginBottom: '20px' }}>
          <h2
            style={{
              color: 'rgba(255, 255, 255, 0.7)',
              fontSize: '10px',
              textTransform: 'uppercase',
              letterSpacing: '1px',
              marginBottom: '12px',
            }}
          >
            Audit Chain
          </h2>
          <Metric label="Total Entries" value={chainStats.totalEntries} />
          <Metric label="Safe States" value={chainStats.convexityStats.safe} />
          <Metric label="Warnings" value={chainStats.convexityStats.warning} />
          <Metric
            label="Violations"
            value={chainStats.convexityStats.violation}
          />
        </div>
      )}

      {/* Controls */}
      <div style={{ marginTop: 'auto' }}>
        <h2
          style={{
            color: 'rgba(255, 255, 255, 0.7)',
            fontSize: '10px',
            textTransform: 'uppercase',
            letterSpacing: '1px',
            marginBottom: '12px',
          }}
        >
          Controls
        </h2>

        <Button onClick={onToggleRun} active={isRunning} variant="success">
          {isRunning ? '⏸ Pause' : '▶ Run'}
        </Button>

        <Button onClick={onStep}>⏭ Single Step</Button>

        <Button onClick={onToggleRotate} active={autoRotate}>
          {autoRotate ? '🔄 Rotating' : '🔄 Auto-Rotate'}
        </Button>

        <Button onClick={onInjectEntropy} variant="default">
          ⚡ Inject Entropy
        </Button>

        <Button onClick={onReset} variant="danger">
          ↺ Reset
        </Button>
      </div>
    </div>
  );
}

export default ControlPanel;
