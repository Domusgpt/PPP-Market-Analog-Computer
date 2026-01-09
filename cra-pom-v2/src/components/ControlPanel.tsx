// File: src/components/ControlPanel.tsx
// Left panel with system status, metrics, and controls

import type { ConvexityResult } from '../core/geometry';
import type { ChainStatistics } from '../core/trace';
import type { CoherenceMetrics } from '../core/analysis';

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
  /** Coherence metrics */
  coherenceMetrics: CoherenceMetrics | null;
  /** Callbacks */
  onToggleRun: () => void;
  onToggleRotate: () => void;
  onStep: () => void;
  onInjectEntropy: () => void;
  onReset: () => void;
  onDownload: () => void;
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
  color,
}: {
  label: string;
  value: string | number;
  unit?: string;
  color?: string;
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
      <span
        style={{
          color: color ?? '#00ffff',
          fontSize: '12px',
          fontFamily: 'monospace',
        }}
      >
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
 * Progress bar for coherence metrics
 */
function CoherenceBar({ label, value }: { label: string; value: number }) {
  const percentage = Math.max(0, Math.min(100, value * 100));
  const color =
    value >= 0.7 ? '#00ff88' : value >= 0.4 ? '#ffaa00' : '#ff3366';

  return (
    <div style={{ marginBottom: '8px' }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          marginBottom: '4px',
        }}
      >
        <span style={{ color: 'rgba(255, 255, 255, 0.6)', fontSize: '10px' }}>
          {label}
        </span>
        <span style={{ color, fontSize: '10px', fontFamily: 'monospace' }}>
          {(value * 100).toFixed(1)}%
        </span>
      </div>
      <div
        style={{
          width: '100%',
          height: '4px',
          backgroundColor: 'rgba(255, 255, 255, 0.1)',
          borderRadius: '2px',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            width: `${percentage}%`,
            height: '100%',
            backgroundColor: color,
            transition: 'width 0.3s ease',
          }}
        />
      </div>
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
  variant?: 'default' | 'danger' | 'success' | 'info';
}) {
  const colors = {
    default: { bg: '#1a1a2e', hover: '#2a2a4e', border: '#333366' },
    danger: { bg: '#2e1a1a', hover: '#4e2a2a', border: '#663333' },
    success: { bg: '#1a2e1a', hover: '#2a4e2a', border: '#336633' },
    info: { bg: '#1a2e2e', hover: '#2a4e4e', border: '#336666' },
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
 * Section header component
 */
function SectionHeader({ children }: { children: React.ReactNode }) {
  return (
    <h2
      style={{
        color: 'rgba(255, 255, 255, 0.7)',
        fontSize: '10px',
        textTransform: 'uppercase',
        letterSpacing: '1px',
        marginBottom: '12px',
      }}
    >
      {children}
    </h2>
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
  coherenceMetrics,
  onToggleRun,
  onToggleRotate,
  onStep,
  onInjectEntropy,
  onReset,
  onDownload,
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
          marginBottom: '16px',
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
      <div style={{ marginBottom: '16px' }}>
        <SectionHeader>Convexity Status</SectionHeader>
        <StatusIndicator
          status={status as 'SAFE' | 'WARNING' | 'VIOLATION'}
          label={status}
        />
      </div>

      {/* Geometry Metrics */}
      <div style={{ marginBottom: '16px' }}>
        <SectionHeader>Geometry</SectionHeader>
        <Metric label="Step" value={stepCount} />
        <Metric label="Distance" value={convexity?.distanceFromOrigin ?? 0} />
        <Metric label="Penetration" value={convexity?.penetrationDepth ?? 0} />
        <Metric
          label="Nearest Vertex"
          value={`#${convexity?.nearestVertex.index ?? '-'}`}
        />
      </div>

      {/* Coherence Metrics */}
      {coherenceMetrics && (
        <div style={{ marginBottom: '16px' }}>
          <SectionHeader>Coherence</SectionHeader>
          <CoherenceBar label="Overall" value={coherenceMetrics.overallCoherence} />
          <CoherenceBar label="Spinor Align" value={coherenceMetrics.spinorAlignment} />
          <CoherenceBar label="Golden Res" value={coherenceMetrics.goldenResonance} />
          <CoherenceBar label="Stability" value={coherenceMetrics.stabilityIndex} />
        </div>
      )}

      {/* Chain Statistics */}
      {chainStats && (
        <div style={{ marginBottom: '16px' }}>
          <SectionHeader>Audit Chain</SectionHeader>
          <Metric label="Entries" value={chainStats.totalEntries} />
          <Metric
            label="Safe"
            value={chainStats.convexityStats.safe}
            color="#00ff88"
          />
          <Metric
            label="Warnings"
            value={chainStats.convexityStats.warning}
            color="#ffaa00"
          />
          <Metric
            label="Violations"
            value={chainStats.convexityStats.violation}
            color="#ff3366"
          />
        </div>
      )}

      {/* Controls */}
      <div style={{ marginTop: 'auto' }}>
        <SectionHeader>Controls</SectionHeader>

        <Button onClick={onToggleRun} active={isRunning} variant="success">
          {isRunning ? '⏸ Pause' : '▶ Run'}
        </Button>

        <Button onClick={onStep}>⏭ Step</Button>

        <Button onClick={onToggleRotate} active={autoRotate}>
          {autoRotate ? '🔄 Rotating' : '🔄 Rotate'}
        </Button>

        <Button onClick={onInjectEntropy} variant="default">
          ⚡ Entropy
        </Button>

        <Button onClick={onDownload} variant="info">
          ⬇ Export
        </Button>

        <Button onClick={onReset} variant="danger">
          ↺ Reset
        </Button>
      </div>
    </div>
  );
}

export default ControlPanel;
