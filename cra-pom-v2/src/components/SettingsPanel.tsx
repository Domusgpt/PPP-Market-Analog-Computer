// File: src/components/SettingsPanel.tsx
// Settings panel with projection parameters, trajectory modes, and advanced controls

import { useState } from 'react';
import type { SessionSettings } from '../hooks';

export type TrajectoryMode = 'free' | 'spiral' | 'boundary' | 'ergodic' | 'oscillate';

export interface SettingsPanelProps {
  settings: SessionSettings;
  trajectoryMode: TrajectoryMode;
  isOpen: boolean;
  onClose: () => void;
  onSettingsChange: (settings: Partial<SessionSettings>) => void;
  onTrajectoryModeChange: (mode: TrajectoryMode) => void;
  onClearData: () => void;
}

/**
 * Slider component
 */
function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  unit,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  unit?: string;
}) {
  return (
    <div style={{ marginBottom: '16px' }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          marginBottom: '6px',
        }}
      >
        <label style={{ color: 'rgba(255, 255, 255, 0.7)', fontSize: '11px' }}>
          {label}
        </label>
        <span style={{ color: '#00ffff', fontSize: '11px', fontFamily: 'monospace' }}>
          {value.toFixed(step < 1 ? 2 : 0)}
          {unit && <span style={{ color: 'rgba(255, 255, 255, 0.4)' }}> {unit}</span>}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{
          width: '100%',
          height: '4px',
          appearance: 'none',
          background: 'linear-gradient(to right, #00ffff, #00ff88)',
          borderRadius: '2px',
          outline: 'none',
          cursor: 'pointer',
        }}
      />
    </div>
  );
}

/**
 * Toggle switch component
 */
function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '12px',
      }}
    >
      <label style={{ color: 'rgba(255, 255, 255, 0.7)', fontSize: '11px' }}>{label}</label>
      <button
        onClick={() => onChange(!checked)}
        style={{
          width: '40px',
          height: '20px',
          borderRadius: '10px',
          border: 'none',
          backgroundColor: checked ? '#00ff88' : '#333366',
          position: 'relative',
          cursor: 'pointer',
          transition: 'background-color 0.2s',
        }}
      >
        <div
          style={{
            width: '16px',
            height: '16px',
            borderRadius: '50%',
            backgroundColor: '#ffffff',
            position: 'absolute',
            top: '2px',
            left: checked ? '22px' : '2px',
            transition: 'left 0.2s',
          }}
        />
      </button>
    </div>
  );
}

/**
 * Mode button component
 */
function ModeButton({
  label,
  description,
  active,
  onClick,
}: {
  label: string;
  description: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        width: '100%',
        padding: '10px 12px',
        marginBottom: '8px',
        backgroundColor: active ? '#1a2e2e' : '#1a1a2e',
        border: `1px solid ${active ? '#00ffff' : '#333366'}`,
        borderRadius: '6px',
        cursor: 'pointer',
        textAlign: 'left',
        transition: 'all 0.2s',
      }}
    >
      <div
        style={{
          color: active ? '#00ffff' : '#ffffff',
          fontSize: '12px',
          fontWeight: 'bold',
          marginBottom: '4px',
        }}
      >
        {label}
      </div>
      <div style={{ color: 'rgba(255, 255, 255, 0.5)', fontSize: '10px' }}>
        {description}
      </div>
    </button>
  );
}

/**
 * Settings Panel Component
 */
export function SettingsPanel({
  settings,
  trajectoryMode,
  isOpen,
  onClose,
  onSettingsChange,
  onTrajectoryModeChange,
  onClearData,
}: SettingsPanelProps) {
  const [activeTab, setActiveTab] = useState<'projection' | 'trajectory' | 'advanced'>('projection');

  if (!isOpen) return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
      }}
      onClick={onClose}
    >
      <div
        style={{
          backgroundColor: '#0d0d15',
          border: '1px solid #1a1a2e',
          borderRadius: '12px',
          width: '100%',
          maxWidth: '420px',
          maxHeight: '80vh',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          style={{
            padding: '16px 20px',
            borderBottom: '1px solid #1a1a2e',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <h2 style={{ color: '#ffffff', fontSize: '16px', margin: 0 }}>Settings</h2>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: 'rgba(255, 255, 255, 0.5)',
              fontSize: '20px',
              cursor: 'pointer',
            }}
          >
            ×
          </button>
        </div>

        {/* Tabs */}
        <div
          style={{
            display: 'flex',
            borderBottom: '1px solid #1a1a2e',
          }}
        >
          {(['projection', 'trajectory', 'advanced'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                flex: 1,
                padding: '12px',
                background: 'none',
                border: 'none',
                borderBottom: activeTab === tab ? '2px solid #00ffff' : '2px solid transparent',
                color: activeTab === tab ? '#00ffff' : 'rgba(255, 255, 255, 0.5)',
                fontSize: '11px',
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
                cursor: 'pointer',
              }}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* Content */}
        <div style={{ padding: '20px', overflowY: 'auto', flex: 1 }}>
          {activeTab === 'projection' && (
            <>
              <Slider
                label="Projection W"
                value={settings.projectionW}
                min={1.5}
                max={5}
                step={0.1}
                onChange={(v) => onSettingsChange({ projectionW: v })}
              />
              <Slider
                label="Camera Distance"
                value={settings.cameraDistance}
                min={2}
                max={10}
                step={0.5}
                onChange={(v) => onSettingsChange({ cameraDistance: v })}
              />
              <Slider
                label="Field of View"
                value={settings.fov}
                min={100}
                max={600}
                step={10}
                onChange={(v) => onSettingsChange({ fov: v })}
              />
              <div style={{ marginTop: '20px' }}>
                <Toggle
                  label="Auto-Rotate"
                  checked={settings.autoRotate}
                  onChange={(v) => onSettingsChange({ autoRotate: v })}
                />
                <Toggle
                  label="Show Grid"
                  checked={settings.showGrid}
                  onChange={(v) => onSettingsChange({ showGrid: v })}
                />
                <Toggle
                  label="Show Trajectory"
                  checked={settings.showTrajectory}
                  onChange={(v) => onSettingsChange({ showTrajectory: v })}
                />
              </div>
            </>
          )}

          {activeTab === 'trajectory' && (
            <>
              <div style={{ marginBottom: '16px' }}>
                <Slider
                  label="Step Rate"
                  value={settings.stepRate}
                  min={1}
                  max={30}
                  step={1}
                  onChange={(v) => onSettingsChange({ stepRate: v })}
                  unit="Hz"
                />
                <Slider
                  label="Trail Length"
                  value={settings.trajectoryLength}
                  min={50}
                  max={500}
                  step={10}
                  onChange={(v) => onSettingsChange({ trajectoryLength: v })}
                  unit="pts"
                />
              </div>

              <div
                style={{
                  color: 'rgba(255, 255, 255, 0.7)',
                  fontSize: '10px',
                  textTransform: 'uppercase',
                  letterSpacing: '1px',
                  marginBottom: '12px',
                }}
              >
                Trajectory Mode
              </div>

              <ModeButton
                label="Free Evolution"
                description="Natural isoclinic rotation with golden ratio axes"
                active={trajectoryMode === 'free'}
                onClick={() => onTrajectoryModeChange('free')}
              />
              <ModeButton
                label="Spiral"
                description="Outward spiral towards boundary, then constrain"
                active={trajectoryMode === 'spiral'}
                onClick={() => onTrajectoryModeChange('spiral')}
              />
              <ModeButton
                label="Boundary Explorer"
                description="Stay near the convexity boundary"
                active={trajectoryMode === 'boundary'}
                onClick={() => onTrajectoryModeChange('boundary')}
              />
              <ModeButton
                label="Ergodic"
                description="Maximize coverage with periodic entropy"
                active={trajectoryMode === 'ergodic'}
                onClick={() => onTrajectoryModeChange('ergodic')}
              />
              <ModeButton
                label="Oscillate"
                description="Periodic motion between vertices"
                active={trajectoryMode === 'oscillate'}
                onClick={() => onTrajectoryModeChange('oscillate')}
              />
            </>
          )}

          {activeTab === 'advanced' && (
            <>
              <div
                style={{
                  padding: '16px',
                  backgroundColor: 'rgba(255, 170, 0, 0.1)',
                  border: '1px solid rgba(255, 170, 0, 0.3)',
                  borderRadius: '8px',
                  marginBottom: '20px',
                }}
              >
                <div
                  style={{
                    color: '#ffaa00',
                    fontSize: '12px',
                    fontWeight: 'bold',
                    marginBottom: '8px',
                  }}
                >
                  Data Management
                </div>
                <p
                  style={{
                    color: 'rgba(255, 255, 255, 0.6)',
                    fontSize: '11px',
                    margin: '0 0 12px 0',
                    lineHeight: '1.5',
                  }}
                >
                  Clear all stored session data including settings, trajectory history, and audit logs.
                </p>
                <button
                  onClick={onClearData}
                  style={{
                    width: '100%',
                    padding: '10px',
                    backgroundColor: '#2e1a1a',
                    border: '1px solid #663333',
                    borderRadius: '6px',
                    color: '#ff6666',
                    fontSize: '11px',
                    fontFamily: 'monospace',
                    cursor: 'pointer',
                    textTransform: 'uppercase',
                  }}
                >
                  Clear All Data
                </button>
              </div>

              <div
                style={{
                  padding: '16px',
                  backgroundColor: 'rgba(0, 255, 255, 0.05)',
                  border: '1px solid rgba(0, 255, 255, 0.2)',
                  borderRadius: '8px',
                }}
              >
                <div
                  style={{
                    color: '#00ffff',
                    fontSize: '12px',
                    fontWeight: 'bold',
                    marginBottom: '8px',
                  }}
                >
                  System Info
                </div>
                <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)' }}>
                  <div style={{ marginBottom: '4px' }}>Version: 2.0.0</div>
                  <div style={{ marginBottom: '4px' }}>
                    Polytope: 24-Cell (D4 Lattice)
                  </div>
                  <div style={{ marginBottom: '4px' }}>Vertices: 24</div>
                  <div style={{ marginBottom: '4px' }}>Edges: 96</div>
                  <div>Hash Algorithm: SHA-256</div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default SettingsPanel;
