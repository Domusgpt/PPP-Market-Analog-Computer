/**
 * PPP v3 Status Panel Component
 *
 * Displays the status of the PPP v3 verified reasoning system.
 *
 * Created: 2026-01-09
 */

import type { V3ReasoningState } from '../hooks/useVerifiedReasoning';

// ============================================================================
// Types
// ============================================================================

export interface V3StatusPanelProps {
  state: V3ReasoningState;
  onValidate?: () => void;
  onExport?: () => void;
  compact?: boolean;
}

// ============================================================================
// Component
// ============================================================================

export function V3StatusPanel({
  state,
  onValidate,
  onExport,
  compact = false,
}: V3StatusPanelProps) {
  const {
    initialized,
    sessionActive,
    session,
    verification,
    grounding,
    recentSteps,
    conclusion,
    error,
    loading,
  } = state;

  if (compact) {
    return (
      <div style={styles.compactContainer}>
        <div style={styles.compactRow}>
          <StatusDot status={initialized ? 'ok' : 'error'} />
          <span style={styles.compactLabel}>v3</span>
          <StatusDot status={verification.chainValid ? 'ok' : 'error'} />
          <span style={styles.compactLabel}>Chain</span>
          <StatusDot status={grounding.semantic ? 'ok' : 'warning'} />
          <span style={styles.compactLabel}>Sem</span>
        </div>
        {sessionActive && (
          <div style={styles.compactSession}>
            Session: {session?.stepCount || 0} steps
          </div>
        )}
      </div>
    );
  }

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <span style={styles.title}>PPP v3 Status</span>
        <span style={styles.version}>3.0.0</span>
      </div>

      {/* Initialization Status */}
      <StatusRow
        label="System"
        status={initialized ? 'ok' : loading ? 'pending' : 'error'}
        value={initialized ? 'Initialized' : loading ? 'Loading...' : 'Not Ready'}
      />

      {/* Verification Status */}
      <StatusRow
        label="Chain"
        status={verification.chainValid ? 'ok' : 'error'}
        value={verification.chainValid ? 'Valid' : 'Invalid'}
      />
      <StatusRow
        label="Signatures"
        status={verification.signaturesValid ? 'ok' : 'error'}
        value={verification.signaturesValid ? 'Valid' : 'Invalid'}
      />

      {/* Grounding Status */}
      <div style={styles.section}>
        <div style={styles.sectionTitle}>Grounding</div>
        <StatusRow
          label="Embeddings"
          status={grounding.semantic ? 'ok' : 'warning'}
          value={grounding.source}
        />
        <StatusRow
          label="Concepts"
          status="info"
          value={`${grounding.conceptCount} loaded`}
        />
        {grounding.warning && (
          <div style={styles.warning}>
            ⚠️ {grounding.warning}
          </div>
        )}
      </div>

      {/* Session Status */}
      {sessionActive && session && (
        <div style={styles.section}>
          <div style={styles.sectionTitle}>Active Session</div>
          <div style={styles.sessionId}>ID: {session.id.substring(0, 20)}...</div>
          <div style={styles.sessionQuery}>Query: {session.query.substring(0, 40)}...</div>
          <div style={styles.sessionSteps}>Steps: {session.stepCount}</div>
        </div>
      )}

      {/* Recent Steps */}
      {recentSteps.length > 0 && (
        <div style={styles.section}>
          <div style={styles.sectionTitle}>Recent Steps</div>
          <div style={styles.stepList}>
            {recentSteps.slice(-5).map((step, i) => (
              <div key={i} style={styles.stepItem}>
                <span style={styles.stepNumber}>#{step.payload.stepNumber}</span>
                <span style={styles.stepOp}>{step.payload.operation}</span>
                <span style={styles.stepSigned}>✓</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Conclusion */}
      {conclusion && (
        <div style={styles.section}>
          <div style={styles.sectionTitle}>Conclusion</div>
          <div style={styles.conclusion}>
            {conclusion.payload.statement.substring(0, 100)}...
          </div>
          <div style={styles.confidence}>
            Confidence: {(conclusion.payload.confidence * 100).toFixed(1)}%
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={styles.error}>
          Error: {error}
        </div>
      )}

      {/* Actions */}
      <div style={styles.actions}>
        {onValidate && (
          <button style={styles.button} onClick={onValidate} disabled={loading}>
            Validate
          </button>
        )}
        {onExport && (
          <button style={styles.button} onClick={onExport} disabled={loading}>
            Export
          </button>
        )}
      </div>

      {/* Timestamp */}
      <div style={styles.timestamp}>
        Last checked: {new Date(verification.lastChecked).toLocaleTimeString()}
      </div>
    </div>
  );
}

// ============================================================================
// Helper Components
// ============================================================================

function StatusDot({ status }: { status: 'ok' | 'warning' | 'error' | 'pending' | 'info' }) {
  const colors = {
    ok: '#00ff88',
    warning: '#ffaa00',
    error: '#ff3366',
    pending: '#6666ff',
    info: '#00ffff',
  };

  return (
    <span
      style={{
        display: 'inline-block',
        width: '8px',
        height: '8px',
        borderRadius: '50%',
        backgroundColor: colors[status],
        marginRight: '4px',
      }}
    />
  );
}

function StatusRow({
  label,
  status,
  value,
}: {
  label: string;
  status: 'ok' | 'warning' | 'error' | 'pending' | 'info';
  value: string;
}) {
  return (
    <div style={styles.statusRow}>
      <StatusDot status={status} />
      <span style={styles.statusLabel}>{label}</span>
      <span style={styles.statusValue}>{value}</span>
    </div>
  );
}

// ============================================================================
// Styles
// ============================================================================

const styles: Record<string, React.CSSProperties> = {
  container: {
    backgroundColor: '#0d0d15',
    border: '1px solid #1a1a2e',
    borderRadius: '8px',
    padding: '12px',
    fontFamily: 'monospace',
    fontSize: '11px',
    color: '#ffffff',
    width: '240px',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '12px',
    paddingBottom: '8px',
    borderBottom: '1px solid #1a1a2e',
  },
  title: {
    fontWeight: 'bold',
    color: '#00ffff',
  },
  version: {
    color: 'rgba(255, 255, 255, 0.5)',
    fontSize: '10px',
  },
  statusRow: {
    display: 'flex',
    alignItems: 'center',
    marginBottom: '6px',
  },
  statusLabel: {
    color: 'rgba(255, 255, 255, 0.7)',
    marginRight: '8px',
    minWidth: '70px',
  },
  statusValue: {
    color: '#ffffff',
    flex: 1,
    textAlign: 'right',
  },
  section: {
    marginTop: '12px',
    paddingTop: '8px',
    borderTop: '1px solid #1a1a2e',
  },
  sectionTitle: {
    color: '#00ffff',
    fontSize: '10px',
    textTransform: 'uppercase',
    marginBottom: '8px',
  },
  warning: {
    backgroundColor: 'rgba(255, 170, 0, 0.1)',
    border: '1px solid rgba(255, 170, 0, 0.3)',
    borderRadius: '4px',
    padding: '6px 8px',
    marginTop: '8px',
    fontSize: '10px',
    color: '#ffaa00',
  },
  error: {
    backgroundColor: 'rgba(255, 51, 102, 0.1)',
    border: '1px solid rgba(255, 51, 102, 0.3)',
    borderRadius: '4px',
    padding: '6px 8px',
    marginTop: '8px',
    fontSize: '10px',
    color: '#ff3366',
  },
  sessionId: {
    color: 'rgba(255, 255, 255, 0.5)',
    fontSize: '9px',
    marginBottom: '4px',
  },
  sessionQuery: {
    color: '#ffffff',
    marginBottom: '4px',
  },
  sessionSteps: {
    color: '#00ff88',
  },
  stepList: {
    maxHeight: '100px',
    overflowY: 'auto',
  },
  stepItem: {
    display: 'flex',
    alignItems: 'center',
    padding: '4px 0',
    borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
  },
  stepNumber: {
    color: '#00ffff',
    marginRight: '8px',
    minWidth: '24px',
  },
  stepOp: {
    flex: 1,
    color: 'rgba(255, 255, 255, 0.7)',
    fontSize: '10px',
  },
  stepSigned: {
    color: '#00ff88',
  },
  conclusion: {
    color: '#ffffff',
    fontSize: '10px',
    lineHeight: '1.4',
    marginBottom: '4px',
  },
  confidence: {
    color: '#00ff88',
    fontSize: '10px',
  },
  actions: {
    display: 'flex',
    gap: '8px',
    marginTop: '12px',
  },
  button: {
    flex: 1,
    backgroundColor: '#1a1a2e',
    border: '1px solid #333366',
    borderRadius: '4px',
    padding: '6px 8px',
    color: '#00ffff',
    fontSize: '10px',
    cursor: 'pointer',
    fontFamily: 'monospace',
  },
  timestamp: {
    marginTop: '8px',
    textAlign: 'center',
    color: 'rgba(255, 255, 255, 0.3)',
    fontSize: '9px',
  },
  // Compact styles
  compactContainer: {
    backgroundColor: 'rgba(13, 13, 21, 0.9)',
    border: '1px solid #1a1a2e',
    borderRadius: '4px',
    padding: '6px 10px',
    fontFamily: 'monospace',
    fontSize: '10px',
  },
  compactRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  compactLabel: {
    color: 'rgba(255, 255, 255, 0.6)',
    marginRight: '4px',
  },
  compactSession: {
    marginTop: '4px',
    color: '#00ffff',
    fontSize: '9px',
  },
};

export default V3StatusPanel;
