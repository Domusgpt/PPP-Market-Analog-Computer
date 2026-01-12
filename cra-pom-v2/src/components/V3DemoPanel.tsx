/**
 * PPP v3 Demo Panel Component
 *
 * Interactive demonstration of PPP v3 verified reasoning capabilities.
 * Shows reasoning sessions, signed steps, and chain verification.
 *
 * Created: 2026-01-09
 */

import { useState, useCallback } from 'react';
import { useVerifiedReasoning } from '../hooks/useVerifiedReasoning';
import { V3StatusPanel } from './V3StatusPanel';

// ============================================================================
// Types
// ============================================================================

export interface V3DemoPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

// ============================================================================
// Component
// ============================================================================

export function V3DemoPanel({ isOpen, onClose }: V3DemoPanelProps) {
  const {
    state,
    startSession,
    endSession,
    lookupConcept,
    querySimilar,
    makeInference,
    conclude,
    addConcept,
    validateChain,
    exportChain,
    reset,
  } = useVerifiedReasoning();

  const [queryInput, setQueryInput] = useState('');
  const [conceptInput, setConceptInput] = useState('');
  const [descriptionInput, setDescriptionInput] = useState('');
  const [inferenceInput, setInferenceInput] = useState('');
  const [demoLog, setDemoLog] = useState<string[]>([]);

  const log = useCallback((message: string) => {
    setDemoLog(prev => [...prev.slice(-50), `[${new Date().toLocaleTimeString()}] ${message}`]);
  }, []);

  // Demo: Run a complete reasoning session
  const runDemo = useCallback(async () => {
    log('Starting demo reasoning session...');

    // Start session
    await startSession('What is the relationship between logic and reasoning?');
    log('Session started');

    // Add some concepts if not present
    await addConcept('logic', 'The systematic study of valid inference and reasoning');
    await addConcept('reasoning', 'The process of drawing conclusions from facts or premises');
    await addConcept('deduction', 'Reasoning from general principles to specific conclusions');
    log('Concepts added');

    // Look up concepts
    const step1 = await lookupConcept('logic');
    if (step1) log(`Step 1: ${step1.payload.operation} - ${step1.payload.outputs[0]?.substring(0, 50)}...`);

    const step2 = await lookupConcept('reasoning');
    if (step2) log(`Step 2: ${step2.payload.operation} - ${step2.payload.outputs[0]?.substring(0, 50)}...`);

    // Query similar
    const step3 = await querySimilar('thinking process', 3);
    if (step3) log(`Step 3: ${step3.payload.operation} - Found ${step3.payload.outputs.length} similar`);

    // Make inference
    const step4 = await makeInference(
      ['Logic is the study of valid inference', 'Reasoning draws conclusions from premises'],
      'Logic provides the rules that govern valid reasoning',
      0.85
    );
    if (step4) log(`Step 4: ${step4.payload.operation} - Confidence: ${step4.payload.confidence}`);

    // Conclude
    const conclusion = await conclude(
      'Logic and reasoning are fundamentally interconnected - logic provides the formal framework for valid reasoning.',
      0.8,
      ['Demo analysis only', 'Simplified for demonstration']
    );
    if (conclusion) log(`Conclusion: ${conclusion.payload.statement.substring(0, 60)}...`);

    // End session
    const result = await endSession();
    if (result) {
      log(`Session ended - Chain valid: ${result.verification.chainValid}`);
      log(`Total steps: ${result.session.steps.length}`);
    }

    log('Demo complete!');
  }, [startSession, addConcept, lookupConcept, querySimilar, makeInference, conclude, endSession, log]);

  // Manual operations
  const handleStartSession = useCallback(async () => {
    if (!queryInput.trim()) return;
    await startSession(queryInput);
    log(`Session started: ${queryInput}`);
    setQueryInput('');
  }, [queryInput, startSession, log]);

  const handleAddConcept = useCallback(async () => {
    if (!conceptInput.trim() || !descriptionInput.trim()) return;
    await addConcept(conceptInput, descriptionInput);
    log(`Concept added: ${conceptInput}`);
    setConceptInput('');
    setDescriptionInput('');
  }, [conceptInput, descriptionInput, addConcept, log]);

  const handleLookup = useCallback(async () => {
    if (!conceptInput.trim()) return;
    const step = await lookupConcept(conceptInput);
    if (step) log(`Lookup: ${step.payload.outputs[0]}`);
  }, [conceptInput, lookupConcept, log]);

  const handleQuerySimilar = useCallback(async () => {
    if (!conceptInput.trim()) return;
    const step = await querySimilar(conceptInput, 5);
    if (step) log(`Similar: ${step.payload.outputs.join(', ')}`);
  }, [conceptInput, querySimilar, log]);

  const handleInference = useCallback(async () => {
    if (!inferenceInput.trim()) return;
    const step = await makeInference(
      ['Premise from demonstration'],
      inferenceInput,
      0.7
    );
    if (step) log(`Inference: ${step.payload.outputs[0]}`);
    setInferenceInput('');
  }, [inferenceInput, makeInference, log]);

  const handleConclude = useCallback(async () => {
    if (!inferenceInput.trim()) return;
    const conclusion = await conclude(inferenceInput, 0.75);
    if (conclusion) log(`Conclusion reached: ${conclusion.payload.statement}`);
    setInferenceInput('');
  }, [inferenceInput, conclude, log]);

  const handleEndSession = useCallback(async () => {
    const result = await endSession();
    if (result) {
      log(`Session ended - Valid: ${result.verification.chainValid}, Steps: ${result.session.steps.length}`);
    }
  }, [endSession, log]);

  const handleValidate = useCallback(async () => {
    const valid = await validateChain();
    log(`Chain validation: ${valid ? 'PASSED' : 'FAILED'}`);
  }, [validateChain, log]);

  const handleExport = useCallback(async () => {
    const chain = await exportChain();
    if (chain) {
      const json = JSON.stringify(chain, null, 2);
      const blob = new Blob([json], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `ppp-v3-chain-${Date.now()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      log('Chain exported to file');
    }
  }, [exportChain, log]);

  const handleReset = useCallback(() => {
    reset();
    setDemoLog([]);
    log('System reset');
  }, [reset, log]);

  if (!isOpen) return null;

  return (
    <div style={styles.overlay} onClick={onClose}>
      <div style={styles.panel} onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div style={styles.header}>
          <h2 style={styles.title}>PPP v3 Interactive Demo</h2>
          <button style={styles.closeButton} onClick={onClose}>×</button>
        </div>

        <div style={styles.content}>
          {/* Left: Controls */}
          <div style={styles.controlsColumn}>
            {/* Quick Demo */}
            <div style={styles.section}>
              <h3 style={styles.sectionTitle}>Quick Demo</h3>
              <button
                style={styles.demoButton}
                onClick={runDemo}
                disabled={state.loading || state.sessionActive}
              >
                ▶ Run Full Demo
              </button>
              <p style={styles.hint}>
                Runs a complete reasoning session with concept lookups,
                similarity queries, inference, and conclusion.
              </p>
            </div>

            {/* Session Control */}
            <div style={styles.section}>
              <h3 style={styles.sectionTitle}>Session Control</h3>
              <div style={styles.inputGroup}>
                <input
                  style={styles.input}
                  type="text"
                  placeholder="Enter query..."
                  value={queryInput}
                  onChange={e => setQueryInput(e.target.value)}
                  disabled={state.sessionActive}
                />
                <button
                  style={styles.button}
                  onClick={handleStartSession}
                  disabled={state.sessionActive || !queryInput.trim()}
                >
                  Start
                </button>
              </div>
              <button
                style={styles.button}
                onClick={handleEndSession}
                disabled={!state.sessionActive}
              >
                End Session
              </button>
            </div>

            {/* Concept Operations */}
            <div style={styles.section}>
              <h3 style={styles.sectionTitle}>Concepts</h3>
              <div style={styles.inputGroup}>
                <input
                  style={styles.input}
                  type="text"
                  placeholder="Concept name..."
                  value={conceptInput}
                  onChange={e => setConceptInput(e.target.value)}
                />
              </div>
              <div style={styles.inputGroup}>
                <input
                  style={styles.input}
                  type="text"
                  placeholder="Description..."
                  value={descriptionInput}
                  onChange={e => setDescriptionInput(e.target.value)}
                />
              </div>
              <div style={styles.buttonRow}>
                <button style={styles.smallButton} onClick={handleAddConcept}>Add</button>
                <button
                  style={styles.smallButton}
                  onClick={handleLookup}
                  disabled={!state.sessionActive}
                >
                  Lookup
                </button>
                <button
                  style={styles.smallButton}
                  onClick={handleQuerySimilar}
                  disabled={!state.sessionActive}
                >
                  Similar
                </button>
              </div>
            </div>

            {/* Reasoning */}
            <div style={styles.section}>
              <h3 style={styles.sectionTitle}>Reasoning</h3>
              <div style={styles.inputGroup}>
                <input
                  style={styles.input}
                  type="text"
                  placeholder="Inference or conclusion..."
                  value={inferenceInput}
                  onChange={e => setInferenceInput(e.target.value)}
                />
              </div>
              <div style={styles.buttonRow}>
                <button
                  style={styles.smallButton}
                  onClick={handleInference}
                  disabled={!state.sessionActive}
                >
                  Infer
                </button>
                <button
                  style={styles.smallButton}
                  onClick={handleConclude}
                  disabled={!state.sessionActive}
                >
                  Conclude
                </button>
              </div>
            </div>

            {/* Actions */}
            <div style={styles.section}>
              <h3 style={styles.sectionTitle}>Verification</h3>
              <div style={styles.buttonRow}>
                <button style={styles.smallButton} onClick={handleValidate}>Validate</button>
                <button style={styles.smallButton} onClick={handleExport}>Export</button>
                <button style={styles.smallButton} onClick={handleReset}>Reset</button>
              </div>
            </div>
          </div>

          {/* Center: Log */}
          <div style={styles.logColumn}>
            <h3 style={styles.sectionTitle}>Activity Log</h3>
            <div style={styles.logContainer}>
              {demoLog.length === 0 ? (
                <div style={styles.logEmpty}>
                  No activity yet. Click "Run Full Demo" to start.
                </div>
              ) : (
                demoLog.map((entry, i) => (
                  <div key={i} style={styles.logEntry}>{entry}</div>
                ))
              )}
            </div>
          </div>

          {/* Right: Status */}
          <div style={styles.statusColumn}>
            <V3StatusPanel
              state={state}
              onValidate={handleValidate}
              onExport={handleExport}
            />

            {/* Important Notes */}
            <div style={styles.notes}>
              <h4 style={styles.notesTitle}>Important Notes</h4>
              <ul style={styles.notesList}>
                <li>Every operation is cryptographically signed</li>
                <li>The audit chain can be exported and verified externally</li>
                <li>⚠️ This does NOT prove correctness or prevent hallucination</li>
                <li>⚠️ Using fallback embeddings (non-semantic)</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Styles
// ============================================================================

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed',
    inset: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.85)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 2000,
  },
  panel: {
    backgroundColor: '#0a0a0f',
    border: '1px solid #333366',
    borderRadius: '12px',
    width: '90vw',
    maxWidth: '1200px',
    maxHeight: '85vh',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '16px 20px',
    borderBottom: '1px solid #1a1a2e',
    backgroundColor: '#0d0d15',
  },
  title: {
    margin: 0,
    fontSize: '18px',
    fontWeight: 'bold',
    color: '#00ffff',
    fontFamily: 'monospace',
  },
  closeButton: {
    background: 'none',
    border: 'none',
    color: 'rgba(255, 255, 255, 0.6)',
    fontSize: '24px',
    cursor: 'pointer',
    padding: '0 8px',
  },
  content: {
    display: 'flex',
    flex: 1,
    overflow: 'hidden',
  },
  controlsColumn: {
    width: '280px',
    borderRight: '1px solid #1a1a2e',
    padding: '16px',
    overflowY: 'auto',
  },
  logColumn: {
    flex: 1,
    padding: '16px',
    display: 'flex',
    flexDirection: 'column',
  },
  statusColumn: {
    width: '280px',
    borderLeft: '1px solid #1a1a2e',
    padding: '16px',
    overflowY: 'auto',
  },
  section: {
    marginBottom: '20px',
  },
  sectionTitle: {
    margin: '0 0 12px 0',
    fontSize: '12px',
    fontWeight: 'bold',
    color: '#00ffff',
    textTransform: 'uppercase',
    fontFamily: 'monospace',
  },
  demoButton: {
    width: '100%',
    padding: '12px',
    backgroundColor: '#00ffff',
    color: '#0a0a0f',
    border: 'none',
    borderRadius: '6px',
    fontSize: '14px',
    fontWeight: 'bold',
    cursor: 'pointer',
    fontFamily: 'monospace',
  },
  hint: {
    marginTop: '8px',
    fontSize: '10px',
    color: 'rgba(255, 255, 255, 0.5)',
    lineHeight: '1.4',
  },
  inputGroup: {
    marginBottom: '8px',
    display: 'flex',
    gap: '8px',
  },
  input: {
    flex: 1,
    padding: '8px 12px',
    backgroundColor: '#1a1a2e',
    border: '1px solid #333366',
    borderRadius: '4px',
    color: '#ffffff',
    fontSize: '12px',
    fontFamily: 'monospace',
  },
  button: {
    padding: '8px 16px',
    backgroundColor: '#1a1a2e',
    border: '1px solid #333366',
    borderRadius: '4px',
    color: '#00ffff',
    fontSize: '12px',
    cursor: 'pointer',
    fontFamily: 'monospace',
  },
  buttonRow: {
    display: 'flex',
    gap: '8px',
  },
  smallButton: {
    flex: 1,
    padding: '6px 8px',
    backgroundColor: '#1a1a2e',
    border: '1px solid #333366',
    borderRadius: '4px',
    color: '#00ffff',
    fontSize: '10px',
    cursor: 'pointer',
    fontFamily: 'monospace',
  },
  logContainer: {
    flex: 1,
    backgroundColor: '#0d0d15',
    border: '1px solid #1a1a2e',
    borderRadius: '8px',
    padding: '12px',
    overflowY: 'auto',
    fontFamily: 'monospace',
    fontSize: '11px',
  },
  logEmpty: {
    color: 'rgba(255, 255, 255, 0.3)',
    textAlign: 'center',
    padding: '20px',
  },
  logEntry: {
    color: 'rgba(255, 255, 255, 0.8)',
    padding: '4px 0',
    borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
  },
  notes: {
    marginTop: '16px',
    backgroundColor: 'rgba(255, 170, 0, 0.05)',
    border: '1px solid rgba(255, 170, 0, 0.2)',
    borderRadius: '8px',
    padding: '12px',
  },
  notesTitle: {
    margin: '0 0 8px 0',
    fontSize: '11px',
    fontWeight: 'bold',
    color: '#ffaa00',
    fontFamily: 'monospace',
  },
  notesList: {
    margin: 0,
    padding: '0 0 0 16px',
    fontSize: '10px',
    color: 'rgba(255, 255, 255, 0.7)',
    lineHeight: '1.6',
  },
};

export default V3DemoPanel;
