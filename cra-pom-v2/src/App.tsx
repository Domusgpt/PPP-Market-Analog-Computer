// File: src/App.tsx
// Main application - Orchestrates the Geometric Cognition Engine

import { useState, useEffect, useCallback, useRef } from 'react';
import { PolytopeCanvas, ControlPanel, AuditLog } from './components';
import type { TrajectoryPoint2D } from './components/PolytopeCanvas';
import {
  CognitiveManifold,
  Quaternion,
  AuditChain,
  ProjectionAnimator,
  TrajectoryHistory,
  projectLattice,
  projectQuaternionTo2D,
  computeCoherenceMetrics,
  type ProjectionResult,
  type ConvexityResult,
  type LogEntryCompact,
  type ChainStatistics,
  type AuditEventType,
  type CoherenceMetrics,
} from './core';

/**
 * Keyboard shortcuts configuration
 */
const KEYBOARD_SHORTCUTS = {
  ' ': 'Toggle run/pause',
  's': 'Single step',
  'r': 'Toggle auto-rotate',
  'e': 'Inject entropy',
  'x': 'Reset system',
  't': 'Toggle trajectory',
  'g': 'Toggle grid',
  'd': 'Download data',
  '?': 'Show shortcuts',
};

/**
 * Main App Component
 */
function App() {
  // Core engine instances (refs to persist across renders)
  const manifoldRef = useRef<CognitiveManifold | null>(null);
  const auditChainRef = useRef<AuditChain | null>(null);
  const projectionAnimatorRef = useRef<ProjectionAnimator | null>(null);
  const trajectoryRef = useRef<TrajectoryHistory | null>(null);

  // UI state
  const [isRunning, setIsRunning] = useState(false);
  const [autoRotate, setAutoRotate] = useState(true);
  const [showTrajectory, setShowTrajectory] = useState(true);
  const [showGrid, setShowGrid] = useState(true);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [canvasSize, setCanvasSize] = useState({ width: 600, height: 400 });

  // Geometric state
  const [projection, setProjection] = useState<ProjectionResult | null>(null);
  const [convexity, setConvexity] = useState<ConvexityResult | null>(null);
  const [stepCount, setStepCount] = useState(0);
  const [trajectoryPoints, setTrajectoryPoints] = useState<TrajectoryPoint2D[]>([]);
  const [coherenceMetrics, setCoherenceMetrics] = useState<CoherenceMetrics | null>(null);

  // Audit state
  const [auditEntries, setAuditEntries] = useState<LogEntryCompact[]>([]);
  const [chainHead, setChainHead] = useState('0'.repeat(64));
  const [chainStats, setChainStats] = useState<ChainStatistics | null>(null);
  const [isChainValid, setIsChainValid] = useState(true);

  // Animation frame reference
  const animationFrameRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);
  const stepAccumulatorRef = useRef<number>(0);

  /**
   * Initialize the engine
   */
  const initializeEngine = useCallback(async () => {
    // Create manifold
    manifoldRef.current = new CognitiveManifold();

    // Create audit chain
    auditChainRef.current = new AuditChain();

    // Create projection animator
    projectionAnimatorRef.current = new ProjectionAnimator();

    // Create trajectory history
    trajectoryRef.current = new TrajectoryHistory(200);

    // Log system init
    const initialState = manifoldRef.current.getGeometricState();
    await auditChainRef.current.initialize(initialState);

    // Update UI
    updateUIState();
  }, []);

  /**
   * Update all UI state from engine
   */
  const updateUIState = useCallback(() => {
    if (!manifoldRef.current || !auditChainRef.current || !projectionAnimatorRef.current) {
      return;
    }

    const manifold = manifoldRef.current;
    const chain = auditChainRef.current;
    const animator = projectionAnimatorRef.current;
    const trajectory = trajectoryRef.current;

    // Update convexity
    const newConvexity = manifold.checkConvexity();
    setConvexity(newConvexity);
    setStepCount(manifold.stepCount);

    // Update projection
    const newProjection = projectLattice(
      manifold.lattice,
      manifold.position,
      canvasSize.width,
      canvasSize.height,
      animator.params
    );
    setProjection(newProjection);

    // Update trajectory points for visualization
    if (trajectory && showTrajectory) {
      const recentPoints = trajectory.getRecent(100);
      const projectedTrajectory: TrajectoryPoint2D[] = recentPoints.map((p, i) => {
        const projected = projectQuaternionTo2D(
          p.position,
          animator.params,
          canvasSize.width,
          canvasSize.height,
          -1
        );
        return {
          point: projected.point2D,
          status: p.status,
          age: recentPoints.length - 1 - i,
        };
      });
      setTrajectoryPoints(projectedTrajectory);
    }

    // Update coherence metrics
    const metrics = computeCoherenceMetrics(
      manifold.position,
      Quaternion.fromArray(manifold.getGeometricState().leftRotor as unknown as [number, number, number, number]),
      Quaternion.fromArray(manifold.getGeometricState().rightRotor as unknown as [number, number, number, number]),
      trajectory ?? undefined
    );
    setCoherenceMetrics(metrics);

    // Update audit log
    setAuditEntries(chain.getRecentCompact(20));
    setChainHead(chain.chainHead);
    setChainStats(chain.getStatistics());
  }, [canvasSize, showTrajectory]);

  /**
   * Log an event to the audit chain
   */
  const logEvent = useCallback(
    async (eventType: AuditEventType, metadata?: Record<string, unknown>) => {
      if (!manifoldRef.current || !auditChainRef.current) return;

      const state = manifoldRef.current.getGeometricState();
      await auditChainRef.current.logEvent(state, eventType, metadata);

      // Update audit UI
      const chain = auditChainRef.current;
      setAuditEntries(chain.getRecentCompact(20));
      setChainHead(chain.chainHead);
      setChainStats(chain.getStatistics());
    },
    []
  );

  /**
   * Perform a single inference step
   */
  const performStep = useCallback(
    async (entropy?: [number, number, number, number]) => {
      if (!manifoldRef.current || !trajectoryRef.current) return;

      const manifold = manifoldRef.current;
      const trajectory = trajectoryRef.current;
      const result = manifold.inferenceStep(1.0, entropy);

      // Add to trajectory
      trajectory.addPoint(
        manifold.position,
        manifold.stepCount,
        result.convexityAfter.status,
        entropy !== undefined
      );

      // Determine event type based on state change
      let eventType: AuditEventType = 'INFERENCE_STEP';
      if (entropy) {
        eventType = 'ENTROPY_INJECT';
      } else if (result.convexityAfter.status === 'VIOLATION') {
        eventType = 'CONVEXITY_VIOLATION';
      } else if (result.convexityAfter.status === 'WARNING') {
        eventType = 'CONVEXITY_WARNING';
      }

      await logEvent(eventType, { delta: result.delta });

      // Auto-constrain if violation
      if (result.convexityAfter.status === 'VIOLATION') {
        manifold.constrainToSafe();
        await logEvent('CONSTRAINT_APPLIED');
      }

      updateUIState();
    },
    [logEvent, updateUIState]
  );

  /**
   * Animation loop
   */
  const animate = useCallback(
    (timestamp: number) => {
      const deltaTime = (timestamp - lastTimeRef.current) / 1000;
      lastTimeRef.current = timestamp;

      if (projectionAnimatorRef.current && autoRotate) {
        projectionAnimatorRef.current.autoRotate(deltaTime, 0.3);
      }

      if (projectionAnimatorRef.current) {
        projectionAnimatorRef.current.update();
      }

      // If running, perform inference steps at ~10 Hz
      if (isRunning && manifoldRef.current) {
        stepAccumulatorRef.current += deltaTime;
        const stepInterval = 0.1; // 10 steps per second

        while (stepAccumulatorRef.current >= stepInterval) {
          stepAccumulatorRef.current -= stepInterval;
          performStep();
        }
      }

      updateUIState();

      animationFrameRef.current = requestAnimationFrame(animate);
    },
    [isRunning, autoRotate, performStep, updateUIState]
  );

  /**
   * Start/stop animation loop
   */
  useEffect(() => {
    lastTimeRef.current = performance.now();
    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [animate]);

  /**
   * Initialize engine on mount
   */
  useEffect(() => {
    initializeEngine();

    // Handle window resize
    const handleResize = () => {
      const container = document.getElementById('canvas-container');
      if (container) {
        setCanvasSize({
          width: container.clientWidth,
          height: container.clientHeight,
        });
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [initializeEngine]);

  /**
   * Keyboard shortcuts
   */
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      switch (e.key.toLowerCase()) {
        case ' ':
          e.preventDefault();
          setIsRunning((prev) => !prev);
          break;
        case 's':
          performStep();
          break;
        case 'r':
          setAutoRotate((prev) => !prev);
          break;
        case 'e':
          handleInjectEntropy();
          break;
        case 'x':
          handleReset();
          break;
        case 't':
          setShowTrajectory((prev) => !prev);
          break;
        case 'g':
          setShowGrid((prev) => !prev);
          break;
        case 'd':
          handleDownload();
          break;
        case '?':
          setShowShortcuts((prev) => !prev);
          break;
        case 'escape':
          setShowShortcuts(false);
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [performStep]);

  /**
   * Validate chain periodically
   */
  useEffect(() => {
    const validateChain = async () => {
      if (auditChainRef.current) {
        const validation = await auditChainRef.current.validateChain();
        setIsChainValid(validation.isValid);
      }
    };

    const interval = setInterval(validateChain, 5000);
    return () => clearInterval(interval);
  }, []);

  /**
   * Toggle running state
   */
  const handleToggleRun = useCallback(() => {
    setIsRunning((prev) => !prev);
  }, []);

  /**
   * Toggle auto-rotate
   */
  const handleToggleRotate = useCallback(() => {
    setAutoRotate((prev) => !prev);
  }, []);

  /**
   * Single step
   */
  const handleStep = useCallback(() => {
    performStep();
  }, [performStep]);

  /**
   * Inject entropy
   */
  const handleInjectEntropy = useCallback(() => {
    // Generate random entropy values
    const entropy: [number, number, number, number] = [
      (Math.random() - 0.5) * 2,
      (Math.random() - 0.5) * 2,
      (Math.random() - 0.5) * 2,
      (Math.random() - 0.5) * 2,
    ];
    performStep(entropy);
  }, [performStep]);

  /**
   * Reset system
   */
  const handleReset = useCallback(async () => {
    if (manifoldRef.current && auditChainRef.current && trajectoryRef.current) {
      manifoldRef.current.reset();
      auditChainRef.current.clear();
      trajectoryRef.current.clear();
      await auditChainRef.current.initialize(manifoldRef.current.getGeometricState());
      await logEvent('POSITION_RESET');
      updateUIState();
    }
  }, [logEvent, updateUIState]);

  /**
   * Download data export
   */
  const handleDownload = useCallback(() => {
    if (!auditChainRef.current || !trajectoryRef.current || !manifoldRef.current) return;

    const exportData = {
      timestamp: new Date().toISOString(),
      version: '2.0.0',
      geometricState: manifoldRef.current.getGeometricState(),
      trajectory: trajectoryRef.current.export(),
      auditChain: JSON.parse(auditChainRef.current.exportChain()),
      coherenceMetrics,
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `cra-pom-v2-export-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [coherenceMetrics]);

  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        display: 'flex',
        flexDirection: 'row',
        backgroundColor: '#0a0a0f',
        overflow: 'hidden',
        position: 'relative',
      }}
    >
      {/* Left Panel - Controls */}
      <ControlPanel
        convexity={convexity}
        isRunning={isRunning}
        autoRotate={autoRotate}
        stepCount={stepCount}
        chainStats={chainStats}
        coherenceMetrics={coherenceMetrics}
        onToggleRun={handleToggleRun}
        onToggleRotate={handleToggleRotate}
        onStep={handleStep}
        onInjectEntropy={handleInjectEntropy}
        onReset={handleReset}
        onDownload={handleDownload}
      />

      {/* Center - Canvas */}
      <div
        id="canvas-container"
        style={{
          flex: 1,
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: '#0a0a0f',
          position: 'relative',
        }}
      >
        <PolytopeCanvas
          projection={projection}
          status={convexity?.status ?? 'SAFE'}
          width={canvasSize.width}
          height={canvasSize.height}
          trajectory={trajectoryPoints}
          showGrid={showGrid}
          showLabels={false}
          showTrajectory={showTrajectory}
          isAnimating={isRunning || autoRotate}
        />

        {/* Keyboard shortcut hint */}
        <div
          style={{
            position: 'absolute',
            bottom: '10px',
            right: '10px',
            fontSize: '10px',
            color: 'rgba(255, 255, 255, 0.3)',
            fontFamily: 'monospace',
          }}
        >
          Press ? for shortcuts
        </div>
      </div>

      {/* Right Panel - Audit Log */}
      <AuditLog entries={auditEntries} chainHead={chainHead} isValid={isChainValid} />

      {/* Keyboard shortcuts modal */}
      {showShortcuts && (
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
          onClick={() => setShowShortcuts(false)}
        >
          <div
            style={{
              backgroundColor: '#1a1a2e',
              border: '1px solid #333366',
              borderRadius: '8px',
              padding: '24px',
              maxWidth: '400px',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <h2
              style={{
                color: '#ffffff',
                fontSize: '16px',
                marginBottom: '16px',
                borderBottom: '1px solid #333366',
                paddingBottom: '8px',
              }}
            >
              Keyboard Shortcuts
            </h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {Object.entries(KEYBOARD_SHORTCUTS).map(([key, description]) => (
                <div
                  key={key}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                  }}
                >
                  <kbd
                    style={{
                      backgroundColor: '#0a0a0f',
                      border: '1px solid #333366',
                      borderRadius: '4px',
                      padding: '4px 8px',
                      color: '#00ffff',
                      fontFamily: 'monospace',
                      fontSize: '12px',
                      minWidth: '32px',
                      textAlign: 'center',
                    }}
                  >
                    {key === ' ' ? 'Space' : key}
                  </kbd>
                  <span style={{ color: 'rgba(255, 255, 255, 0.7)', fontSize: '12px' }}>
                    {description}
                  </span>
                </div>
              ))}
            </div>
            <div
              style={{
                marginTop: '16px',
                textAlign: 'center',
                color: 'rgba(255, 255, 255, 0.4)',
                fontSize: '10px',
              }}
            >
              Press Escape or click outside to close
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
