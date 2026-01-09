// File: src/App.tsx
// Main application - Orchestrates the Geometric Cognition Engine

import { useState, useEffect, useCallback, useRef } from 'react';
import { PolytopeCanvas, ControlPanel, AuditLog } from './components';
import {
  CognitiveManifold,
  AuditChain,
  ProjectionAnimator,
  projectLattice,
  type ProjectionResult,
  type ConvexityResult,
  type LogEntryCompact,
  type ChainStatistics,
  type AuditEventType,
} from './core';

/**
 * Main App Component
 */
function App() {
  // Core engine instances (refs to persist across renders)
  const manifoldRef = useRef<CognitiveManifold | null>(null);
  const auditChainRef = useRef<AuditChain | null>(null);
  const projectionAnimatorRef = useRef<ProjectionAnimator | null>(null);

  // UI state
  const [isRunning, setIsRunning] = useState(false);
  const [autoRotate, setAutoRotate] = useState(true);
  const [canvasSize, setCanvasSize] = useState({ width: 600, height: 400 });

  // Geometric state
  const [projection, setProjection] = useState<ProjectionResult | null>(null);
  const [convexity, setConvexity] = useState<ConvexityResult | null>(null);
  const [stepCount, setStepCount] = useState(0);

  // Audit state
  const [auditEntries, setAuditEntries] = useState<LogEntryCompact[]>([]);
  const [chainHead, setChainHead] = useState('0'.repeat(64));
  const [chainStats, setChainStats] = useState<ChainStatistics | null>(null);
  const [isChainValid, setIsChainValid] = useState(true);

  // Animation frame reference
  const animationFrameRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);

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

    // Update audit log
    setAuditEntries(chain.getRecentCompact(20));
    setChainHead(chain.chainHead);
    setChainStats(chain.getStatistics());
  }, [canvasSize]);

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
      if (!manifoldRef.current) return;

      const manifold = manifoldRef.current;
      const result = manifold.inferenceStep(1.0, entropy);

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

      // If running, perform inference steps
      if (isRunning && manifoldRef.current) {
        // Throttle to ~10 steps per second
        if (Math.random() < deltaTime * 10) {
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
    if (manifoldRef.current && auditChainRef.current) {
      manifoldRef.current.reset();
      auditChainRef.current.clear();
      await auditChainRef.current.initialize(
        manifoldRef.current.getGeometricState()
      );
      await logEvent('POSITION_RESET');
      updateUIState();
    }
  }, [logEvent, updateUIState]);

  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        display: 'flex',
        flexDirection: 'row',
        backgroundColor: '#0a0a0f',
        overflow: 'hidden',
      }}
    >
      {/* Left Panel - Controls */}
      <ControlPanel
        convexity={convexity}
        isRunning={isRunning}
        autoRotate={autoRotate}
        stepCount={stepCount}
        chainStats={chainStats}
        onToggleRun={handleToggleRun}
        onToggleRotate={handleToggleRotate}
        onStep={handleStep}
        onInjectEntropy={handleInjectEntropy}
        onReset={handleReset}
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
        }}
      >
        <PolytopeCanvas
          projection={projection}
          status={convexity?.status ?? 'SAFE'}
          width={canvasSize.width}
          height={canvasSize.height}
          showGrid={true}
          showLabels={false}
          isAnimating={isRunning || autoRotate}
        />
      </div>

      {/* Right Panel - Audit Log */}
      <AuditLog
        entries={auditEntries}
        chainHead={chainHead}
        isValid={isChainValid}
      />
    </div>
  );
}

export default App;
