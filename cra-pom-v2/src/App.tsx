// File: src/App.tsx
// Main application - Orchestrates the Geometric Cognition Engine

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  PolytopeCanvas,
  ControlPanel,
  AuditLog,
  SettingsPanel,
  Sparkline,
  ChatPanel,
  ChatToggleButton,
  V3DemoPanel,
  V3StatusPanel,
  type TrajectoryPoint2D,
  type TrajectoryMode,
  type SparklineDataPoint,
} from './components';
import {
  CognitiveManifold,
  Quaternion,
  AuditChain,
  ProjectionAnimator,
  TrajectoryHistory,
  projectLattice,
  projectQuaternionTo2D,
  computeCoherenceMetrics,
  PHI,
  type ProjectionResult,
  type ConvexityResult,
  type LogEntryCompact,
  type ChainStatistics,
  type AuditEventType,
  type CoherenceMetrics,
} from './core';
import {
  DemoLLMService,
  LLMService,
  buildGeometricContext,
  type LLMProviderConfig,
  type ChatMessage,
  type SimulationCommand,
} from './services';
import {
  useSessionSettings,
  useTouchControls,
  clearAllStoredData,
  DEFAULT_SESSION_SETTINGS,
  useVerifiedReasoning,
} from './hooks';

/**
 * Keyboard shortcuts configuration
 */
const KEYBOARD_SHORTCUTS = {
  ' ': 'Toggle run/pause',
  s: 'Single step',
  r: 'Toggle auto-rotate',
  e: 'Inject entropy',
  x: 'Reset system',
  t: 'Toggle trajectory',
  g: 'Toggle grid',
  d: 'Download data',
  ',': 'Open settings',
  c: 'Toggle chat',
  v: 'PPP v3 Demo',
  '?': 'Show shortcuts',
};

/**
 * Main App Component
 */
function App() {
  // Persistent settings
  const [settings, setSettings, clearSettings] = useSessionSettings();

  // Core engine instances
  const manifoldRef = useRef<CognitiveManifold | null>(null);
  const auditChainRef = useRef<AuditChain | null>(null);
  const projectionAnimatorRef = useRef<ProjectionAnimator | null>(null);
  const trajectoryRef = useRef<TrajectoryHistory | null>(null);
  const canvasContainerRef = useRef<HTMLDivElement>(null);

  // UI state
  const [isRunning, setIsRunning] = useState(false);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [trajectoryMode, setTrajectoryMode] = useState<TrajectoryMode>('free');
  const [canvasSize, setCanvasSize] = useState({ width: 600, height: 400 });

  // Geometric state
  const [projection, setProjection] = useState<ProjectionResult | null>(null);
  const [convexity, setConvexity] = useState<ConvexityResult | null>(null);
  const [stepCount, setStepCount] = useState(0);
  const [trajectoryPoints, setTrajectoryPoints] = useState<TrajectoryPoint2D[]>([]);
  const [coherenceMetrics, setCoherenceMetrics] = useState<CoherenceMetrics | null>(null);
  const [coherenceHistory, setCoherenceHistory] = useState<SparklineDataPoint[]>([]);

  // Audit state
  const [auditEntries, setAuditEntries] = useState<LogEntryCompact[]>([]);
  const [chainHead, setChainHead] = useState('0'.repeat(64));
  const [chainStats, setChainStats] = useState<ChainStatistics | null>(null);
  const [isChainValid, setIsChainValid] = useState(true);

  // Animation state
  const animationFrameRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);
  const stepAccumulatorRef = useRef<number>(0);
  const modeStepCountRef = useRef<number>(0);

  // LLM Chat state
  const llmServiceRef = useRef<LLMService | DemoLLMService | null>(null);
  const [showChat, setShowChat] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [llmConfig, setLlmConfig] = useState<LLMProviderConfig>({
    provider: 'custom',
    model: 'demo',
    baseUrl: '',
  });

  // PPP v3 state
  const [showV3Demo, setShowV3Demo] = useState(false);
  const v3Reasoning = useVerifiedReasoning();

  /**
   * Touch controls for canvas rotation
   */
  useTouchControls(canvasContainerRef, {
    onRotate: (deltaX, deltaY) => {
      if (projectionAnimatorRef.current) {
        const animator = projectionAnimatorRef.current;
        animator.setTarget({
          rotationY: animator.params.rotationY + deltaX,
          rotationX: animator.params.rotationX + deltaY,
        });
      }
    },
    onZoom: (delta) => {
      if (projectionAnimatorRef.current) {
        const animator = projectionAnimatorRef.current;
        const newDistance = Math.max(2, Math.min(10, animator.params.cameraDistance - delta * 10));
        animator.setTarget({ cameraDistance: newDistance });
      }
    },
    onTap: () => {
      // Single tap = single step
      performStep();
    },
  });

  /**
   * Apply trajectory mode modifiers
   */
  const getTrajectoryModifiers = useCallback((): {
    entropy?: [number, number, number, number];
    delta: number;
  } => {
    const step = modeStepCountRef.current;

    switch (trajectoryMode) {
      case 'spiral': {
        // Spiral outward with increasing delta
        const spiralFactor = 1 + (step % 100) * 0.01;
        return { delta: spiralFactor };
      }
      case 'boundary': {
        // Stay near boundary by adding outward entropy
        if (convexity && convexity.penetrationDepth < -0.3) {
          const outward: [number, number, number, number] = [0.2, 0.2, 0.2, 0.2];
          return { entropy: outward, delta: 1 };
        }
        return { delta: 0.5 };
      }
      case 'ergodic': {
        // Periodic entropy injection for coverage
        if (step % 20 === 0) {
          const randomEntropy: [number, number, number, number] = [
            (Math.random() - 0.5) * 0.5,
            (Math.random() - 0.5) * 0.5,
            (Math.random() - 0.5) * 0.5,
            (Math.random() - 0.5) * 0.5,
          ];
          return { entropy: randomEntropy, delta: 1 };
        }
        return { delta: 1 };
      }
      case 'oscillate': {
        // Oscillate with sine wave modulation
        const oscillation = Math.sin(step * PHI * 0.1);
        return { delta: 0.5 + oscillation * 0.5 };
      }
      default:
        return { delta: 1 };
    }
  }, [trajectoryMode, convexity]);

  /**
   * Initialize the engine
   */
  const initializeEngine = useCallback(async () => {
    manifoldRef.current = new CognitiveManifold();
    auditChainRef.current = new AuditChain();
    projectionAnimatorRef.current = new ProjectionAnimator({
      ...DEFAULT_SESSION_SETTINGS,
      rotationX: 0.4,
      rotationY: 0.3,
      rotationZ: 0,
    });
    trajectoryRef.current = new TrajectoryHistory(settings.trajectoryLength);

    const initialState = manifoldRef.current.getGeometricState();
    await auditChainRef.current.initialize(initialState);

    updateUIState();
  }, [settings.trajectoryLength]);

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

    // Update projection with current settings
    animator.setTarget({
      projectionW: settings.projectionW,
      cameraDistance: settings.cameraDistance,
      fov: settings.fov,
    });

    const newProjection = projectLattice(
      manifold.lattice,
      manifold.position,
      canvasSize.width,
      canvasSize.height,
      animator.params
    );
    setProjection(newProjection);

    // Update trajectory points for visualization
    if (trajectory && settings.showTrajectory) {
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
    } else {
      setTrajectoryPoints([]);
    }

    // Update coherence metrics
    const geoState = manifold.getGeometricState();
    const metrics = computeCoherenceMetrics(
      manifold.position,
      new Quaternion(geoState.leftRotor.w, geoState.leftRotor.x, geoState.leftRotor.y, geoState.leftRotor.z),
      new Quaternion(geoState.rightRotor.w, geoState.rightRotor.x, geoState.rightRotor.y, geoState.rightRotor.z),
      trajectory ?? undefined
    );
    setCoherenceMetrics(metrics);

    // Update coherence history for sparkline
    setCoherenceHistory((prev) => {
      const newHistory = [
        ...prev.slice(-99),
        { value: metrics.overallCoherence, timestamp: Date.now() },
      ];
      return newHistory;
    });

    // Update audit log
    setAuditEntries(chain.getRecentCompact(20));
    setChainHead(chain.chainHead);
    setChainStats(chain.getStatistics());
  }, [canvasSize, settings]);

  /**
   * Log an event to the audit chain
   */
  const logEvent = useCallback(
    async (eventType: AuditEventType, metadata?: Record<string, unknown>) => {
      if (!manifoldRef.current || !auditChainRef.current) return;

      const state = manifoldRef.current.getGeometricState();
      await auditChainRef.current.logEvent(state, eventType, metadata);

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

      // Get mode modifiers
      const modifiers = getTrajectoryModifiers();
      const effectiveEntropy = entropy ?? modifiers.entropy;

      modeStepCountRef.current++;
      const result = manifold.inferenceStep(modifiers.delta, effectiveEntropy);

      // Add to trajectory
      trajectory.addPoint(
        manifold.position,
        manifold.stepCount,
        result.convexityAfter.status,
        effectiveEntropy !== undefined
      );

      // Determine event type
      let eventType: AuditEventType = 'INFERENCE_STEP';
      if (effectiveEntropy) {
        eventType = 'ENTROPY_INJECT';
      } else if (result.convexityAfter.status === 'VIOLATION') {
        eventType = 'CONVEXITY_VIOLATION';
      } else if (result.convexityAfter.status === 'WARNING') {
        eventType = 'CONVEXITY_WARNING';
      }

      await logEvent(eventType, { delta: modifiers.delta, mode: trajectoryMode });

      // Auto-constrain if violation
      if (result.convexityAfter.status === 'VIOLATION') {
        manifold.constrainToSafe();
        await logEvent('CONSTRAINT_APPLIED');
      }

      updateUIState();
    },
    [logEvent, updateUIState, getTrajectoryModifiers, trajectoryMode]
  );

  /**
   * Animation loop
   */
  const animate = useCallback(
    (timestamp: number) => {
      const deltaTime = (timestamp - lastTimeRef.current) / 1000;
      lastTimeRef.current = timestamp;

      if (projectionAnimatorRef.current) {
        if (settings.autoRotate && !isRunning) {
          projectionAnimatorRef.current.autoRotate(deltaTime, 0.3);
        }
        projectionAnimatorRef.current.update();
      }

      // Run inference steps
      if (isRunning && manifoldRef.current) {
        stepAccumulatorRef.current += deltaTime;
        const stepInterval = 1 / settings.stepRate;

        while (stepAccumulatorRef.current >= stepInterval) {
          stepAccumulatorRef.current -= stepInterval;
          performStep();
        }
      }

      updateUIState();
      animationFrameRef.current = requestAnimationFrame(animate);
    },
    [isRunning, settings.autoRotate, settings.stepRate, performStep, updateUIState]
  );

  // Start animation loop
  useEffect(() => {
    lastTimeRef.current = performance.now();
    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [animate]);

  // Initialize engine on mount
  useEffect(() => {
    initializeEngine();

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
    return () => window.removeEventListener('resize', handleResize);
  }, [initializeEngine]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
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
          setSettings((prev) => ({ ...prev, autoRotate: !prev.autoRotate }));
          break;
        case 'e':
          handleInjectEntropy();
          break;
        case 'x':
          handleReset();
          break;
        case 't':
          setSettings((prev) => ({ ...prev, showTrajectory: !prev.showTrajectory }));
          break;
        case 'g':
          setSettings((prev) => ({ ...prev, showGrid: !prev.showGrid }));
          break;
        case 'd':
          handleDownload();
          break;
        case ',':
          setShowSettings((prev) => !prev);
          break;
        case 'c':
          setShowChat((prev) => !prev);
          break;
        case 'v':
          setShowV3Demo((prev) => !prev);
          break;
        case '?':
          setShowShortcuts((prev) => !prev);
          break;
        case 'escape':
          setShowShortcuts(false);
          setShowSettings(false);
          setShowChat(false);
          setShowV3Demo(false);
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [performStep, setSettings]);

  // Validate chain periodically
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

  // Handlers
  const handleToggleRun = useCallback(() => setIsRunning((prev) => !prev), []);
  const handleToggleRotate = useCallback(
    () => setSettings((prev) => ({ ...prev, autoRotate: !prev.autoRotate })),
    [setSettings]
  );
  const handleStep = useCallback(() => performStep(), [performStep]);

  const handleInjectEntropy = useCallback(() => {
    const entropy: [number, number, number, number] = [
      (Math.random() - 0.5) * 2,
      (Math.random() - 0.5) * 2,
      (Math.random() - 0.5) * 2,
      (Math.random() - 0.5) * 2,
    ];
    performStep(entropy);
  }, [performStep]);

  const handleReset = useCallback(async () => {
    if (manifoldRef.current && auditChainRef.current && trajectoryRef.current) {
      manifoldRef.current.reset();
      auditChainRef.current.clear();
      trajectoryRef.current.clear();
      modeStepCountRef.current = 0;
      setCoherenceHistory([]);
      await auditChainRef.current.initialize(manifoldRef.current.getGeometricState());
      await logEvent('POSITION_RESET');
      updateUIState();
    }
  }, [logEvent, updateUIState]);

  const handleDownload = useCallback(() => {
    if (!auditChainRef.current || !trajectoryRef.current || !manifoldRef.current) return;

    const exportData = {
      timestamp: new Date().toISOString(),
      version: '2.0.0',
      settings,
      trajectoryMode,
      geometricState: manifoldRef.current.getGeometricState(),
      trajectory: trajectoryRef.current.export(),
      auditChain: JSON.parse(auditChainRef.current.exportChain()),
      coherenceMetrics,
      coherenceHistory,
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `cra-pom-v2-export-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [settings, trajectoryMode, coherenceMetrics, coherenceHistory]);

  const handleClearData = useCallback(() => {
    clearAllStoredData();
    clearSettings();
    handleReset();
  }, [clearSettings, handleReset]);

  /**
   * Initialize or update LLM service
   */
  useEffect(() => {
    if (llmConfig.model === 'demo' && llmConfig.provider === 'custom') {
      llmServiceRef.current = new DemoLLMService();
    } else {
      llmServiceRef.current = new LLMService(llmConfig);
    }
    // Sync history
    setChatMessages(llmServiceRef.current.getHistory());
  }, [llmConfig]);

  /**
   * Execute simulation command from LLM
   */
  const executeCommand = useCallback(
    async (command: SimulationCommand) => {
      switch (command.type) {
        case 'set_mode':
          setTrajectoryMode(command.payload as TrajectoryMode);
          break;
        case 'inject_entropy':
          if (command.payload === 'random') {
            handleInjectEntropy();
          } else if (Array.isArray(command.payload)) {
            performStep(command.payload as [number, number, number, number]);
          }
          break;
        case 'reset':
          await handleReset();
          break;
        case 'toggle_run':
          setIsRunning(command.payload as boolean);
          break;
        case 'step':
          await performStep();
          break;
      }
    },
    [handleInjectEntropy, handleReset, performStep]
  );

  /**
   * Send message to LLM
   */
  const handleSendMessage = useCallback(
    async (message: string) => {
      if (!llmServiceRef.current || !manifoldRef.current) return;

      setIsChatLoading(true);

      try {
        // Build geometric context
        const geoState = manifoldRef.current.getGeometricState();
        const context = buildGeometricContext(
          geoState,
          convexity,
          coherenceMetrics,
          trajectoryMode,
          stepCount
        );

        // Send to LLM
        const response = await llmServiceRef.current.sendMessage(message, context);

        // Update messages
        setChatMessages(llmServiceRef.current.getHistory());

        // Execute any commands
        if (response.commands) {
          for (const cmd of response.commands) {
            await executeCommand(cmd);
          }
        }
      } catch (error) {
        console.error('LLM error:', error);
        // Add error message to chat
        if (llmServiceRef.current) {
          llmServiceRef.current.addMessage({
            role: 'assistant',
            content: `Error: ${error instanceof Error ? error.message : 'Failed to get response'}`,
          });
          setChatMessages(llmServiceRef.current.getHistory());
        }
      } finally {
        setIsChatLoading(false);
      }
    },
    [convexity, coherenceMetrics, trajectoryMode, stepCount, executeCommand]
  );

  /**
   * Update LLM provider config
   */
  const handleLlmConfigChange = useCallback((config: Partial<LLMProviderConfig>) => {
    setLlmConfig((prev) => ({ ...prev, ...config }));
  }, []);

  /**
   * Clear chat history
   */
  const handleClearChat = useCallback(() => {
    if (llmServiceRef.current) {
      llmServiceRef.current.clearHistory();
      setChatMessages([]);
    }
  }, []);

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
        autoRotate={settings.autoRotate}
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
        ref={canvasContainerRef}
        style={{
          flex: 1,
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          backgroundColor: '#0a0a0f',
          position: 'relative',
        }}
      >
        <div style={{ flex: 1, position: 'relative' }}>
          <PolytopeCanvas
            projection={projection}
            status={convexity?.status ?? 'SAFE'}
            width={canvasSize.width}
            height={canvasSize.height}
            trajectory={trajectoryPoints}
            showGrid={settings.showGrid}
            showLabels={false}
            showTrajectory={settings.showTrajectory}
            isAnimating={isRunning || settings.autoRotate}
          />
        </div>

        {/* Coherence Sparkline */}
        <div
          style={{
            padding: '12px 16px',
            backgroundColor: '#0d0d15',
            borderTop: '1px solid #1a1a2e',
          }}
        >
          <Sparkline
            data={coherenceHistory}
            width={canvasSize.width - 32}
            height={40}
            color="#00ffff"
            label="Coherence"
            showValue={true}
            showRange={true}
          />
        </div>

        {/* Hints */}
        <div
          style={{
            position: 'absolute',
            bottom: '70px',
            right: '10px',
            fontSize: '10px',
            color: 'rgba(255, 255, 255, 0.3)',
            fontFamily: 'monospace',
            textAlign: 'right',
          }}
        >
          <div>Drag to rotate • Scroll to zoom</div>
          <div>Press ? for shortcuts • , for settings</div>
        </div>

        {/* Mode indicator */}
        {trajectoryMode !== 'free' && (
          <div
            style={{
              position: 'absolute',
              top: '10px',
              right: '10px',
              backgroundColor: 'rgba(0, 255, 255, 0.1)',
              border: '1px solid rgba(0, 255, 255, 0.3)',
              borderRadius: '4px',
              padding: '6px 10px',
              fontSize: '10px',
              color: '#00ffff',
              fontFamily: 'monospace',
              textTransform: 'uppercase',
            }}
          >
            Mode: {trajectoryMode}
          </div>
        )}
      </div>

      {/* Right Panel - Audit Log */}
      <AuditLog entries={auditEntries} chainHead={chainHead} isValid={isChainValid} />

      {/* Settings Panel */}
      <SettingsPanel
        settings={settings}
        trajectoryMode={trajectoryMode}
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
        onSettingsChange={(newSettings) => setSettings((prev) => ({ ...prev, ...newSettings }))}
        onTrajectoryModeChange={setTrajectoryMode}
        onClearData={handleClearData}
      />

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
              Press Escape to close
            </div>
          </div>
        </div>
      )}

      {/* LLM Chat Panel */}
      <ChatPanel
        messages={chatMessages}
        isLoading={isChatLoading}
        isOpen={showChat}
        provider={llmConfig}
        onSendMessage={handleSendMessage}
        onProviderChange={handleLlmConfigChange}
        onClearHistory={handleClearChat}
        onClose={() => setShowChat(false)}
      />

      {/* Chat Toggle Button */}
      <ChatToggleButton
        isOpen={showChat}
        hasUnread={false}
        onClick={() => setShowChat(true)}
      />

      {/* PPP v3 Demo Panel */}
      <V3DemoPanel isOpen={showV3Demo} onClose={() => setShowV3Demo(false)} />

      {/* PPP v3 Status Indicator */}
      {!showV3Demo && (
        <div
          style={{
            position: 'fixed',
            bottom: '60px',
            left: '10px',
            zIndex: 100,
            cursor: 'pointer',
          }}
          onClick={() => setShowV3Demo(true)}
          title="Press 'v' or click to open PPP v3 Demo"
        >
          <V3StatusPanel state={v3Reasoning.state} compact />
        </div>
      )}
    </div>
  );
}

export default App;
