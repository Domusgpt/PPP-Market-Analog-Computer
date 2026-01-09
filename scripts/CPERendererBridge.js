/**
 * CPE Renderer Bridge - WebGL Integration Module
 *
 * @package @clear-seas/cpe
 * @version 1.0.0
 * @license MIT
 * @author Clear Seas Solutions LLC
 *
 * This module bridges the Chronomorphic Polytopal Engine physics to the
 * existing HypercubeRenderer visualization system. It provides:
 *
 * 1. State-to-uniform mapping (CPE state → WebGL shader uniforms)
 * 2. Telemetry integration (CPE events → PPP.sonicGeometry API)
 * 3. Visual feedback (coherence → glitch, transitions → flash)
 * 4. Animation loop management (CPE update → render frame)
 *
 * Integration Tasks:
 * - Replace SonicGeometryEngine interpolation with CPE physics output
 * - Map CPE state to shader uniforms (rotations, glitch intensity)
 * - Wire telemetry to existing PPP channels
 * - Update app.js initialization patterns
 *
 * Visual Feedback:
 * - Coherence < 0.5 → increasing visual distortion (glitch)
 * - Lattice transition → flash/pulse effect
 * - Topology violation → color shift warning
 */

// =============================================================================
// MODULE PATTERN WITH ES6 CLASS
// =============================================================================

/**
 * Configuration for the CPE-Renderer bridge.
 */
const DEFAULT_BRIDGE_CONFIG = {
    // Coherence thresholds for visual feedback
    coherenceWarningThreshold: 0.5,
    coherenceCriticalThreshold: 0.3,

    // Glitch intensity scaling
    glitchIntensityScale: 2.0,
    maxGlitchIntensity: 1.0,

    // Flash effect for transitions
    transitionFlashDuration: 200, // ms
    transitionFlashIntensity: 0.5,

    // Color shift for violations
    violationColorShift: 0.3,
    violationColorDuration: 500, // ms

    // Rotation mapping scale
    rotationScale: 1.0,
    angularVelocityScale: 0.5,

    // Update rate limiting
    minUpdateInterval: 16, // ~60fps max

    // Debug mode
    debug: false
};

/**
 * CPERendererBridge connects the CPE physics engine to WebGL visualization.
 *
 * Usage:
 * ```javascript
 * // In app.js initialization
 * const bridge = new CPERendererBridge(renderer, engine);
 * bridge.start();
 *
 * // From HDC encoder
 * const force = encoder.textToForce("reasoning about causality");
 * engine.applyForce(force);
 * // Bridge automatically updates renderer on next frame
 * ```
 */
class CPERendererBridge {
    /**
     * Create a new CPE-Renderer bridge.
     *
     * @param {Object} renderer - HypercubeRenderer or compatible renderer instance
     * @param {Object} engine - CausalReasoningEngine instance
     * @param {Object} config - Optional configuration overrides
     */
    constructor(renderer, engine, config = {}) {
        this._renderer = renderer;
        this._engine = engine;
        this._config = { ...DEFAULT_BRIDGE_CONFIG, ...config };

        // State tracking
        this._isRunning = false;
        this._lastUpdateTime = 0;
        this._animationFrameId = null;

        // Visual effect state
        this._currentGlitchIntensity = 0;
        this._transitionFlashActive = false;
        this._transitionFlashStart = 0;
        this._violationColorActive = false;
        this._violationColorStart = 0;

        // Previous state for change detection
        this._previousNearestVertex = -1;
        this._previousCoherence = 1.0;

        // Telemetry subscribers
        this._telemetryUnsubscribe = null;

        // PPP API reference (if available)
        this._pppApi = typeof PPP !== 'undefined' ? PPP : null;

        // Bind methods
        this._updateLoop = this._updateLoop.bind(this);
        this._handleTelemetry = this._handleTelemetry.bind(this);
    }

    // =========================================================================
    // LIFECYCLE
    // =========================================================================

    /**
     * Start the bridge (begins updating renderer from engine state).
     */
    start() {
        if (this._isRunning) return;
        this._isRunning = true;

        // Subscribe to engine telemetry
        if (this._engine && typeof this._engine.subscribe === 'function') {
            this._telemetryUnsubscribe = this._engine.subscribe(this._handleTelemetry);
        }

        // Start update loop
        this._lastUpdateTime = performance.now();
        this._animationFrameId = requestAnimationFrame(this._updateLoop);

        this._log('Bridge started');
    }

    /**
     * Stop the bridge.
     */
    stop() {
        if (!this._isRunning) return;
        this._isRunning = false;

        // Unsubscribe from telemetry
        if (this._telemetryUnsubscribe) {
            this._telemetryUnsubscribe();
            this._telemetryUnsubscribe = null;
        }

        // Cancel animation frame
        if (this._animationFrameId) {
            cancelAnimationFrame(this._animationFrameId);
            this._animationFrameId = null;
        }

        this._log('Bridge stopped');
    }

    /**
     * Check if bridge is running.
     */
    get isRunning() {
        return this._isRunning;
    }

    // =========================================================================
    // UPDATE LOOP
    // =========================================================================

    /**
     * Main update loop - called each animation frame.
     */
    _updateLoop(timestamp) {
        if (!this._isRunning) return;

        // Rate limiting
        const elapsed = timestamp - this._lastUpdateTime;
        if (elapsed < this._config.minUpdateInterval) {
            this._animationFrameId = requestAnimationFrame(this._updateLoop);
            return;
        }
        this._lastUpdateTime = timestamp;

        // Get current engine state
        const state = this._engine?.state;
        if (!state) {
            this._animationFrameId = requestAnimationFrame(this._updateLoop);
            return;
        }

        // Get validity information
        const convexity = this._engine.checkConvexity?.() || {
            isValid: true,
            coherence: 1.0,
            nearestVertex: 0
        };

        // Update visual effects
        this._updateVisualEffects(convexity, timestamp);

        // Map state to shader uniforms
        this._updateShaderUniforms(state, convexity);

        // Update SpinorResonanceAtlas if available
        this._updateSpinorResonanceAtlas(state, convexity);

        // Continue loop
        this._animationFrameId = requestAnimationFrame(this._updateLoop);
    }

    // =========================================================================
    // STATE-TO-UNIFORM MAPPING
    // =========================================================================

    /**
     * Map CPE state to WebGL shader uniforms.
     */
    _updateShaderUniforms(state, convexity) {
        if (!this._renderer) return;

        // Map position/orientation to 6-plane rotations
        // The 6 rotation planes correspond to bivector components
        const rotations = this._stateToRotations(state);

        // Set rotation uniforms if renderer supports it
        if (typeof this._renderer.setUniform === 'function') {
            this._renderer.setUniform('u_rotXY', rotations[0]);
            this._renderer.setUniform('u_rotXZ', rotations[1]);
            this._renderer.setUniform('u_rotXW', rotations[2]);
            this._renderer.setUniform('u_rotYZ', rotations[3]);
            this._renderer.setUniform('u_rotYW', rotations[4]);
            this._renderer.setUniform('u_rotZW', rotations[5]);

            // Set glitch intensity based on coherence
            this._renderer.setUniform('u_glitchIntensity', this._currentGlitchIntensity);

            // Set additional CPE uniforms if shader supports them
            this._renderer.setUniform('u_coherence', convexity.coherence);
            this._renderer.setUniform('u_isValid', convexity.isValid ? 1.0 : 0.0);
        }

        // Alternative: use rotation array if that's the interface
        if (typeof this._renderer.setRotations === 'function') {
            this._renderer.setRotations(rotations);
        }

        // Set data mapper uniforms if available (PPP integration)
        if (this._renderer.dataMapper && typeof this._renderer.dataMapper.setRotations === 'function') {
            this._renderer.dataMapper.setRotations(rotations);
        }
    }

    /**
     * Convert CPE state (position, orientation, angular velocity) to 6 rotation angles.
     */
    _stateToRotations(state) {
        const scale = this._config.rotationScale;
        const angScale = this._config.angularVelocityScale;

        // Extract orientation bivector (6 components)
        const orientationBivector = state.orientation?.bivector || [0, 0, 0, 0, 0, 0];

        // Extract angular velocity (6 components)
        const angularVelocity = state.angularVelocity || [0, 0, 0, 0, 0, 0];

        // Combine orientation and angular velocity into rotation angles
        // Orientation provides base angle, angular velocity adds dynamic component
        const rotations = [];
        for (let i = 0; i < 6; i++) {
            // Convert bivector component to angle (simplified: treat as angle directly)
            // In full implementation, would extract angle from rotor properly
            const baseAngle = Math.atan2(orientationBivector[i], state.orientation?.scalar || 1) * 2;
            const dynamicAngle = angularVelocity[i] * angScale;

            rotations.push((baseAngle + dynamicAngle) * scale);
        }

        return rotations;
    }

    // =========================================================================
    // VISUAL EFFECTS
    // =========================================================================

    /**
     * Update visual effects based on coherence and state changes.
     */
    _updateVisualEffects(convexity, timestamp) {
        // 1. Glitch intensity based on coherence
        this._updateGlitchIntensity(convexity.coherence);

        // 2. Transition flash when nearest vertex changes
        if (convexity.nearestVertex !== this._previousNearestVertex) {
            this._triggerTransitionFlash(timestamp);
            this._previousNearestVertex = convexity.nearestVertex;
        }

        // 3. Color shift for topology violations
        if (!convexity.isValid && !this._violationColorActive) {
            this._triggerViolationColor(timestamp);
        }

        // 4. Update active effects
        this._updateActiveEffects(timestamp);

        // Track previous coherence
        this._previousCoherence = convexity.coherence;
    }

    /**
     * Update glitch intensity based on coherence.
     * Low coherence → high glitch.
     */
    _updateGlitchIntensity(coherence) {
        const { coherenceWarningThreshold, coherenceCriticalThreshold,
                glitchIntensityScale, maxGlitchIntensity } = this._config;

        let targetGlitch = 0;

        if (coherence < coherenceCriticalThreshold) {
            // Critical: high glitch
            targetGlitch = (coherenceCriticalThreshold - coherence) /
                coherenceCriticalThreshold * glitchIntensityScale;
        } else if (coherence < coherenceWarningThreshold) {
            // Warning: medium glitch
            const range = coherenceWarningThreshold - coherenceCriticalThreshold;
            const level = (coherenceWarningThreshold - coherence) / range;
            targetGlitch = level * glitchIntensityScale * 0.5;
        }

        // Smooth transition
        this._currentGlitchIntensity = this._currentGlitchIntensity * 0.9 +
            Math.min(targetGlitch, maxGlitchIntensity) * 0.1;
    }

    /**
     * Trigger transition flash effect.
     */
    _triggerTransitionFlash(timestamp) {
        this._transitionFlashActive = true;
        this._transitionFlashStart = timestamp;

        this._log(`Lattice transition to vertex ${this._previousNearestVertex}`);
    }

    /**
     * Trigger violation color shift effect.
     */
    _triggerViolationColor(timestamp) {
        this._violationColorActive = true;
        this._violationColorStart = timestamp;

        this._log('Topology violation detected');
    }

    /**
     * Update active effects (flash, color shift decay).
     */
    _updateActiveEffects(timestamp) {
        // Update transition flash
        if (this._transitionFlashActive) {
            const elapsed = timestamp - this._transitionFlashStart;
            if (elapsed > this._config.transitionFlashDuration) {
                this._transitionFlashActive = false;
            } else {
                // Apply flash effect to glitch
                const progress = elapsed / this._config.transitionFlashDuration;
                const flashAmount = Math.sin(progress * Math.PI) *
                    this._config.transitionFlashIntensity;
                this._currentGlitchIntensity += flashAmount;
            }
        }

        // Update violation color
        if (this._violationColorActive) {
            const elapsed = timestamp - this._violationColorStart;
            if (elapsed > this._config.violationColorDuration) {
                this._violationColorActive = false;
            } else if (this._renderer && typeof this._renderer.setUniform === 'function') {
                // Apply color shift
                const progress = elapsed / this._config.violationColorDuration;
                const shift = Math.sin(progress * Math.PI) *
                    this._config.violationColorShift;
                this._renderer.setUniform('u_colorShift', shift);
            }
        }
    }

    // =========================================================================
    // TELEMETRY INTEGRATION
    // =========================================================================

    /**
     * Handle telemetry events from the engine.
     */
    _handleTelemetry(event) {
        // Forward to PPP.sonicGeometry API if available
        this._forwardToPPP(event);

        // Log specific events
        switch (event.eventType) {
            case 'STATE_UPDATE':
                // High frequency - don't log
                break;

            case 'FORCE_APPLIED':
                this._log(`Force applied: ${event.payload.force?.source}`);
                break;

            case 'TOPOLOGY_VIOLATION':
                this._log(`Topology violation at coherence ${event.payload.coherence?.toFixed(3)}`);
                break;

            case 'COHERENCE_CHANGE':
                this._log(`Coherence changed: ${event.payload.previous?.toFixed(3)} → ${event.payload.current?.toFixed(3)}`);
                break;

            case 'LATTICE_TRANSITION':
                this._log(`Lattice transition: vertex ${event.payload.from} → ${event.payload.to}`);
                break;

            case 'ENGINE_INITIALIZED':
                this._log('Engine initialized');
                break;

            case 'ENGINE_RESET':
                this._log('Engine reset');
                break;
        }
    }

    /**
     * Forward telemetry to PPP.sonicGeometry API.
     */
    _forwardToPPP(event) {
        if (!this._pppApi || !this._pppApi.sonicGeometry) return;

        // Map CPE events to sonicGeometry format
        const sonicEvent = {
            timestamp: event.timestamp,
            type: event.eventType,
            cpe: event.payload
        };

        // If PPP has event emitter, emit
        if (typeof this._pppApi.sonicGeometry.emit === 'function') {
            this._pppApi.sonicGeometry.emit('cpe:telemetry', sonicEvent);
        }

        // If PPP has analysis update, push state
        if (event.eventType === 'STATE_UPDATE' &&
            typeof this._pppApi.sonicGeometry.updateAnalysis === 'function') {
            this._pppApi.sonicGeometry.updateAnalysis({
                cpeCoherence: event.payload.coherence,
                cpePosition: event.payload.position,
                cpeValid: event.payload.isValid
            });
        }
    }

    /**
     * Update SpinorResonanceAtlas with CPE state.
     */
    _updateSpinorResonanceAtlas(state, convexity) {
        if (!this._pppApi || !this._pppApi.sonicGeometry) return;

        // Push to resonance atlas if available
        if (typeof this._pppApi.sonicGeometry.pushResonance === 'function') {
            this._pppApi.sonicGeometry.pushResonance({
                position: state.position,
                orientation: state.orientation,
                coherence: convexity.coherence,
                nearestVertex: convexity.nearestVertex,
                angularVelocity: state.angularVelocity
            });
        }
    }

    // =========================================================================
    // UTILITY METHODS
    // =========================================================================

    /**
     * Get bridge configuration.
     */
    get config() {
        return { ...this._config };
    }

    /**
     * Update configuration.
     */
    setConfig(config) {
        this._config = { ...this._config, ...config };
    }

    /**
     * Get current visual state.
     */
    getVisualState() {
        return {
            glitchIntensity: this._currentGlitchIntensity,
            transitionFlashActive: this._transitionFlashActive,
            violationColorActive: this._violationColorActive,
            previousNearestVertex: this._previousNearestVertex,
            previousCoherence: this._previousCoherence
        };
    }

    /**
     * Get bridge statistics.
     */
    getStats() {
        return {
            isRunning: this._isRunning,
            hasRenderer: !!this._renderer,
            hasEngine: !!this._engine,
            hasPPP: !!this._pppApi,
            glitchIntensity: this._currentGlitchIntensity,
            previousCoherence: this._previousCoherence
        };
    }

    /**
     * Force immediate update (bypass rate limiting).
     */
    forceUpdate() {
        if (!this._engine) return;

        const state = this._engine.state;
        const convexity = this._engine.checkConvexity?.() || {
            isValid: true,
            coherence: 1.0,
            nearestVertex: 0
        };

        this._updateVisualEffects(convexity, performance.now());
        this._updateShaderUniforms(state, convexity);
    }

    /**
     * Debug logging.
     */
    _log(message) {
        if (this._config.debug) {
            console.log(`[CPERendererBridge] ${message}`);
        }
    }
}

// =============================================================================
// FACTORY FUNCTIONS
// =============================================================================

/**
 * Create a new CPE-Renderer bridge.
 *
 * @param {Object} renderer - HypercubeRenderer instance
 * @param {Object} engine - CausalReasoningEngine instance
 * @param {Object} config - Optional configuration
 * @returns {CPERendererBridge} Bridge instance
 */
function createBridge(renderer, engine, config = {}) {
    return new CPERendererBridge(renderer, engine, config);
}

/**
 * Initialize CPE integration with existing PPP app.
 * Call this from app.js after creating renderer.
 *
 * @param {Object} options - Initialization options
 * @param {Object} options.renderer - HypercubeRenderer instance
 * @param {Object} options.CausalReasoningEngine - Engine class
 * @param {Object} options.HDCEncoder - Encoder class (optional)
 * @param {Object} options.config - Bridge configuration
 * @returns {Object} Integration object with engine, encoder, and bridge
 */
function initializeCPEIntegration(options = {}) {
    const {
        renderer,
        CausalReasoningEngine,
        HDCEncoder,
        config = {}
    } = options;

    // Create engine
    const engine = CausalReasoningEngine ? new CausalReasoningEngine() : null;

    // Create encoder
    const encoder = HDCEncoder ? new HDCEncoder() : null;

    // Create bridge
    const bridge = engine && renderer
        ? new CPERendererBridge(renderer, engine, config)
        : null;

    // Wire up if all components present
    if (bridge) {
        bridge.start();
    }

    // Expose to PPP API if available
    if (typeof PPP !== 'undefined') {
        PPP.cpe = {
            engine,
            encoder,
            bridge,
            applyText: (text) => {
                if (encoder && engine) {
                    const force = encoder.textToForce(text);
                    engine.applyForce(force);
                    return force;
                }
                return null;
            },
            applyEmbedding: (embedding) => {
                if (encoder && engine) {
                    const force = encoder.embeddingToForce(embedding);
                    engine.applyForce(force);
                    return force;
                }
                return null;
            },
            getState: () => engine?.state,
            getCoherence: () => engine?.checkConvexity?.()?.coherence,
            reset: (position) => engine?.reset(position)
        };
    }

    return { engine, encoder, bridge };
}

// =============================================================================
// EXPORTS
// =============================================================================

// ES module exports
export {
    CPERendererBridge,
    createBridge,
    initializeCPEIntegration,
    DEFAULT_BRIDGE_CONFIG
};

// Also expose as global for script tag usage
if (typeof window !== 'undefined') {
    window.CPERendererBridge = CPERendererBridge;
    window.createCPEBridge = createBridge;
    window.initializeCPEIntegration = initializeCPEIntegration;
}
