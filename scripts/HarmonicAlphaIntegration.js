/**
 * Harmonic Alpha Integration Bridge
 *
 * This module integrates all components of the "Harmonic Alpha" system:
 * 1. Rust/WASM Market Larynx (cognitive engine)
 * 2. Voyage AI MCP Tools (semantic bridge)
 * 3. Market Larynx Sonification (audio)
 * 4. WebGPU Visualization (crash void rendering)
 *
 * Architecture Overview:
 * ```
 *   [Market Data Provider]
 *           │
 *           ▼
 *   [Voyage AI Embeddings] ──────────────┐
 *           │                            │
 *           ▼                            ▼
 *   [WASM Market Larynx] ◄──────────► [TDA Analysis]
 *           │                            │
 *           ├────────────────────────────┤
 *           │                            │
 *           ▼                            ▼
 *   [Audio Sonification]         [WebGPU Visualization]
 *           │                            │
 *           └────────────────────────────┘
 *                       │
 *                       ▼
 *                  [User Output]
 * ```
 */

import { MarketSentimentTools, MarketLarynxBridge } from '../src/agent/mcp/tools.js';
import { MarketLarynxSonification } from './MarketLarynxSonification.js';

/**
 * Configuration for the Harmonic Alpha system
 */
const DEFAULT_CONFIG = {
    // WASM Engine settings
    wasmPath: './rust-engine/target/wasm32-unknown-unknown/release/geometric_cognition.wasm',

    // Voyage AI settings
    voyageApiKey: null, // Must be provided or set via environment

    // Market data settings
    updateIntervalMs: 1000,
    priceNormalizationMin: 0,
    priceNormalizationMax: 100000,

    // Audio settings
    audioEnabled: true,
    audioBaseFrequency: 110,
    audioMasterVolume: 0.3,

    // Visualization settings
    visualizationEnabled: true,
    crashVoidEnabled: true,

    // Thresholds
    crashWarningThreshold: 0.7,
    gammaThreshold: 0.85
};

/**
 * Market data snapshot
 */
class MarketSnapshot {
    constructor() {
        this.price = 0;
        this.sentiment = 0.5;
        this.tension = 0;
        this.regime = 'Neutral';
        this.gammaActive = false;
        this.crashProbability = 0;
        this.timestamp = Date.now();
    }

    static fromWasmState(wasmEngine) {
        const snapshot = new MarketSnapshot();
        if (wasmEngine) {
            snapshot.tension = wasmEngine.get_market_tension?.() || 0;
            snapshot.regime = wasmEngine.get_market_regime?.() || 'Neutral';
            snapshot.gammaActive = wasmEngine.is_market_gamma_active?.() || false;
            snapshot.crashProbability = wasmEngine.get_crash_probability?.() || 0;
        }
        snapshot.timestamp = Date.now();
        return snapshot;
    }
}

/**
 * Event types emitted by the integration
 */
export const HarmonicAlphaEvents = {
    TENSION_UPDATE: 'tension_update',
    REGIME_CHANGE: 'regime_change',
    GAMMA_EVENT: 'gamma_event',
    CRASH_WARNING: 'crash_warning',
    AUDIO_STATE_CHANGE: 'audio_state_change',
    ERROR: 'error'
};

/**
 * Main Harmonic Alpha Integration class
 */
export class HarmonicAlphaIntegration {
    /**
     * @param {Object} config - Configuration options
     */
    constructor(config = {}) {
        this.config = { ...DEFAULT_CONFIG, ...config };

        // Core components
        this.wasmEngine = null;
        this.sentimentTools = null;
        this.larynxBridge = null;
        this.sonification = null;

        // State
        this.isInitialized = false;
        this.isRunning = false;
        this.currentSnapshot = new MarketSnapshot();
        this.snapshotHistory = [];
        this.maxHistoryLength = 1000;

        // Event listeners
        this.listeners = new Map();

        // Update loop
        this.updateLoopId = null;
        this.lastUpdateTime = 0;
    }

    /**
     * Initialize all components
     */
    async initialize() {
        try {
            console.log('[HarmonicAlpha] Initializing system...');

            // Initialize Voyage AI tools
            this.sentimentTools = new MarketSentimentTools({
                voyageApiKey: this.config.voyageApiKey
            });

            // Initialize sonification
            if (this.config.audioEnabled) {
                this.sonification = new MarketLarynxSonification({
                    baseFrequency: this.config.audioBaseFrequency,
                    masterVolume: this.config.audioMasterVolume,
                    onRegimeChange: (newRegime, oldRegime) => {
                        this.emit(HarmonicAlphaEvents.REGIME_CHANGE, { newRegime, oldRegime });
                    },
                    onGammaEvent: (tension, regime) => {
                        this.emit(HarmonicAlphaEvents.GAMMA_EVENT, { tension, regime });
                    }
                });
                await this.sonification.initialize();
            }

            this.isInitialized = true;
            console.log('[HarmonicAlpha] System initialized');
            return true;

        } catch (error) {
            console.error('[HarmonicAlpha] Initialization failed:', error);
            this.emit(HarmonicAlphaEvents.ERROR, { error, phase: 'initialization' });
            return false;
        }
    }

    /**
     * Connect to a WASM engine instance
     * @param {Object} wasmEngine - WebEngine WASM instance
     */
    connectWasmEngine(wasmEngine) {
        this.wasmEngine = wasmEngine;

        // Enable market mode in WASM
        if (wasmEngine && typeof wasmEngine.enable_market_mode === 'function') {
            wasmEngine.enable_market_mode();
        }

        // Create bridge
        if (this.sentimentTools) {
            this.larynxBridge = new MarketLarynxBridge(this.sentimentTools, wasmEngine);
            this.larynxBridge.initialize();
        }

        console.log('[HarmonicAlpha] WASM engine connected');
    }

    /**
     * Start the Harmonic Alpha system
     */
    async start() {
        if (!this.isInitialized) {
            await this.initialize();
        }

        if (this.isRunning) return;

        // Enable audio
        if (this.sonification) {
            await this.sonification.enable();
        }

        this.isRunning = true;
        this.lastUpdateTime = Date.now();

        console.log('[HarmonicAlpha] System started');
    }

    /**
     * Stop the Harmonic Alpha system
     */
    stop() {
        this.isRunning = false;

        if (this.updateLoopId) {
            clearInterval(this.updateLoopId);
            this.updateLoopId = null;
        }

        if (this.sonification) {
            this.sonification.disable();
        }

        if (this.larynxBridge) {
            this.larynxBridge.stopAutoUpdate();
        }

        console.log('[HarmonicAlpha] System stopped');
    }

    /**
     * Update market data
     * @param {number} price - Current price (raw value)
     * @param {string} newsText - Current market news for sentiment analysis
     */
    async updateMarketData(price, newsText) {
        if (!this.isRunning) return null;

        const normalizedPrice = this.normalizePrice(price);

        // Update WASM engine with price
        if (this.wasmEngine) {
            this.wasmEngine.set_market_price(normalizedPrice);
        }

        // Get sentiment from Voyage AI
        if (this.sentimentTools && newsText) {
            try {
                const sentimentResult = await this.sentimentTools.fetchMarketSentiment(newsText);

                // Update WASM engine with sentiment embedding
                if (this.wasmEngine && sentimentResult.embedding) {
                    this.wasmEngine.set_market_sentiment_embedding(Array.from(sentimentResult.embedding));
                }

                this.currentSnapshot.sentiment = sentimentResult.sentiment;
            } catch (error) {
                console.warn('[HarmonicAlpha] Sentiment fetch failed:', error);
            }
        }

        // Update WASM engine to process
        if (this.wasmEngine) {
            this.wasmEngine.update(1.0 / 60.0);
        }

        // Get updated state
        this.currentSnapshot = MarketSnapshot.fromWasmState(this.wasmEngine);
        this.currentSnapshot.price = normalizedPrice;

        // Record history
        this.snapshotHistory.push({ ...this.currentSnapshot });
        if (this.snapshotHistory.length > this.maxHistoryLength) {
            this.snapshotHistory.shift();
        }

        // Update sonification
        if (this.sonification) {
            this.sonification.updateFromWasm(this.wasmEngine);
        }

        // Check for warnings
        this.checkWarnings();

        // Emit update event
        this.emit(HarmonicAlphaEvents.TENSION_UPDATE, this.currentSnapshot);

        return this.currentSnapshot;
    }

    /**
     * Normalize price to 0-1 range
     */
    normalizePrice(price) {
        const { priceNormalizationMin, priceNormalizationMax } = this.config;
        const normalized = (price - priceNormalizationMin) / (priceNormalizationMax - priceNormalizationMin);
        return Math.max(0, Math.min(1, normalized));
    }

    /**
     * Check for warning conditions
     */
    checkWarnings() {
        const { crashWarningThreshold } = this.config;
        const { tension, crashProbability, gammaActive } = this.currentSnapshot;

        if (!gammaActive && (tension > crashWarningThreshold || crashProbability > 0.6)) {
            this.emit(HarmonicAlphaEvents.CRASH_WARNING, {
                tension,
                crashProbability,
                level: crashProbability > 0.8 ? 'critical' : crashProbability > 0.6 ? 'high' : 'moderate'
            });
        }
    }

    /**
     * Start automatic updates with data providers
     * @param {Function} priceProvider - Async function returning current price
     * @param {Function} newsProvider - Async function returning current news
     */
    startAutoUpdate(priceProvider, newsProvider) {
        this.stopAutoUpdate();

        this.updateLoopId = setInterval(async () => {
            try {
                const price = await priceProvider();
                const news = await newsProvider();
                await this.updateMarketData(price, news);
            } catch (error) {
                console.error('[HarmonicAlpha] Auto-update error:', error);
            }
        }, this.config.updateIntervalMs);
    }

    /**
     * Stop automatic updates
     */
    stopAutoUpdate() {
        if (this.updateLoopId) {
            clearInterval(this.updateLoopId);
            this.updateLoopId = null;
        }
    }

    /**
     * Get current system state
     */
    getState() {
        return {
            isInitialized: this.isInitialized,
            isRunning: this.isRunning,
            snapshot: { ...this.currentSnapshot },
            audio: this.sonification?.getState() || null,
            historyLength: this.snapshotHistory.length,
            config: { ...this.config, voyageApiKey: '***' } // Mask API key
        };
    }

    /**
     * Get visualization uniforms for WebGPU shader
     * @returns {Object} Uniform values for market-aware shaders
     */
    getVisualizationUniforms() {
        const snapshot = this.currentSnapshot;
        const audioState = this.sonification?.getState() || {};

        // Map regime to numeric value
        const regimeMap = {
            'Bull': 0, 'MildBull': 1, 'Neutral': 2,
            'MildBear': 3, 'Bear': 4, 'CrashRisk': 5, 'GammaEvent': 6
        };

        return {
            tension: snapshot.tension,
            consonance: audioState.interval?.consonance || (1 - snapshot.tension),
            gamma_active: snapshot.gammaActive ? 1.0 : 0.0,
            crash_probability: snapshot.crashProbability,
            freq_fundamental: audioState.frequencies?.[0] || 110,
            freq_harmonic1: audioState.frequencies?.[1] || 165,
            freq_harmonic2: audioState.frequencies?.[2] || 220,
            regime: regimeMap[snapshot.regime] || 2
        };
    }

    /**
     * Get tension history for analysis
     * @param {number} length - Number of entries to return
     */
    getTensionHistory(length = 100) {
        return this.snapshotHistory.slice(-length).map(s => ({
            tension: s.tension,
            regime: s.regime,
            timestamp: s.timestamp
        }));
    }

    /**
     * Add event listener
     */
    on(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, []);
        }
        this.listeners.get(event).push(callback);
    }

    /**
     * Remove event listener
     */
    off(event, callback) {
        const callbacks = this.listeners.get(event);
        if (callbacks) {
            const index = callbacks.indexOf(callback);
            if (index !== -1) {
                callbacks.splice(index, 1);
            }
        }
    }

    /**
     * Emit event
     */
    emit(event, data) {
        const callbacks = this.listeners.get(event) || [];
        for (const callback of callbacks) {
            try {
                callback(data);
            } catch (error) {
                console.error(`[HarmonicAlpha] Event handler error for ${event}:`, error);
            }
        }
    }

    /**
     * Dispose of all resources
     */
    dispose() {
        this.stop();

        if (this.sonification) {
            this.sonification.dispose();
            this.sonification = null;
        }

        if (this.larynxBridge) {
            this.larynxBridge.shutdown();
            this.larynxBridge = null;
        }

        if (this.wasmEngine && typeof this.wasmEngine.disable_market_mode === 'function') {
            this.wasmEngine.disable_market_mode();
        }

        this.listeners.clear();
        this.snapshotHistory = [];
        this.isInitialized = false;

        console.log('[HarmonicAlpha] System disposed');
    }
}

/**
 * Demo data providers for testing
 */
export const DemoProviders = {
    /**
     * Simulated price provider with random walk
     */
    createPriceProvider(initialPrice = 50000, volatility = 0.02) {
        let price = initialPrice;
        return async () => {
            const change = (Math.random() - 0.5) * 2 * volatility;
            price = price * (1 + change);
            return price;
        };
    },

    /**
     * Simulated news provider cycling through different sentiments
     */
    createNewsProvider() {
        const headlines = [
            // Bullish
            'Market rallies on strong earnings reports',
            'Institutional buying drives prices higher',
            'Positive regulatory news boosts confidence',
            // Bearish
            'Market tumbles on economic concerns',
            'Selling pressure increases amid uncertainty',
            'Risk-off sentiment dominates trading',
            // Neutral
            'Market consolidates after recent moves',
            'Mixed signals keep traders cautious',
            'Trading range continues as markets await data'
        ];
        let index = 0;
        return async () => {
            const headline = headlines[index % headlines.length];
            index++;
            return headline;
        };
    }
};

/**
 * Quick start helper
 */
export async function createHarmonicAlphaSystem(wasmEngine, config = {}) {
    const system = new HarmonicAlphaIntegration(config);
    await system.initialize();
    system.connectWasmEngine(wasmEngine);
    return system;
}

export default HarmonicAlphaIntegration;
