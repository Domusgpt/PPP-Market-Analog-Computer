/**
 * Mock WASM Engine for Testing Harmonic Alpha
 *
 * This module provides a JavaScript simulation of the Rust WASM engine,
 * allowing testing of the full Harmonic Alpha system without WASM compilation.
 *
 * The mock implements the same interface as the real WebEngine from web.rs
 */

/**
 * Musical intervals for tension mapping
 */
const INTERVALS = {
    UNISON: { ratio: 1.0, consonance: 1.0 },
    PERFECT_FIFTH: { ratio: 1.5, consonance: 0.95 },
    PERFECT_FOURTH: { ratio: 4/3, consonance: 0.9 },
    MAJOR_THIRD: { ratio: 1.25, consonance: 0.8 },
    MINOR_THIRD: { ratio: 1.2, consonance: 0.7 },
    MINOR_SECOND: { ratio: 16/15, consonance: 0.2 },
    TRITONE: { ratio: Math.SQRT2, consonance: 0.0 }
};

function getIntervalFromTension(tension) {
    const t = Math.max(0, Math.min(1, tension));
    if (t < 0.15) return INTERVALS.UNISON;
    if (t < 0.3) return INTERVALS.PERFECT_FIFTH;
    if (t < 0.45) return INTERVALS.PERFECT_FOURTH;
    if (t < 0.55) return INTERVALS.MAJOR_THIRD;
    if (t < 0.65) return INTERVALS.MINOR_THIRD;
    if (t < 0.8) return INTERVALS.MINOR_SECOND;
    return INTERVALS.TRITONE;
}

/**
 * Market regime from tension
 */
function getRegimeFromTension(tension) {
    if (tension < 0.15) return 'Bull';
    if (tension < 0.3) return 'MildBull';
    if (tension < 0.45) return 'Neutral';
    if (tension < 0.6) return 'MildBear';
    if (tension < 0.75) return 'Bear';
    if (tension < 0.9) return 'CrashRisk';
    return 'GammaEvent';
}

/**
 * Simple TDA feature detector
 */
class SimpleTDA {
    constructor() {
        this.priceHistory = [];
        this.sentimentHistory = [];
        this.maxHistory = 50;
    }

    addPoint(price, sentiment) {
        this.priceHistory.push(price);
        this.sentimentHistory.push(sentiment);
        if (this.priceHistory.length > this.maxHistory) {
            this.priceHistory.shift();
            this.sentimentHistory.shift();
        }
    }

    computeCrashProbability() {
        if (this.priceHistory.length < 10) return 0;

        // Compute volatility
        const n = this.priceHistory.length;
        const mean = this.priceHistory.reduce((a, b) => a + b, 0) / n;
        const variance = this.priceHistory.reduce((sum, p) => sum + Math.pow(p - mean, 2), 0) / n;
        const volatility = Math.sqrt(variance);

        // Compute divergence
        const recentDivergence = Math.abs(
            this.priceHistory[n - 1] - this.sentimentHistory[n - 1]
        );

        // Combine into crash probability
        return Math.min(1, volatility * 2 + recentDivergence * 0.5);
    }

    detectVoids() {
        // Count low-density regions (simplified)
        return this.priceHistory.length > 20 ? Math.floor(Math.random() * 3) : 0;
    }

    detectLoops() {
        // Count high-volatility periods (simplified)
        if (this.priceHistory.length < 10) return 0;
        let loops = 0;
        for (let i = 5; i < this.priceHistory.length; i++) {
            const window = this.priceHistory.slice(i - 5, i);
            const mean = window.reduce((a, b) => a + b, 0) / 5;
            const variance = window.reduce((sum, p) => sum + Math.pow(p - mean, 2), 0) / 5;
            if (variance > 0.05) loops++;
        }
        return Math.min(loops, 5);
    }
}

/**
 * Mock Market Larynx (mirrors Rust implementation)
 */
class MockMarketLarynx {
    constructor() {
        this.priceAlpha = 0.5;
        this.sentimentBeta = 0.5;
        this.tension = 0;
        this.tensionHistory = [];
        this.smoothingWindow = 30;
        this.gammaThreshold = 0.85;
        this.tensionDecay = 0.02;
        this.sentimentSensitivity = 1.5;
        this.baseFrequency = 110;
        this.gammaActive = false;
        this.gammaStartFrame = 0;
        this.frame = 0;
        this.tda = new SimpleTDA();
    }

    setPrice(price) {
        this.priceAlpha = Math.max(0, Math.min(1, price));
        this.tda.addPoint(this.priceAlpha, this.sentimentBeta);
    }

    setSentiment(sentiment) {
        this.sentimentBeta = Math.max(0, Math.min(1, sentiment));
    }

    setSentimentFromEmbedding(embedding) {
        if (!embedding || embedding.length === 0) return;

        // Convert embedding to scalar (simplified)
        const mean = embedding.reduce((a, b) => a + b, 0) / embedding.length;
        const variance = embedding.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / embedding.length;
        const rawSentiment = mean + Math.sqrt(variance) * 0.5;
        this.sentimentBeta = 1 / (1 + Math.exp(-rawSentiment * 2));
        this.sentimentBeta = Math.max(0, Math.min(1, this.sentimentBeta));
    }

    step(deltaTime) {
        this.frame++;

        // Calculate raw tension
        const rawTension = Math.abs(this.priceAlpha - this.sentimentBeta);
        const targetTension = Math.min(1, rawTension * this.sentimentSensitivity);

        // Smooth transition
        const decayFactor = 1 - Math.exp(-deltaTime / this.tensionDecay);
        this.tension = this.tension + (targetTension - this.tension) * decayFactor;
        this.tension = Math.max(0, Math.min(1, this.tension));

        // Record history
        this.tensionHistory.push(this.tension);
        if (this.tensionHistory.length > this.smoothingWindow) {
            this.tensionHistory.shift();
        }

        // Smoothed tension
        const smoothedTension = this.tensionHistory.reduce((a, b) => a + b, 0) / this.tensionHistory.length;

        // Crash probability from TDA
        const crashProbability = this.tda.computeCrashProbability();

        // Adjusted threshold
        const adjustedThreshold = this.gammaThreshold * (1 - crashProbability * 0.3);

        // Check for gamma event
        if (smoothedTension > adjustedThreshold && !this.gammaActive) {
            this.gammaActive = true;
            this.gammaStartFrame = this.frame;
        } else if (smoothedTension < this.gammaThreshold * 0.7 && this.gammaActive) {
            this.gammaActive = false;
        }

        // Get interval and frequencies
        const interval = getIntervalFromTension(smoothedTension);
        const base = this.baseFrequency;
        const frequencies = this.gammaActive
            ? [base, base * 4/3, base * 3/2]  // Resolution chord
            : [base, base * interval.ratio, base * interval.ratio * interval.ratio];

        return {
            tension: this.tension,
            smoothedTension,
            regime: getRegimeFromTension(smoothedTension),
            intervalRatio: interval.ratio,
            consonance: interval.consonance,
            gammaActive: this.gammaActive,
            sonificationFrequencies: frequencies,
            crashProbability,
            tdaFeatures: {
                voids: this.tda.detectVoids(),
                loops: this.tda.detectLoops()
            }
        };
    }

    reset() {
        this.priceAlpha = 0.5;
        this.sentimentBeta = 0.5;
        this.tension = 0;
        this.tensionHistory = [];
        this.gammaActive = false;
        this.frame = 0;
        this.tda = new SimpleTDA();
    }
}

/**
 * Mock WebEngine (mirrors Rust WASM interface)
 */
export class MockWebEngine {
    constructor(canvasId = 'test-canvas') {
        this.canvasId = canvasId;
        this.frameCount = 0;
        this.marketMode = false;
        this.marketLarynx = new MockMarketLarynx();
        this.lastMarketResult = null;

        // Geometry state (simplified)
        this.rotationSpeeds = [0.3, 0.2, 0.1, 0.15, 0.25, 0.05];
        this.mode = 'trinity';

        console.log('[MockWebEngine] Initialized for canvas:', canvasId);
    }

    // === Standard Engine Methods ===

    update(deltaTime) {
        this.frameCount++;

        if (this.marketMode) {
            this.lastMarketResult = this.marketLarynx.step(deltaTime);
        }
    }

    frame_count() {
        return this.frameCount;
    }

    set_mode(mode) {
        this.mode = mode;
    }

    get_mode() {
        return this.mode;
    }

    set_rotation_speeds(xy, xz, xw, yz, yw, zw) {
        this.rotationSpeeds = [xy, xz, xw, yz, yw, zw];
    }

    inject_channel(channel, value) {
        // Mock data injection
    }

    inject_channels(values) {
        // Mock batch injection
    }

    // === Market Larynx Methods ===

    enable_market_mode() {
        this.marketMode = true;
        console.log('[MockWebEngine] Market mode enabled');
    }

    disable_market_mode() {
        this.marketMode = false;
        console.log('[MockWebEngine] Market mode disabled');
    }

    is_market_mode() {
        return this.marketMode;
    }

    set_market_price(price) {
        this.marketLarynx.setPrice(price);
    }

    set_market_sentiment(sentiment) {
        this.marketLarynx.setSentiment(sentiment);
    }

    set_market_sentiment_embedding(embedding) {
        this.marketLarynx.setSentimentFromEmbedding(embedding);
    }

    get_market_tension() {
        return this.marketLarynx.tension;
    }

    is_market_gamma_active() {
        return this.marketLarynx.gammaActive;
    }

    get_market_regime() {
        const result = this.lastMarketResult || this.marketLarynx.step(1/60);
        return result.regime;
    }

    get_market_sonification_frequencies() {
        const result = this.lastMarketResult || this.marketLarynx.step(1/60);
        return result.sonificationFrequencies;
    }

    get_crash_probability() {
        const result = this.lastMarketResult || this.marketLarynx.step(1/60);
        return result.crashProbability;
    }

    get_market_state() {
        const result = this.lastMarketResult || this.marketLarynx.step(1/60);
        return JSON.stringify({
            tension: result.tension,
            smoothed_tension: result.smoothedTension,
            regime: result.regime,
            consonance: result.consonance,
            interval_ratio: result.intervalRatio,
            gamma_active: result.gammaActive,
            crash_probability: result.crashProbability,
            sonification: {
                frequencies: result.sonificationFrequencies,
                fundamental: result.sonificationFrequencies[0],
                harmonic1: result.sonificationFrequencies[1],
                harmonic2: result.sonificationFrequencies[2]
            },
            tda: {
                feature_count: result.tdaFeatures.voids + result.tdaFeatures.loops,
                voids: result.tdaFeatures.voids,
                loops: result.tdaFeatures.loops
            }
        });
    }

    configure_market_larynx(gammaThreshold, smoothingWindow, tensionDecay, sentimentSensitivity, baseFrequency) {
        this.marketLarynx.gammaThreshold = gammaThreshold;
        this.marketLarynx.smoothingWindow = smoothingWindow;
        this.marketLarynx.tensionDecay = tensionDecay;
        this.marketLarynx.sentimentSensitivity = sentimentSensitivity;
        this.marketLarynx.baseFrequency = baseFrequency;
    }

    reset_market_larynx() {
        this.marketLarynx.reset();
    }

    // === Analysis Methods (Mock) ===

    get_trinity_state() {
        return JSON.stringify({
            alpha: { level: 0.5 + Math.random() * 0.3, dominant: false },
            beta: { level: 0.5 + Math.random() * 0.3, dominant: false },
            gamma: { level: 0.3 + Math.random() * 0.2, dominant: true },
            tension: this.marketLarynx.tension,
            coherence: 1 - this.marketLarynx.tension,
            resonant: this.marketLarynx.tension < 0.3
        });
    }

    get_betti_numbers() {
        return JSON.stringify({
            b0: 1,
            b1: 24,
            b2: 24,
            b3: 1,
            euler: 0
        });
    }

    get_dialectic_distance() {
        return this.marketLarynx.tension * 2;
    }

    is_synthesis_detected() {
        return this.marketLarynx.gammaActive;
    }
}

export default MockWebEngine;
