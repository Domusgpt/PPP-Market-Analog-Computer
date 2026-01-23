#!/usr/bin/env node
/**
 * Run Harmonic Alpha Prediction (Node.js version)
 *
 * This generates the actual prediction from current market data.
 */

// Mock WASM Engine (inline for Node.js)
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

function getRegimeFromTension(tension) {
    if (tension < 0.15) return 'Bull';
    if (tension < 0.3) return 'MildBull';
    if (tension < 0.45) return 'Neutral';
    if (tension < 0.6) return 'MildBear';
    if (tension < 0.75) return 'Bear';
    if (tension < 0.9) return 'CrashRisk';
    return 'GammaEvent';
}

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
        this.frame = 0;
    }

    step(deltaTime) {
        this.frame++;
        const rawTension = Math.abs(this.priceAlpha - this.sentimentBeta);
        const targetTension = Math.min(1, rawTension * this.sentimentSensitivity);
        const decayFactor = 1 - Math.exp(-deltaTime / this.tensionDecay);
        this.tension = this.tension + (targetTension - this.tension) * decayFactor;
        this.tension = Math.max(0, Math.min(1, this.tension));

        this.tensionHistory.push(this.tension);
        if (this.tensionHistory.length > this.smoothingWindow) {
            this.tensionHistory.shift();
        }

        const smoothedTension = this.tensionHistory.reduce((a, b) => a + b, 0) / this.tensionHistory.length;

        if (smoothedTension > this.gammaThreshold && !this.gammaActive) {
            this.gammaActive = true;
        } else if (smoothedTension < this.gammaThreshold * 0.7 && this.gammaActive) {
            this.gammaActive = false;
        }

        const interval = getIntervalFromTension(smoothedTension);
        const base = this.baseFrequency;
        const frequencies = this.gammaActive
            ? [base, base * 4/3, base * 3/2]
            : [base, base * interval.ratio, base * interval.ratio * interval.ratio];

        return {
            tension: this.tension,
            smoothedTension,
            regime: getRegimeFromTension(smoothedTension),
            intervalRatio: interval.ratio,
            consonance: interval.consonance,
            gammaActive: this.gammaActive,
            sonificationFrequencies: frequencies,
            crashProbability: Math.min(1, smoothedTension * 0.8 + (this.tensionHistory.length > 10 ? 0.1 : 0))
        };
    }
}

class MockWebEngine {
    constructor() {
        this.marketMode = false;
        this.marketLarynx = new MockMarketLarynx();
        this.lastMarketResult = null;
    }

    enable_market_mode() { this.marketMode = true; }

    configure_market_larynx(gammaThreshold, smoothingWindow, tensionDecay, sentimentSensitivity, baseFrequency) {
        this.marketLarynx.gammaThreshold = gammaThreshold;
        this.marketLarynx.smoothingWindow = smoothingWindow;
        this.marketLarynx.tensionDecay = tensionDecay;
        this.marketLarynx.sentimentSensitivity = sentimentSensitivity;
        this.marketLarynx.baseFrequency = baseFrequency;
    }

    set_market_price(price) { this.marketLarynx.priceAlpha = Math.max(0, Math.min(1, price)); }
    set_market_sentiment(sentiment) { this.marketLarynx.sentimentBeta = Math.max(0, Math.min(1, sentiment)); }

    update(deltaTime) {
        if (this.marketMode) {
            this.lastMarketResult = this.marketLarynx.step(deltaTime);
        }
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
                frequencies: result.sonificationFrequencies
            }
        });
    }
}

// ============ CURRENT MARKET DATA (REAL - January 23, 2026) ============

const CURRENT_MARKET_DATA = {
    timestamp: '2026-01-23T14:00:00Z',

    btc: {
        price: 89000,       // CORRECTED: $89k as reported
        change24h: -0.035,  // Down ~3.5% from recent highs
        weeklyTrend: 'declining',
        support: 87000,
        resistance: 92000
    },

    cryptoFearGreed: 44,  // Fear
    vix: 15.06,
    vixChange24h: 0.0107,
    vixPreviousDay: 14.90,
    vix52WeekLow: 13.38,
    vix52WeekHigh: 60.13
};

// ============ ANALYSIS FUNCTIONS ============

function normalizeMarketData(data) {
    const btcMin = 80000;   // Adjusted range for current market
    const btcMax = 110000;
    const priceAlpha = Math.max(0, Math.min(1, (data.btc.price - btcMin) / (btcMax - btcMin)));

    const fgNormalized = data.cryptoFearGreed / 100;
    const vixMin = 10;
    const vixMax = 40;
    const vixNormalized = 1 - Math.max(0, Math.min(1, (data.vix - vixMin) / (vixMax - vixMin)));
    const sentimentBeta = fgNormalized * 0.6 + vixNormalized * 0.4;

    return { priceAlpha, sentimentBeta, fgNormalized, vixNormalized };
}

function detectDivergence(data) {
    const vixRange = data.vix52WeekHigh - data.vix52WeekLow;
    const vixPercentile = (data.vix - data.vix52WeekLow) / vixRange;
    const fgInFear = data.cryptoFearGreed < 50;
    const vixComplacent = vixPercentile < 0.10;

    if (fgInFear && vixComplacent) {
        return {
            detected: true,
            type: 'SENTIMENT_VIX_DIVERGENCE',
            severity: (1 - vixPercentile) * 0.5,
            vixPercentile,
            interpretation: `Crypto Fear & Greed at ${data.cryptoFearGreed} (Fear) but VIX at ${data.vix} is only ${(vixPercentile * 100).toFixed(1)}% above 52-week low. Potential complacency.`,
            riskMultiplier: 1.3
        };
    }

    return {
        detected: false,
        type: 'NONE',
        severity: 0,
        vixPercentile,
        interpretation: 'No significant divergence',
        riskMultiplier: 1.0
    };
}

// ============ MAIN PREDICTION ============

console.log('='.repeat(70));
console.log('  HARMONIC ALPHA LIVE MARKET PREDICTION');
console.log('  Date: ' + new Date().toISOString());
console.log('  THIS IS A FORWARD PREDICTION, NOT BACKTESTING');
console.log('='.repeat(70));

// Step 1: Current Market Data
console.log('\n[1] CURRENT MARKET DATA (Real values from web search)');
console.log('─'.repeat(50));
console.log(`  BTC Price:           $${CURRENT_MARKET_DATA.btc.price.toLocaleString()}`);
console.log(`  BTC 24h Change:      +${(CURRENT_MARKET_DATA.btc.change24h * 100).toFixed(2)}%`);
console.log(`  Fear & Greed Index:  ${CURRENT_MARKET_DATA.cryptoFearGreed} (Fear)`);
console.log(`  VIX:                 ${CURRENT_MARKET_DATA.vix}`);
console.log(`  VIX 52-week range:   ${CURRENT_MARKET_DATA.vix52WeekLow} - ${CURRENT_MARKET_DATA.vix52WeekHigh}`);

// Step 2: Normalize
console.log('\n[2] NORMALIZATION');
console.log('─'.repeat(50));
const normalized = normalizeMarketData(CURRENT_MARKET_DATA);
console.log(`  Price Alpha (BTC position):    ${normalized.priceAlpha.toFixed(3)}`);
console.log(`  Fear/Greed normalized:         ${normalized.fgNormalized.toFixed(3)}`);
console.log(`  VIX inverse normalized:        ${normalized.vixNormalized.toFixed(3)}`);
console.log(`  Combined Sentiment Beta:       ${normalized.sentimentBeta.toFixed(3)}`);

// Step 3: Divergence Analysis
console.log('\n[3] DIVERGENCE ANALYSIS');
console.log('─'.repeat(50));
const divergence = detectDivergence(CURRENT_MARKET_DATA);
console.log(`  Divergence detected:  ${divergence.detected}`);
console.log(`  Type:                 ${divergence.type}`);
console.log(`  VIX percentile:       ${(divergence.vixPercentile * 100).toFixed(1)}%`);
console.log(`  Severity:             ${divergence.severity.toFixed(3)}`);
console.log(`  Risk multiplier:      ${divergence.riskMultiplier}x`);
if (divergence.detected) {
    console.log(`  \n  ⚠️  ${divergence.interpretation}`);
}

// Step 4: Run Engine
console.log('\n[4] HARMONIC ALPHA ENGINE');
console.log('─'.repeat(50));
const engine = new MockWebEngine();
engine.enable_market_mode();
engine.configure_market_larynx(0.85, 30, 0.02, divergence.detected ? 2.0 : 1.5, 110);
engine.set_market_price(normalized.priceAlpha);
engine.set_market_sentiment(normalized.sentimentBeta);

// Run simulation
for (let i = 0; i < 60; i++) {
    engine.update(1/60);
}

const state = JSON.parse(engine.get_market_state());
const adjustedCrashProb = Math.min(1, state.crash_probability * divergence.riskMultiplier);

console.log(`  Tension:              ${state.tension.toFixed(3)}`);
console.log(`  Smoothed Tension:     ${state.smoothed_tension.toFixed(3)}`);
console.log(`  Regime:               ${state.regime}`);
console.log(`  Consonance:           ${state.consonance.toFixed(3)}`);
console.log(`  Raw Crash Prob:       ${state.crash_probability.toFixed(3)}`);
console.log(`  Adjusted Crash Prob:  ${adjustedCrashProb.toFixed(3)}`);
console.log(`  Gamma Active:         ${state.gamma_active}`);

// Step 5: Musical Mapping
console.log('\n[5] MUSICAL MAPPING');
console.log('─'.repeat(50));
const freqs = state.sonification.frequencies;
console.log(`  Fundamental:   ${freqs[0].toFixed(1)} Hz`);
console.log(`  Harmonic 1:    ${freqs[1].toFixed(1)} Hz`);
console.log(`  Harmonic 2:    ${freqs[2].toFixed(1)} Hz`);
console.log(`  Interval:      ${state.interval_ratio.toFixed(3)}`);
const musicalInterp = state.consonance > 0.7 ? 'CONSONANT (stable, like a major chord)' :
                      state.consonance > 0.4 ? 'MIXED (tension building, like a suspended chord)' :
                      'DISSONANT (unstable, like a tritone)';
console.log(`  Interpretation: ${musicalInterp}`);

// Step 6: Final Prediction
console.log('\n' + '='.repeat(70));
console.log('  PREDICTION RESULT');
console.log('='.repeat(70));

let riskLevel, outlook, timeframe, confidence;
if (adjustedCrashProb > 0.7) {
    riskLevel = 'HIGH';
    outlook = 'BEARISH - Significant correction likely';
    timeframe = '1-5 days';
    confidence = 'High';
} else if (adjustedCrashProb > 0.5) {
    riskLevel = 'ELEVATED';
    outlook = 'CAUTIOUS - Increased downside risk';
    timeframe = '1-2 weeks';
    confidence = 'Medium-High';
} else if (adjustedCrashProb > 0.3) {
    riskLevel = 'MODERATE';
    outlook = 'NEUTRAL-CAUTIOUS - Watch for divergence resolution';
    timeframe = '1-4 weeks';
    confidence = 'Medium';
} else {
    riskLevel = 'LOW';
    outlook = 'STABLE - No immediate crash signals';
    timeframe = 'Ongoing';
    confidence = 'Medium';
}

console.log(`\n  🎯 RISK LEVEL:    ${riskLevel}`);
console.log(`  📊 OUTLOOK:       ${outlook}`);
console.log(`  ⏱️  TIMEFRAME:     ${timeframe}`);
console.log(`  🎯 CONFIDENCE:    ${confidence}`);

console.log('\n  KEY FACTORS:');
console.log(`  • Price-Sentiment Divergence: ${(Math.abs(normalized.priceAlpha - normalized.sentimentBeta) * 100).toFixed(1)}%`);
console.log(`  • VIX near 52-week low (${(divergence.vixPercentile * 100).toFixed(1)}% above bottom) = complacency risk`);
console.log(`  • Fear & Greed at ${CURRENT_MARKET_DATA.cryptoFearGreed} indicates investor caution`);
console.log(`  • Musical consonance ${state.consonance.toFixed(2)} = ${musicalInterp.split(' ')[0].toLowerCase()}`);

console.log('\n  VERIFICATION CRITERIA:');
if (riskLevel === 'HIGH' || riskLevel === 'ELEVATED') {
    console.log('  → If correct: BTC drops >3% within ' + timeframe);
    console.log('  → If wrong: BTC stays above $92,000 support');
} else {
    console.log('  → If correct: BTC stays within ±5% range for 1 week');
    console.log('  → If wrong: BTC breaks below $90,000');
}

console.log('\n  MONITOR AT:');
console.log('  • https://www.coingecko.com/en/coins/bitcoin');
console.log('  • https://alternative.me/crypto/fear-and-greed-index/');
console.log('  • https://finance.yahoo.com/quote/^VIX/');

console.log('\n' + '='.repeat(70));
console.log('  Prediction generated: ' + new Date().toISOString());
console.log('  Valid until: ' + new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString());
console.log('='.repeat(70));
