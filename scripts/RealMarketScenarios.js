/**
 * Real Market Scenarios for Harmonic Alpha Testing
 *
 * Based on actual historical market data and documented events.
 * These scenarios test the Market Larynx crash detection system
 * against real-world price/sentiment divergences.
 *
 * Sources:
 * - CoinMarketCap Fear & Greed Index
 * - VIX historical data
 * - Academic research on sentiment-crash relationships
 */

/**
 * Normalize a value to 0-1 range
 */
function normalize(value, min, max) {
    return Math.max(0, Math.min(1, (value - min) / (max - min)));
}

/**
 * Convert Fear & Greed Index (0-100) to sentiment (0-1)
 * Higher F&G = more greed = higher sentiment
 */
function fearGreedToSentiment(fg) {
    return fg / 100;
}

/**
 * Convert VIX to tension indicator (inverted - high VIX = high tension)
 * VIX range typically 10-80, with spikes to 65+
 */
function vixToTension(vix) {
    // Normal VIX ~12-20, elevated 20-30, panic 30+
    return normalize(vix, 10, 65);
}

/**
 * SCENARIO 1: August 5, 2024 "Black Monday" - Japan Crash
 *
 * The largest single-day point drop in Nikkei 225 history.
 * VIX spiked from ~12 to 65 (highest since COVID-19 crash).
 *
 * Timeline:
 * - July 31: BoJ raises rates unexpectedly
 * - Aug 1-2: Markets start unwinding yen carry trade
 * - Aug 5: Crash - Nikkei falls 12.4%, VIX hits 65
 * - Aug 6-7: Partial recovery
 *
 * This tests: Rapid crash detection, gamma event triggering, recovery
 */
export const AUGUST_2024_BLACK_MONDAY = {
    name: "August 5, 2024 Black Monday (Japan Crash)",
    description: "Nikkei 225 crashed 12.4% in one day, VIX spiked to 65. Tests rapid crash detection.",
    duration: 8, // days
    updateIntervalMs: 500, // Speed up for demo

    // Data points: [day, price_normalized, sentiment_normalized, expected_regime]
    // Price: S&P 500 normalized (high ~5600, low ~5100)
    // Sentiment: Inverted VIX (high VIX = low sentiment)
    timeline: [
        // Day 0 (July 31) - Pre-crash, markets complacent
        { day: 0, price: 0.85, sentiment: 0.88, vix: 12, note: "Markets complacent, VIX low" },

        // Day 1 (Aug 1) - BoJ rate hike, first cracks
        { day: 1, price: 0.82, sentiment: 0.75, vix: 16, note: "BoJ rate hike, yen strengthens" },

        // Day 2 (Aug 2) - Carry trade unwinding begins
        { day: 2, price: 0.78, sentiment: 0.60, vix: 23, note: "Carry trade unwinding, fear rising" },

        // Day 3 (Aug 3-4 weekend) - Tension builds
        { day: 3, price: 0.75, sentiment: 0.45, vix: 29, note: "Weekend tension, futures selling" },

        // Day 4 (Aug 5) - THE CRASH - VIX spikes to 65
        { day: 4, price: 0.55, sentiment: 0.15, vix: 65, note: "BLACK MONDAY - VIX 65, Nikkei -12.4%" },

        // Day 5 (Aug 6) - Partial recovery, still volatile
        { day: 5, price: 0.62, sentiment: 0.35, vix: 38, note: "Dead cat bounce, high volatility" },

        // Day 6 (Aug 7) - Recovery continues
        { day: 6, price: 0.70, sentiment: 0.50, vix: 27, note: "Recovery phase, fear subsiding" },

        // Day 7 (Aug 8) - Stabilization
        { day: 7, price: 0.75, sentiment: 0.62, vix: 20, note: "Markets stabilizing" },
    ],

    // Expected system behavior
    expectedBehavior: {
        crashDetectionDay: 4,
        gammaEventDays: [4, 5],
        maxTension: 0.85,
        recoveryStartDay: 5
    }
};

/**
 * SCENARIO 2: Bitcoin 2022-2023 Extreme Fear to Extreme Greed
 *
 * The complete sentiment cycle from FTX collapse to new ATH.
 * Fear & Greed went from 8 (extreme fear) to 90 (extreme greed).
 *
 * Timeline:
 * - Nov 2022: FTX collapse, BTC ~$16,000, F&G = 8-12
 * - Jan 2023: Recovery begins, F&G rising
 * - Oct 2023: BTC ETF speculation, F&G ~60
 * - Mar 2024: BTC ATH $73,500, F&G = 90
 *
 * This tests: Long-term regime transitions, euphoria detection
 */
export const BITCOIN_2022_2024_CYCLE = {
    name: "Bitcoin 2022-2024: Fear to Greed Cycle",
    description: "From FTX collapse extreme fear to ATH extreme greed. Tests regime transitions.",
    duration: 16, // Compressed months
    updateIntervalMs: 800,

    timeline: [
        // Nov 2022 - FTX Collapse (Extreme Fear)
        { day: 0, price: 0.10, sentiment: 0.08, fg: 8, note: "FTX collapse, BTC $16k, Extreme Fear" },
        { day: 1, price: 0.08, sentiment: 0.12, fg: 12, note: "Capitulation continues" },

        // Dec 2022 - Bottom formation
        { day: 2, price: 0.09, sentiment: 0.15, fg: 15, note: "Bottom forming, still fearful" },
        { day: 3, price: 0.11, sentiment: 0.22, fg: 22, note: "Early recovery signs" },

        // Q1 2023 - Recovery begins
        { day: 4, price: 0.18, sentiment: 0.35, fg: 35, note: "Recovery underway" },
        { day: 5, price: 0.25, sentiment: 0.45, fg: 45, note: "Neutral territory" },

        // Q2 2023 - Steady climb
        { day: 6, price: 0.32, sentiment: 0.50, fg: 50, note: "Momentum building" },
        { day: 7, price: 0.38, sentiment: 0.52, fg: 52, note: "BTC ~$30k" },

        // Q3 2023 - ETF speculation
        { day: 8, price: 0.42, sentiment: 0.58, fg: 58, note: "ETF speculation begins" },
        { day: 9, price: 0.48, sentiment: 0.62, fg: 62, note: "Optimism growing" },

        // Q4 2023 - Pre-approval rally
        { day: 10, price: 0.55, sentiment: 0.68, fg: 68, note: "ETF approval imminent" },
        { day: 11, price: 0.62, sentiment: 0.72, fg: 72, note: "Greed territory" },

        // Jan-Feb 2024 - ETF approved
        { day: 12, price: 0.70, sentiment: 0.78, fg: 78, note: "ETF approved!" },
        { day: 13, price: 0.82, sentiment: 0.82, fg: 82, note: "Institutional inflows" },

        // March 2024 - ATH Euphoria (DIVERGENCE WARNING)
        { day: 14, price: 0.95, sentiment: 0.88, fg: 88, note: "Approaching ATH, extreme greed" },
        { day: 15, price: 1.00, sentiment: 0.90, fg: 90, note: "ATH $73,500 - EXTREME GREED TOP" },
    ],

    expectedBehavior: {
        fearBottomDay: 1,
        neutralCrossDay: 5,
        greedStartDay: 11,
        extremeGreedDay: 15,
        warningSignal: "Price/sentiment aligned at top = potential reversal"
    }
};

/**
 * SCENARIO 3: July 2024 Crypto Crash - Price/Sentiment Divergence
 *
 * Perfect example of price-sentiment divergence.
 * Price dropped 25-30% while sentiment lagged, then crashed to F&G = 29.
 *
 * Catalysts:
 * - German government selling seized BTC
 * - Mt. Gox distribution fears
 * - Risk-off sentiment
 *
 * This tests: Divergence detection, TDA void formation
 */
export const JULY_2024_CRYPTO_CRASH = {
    name: "July 2024 Crypto Crash",
    description: "BTC fell from $73k to $54k. F&G crashed to 29. Tests divergence detection.",
    duration: 12,
    updateIntervalMs: 600,

    timeline: [
        // Early July - Post-ATH correction
        { day: 0, price: 0.85, sentiment: 0.70, fg: 70, note: "Post-ATH, still greedy" },
        { day: 1, price: 0.80, sentiment: 0.68, fg: 68, note: "Light selling" },

        // DIVERGENCE FORMING - Price drops, sentiment lags
        { day: 2, price: 0.72, sentiment: 0.62, fg: 62, note: "German govt selling BTC" },
        { day: 3, price: 0.65, sentiment: 0.58, fg: 58, note: "DIVERGENCE: Price falling faster" },
        { day: 4, price: 0.58, sentiment: 0.52, fg: 52, note: "Mt. Gox fears" },

        // July 5 - Crash acceleration
        { day: 5, price: 0.48, sentiment: 0.40, fg: 40, note: "Crash accelerating" },
        { day: 6, price: 0.42, sentiment: 0.32, fg: 32, note: "BTC below $54k" },
        { day: 7, price: 0.38, sentiment: 0.29, fg: 29, note: "F&G = 29, lowest since Jan 2023" },

        // Stabilization
        { day: 8, price: 0.40, sentiment: 0.32, fg: 32, note: "Finding support" },
        { day: 9, price: 0.45, sentiment: 0.38, fg: 38, note: "Relief bounce" },
        { day: 10, price: 0.50, sentiment: 0.42, fg: 42, note: "Recovery attempt" },
        { day: 11, price: 0.52, sentiment: 0.45, fg: 45, note: "Stabilizing in fear" },
    ],

    expectedBehavior: {
        divergenceStartDay: 2,
        maxDivergence: 0.15, // Price 0.58, Sentiment 0.52 = divergence
        crashDay: 7,
        voidDetectionDays: [5, 6, 7]
    }
};

/**
 * SCENARIO 4: 2022 Crypto Winter (Extended Bear)
 *
 * 47 consecutive days of extreme fear.
 * Tests sustained high tension without immediate resolution.
 */
export const CRYPTO_WINTER_2022 = {
    name: "2022 Crypto Winter",
    description: "47 days of extreme fear. Tests sustained tension and regime persistence.",
    duration: 20,
    updateIntervalMs: 400,

    timeline: [
        // LUNA/UST Collapse (May 2022)
        { day: 0, price: 0.55, sentiment: 0.45, fg: 45, note: "Pre-LUNA collapse" },
        { day: 1, price: 0.42, sentiment: 0.28, fg: 28, note: "LUNA death spiral begins" },
        { day: 2, price: 0.30, sentiment: 0.15, fg: 15, note: "UST depeg, panic" },
        { day: 3, price: 0.25, sentiment: 0.10, fg: 10, note: "EXTREME FEAR" },

        // Summer of Fear
        { day: 4, price: 0.28, sentiment: 0.12, fg: 12, note: "Dead market" },
        { day: 5, price: 0.26, sentiment: 0.08, fg: 8, note: "Peak fear" },
        { day: 6, price: 0.27, sentiment: 0.10, fg: 10, note: "Flatline" },
        { day: 7, price: 0.25, sentiment: 0.09, fg: 9, note: "Still extreme fear" },

        // 3AC/Celsius collapse
        { day: 8, price: 0.22, sentiment: 0.07, fg: 7, note: "3AC/Celsius cascade" },
        { day: 9, price: 0.18, sentiment: 0.06, fg: 6, note: "More contagion" },
        { day: 10, price: 0.20, sentiment: 0.08, fg: 8, note: "Slight bounce" },

        // FTX Collapse (Nov 2022)
        { day: 11, price: 0.22, sentiment: 0.15, fg: 15, note: "Pre-FTX, cautious hope" },
        { day: 12, price: 0.18, sentiment: 0.10, fg: 10, note: "FTX concerns emerge" },
        { day: 13, price: 0.12, sentiment: 0.05, fg: 5, note: "FTX bank run" },
        { day: 14, price: 0.08, sentiment: 0.06, fg: 6, note: "FTX collapse confirmed" },
        { day: 15, price: 0.06, sentiment: 0.08, fg: 8, note: "BTC $15,500 bottom" },

        // Capitulation bottom
        { day: 16, price: 0.07, sentiment: 0.10, fg: 10, note: "Capitulation" },
        { day: 17, price: 0.08, sentiment: 0.12, fg: 12, note: "Finding floor" },
        { day: 18, price: 0.10, sentiment: 0.15, fg: 15, note: "Early signs of stabilization" },
        { day: 19, price: 0.12, sentiment: 0.18, fg: 18, note: "Bottom formation" },
    ],

    expectedBehavior: {
        sustainedFearDays: 15,
        multipleGammaEvents: true,
        cascadePattern: [3, 9, 14],
        finalBottom: 15
    }
};

/**
 * Scenario runner - feeds data to Harmonic Alpha system
 */
export class ScenarioRunner {
    constructor(wasmEngine, sonification = null) {
        this.engine = wasmEngine;
        this.sonification = sonification;
        this.currentScenario = null;
        this.currentStep = 0;
        this.isRunning = false;
        this.intervalId = null;
        this.results = [];
        this.onUpdate = null;
        this.onComplete = null;
        this.onGammaEvent = null;
    }

    /**
     * Load a scenario
     */
    load(scenario) {
        this.currentScenario = scenario;
        this.currentStep = 0;
        this.results = [];
        console.log(`[ScenarioRunner] Loaded: ${scenario.name}`);
    }

    /**
     * Run the scenario
     */
    async run() {
        if (!this.currentScenario || !this.engine) {
            console.error('[ScenarioRunner] No scenario or engine loaded');
            return;
        }

        // Enable market mode
        if (this.engine.enable_market_mode) {
            this.engine.enable_market_mode();
        }

        this.isRunning = true;
        const scenario = this.currentScenario;

        console.log(`[ScenarioRunner] Running: ${scenario.name}`);
        console.log(`[ScenarioRunner] ${scenario.description}`);

        return new Promise((resolve) => {
            this.intervalId = setInterval(() => {
                if (this.currentStep >= scenario.timeline.length) {
                    this.stop();
                    console.log(`[ScenarioRunner] Scenario complete`);
                    if (this.onComplete) this.onComplete(this.results);
                    resolve(this.results);
                    return;
                }

                const dataPoint = scenario.timeline[this.currentStep];
                this.processDataPoint(dataPoint);
                this.currentStep++;

            }, scenario.updateIntervalMs);
        });
    }

    /**
     * Process a single data point
     */
    processDataPoint(dataPoint) {
        // Set price and sentiment
        this.engine.set_market_price(dataPoint.price);
        this.engine.set_market_sentiment(dataPoint.sentiment);

        // Run several engine updates to allow tension to build
        for (let i = 0; i < 30; i++) {
            this.engine.update(1/60);
        }

        // Get state
        const tension = this.engine.get_market_tension?.() || 0;
        const regime = this.engine.get_market_regime?.() || 'Unknown';
        const gammaActive = this.engine.is_market_gamma_active?.() || false;
        const crashProb = this.engine.get_crash_probability?.() || 0;

        // Record result
        const result = {
            day: dataPoint.day,
            input: { price: dataPoint.price, sentiment: dataPoint.sentiment },
            output: { tension, regime, gammaActive, crashProb },
            note: dataPoint.note,
            timestamp: Date.now()
        };
        this.results.push(result);

        // Log
        const gammaStr = gammaActive ? ' [GAMMA EVENT]' : '';
        console.log(
            `Day ${dataPoint.day}: Price=${dataPoint.price.toFixed(2)} ` +
            `Sentiment=${dataPoint.sentiment.toFixed(2)} → ` +
            `Tension=${tension.toFixed(2)} Regime=${regime} CrashProb=${(crashProb*100).toFixed(0)}%${gammaStr}`
        );

        // Callbacks
        if (this.onUpdate) {
            this.onUpdate(result, this.currentStep, this.currentScenario.timeline.length);
        }

        if (gammaActive && this.onGammaEvent) {
            this.onGammaEvent(result);
        }

        // Update sonification if available
        if (this.sonification) {
            this.sonification.updateFromWasm(this.engine);
        }
    }

    /**
     * Stop the scenario
     */
    stop() {
        this.isRunning = false;
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
    }

    /**
     * Get analysis of results
     */
    analyze() {
        if (this.results.length === 0) return null;

        const maxTension = Math.max(...this.results.map(r => r.output.tension));
        const gammaEvents = this.results.filter(r => r.output.gammaActive);
        const regimeCounts = {};

        this.results.forEach(r => {
            regimeCounts[r.output.regime] = (regimeCounts[r.output.regime] || 0) + 1;
        });

        // Detect divergence points
        const divergences = this.results.filter(r =>
            Math.abs(r.input.price - r.input.sentiment) > 0.15
        );

        return {
            scenarioName: this.currentScenario?.name,
            totalSteps: this.results.length,
            maxTension,
            gammaEventCount: gammaEvents.length,
            gammaEventDays: gammaEvents.map(r => r.day),
            regimeDistribution: regimeCounts,
            divergenceDetections: divergences.length,
            crashDetected: gammaEvents.length > 0 || maxTension > 0.7
        };
    }
}

/**
 * All available scenarios
 */
export const ALL_SCENARIOS = {
    AUGUST_2024_BLACK_MONDAY,
    BITCOIN_2022_2024_CYCLE,
    JULY_2024_CRYPTO_CRASH,
    CRYPTO_WINTER_2022
};

export default ALL_SCENARIOS;
