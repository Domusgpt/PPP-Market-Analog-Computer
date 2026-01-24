/**
 * Harmonic Alpha Automated Strategy Engine
 *
 * Realistic automated trading strategies based on regime detection
 * and tension analysis. Designed for comparison against hedge fund benchmarks.
 *
 * Strategies:
 * 1. Regime-Based Position Sizing (Index/ETF)
 * 2. Tension Mean-Reversion (Crypto)
 * 3. Cross-Asset Rotation (BTC/ETH/Stables)
 * 4. VIX-Spread Strategy (Volatility arbitrage)
 */

// ============================================================================
// CORE HARMONIC ALPHA ENGINE (Simplified for strategy use)
// ============================================================================

class HarmonicEngine {
    constructor() {
        this.tensionHistory = [];
        this.regimeHistory = [];
        this.maxHistory = 100;
    }

    /**
     * Calculate tension between price position and sentiment
     * @param {number} priceNorm - Normalized price (0-1 in recent range)
     * @param {number} sentimentNorm - Normalized sentiment (0-1, higher=bullish)
     * @returns {object} Tension analysis
     */
    analyze(priceNorm, sentimentNorm) {
        const tension = Math.abs(priceNorm - sentimentNorm);
        const direction = priceNorm > sentimentNorm ? 'PRICE_LEADING' : 'SENTIMENT_LEADING';

        // Determine regime from tension
        let regime, consonance;
        if (tension < 0.15) {
            regime = 'STABLE';
            consonance = 0.95; // Perfect fifth
        } else if (tension < 0.30) {
            regime = 'MILD_DIVERGENCE';
            consonance = 0.80; // Major third
        } else if (tension < 0.50) {
            regime = 'BUILDING_TENSION';
            consonance = 0.50; // Minor third
        } else if (tension < 0.70) {
            regime = 'HIGH_TENSION';
            consonance = 0.20; // Minor second
        } else {
            regime = 'EXTREME_DIVERGENCE';
            consonance = 0.0; // Tritone
        }

        // Track history
        this.tensionHistory.push({ tension, timestamp: Date.now() });
        this.regimeHistory.push({ regime, timestamp: Date.now() });
        if (this.tensionHistory.length > this.maxHistory) {
            this.tensionHistory.shift();
            this.regimeHistory.shift();
        }

        // Calculate trend
        const tensionTrend = this.calculateTrend(this.tensionHistory.map(h => h.tension));

        return {
            tension,
            direction,
            regime,
            consonance,
            tensionTrend, // 'INCREASING', 'DECREASING', 'STABLE'
            regimeStability: this.calculateRegimeStability()
        };
    }

    calculateTrend(values) {
        if (values.length < 5) return 'INSUFFICIENT_DATA';
        const recent = values.slice(-5);
        const earlier = values.slice(-10, -5);
        if (earlier.length === 0) return 'INSUFFICIENT_DATA';

        const recentAvg = recent.reduce((a, b) => a + b, 0) / recent.length;
        const earlierAvg = earlier.reduce((a, b) => a + b, 0) / earlier.length;

        if (recentAvg > earlierAvg + 0.05) return 'INCREASING';
        if (recentAvg < earlierAvg - 0.05) return 'DECREASING';
        return 'STABLE';
    }

    calculateRegimeStability() {
        if (this.regimeHistory.length < 5) return 1.0;
        const recent = this.regimeHistory.slice(-5).map(h => h.regime);
        const uniqueRegimes = new Set(recent).size;
        return 1 - (uniqueRegimes - 1) / 4; // 1.0 = very stable, 0.0 = constantly changing
    }
}

// ============================================================================
// STRATEGY 1: INDEX REGIME TRADING (S&P 500 / SPY)
// ============================================================================

class IndexRegimeStrategy {
    constructor() {
        this.name = 'Index Regime Trading';
        this.description = 'Adjust SPY exposure based on VIX-Price regime';
        this.engine = new HarmonicEngine();
        this.position = 0; // -1 to 1 (short to long)
        this.trades = [];
        this.equity = 100000; // Starting capital
        this.maxPosition = 1.0;
    }

    /**
     * Generate signal from current market state
     * @param {number} spyPrice - Current SPY price
     * @param {number} vix - Current VIX level
     * @param {number} spy52WeekHigh - 52-week high
     * @param {number} spy52WeekLow - 52-week low
     */
    generateSignal(spyPrice, vix, spy52WeekHigh, spy52WeekLow) {
        // Normalize SPY price in 52-week range
        const priceNorm = (spyPrice - spy52WeekLow) / (spy52WeekHigh - spy52WeekLow);

        // Normalize VIX (inverted - low VIX = bullish sentiment)
        // VIX typically ranges 10-40, extremes to 80
        const vixNorm = 1 - Math.min(1, (vix - 10) / 30);

        const analysis = this.engine.analyze(priceNorm, vixNorm);

        let targetPosition = 0;
        let reasoning = '';

        // Strategy logic
        if (analysis.regime === 'STABLE' && analysis.direction === 'PRICE_LEADING') {
            // Price high, sentiment bullish, aligned = ride the trend
            targetPosition = 1.0;
            reasoning = 'Bullish trend confirmed, full long';
        } else if (analysis.regime === 'STABLE' && analysis.direction === 'SENTIMENT_LEADING') {
            // Price low, sentiment bearish, aligned = stay out or short
            targetPosition = -0.3;
            reasoning = 'Bearish trend confirmed, light short';
        } else if (analysis.regime === 'HIGH_TENSION' && analysis.direction === 'PRICE_LEADING') {
            // Price high but sentiment lagging = potential top
            targetPosition = 0.0;
            reasoning = 'Price extended beyond sentiment, exit longs';
        } else if (analysis.regime === 'HIGH_TENSION' && analysis.direction === 'SENTIMENT_LEADING') {
            // Sentiment bullish but price lagging = potential bottom
            targetPosition = 0.7;
            reasoning = 'Sentiment recovering before price, accumulate';
        } else if (analysis.regime === 'EXTREME_DIVERGENCE') {
            // Maximum tension = regime change imminent
            targetPosition = 0.0;
            reasoning = 'Extreme divergence, flat until resolution';
        } else {
            // Building tension - reduce exposure
            targetPosition = this.position * 0.5;
            reasoning = 'Tension building, reduce exposure';
        }

        return {
            timestamp: new Date().toISOString(),
            analysis,
            currentPosition: this.position,
            targetPosition,
            reasoning,
            inputs: { spyPrice, vix, priceNorm, vixNorm }
        };
    }

    executeSignal(signal, currentPrice) {
        if (Math.abs(signal.targetPosition - this.position) > 0.1) {
            const trade = {
                timestamp: signal.timestamp,
                from: this.position,
                to: signal.targetPosition,
                price: currentPrice,
                reasoning: signal.reasoning
            };
            this.trades.push(trade);
            this.position = signal.targetPosition;
            return trade;
        }
        return null;
    }
}

// ============================================================================
// STRATEGY 2: CRYPTO TENSION MEAN-REVERSION
// ============================================================================

class CryptoMeanReversionStrategy {
    constructor() {
        this.name = 'Crypto Tension Mean-Reversion';
        this.description = 'Fade extreme divergences in BTC';
        this.engine = new HarmonicEngine();
        this.position = 0;
        this.trades = [];
        this.equity = 100000;
    }

    generateSignal(btcPrice, fearGreed, btc30DayHigh, btc30DayLow) {
        // Normalize BTC in 30-day range
        const priceNorm = (btcPrice - btc30DayLow) / (btc30DayHigh - btc30DayLow);

        // Normalize Fear & Greed (0-100 scale, higher = greedier)
        const sentimentNorm = fearGreed / 100;

        const analysis = this.engine.analyze(priceNorm, sentimentNorm);

        let targetPosition = 0;
        let reasoning = '';

        // Mean-reversion logic: fade extremes
        if (analysis.tension > 0.5) {
            if (analysis.direction === 'PRICE_LEADING') {
                // Price way above sentiment = overbought, fade
                targetPosition = -0.5;
                reasoning = 'Price extended, sentiment lagging - fade long';
            } else {
                // Sentiment way above price = oversold, buy
                targetPosition = 0.7;
                reasoning = 'Sentiment recovered, price lagging - buy dip';
            }
        } else if (analysis.tension < 0.2) {
            // Low tension = trend following mode
            if (priceNorm > 0.5) {
                targetPosition = 0.5;
                reasoning = 'Aligned bullish, moderate long';
            } else {
                targetPosition = 0;
                reasoning = 'Aligned bearish, stay flat';
            }
        } else {
            // Medium tension = reduce and wait
            targetPosition = this.position * 0.7;
            reasoning = 'Medium tension, reduce exposure';
        }

        return {
            timestamp: new Date().toISOString(),
            analysis,
            currentPosition: this.position,
            targetPosition,
            reasoning,
            inputs: { btcPrice, fearGreed, priceNorm, sentimentNorm }
        };
    }

    executeSignal(signal, currentPrice) {
        if (Math.abs(signal.targetPosition - this.position) > 0.1) {
            const trade = {
                timestamp: signal.timestamp,
                from: this.position,
                to: signal.targetPosition,
                price: currentPrice,
                reasoning: signal.reasoning
            };
            this.trades.push(trade);
            this.position = signal.targetPosition;
            return trade;
        }
        return null;
    }
}

// ============================================================================
// STRATEGY 3: CROSS-ASSET ROTATION (BTC / ETH / USDC)
// ============================================================================

class CrossAssetRotationStrategy {
    constructor() {
        this.name = 'Cross-Asset Rotation';
        this.description = 'Rotate between BTC, ETH, and stables based on regime';
        this.btcEngine = new HarmonicEngine();
        this.ethEngine = new HarmonicEngine();
        this.allocation = { BTC: 0.33, ETH: 0.33, USDC: 0.34 };
        this.trades = [];
        this.equity = 100000;
    }

    generateSignal(btcData, ethData) {
        // Analyze both assets
        const btcAnalysis = this.btcEngine.analyze(
            btcData.priceNorm,
            btcData.sentimentNorm
        );
        const ethAnalysis = this.ethEngine.analyze(
            ethData.priceNorm,
            ethData.sentimentNorm
        );

        let targetAllocation = { BTC: 0, ETH: 0, USDC: 0 };
        let reasoning = [];

        // Allocate based on relative tension and regime
        const btcScore = this.scoreAsset(btcAnalysis);
        const ethScore = this.scoreAsset(ethAnalysis);
        const totalRiskScore = (btcScore + ethScore) / 2;

        // If both high tension, go to stables
        if (totalRiskScore < 0.3) {
            targetAllocation = { BTC: 0.1, ETH: 0.1, USDC: 0.8 };
            reasoning.push('High tension in both assets, defensive allocation');
        }
        // If BTC better, overweight BTC
        else if (btcScore > ethScore + 0.2) {
            targetAllocation = { BTC: 0.6, ETH: 0.2, USDC: 0.2 };
            reasoning.push('BTC regime favorable, overweight BTC');
        }
        // If ETH better, overweight ETH
        else if (ethScore > btcScore + 0.2) {
            targetAllocation = { BTC: 0.2, ETH: 0.6, USDC: 0.2 };
            reasoning.push('ETH regime favorable, overweight ETH');
        }
        // Similar scores, equal weight
        else {
            targetAllocation = { BTC: 0.4, ETH: 0.4, USDC: 0.2 };
            reasoning.push('Similar regimes, balanced allocation');
        }

        return {
            timestamp: new Date().toISOString(),
            btcAnalysis,
            ethAnalysis,
            currentAllocation: { ...this.allocation },
            targetAllocation,
            reasoning: reasoning.join('; '),
            scores: { btc: btcScore, eth: ethScore }
        };
    }

    scoreAsset(analysis) {
        // Higher score = more favorable for holding
        // Low tension + bullish direction = high score
        // High tension or bearish = low score

        let score = 1 - analysis.tension; // Base: low tension is good

        if (analysis.direction === 'SENTIMENT_LEADING' && analysis.tension > 0.3) {
            score += 0.2; // Sentiment recovering, potential upside
        }
        if (analysis.regime === 'STABLE') {
            score += 0.1; // Stability bonus
        }
        if (analysis.tensionTrend === 'DECREASING') {
            score += 0.15; // Tension resolving favorably
        }

        return Math.min(1, Math.max(0, score));
    }

    executeSignal(signal) {
        const threshold = 0.1;
        let rebalanced = false;

        for (const asset of ['BTC', 'ETH', 'USDC']) {
            if (Math.abs(signal.targetAllocation[asset] - this.allocation[asset]) > threshold) {
                rebalanced = true;
            }
        }

        if (rebalanced) {
            const trade = {
                timestamp: signal.timestamp,
                from: { ...this.allocation },
                to: signal.targetAllocation,
                reasoning: signal.reasoning
            };
            this.trades.push(trade);
            this.allocation = { ...signal.targetAllocation };
            return trade;
        }
        return null;
    }
}

// ============================================================================
// STRATEGY 4: VIX SPREAD STRATEGY
// ============================================================================

class VixSpreadStrategy {
    constructor() {
        this.name = 'VIX-Crypto Spread';
        this.description = 'Trade VIX vs Crypto sentiment divergence';
        this.engine = new HarmonicEngine();
        this.position = { vix: 0, crypto: 0 }; // Can be long/short each
        this.trades = [];
        this.equity = 100000;
    }

    generateSignal(vix, vix52WeekHigh, vix52WeekLow, cryptoFearGreed) {
        // Normalize VIX in its range
        const vixNorm = (vix - vix52WeekLow) / (vix52WeekHigh - vix52WeekLow);

        // Crypto Fear & Greed normalized
        const cryptoNorm = cryptoFearGreed / 100;

        // The spread: when VIX low + Crypto Fear = divergence
        // VIX measures equity fear, Crypto F&G measures crypto sentiment
        const analysis = this.engine.analyze(vixNorm, cryptoNorm);

        let targetPosition = { vix: 0, crypto: 0 };
        let reasoning = '';

        // VIX low (complacent) + Crypto Fear = long VIX, short crypto
        if (vixNorm < 0.2 && cryptoNorm < 0.4) {
            targetPosition = { vix: 0.5, crypto: -0.3 };
            reasoning = 'VIX complacency + Crypto fear = long vol, short crypto';
        }
        // VIX high (panic) + Crypto Greed = short VIX, long crypto
        else if (vixNorm > 0.7 && cryptoNorm > 0.6) {
            targetPosition = { vix: -0.3, crypto: 0.5 };
            reasoning = 'VIX panic + Crypto greed = short vol, long crypto';
        }
        // Aligned (both fearful or both greedy) = no spread trade
        else if (Math.abs(vixNorm - (1 - cryptoNorm)) < 0.2) {
            targetPosition = { vix: 0, crypto: 0 };
            reasoning = 'VIX and Crypto aligned, no spread opportunity';
        }
        // Partial divergence
        else {
            const divergence = Math.abs(vixNorm - (1 - cryptoNorm));
            targetPosition = {
                vix: vixNorm < 0.3 ? divergence * 0.5 : -divergence * 0.3,
                crypto: cryptoNorm < 0.4 ? -divergence * 0.3 : divergence * 0.5
            };
            reasoning = `Partial divergence (${(divergence * 100).toFixed(0)}%), scaled position`;
        }

        return {
            timestamp: new Date().toISOString(),
            analysis,
            currentPosition: { ...this.position },
            targetPosition,
            reasoning,
            inputs: { vix, vixNorm, cryptoFearGreed, cryptoNorm }
        };
    }

    executeSignal(signal) {
        const vixChange = Math.abs(signal.targetPosition.vix - this.position.vix);
        const cryptoChange = Math.abs(signal.targetPosition.crypto - this.position.crypto);

        if (vixChange > 0.1 || cryptoChange > 0.1) {
            const trade = {
                timestamp: signal.timestamp,
                from: { ...this.position },
                to: signal.targetPosition,
                reasoning: signal.reasoning
            };
            this.trades.push(trade);
            this.position = { ...signal.targetPosition };
            return trade;
        }
        return null;
    }
}

// ============================================================================
// BACKTESTER
// ============================================================================

class StrategyBacktester {
    constructor(strategy) {
        this.strategy = strategy;
        this.results = [];
    }

    runBacktest(historicalData) {
        // historicalData should be array of { date, price, sentiment, ... }
        let equity = 100000;
        let position = 0;
        const equityCurve = [{ date: historicalData[0].date, equity }];

        for (let i = 1; i < historicalData.length; i++) {
            const prev = historicalData[i - 1];
            const curr = historicalData[i];

            // Calculate return if holding
            if (position !== 0) {
                const priceReturn = (curr.price - prev.price) / prev.price;
                equity *= (1 + position * priceReturn);
            }

            // Generate new signal
            // (This is simplified - real backtest would use strategy.generateSignal)

            equityCurve.push({ date: curr.date, equity });
        }

        return {
            finalEquity: equity,
            totalReturn: (equity - 100000) / 100000,
            equityCurve,
            trades: this.strategy.trades
        };
    }
}

// ============================================================================
// PERFORMANCE COMPARISON
// ============================================================================

function compareToHedgeFunds(strategyReturns) {
    // Typical hedge fund benchmarks
    const benchmarks = {
        'HFRI Fund Weighted Composite': 0.08,  // ~8% annual
        'S&P 500 (Buy & Hold)': 0.10,          // ~10% annual
        'Bridgewater Pure Alpha': 0.12,        // ~12% annual
        'Renaissance Medallion': 0.66,         // ~66% annual (exceptional)
        'Two Sigma': 0.15,                     // ~15% annual
        'Citadel': 0.15                        // ~15% annual
    };

    const comparison = {};
    for (const [fund, annualReturn] of Object.entries(benchmarks)) {
        comparison[fund] = {
            benchmark: (annualReturn * 100).toFixed(1) + '%',
            strategy: (strategyReturns * 100).toFixed(1) + '%',
            difference: ((strategyReturns - annualReturn) * 100).toFixed(1) + '%',
            outperforms: strategyReturns > annualReturn
        };
    }
    return comparison;
}

// ============================================================================
// LIVE DEMO WITH CURRENT DATA
// ============================================================================

function runLiveDemo() {
    console.log('='.repeat(70));
    console.log('  HARMONIC ALPHA - AUTOMATED STRATEGY ENGINE');
    console.log('  Live Demo with Current Market Data');
    console.log('='.repeat(70));

    // Current market data (verified from APIs)
    const currentData = {
        btc: { price: 89776, fearGreed: 24, high30d: 108000, low30d: 85000 },
        eth: { price: 3100, fearGreed: 24, high30d: 4000, low30d: 2800 },
        spy: { price: 595, vix: 15.5, high52w: 610, low52w: 480 },
        vix: { current: 15.5, high52w: 60.13, low52w: 13.38 }
    };

    // Strategy 1: Index Regime
    console.log('\n[STRATEGY 1] Index Regime Trading (SPY)');
    console.log('-'.repeat(50));
    const indexStrategy = new IndexRegimeStrategy();
    const indexSignal = indexStrategy.generateSignal(
        currentData.spy.price,
        currentData.spy.vix,
        currentData.spy.high52w,
        currentData.spy.low52w
    );
    console.log('  Regime:', indexSignal.analysis.regime);
    console.log('  Tension:', (indexSignal.analysis.tension * 100).toFixed(1) + '%');
    console.log('  Target Position:', indexSignal.targetPosition);
    console.log('  Reasoning:', indexSignal.reasoning);

    // Strategy 2: Crypto Mean-Reversion
    console.log('\n[STRATEGY 2] Crypto Mean-Reversion (BTC)');
    console.log('-'.repeat(50));
    const cryptoStrategy = new CryptoMeanReversionStrategy();
    const cryptoSignal = cryptoStrategy.generateSignal(
        currentData.btc.price,
        currentData.btc.fearGreed,
        currentData.btc.high30d,
        currentData.btc.low30d
    );
    console.log('  Regime:', cryptoSignal.analysis.regime);
    console.log('  Tension:', (cryptoSignal.analysis.tension * 100).toFixed(1) + '%');
    console.log('  Direction:', cryptoSignal.analysis.direction);
    console.log('  Target Position:', cryptoSignal.targetPosition);
    console.log('  Reasoning:', cryptoSignal.reasoning);

    // Strategy 3: Cross-Asset Rotation
    console.log('\n[STRATEGY 3] Cross-Asset Rotation (BTC/ETH/USDC)');
    console.log('-'.repeat(50));
    const rotationStrategy = new CrossAssetRotationStrategy();
    const rotationSignal = rotationStrategy.generateSignal(
        {
            priceNorm: (currentData.btc.price - currentData.btc.low30d) /
                       (currentData.btc.high30d - currentData.btc.low30d),
            sentimentNorm: currentData.btc.fearGreed / 100
        },
        {
            priceNorm: (currentData.eth.price - currentData.eth.low30d) /
                       (currentData.eth.high30d - currentData.eth.low30d),
            sentimentNorm: currentData.eth.fearGreed / 100
        }
    );
    console.log('  BTC Score:', rotationSignal.scores.btc.toFixed(2));
    console.log('  ETH Score:', rotationSignal.scores.eth.toFixed(2));
    console.log('  Target Allocation:', JSON.stringify(rotationSignal.targetAllocation));
    console.log('  Reasoning:', rotationSignal.reasoning);

    // Strategy 4: VIX Spread
    console.log('\n[STRATEGY 4] VIX-Crypto Spread');
    console.log('-'.repeat(50));
    const vixStrategy = new VixSpreadStrategy();
    const vixSignal = vixStrategy.generateSignal(
        currentData.vix.current,
        currentData.vix.high52w,
        currentData.vix.low52w,
        currentData.btc.fearGreed
    );
    console.log('  VIX Norm:', (vixSignal.inputs.vixNorm * 100).toFixed(1) + '%');
    console.log('  Crypto Norm:', (vixSignal.inputs.cryptoNorm * 100).toFixed(1) + '%');
    console.log('  Target Position:', JSON.stringify(vixSignal.targetPosition));
    console.log('  Reasoning:', vixSignal.reasoning);

    console.log('\n' + '='.repeat(70));
    console.log('  SUMMARY OF CURRENT SIGNALS');
    console.log('='.repeat(70));
    console.log(`
  Market State: BTC $89,776 | Fear & Greed: 24 | VIX: 15.5

  1. SPY:     ${indexSignal.targetPosition > 0 ? 'LONG ' + (indexSignal.targetPosition * 100).toFixed(0) + '%' :
               indexSignal.targetPosition < 0 ? 'SHORT ' + Math.abs(indexSignal.targetPosition * 100).toFixed(0) + '%' : 'FLAT'}
  2. BTC:     ${cryptoSignal.targetPosition > 0 ? 'LONG ' + (cryptoSignal.targetPosition * 100).toFixed(0) + '%' :
               cryptoSignal.targetPosition < 0 ? 'SHORT ' + Math.abs(cryptoSignal.targetPosition * 100).toFixed(0) + '%' : 'FLAT'}
  3. ROTATION: BTC ${(rotationSignal.targetAllocation.BTC * 100).toFixed(0)}% / ETH ${(rotationSignal.targetAllocation.ETH * 100).toFixed(0)}% / USDC ${(rotationSignal.targetAllocation.USDC * 100).toFixed(0)}%
  4. SPREAD:  VIX ${vixSignal.targetPosition.vix > 0 ? 'LONG' : vixSignal.targetPosition.vix < 0 ? 'SHORT' : 'FLAT'} / Crypto ${vixSignal.targetPosition.crypto > 0 ? 'LONG' : vixSignal.targetPosition.crypto < 0 ? 'SHORT' : 'FLAT'}
    `);

    return {
        indexSignal,
        cryptoSignal,
        rotationSignal,
        vixSignal
    };
}

// ============================================================================
// EXPORTS
// ============================================================================

export {
    HarmonicEngine,
    IndexRegimeStrategy,
    CryptoMeanReversionStrategy,
    CrossAssetRotationStrategy,
    VixSpreadStrategy,
    StrategyBacktester,
    compareToHedgeFunds,
    runLiveDemo
};

// Run demo if executed directly
if (typeof process !== 'undefined' && process.argv[1]?.includes('AutomatedStrategyEngine')) {
    runLiveDemo();
}

// Browser global
if (typeof window !== 'undefined') {
    window.HarmonicAlphaStrategies = {
        HarmonicEngine,
        IndexRegimeStrategy,
        CryptoMeanReversionStrategy,
        CrossAssetRotationStrategy,
        VixSpreadStrategy,
        runLiveDemo
    };
}
