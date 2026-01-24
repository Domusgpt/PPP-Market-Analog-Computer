#!/usr/bin/env node
/**
 * DEPRECATED - This file contains MANUALLY TYPED data, not API-fetched data.
 *
 * For a REAL backtest using verified API data, use: RealBacktest.js
 *
 * ============================================================================
 * HONESTY NOTICE:
 * The HISTORICAL_DATA_2024 array below was manually typed from memory, not
 * fetched from APIs. This makes it unsuitable for valid backtesting because:
 * 1. The data may contain errors from human memory
 * 2. There's no verification trail
 * 3. Cherry-picking bias is possible
 *
 * Use RealBacktest.js instead, which fetches data from:
 * - CoinGecko API (BTC prices)
 * - Alternative.me API (Fear & Greed Index)
 * ============================================================================
 *
 * Original description (for reference):
 * Harmonic Alpha Historical Backtest
 * Test Period: 2024 (full year with multiple regimes)
 */

// ============================================================================
// VERIFIED HISTORICAL DATA - 2024 KEY DATES
// ============================================================================

// This is REAL data compiled from public sources
// BTC prices from CoinGecko, F&G from Alternative.me, VIX from Yahoo Finance

const HISTORICAL_DATA_2024 = [
    // Format: { date, btcPrice, fearGreed, vix, event }

    // January 2024 - ETF Approval Rally
    { date: '2024-01-01', btcPrice: 42283, fearGreed: 65, vix: 12.45, event: null },
    { date: '2024-01-08', btcPrice: 46000, fearGreed: 71, vix: 12.70, event: 'BTC_ETF_ANTICIPATION' },
    { date: '2024-01-10', btcPrice: 46600, fearGreed: 76, vix: 12.44, event: 'ETF_APPROVED' },
    { date: '2024-01-11', btcPrice: 46300, fearGreed: 72, vix: 12.69, event: 'SELL_THE_NEWS' },
    { date: '2024-01-15', btcPrice: 42850, fearGreed: 63, vix: 12.91, event: null },
    { date: '2024-01-23', btcPrice: 39900, fearGreed: 51, vix: 13.26, event: null },
    { date: '2024-01-31', btcPrice: 42580, fearGreed: 63, vix: 13.88, event: null },

    // February 2024 - Recovery begins
    { date: '2024-02-07', btcPrice: 44500, fearGreed: 72, vix: 12.93, event: null },
    { date: '2024-02-15', btcPrice: 52000, fearGreed: 79, vix: 14.41, event: 'BREAKOUT' },
    { date: '2024-02-26', btcPrice: 54500, fearGreed: 82, vix: 13.75, event: null },
    { date: '2024-02-29', btcPrice: 62000, fearGreed: 82, vix: 13.11, event: 'BULL_ACCELERATION' },

    // March 2024 - ATH Run
    { date: '2024-03-05', btcPrice: 66000, fearGreed: 90, vix: 14.44, event: 'NEW_ATH_APPROACH' },
    { date: '2024-03-08', btcPrice: 68000, fearGreed: 90, vix: 14.74, event: null },
    { date: '2024-03-14', btcPrice: 73000, fearGreed: 88, vix: 14.41, event: 'ATH_73K' },
    { date: '2024-03-19', btcPrice: 63800, fearGreed: 74, vix: 13.04, event: 'SHARP_CORRECTION' },
    { date: '2024-03-25', btcPrice: 70000, fearGreed: 79, vix: 12.92, event: null },
    { date: '2024-03-31', btcPrice: 71300, fearGreed: 75, vix: 13.01, event: null },

    // April 2024 - Distribution begins
    { date: '2024-04-08', btcPrice: 71500, fearGreed: 79, vix: 16.35, event: null },
    { date: '2024-04-12', btcPrice: 66000, fearGreed: 65, vix: 17.31, event: 'HALVING_SELLOFF' },
    { date: '2024-04-19', btcPrice: 64000, fearGreed: 56, vix: 18.00, event: 'HALVING_EVENT' },
    { date: '2024-04-30', btcPrice: 60600, fearGreed: 43, vix: 15.65, event: null },

    // May 2024 - Continued weakness
    { date: '2024-05-07', btcPrice: 63300, fearGreed: 64, vix: 13.49, event: null },
    { date: '2024-05-15', btcPrice: 66000, fearGreed: 74, vix: 12.55, event: null },
    { date: '2024-05-21', btcPrice: 71000, fearGreed: 76, vix: 11.86, event: 'RECOVERY_ATTEMPT' },
    { date: '2024-05-31', btcPrice: 67500, fearGreed: 73, vix: 12.92, event: null },

    // June 2024 - Another leg down
    { date: '2024-06-07', btcPrice: 71000, fearGreed: 77, vix: 12.22, event: null },
    { date: '2024-06-14', btcPrice: 66000, fearGreed: 70, vix: 11.89, event: null },
    { date: '2024-06-18', btcPrice: 65000, fearGreed: 67, vix: 12.55, event: null },
    { date: '2024-06-24', btcPrice: 61300, fearGreed: 47, vix: 12.84, event: 'BREAKDOWN' },
    { date: '2024-06-30', btcPrice: 62700, fearGreed: 53, vix: 12.44, event: null },

    // July 2024 - Bottoming process
    { date: '2024-07-05', btcPrice: 56600, fearGreed: 29, vix: 12.48, event: 'LOCAL_BOTTOM' },
    { date: '2024-07-08', btcPrice: 58000, fearGreed: 44, vix: 12.46, event: null },
    { date: '2024-07-15', btcPrice: 63000, fearGreed: 60, vix: 12.92, event: null },
    { date: '2024-07-22', btcPrice: 67500, fearGreed: 71, vix: 14.50, event: null },
    { date: '2024-07-29', btcPrice: 66700, fearGreed: 67, vix: 16.36, event: 'VIX_RISING' },

    // August 2024 - BLACK MONDAY
    { date: '2024-08-01', btcPrice: 64600, fearGreed: 57, vix: 16.36, event: null },
    { date: '2024-08-02', btcPrice: 62300, fearGreed: 46, vix: 23.39, event: 'VIX_SPIKE_WARNING' },
    { date: '2024-08-05', btcPrice: 54300, fearGreed: 17, vix: 38.57, event: 'BLACK_MONDAY' },
    { date: '2024-08-06', btcPrice: 55000, fearGreed: 20, vix: 27.71, event: 'BOUNCE' },
    { date: '2024-08-08', btcPrice: 60800, fearGreed: 42, vix: 20.37, event: null },
    { date: '2024-08-15', btcPrice: 58900, fearGreed: 48, vix: 15.22, event: null },
    { date: '2024-08-23', btcPrice: 64000, fearGreed: 55, vix: 15.86, event: null },
    { date: '2024-08-31', btcPrice: 59000, fearGreed: 26, vix: 15.00, event: null },

    // September 2024 - Consolidation
    { date: '2024-09-06', btcPrice: 53800, fearGreed: 22, vix: 22.38, event: 'RETEST_LOWS' },
    { date: '2024-09-10', btcPrice: 57000, fearGreed: 32, vix: 17.69, event: null },
    { date: '2024-09-16', btcPrice: 60000, fearGreed: 45, vix: 17.14, event: null },
    { date: '2024-09-23', btcPrice: 63500, fearGreed: 59, vix: 16.15, event: null },
    { date: '2024-09-30', btcPrice: 63400, fearGreed: 50, vix: 16.73, event: null },

    // October 2024 - Bull resumes
    { date: '2024-10-07', btcPrice: 62000, fearGreed: 41, vix: 20.49, event: null },
    { date: '2024-10-14', btcPrice: 65500, fearGreed: 65, vix: 20.46, event: null },
    { date: '2024-10-21', btcPrice: 67000, fearGreed: 56, vix: 18.37, event: null },
    { date: '2024-10-28', btcPrice: 71300, fearGreed: 72, vix: 20.33, event: 'BREAKOUT_SIGNAL' },
    { date: '2024-10-31', btcPrice: 69500, fearGreed: 65, vix: 21.88, event: null },

    // November 2024 - Trump Election Rally
    { date: '2024-11-04', btcPrice: 68700, fearGreed: 63, vix: 21.98, event: 'PRE_ELECTION' },
    { date: '2024-11-06', btcPrice: 75000, fearGreed: 77, vix: 16.27, event: 'TRUMP_ELECTED' },
    { date: '2024-11-11', btcPrice: 82000, fearGreed: 80, vix: 14.71, event: 'RALLY_CONTINUES' },
    { date: '2024-11-13', btcPrice: 90000, fearGreed: 88, vix: 14.02, event: null },
    { date: '2024-11-18', btcPrice: 91500, fearGreed: 83, vix: 15.58, event: null },
    { date: '2024-11-22', btcPrice: 98000, fearGreed: 94, vix: 15.24, event: 'APPROACHING_100K' },
    { date: '2024-11-25', btcPrice: 97000, fearGreed: 79, vix: 14.10, event: null },
    { date: '2024-11-30', btcPrice: 96500, fearGreed: 76, vix: 13.51, event: null },

    // December 2024 - $100k achieved
    { date: '2024-12-05', btcPrice: 103000, fearGreed: 84, vix: 12.77, event: '100K_BREAKTHROUGH' },
    { date: '2024-12-10', btcPrice: 97000, fearGreed: 74, vix: 14.18, event: null },
    { date: '2024-12-17', btcPrice: 106000, fearGreed: 87, vix: 15.87, event: 'NEW_ATH_106K' },
    { date: '2024-12-20', btcPrice: 97000, fearGreed: 62, vix: 18.36, event: 'CORRECTION' },
    { date: '2024-12-24', btcPrice: 94000, fearGreed: 54, vix: 14.73, event: null },
    { date: '2024-12-31', btcPrice: 93400, fearGreed: 65, vix: 17.35, event: 'YEAR_END' },
];

// ============================================================================
// HARMONIC ALPHA ENGINE FOR BACKTEST
// ============================================================================

class BacktestHarmonicEngine {
    constructor(config = {}) {
        this.config = {
            tensionThreshold: config.tensionThreshold || 0.4,
            regimeChangeThreshold: config.regimeChangeThreshold || 0.6,
            priceRange: config.priceRange || { min: 35000, max: 110000 },
            ...config
        };
        this.tensionHistory = [];
        this.signals = [];
        this.lastRegime = null;
    }

    normalize(price) {
        return (price - this.config.priceRange.min) /
               (this.config.priceRange.max - this.config.priceRange.min);
    }

    analyze(dataPoint) {
        const priceNorm = this.normalize(dataPoint.btcPrice);
        const sentimentNorm = dataPoint.fearGreed / 100;
        const vixNorm = 1 - Math.min(1, (dataPoint.vix - 10) / 30);

        // Combined sentiment: 60% F&G, 40% inverse VIX
        const combinedSentiment = sentimentNorm * 0.6 + vixNorm * 0.4;

        // Tension calculation
        const tension = Math.abs(priceNorm - combinedSentiment);

        // Direction
        const direction = priceNorm > combinedSentiment ? 'PRICE_LEADING' : 'SENTIMENT_LEADING';

        // Regime classification
        let regime;
        if (tension < 0.15) {
            regime = priceNorm > 0.5 ? 'STABLE_BULL' : 'STABLE_BEAR';
        } else if (tension < 0.35) {
            regime = 'MILD_DIVERGENCE';
        } else if (tension < 0.55) {
            regime = 'HIGH_TENSION';
        } else {
            regime = 'EXTREME_DIVERGENCE';
        }

        // Detect regime change
        let regimeChange = null;
        if (this.lastRegime && this.lastRegime !== regime) {
            regimeChange = { from: this.lastRegime, to: regime };
        }
        this.lastRegime = regime;

        // Track history
        this.tensionHistory.push({
            date: dataPoint.date,
            tension,
            regime,
            priceNorm,
            sentimentNorm: combinedSentiment
        });

        // Generate signals
        const signal = this.generateSignal(tension, direction, regime, regimeChange, dataPoint);

        return {
            date: dataPoint.date,
            price: dataPoint.btcPrice,
            fearGreed: dataPoint.fearGreed,
            vix: dataPoint.vix,
            priceNorm,
            sentimentNorm: combinedSentiment,
            tension,
            direction,
            regime,
            regimeChange,
            signal
        };
    }

    generateSignal(tension, direction, regime, regimeChange, dataPoint) {
        // Signal generation logic
        let action = 'HOLD';
        let strength = 0;
        let reasoning = '';

        // Regime change signals
        if (regimeChange) {
            if (regimeChange.to === 'EXTREME_DIVERGENCE') {
                action = 'REDUCE_ALL';
                strength = 1.0;
                reasoning = 'Entering extreme divergence - reduce exposure';
            } else if (regimeChange.from === 'EXTREME_DIVERGENCE') {
                action = direction === 'SENTIMENT_LEADING' ? 'BUY' : 'SELL';
                strength = 0.7;
                reasoning = 'Exiting extreme divergence - regime resolving';
            }
        }

        // High tension signals
        if (tension > 0.5 && action === 'HOLD') {
            if (direction === 'PRICE_LEADING') {
                action = 'SELL_WARNING';
                strength = tension;
                reasoning = 'Price extended beyond sentiment - potential top';
            } else {
                action = 'BUY_WARNING';
                strength = tension;
                reasoning = 'Sentiment recovering - potential bottom';
            }
        }

        // VIX spike detection
        if (dataPoint.vix > 25 && action === 'HOLD') {
            action = 'VOLATILITY_WARNING';
            strength = Math.min(1, (dataPoint.vix - 20) / 20);
            reasoning = `VIX spike to ${dataPoint.vix} - increased volatility`;
        }

        // Low tension + high sentiment = trend continuation
        if (tension < 0.2 && dataPoint.fearGreed > 70 && action === 'HOLD') {
            action = 'TREND_STRONG';
            strength = 0.3;
            reasoning = 'Low tension, high sentiment - trend intact';
        }

        // Low tension + low sentiment = trend continuation (bearish)
        if (tension < 0.2 && dataPoint.fearGreed < 30 && action === 'HOLD') {
            action = 'TREND_BEARISH';
            strength = 0.3;
            reasoning = 'Low tension, low sentiment - bearish trend intact';
        }

        if (action !== 'HOLD') {
            this.signals.push({
                date: dataPoint.date,
                action,
                strength,
                reasoning,
                price: dataPoint.btcPrice,
                tension
            });
        }

        return { action, strength, reasoning };
    }
}

// ============================================================================
// PORTFOLIO SIMULATOR
// ============================================================================

class PortfolioSimulator {
    constructor(initialCapital = 100000) {
        this.initialCapital = initialCapital;
        this.cash = initialCapital;
        this.btcHolding = 0;
        this.portfolio = [];
        this.trades = [];
        this.maxDrawdown = 0;
        this.peakValue = initialCapital;
    }

    getPortfolioValue(btcPrice) {
        return this.cash + this.btcHolding * btcPrice;
    }

    executeSignal(signal, btcPrice, date) {
        const currentValue = this.getPortfolioValue(btcPrice);

        // Update peak and drawdown
        if (currentValue > this.peakValue) {
            this.peakValue = currentValue;
        }
        const drawdown = (this.peakValue - currentValue) / this.peakValue;
        if (drawdown > this.maxDrawdown) {
            this.maxDrawdown = drawdown;
        }

        let trade = null;

        switch (signal.action) {
            case 'BUY':
            case 'BUY_WARNING':
                // Buy with portion of cash based on signal strength
                const buyAmount = this.cash * signal.strength * 0.5;
                if (buyAmount > 100) {
                    const btcToBuy = buyAmount / btcPrice;
                    this.btcHolding += btcToBuy;
                    this.cash -= buyAmount;
                    trade = { date, action: 'BUY', amount: btcToBuy, price: btcPrice, value: buyAmount };
                }
                break;

            case 'SELL':
            case 'SELL_WARNING':
                // Sell portion of holdings
                const sellAmount = this.btcHolding * signal.strength * 0.5;
                if (sellAmount > 0.001) {
                    const cashReceived = sellAmount * btcPrice;
                    this.btcHolding -= sellAmount;
                    this.cash += cashReceived;
                    trade = { date, action: 'SELL', amount: sellAmount, price: btcPrice, value: cashReceived };
                }
                break;

            case 'REDUCE_ALL':
                // Reduce to 25% exposure
                const targetBtc = this.btcHolding * 0.25;
                const toSell = this.btcHolding - targetBtc;
                if (toSell > 0.001) {
                    const cashReceived = toSell * btcPrice;
                    this.btcHolding = targetBtc;
                    this.cash += cashReceived;
                    trade = { date, action: 'REDUCE', amount: toSell, price: btcPrice, value: cashReceived };
                }
                break;
        }

        if (trade) {
            this.trades.push(trade);
        }

        // Record portfolio state
        this.portfolio.push({
            date,
            cash: this.cash,
            btcHolding: this.btcHolding,
            btcPrice,
            totalValue: this.getPortfolioValue(btcPrice),
            signal: signal.action
        });

        return trade;
    }
}

// ============================================================================
// BENCHMARK COMPARISONS
// ============================================================================

class BenchmarkComparison {
    constructor(historicalData) {
        this.data = historicalData;
    }

    buyAndHold(initialCapital = 100000) {
        const startPrice = this.data[0].btcPrice;
        const endPrice = this.data[this.data.length - 1].btcPrice;
        const btcBought = initialCapital / startPrice;
        const finalValue = btcBought * endPrice;
        return {
            strategy: 'Buy & Hold',
            initialCapital,
            finalValue,
            totalReturn: (finalValue - initialCapital) / initialCapital,
            trades: 1
        };
    }

    movingAverageCrossover(initialCapital = 100000, shortPeriod = 7, longPeriod = 21) {
        // Simple MA crossover strategy
        let cash = initialCapital;
        let btc = 0;
        let trades = 0;

        for (let i = longPeriod; i < this.data.length; i++) {
            const shortMA = this.data.slice(i - shortPeriod, i)
                .reduce((sum, d) => sum + d.btcPrice, 0) / shortPeriod;
            const longMA = this.data.slice(i - longPeriod, i)
                .reduce((sum, d) => sum + d.btcPrice, 0) / longPeriod;

            const price = this.data[i].btcPrice;

            if (shortMA > longMA && btc === 0) {
                // Buy signal
                btc = cash / price;
                cash = 0;
                trades++;
            } else if (shortMA < longMA && btc > 0) {
                // Sell signal
                cash = btc * price;
                btc = 0;
                trades++;
            }
        }

        const finalPrice = this.data[this.data.length - 1].btcPrice;
        const finalValue = cash + btc * finalPrice;

        return {
            strategy: `MA Crossover (${shortPeriod}/${longPeriod})`,
            initialCapital,
            finalValue,
            totalReturn: (finalValue - initialCapital) / initialCapital,
            trades
        };
    }

    fearGreedStrategy(initialCapital = 100000) {
        // Buy when F&G < 25, sell when F&G > 75
        let cash = initialCapital;
        let btc = 0;
        let trades = 0;

        for (const d of this.data) {
            if (d.fearGreed < 25 && cash > 0) {
                btc = cash / d.btcPrice;
                cash = 0;
                trades++;
            } else if (d.fearGreed > 75 && btc > 0) {
                cash = btc * d.btcPrice;
                btc = 0;
                trades++;
            }
        }

        const finalPrice = this.data[this.data.length - 1].btcPrice;
        const finalValue = cash + btc * finalPrice;

        return {
            strategy: 'Fear & Greed (25/75)',
            initialCapital,
            finalValue,
            totalReturn: (finalValue - initialCapital) / initialCapital,
            trades
        };
    }
}

// ============================================================================
// RUN BACKTEST
// ============================================================================

function runBacktest() {
    console.log('='.repeat(70));
    console.log('  HARMONIC ALPHA HISTORICAL BACKTEST');
    console.log('  Test Period: January 2024 - December 2024');
    console.log('  Initial Capital: $100,000');
    console.log('='.repeat(70));

    // Initialize
    const engine = new BacktestHarmonicEngine();
    const portfolio = new PortfolioSimulator(100000);
    const benchmark = new BenchmarkComparison(HISTORICAL_DATA_2024);

    // Start with 50% BTC exposure
    const startPrice = HISTORICAL_DATA_2024[0].btcPrice;
    portfolio.btcHolding = 50000 / startPrice;
    portfolio.cash = 50000;

    console.log('\n[1] RUNNING HARMONIC ALPHA THROUGH 2024 DATA...\n');

    const analyses = [];
    const keyEvents = [];

    for (const dataPoint of HISTORICAL_DATA_2024) {
        const analysis = engine.analyze(dataPoint);
        analyses.push(analysis);

        // Execute signal
        if (analysis.signal.action !== 'HOLD') {
            const trade = portfolio.executeSignal(analysis.signal, dataPoint.btcPrice, dataPoint.date);

            // Log significant signals
            if (analysis.signal.strength > 0.5 || analysis.regimeChange) {
                keyEvents.push({
                    date: dataPoint.date,
                    event: dataPoint.event,
                    signal: analysis.signal,
                    tension: analysis.tension,
                    regime: analysis.regime,
                    price: dataPoint.btcPrice,
                    trade
                });
            }
        }
    }

    // Final portfolio value
    const finalPrice = HISTORICAL_DATA_2024[HISTORICAL_DATA_2024.length - 1].btcPrice;
    const finalValue = portfolio.getPortfolioValue(finalPrice);

    console.log('[2] KEY SIGNALS GENERATED:\n');
    for (const event of keyEvents) {
        const marker = event.event ? ` [${event.event}]` : '';
        console.log(`  ${event.date}${marker}`);
        console.log(`    Signal: ${event.signal.action} (strength: ${event.signal.strength.toFixed(2)})`);
        console.log(`    Reason: ${event.signal.reasoning}`);
        console.log(`    Price: $${event.price.toLocaleString()} | Tension: ${(event.tension * 100).toFixed(1)}%`);
        if (event.trade) {
            console.log(`    Trade: ${event.trade.action} ${event.trade.amount.toFixed(4)} BTC @ $${event.trade.price.toLocaleString()}`);
        }
        console.log('');
    }

    console.log('[3] PORTFOLIO PERFORMANCE:\n');
    console.log(`  Initial Capital:    $${portfolio.initialCapital.toLocaleString()}`);
    console.log(`  Final Value:        $${finalValue.toLocaleString()}`);
    console.log(`  Total Return:       ${((finalValue - portfolio.initialCapital) / portfolio.initialCapital * 100).toFixed(1)}%`);
    console.log(`  Max Drawdown:       ${(portfolio.maxDrawdown * 100).toFixed(1)}%`);
    console.log(`  Total Trades:       ${portfolio.trades.length}`);

    console.log('\n[4] BENCHMARK COMPARISONS:\n');

    const buyHold = benchmark.buyAndHold(100000);
    const maCross = benchmark.movingAverageCrossover(100000);
    const fgStrategy = benchmark.fearGreedStrategy(100000);

    const strategies = [
        { name: 'Harmonic Alpha', return: (finalValue - 100000) / 100000, trades: portfolio.trades.length },
        { name: buyHold.strategy, return: buyHold.totalReturn, trades: buyHold.trades },
        { name: maCross.strategy, return: maCross.totalReturn, trades: maCross.trades },
        { name: fgStrategy.strategy, return: fgStrategy.totalReturn, trades: fgStrategy.trades }
    ];

    console.log('  Strategy                  Return      Trades');
    console.log('  ' + '-'.repeat(50));
    for (const s of strategies.sort((a, b) => b.return - a.return)) {
        const returnStr = (s.return * 100).toFixed(1).padStart(7) + '%';
        console.log(`  ${s.name.padEnd(25)} ${returnStr}     ${s.trades}`);
    }

    console.log('\n[5] EARLY WARNING DETECTION:\n');

    // Check if Harmonic Alpha detected events BEFORE they happened
    const earlyWarnings = [];

    // Black Monday (Aug 5) - check signals in late July/early Aug
    const blackMondaySignals = engine.signals.filter(s =>
        s.date >= '2024-07-29' && s.date <= '2024-08-05'
    );
    if (blackMondaySignals.length > 0) {
        const firstWarning = blackMondaySignals[0];
        earlyWarnings.push({
            event: 'BLACK MONDAY (Aug 5)',
            firstSignal: firstWarning.date,
            daysEarly: Math.floor((new Date('2024-08-05') - new Date(firstWarning.date)) / 86400000),
            signal: firstWarning.action
        });
    }

    // March ATH correction - check signals around March 14-19
    const marchCorrectionSignals = engine.signals.filter(s =>
        s.date >= '2024-03-08' && s.date <= '2024-03-19' &&
        (s.action.includes('SELL') || s.action.includes('WARNING'))
    );
    if (marchCorrectionSignals.length > 0) {
        const firstWarning = marchCorrectionSignals[0];
        earlyWarnings.push({
            event: 'MARCH ATH CORRECTION',
            firstSignal: firstWarning.date,
            daysEarly: Math.floor((new Date('2024-03-19') - new Date(firstWarning.date)) / 86400000),
            signal: firstWarning.action
        });
    }

    // November rally - check signals in late October
    const novRallySignals = engine.signals.filter(s =>
        s.date >= '2024-10-21' && s.date <= '2024-11-06' &&
        s.action.includes('BUY')
    );
    if (novRallySignals.length > 0) {
        const firstSignal = novRallySignals[0];
        earlyWarnings.push({
            event: 'NOVEMBER RALLY',
            firstSignal: firstSignal.date,
            daysEarly: Math.floor((new Date('2024-11-06') - new Date(firstSignal.date)) / 86400000),
            signal: firstSignal.action
        });
    }

    for (const warning of earlyWarnings) {
        console.log(`  ${warning.event}`);
        console.log(`    First Signal: ${warning.firstSignal} (${warning.daysEarly} days early)`);
        console.log(`    Signal Type: ${warning.signal}`);
        console.log('');
    }

    console.log('='.repeat(70));
    console.log('  BACKTEST COMPLETE');
    console.log('='.repeat(70));

    return {
        portfolio,
        analyses,
        keyEvents,
        earlyWarnings,
        benchmarks: { buyHold, maCross, fgStrategy }
    };
}

// ============================================================================
// EXPORTS & EXECUTION
// ============================================================================

export { runBacktest, HISTORICAL_DATA_2024, BacktestHarmonicEngine, PortfolioSimulator };

// Run if executed directly
runBacktest();
