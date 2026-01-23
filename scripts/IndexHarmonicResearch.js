#!/usr/bin/env node
/**
 * Harmonic Alpha Research - Stock Index Focus
 *
 * ACADEMIC APPROACH:
 * 1. Fetch real historical data from Yahoo Finance
 * 2. Apply harmonic tension model to index + fear indicators
 * 3. Test against known regime changes (2022 bear, 2023-24 recovery)
 * 4. Measure predictive accuracy with proper train/test split
 *
 * Assets:
 * - S&P 500 (^GSPC) - primary price signal
 * - VIX (^VIX) - fear/volatility
 * - Gold (GLD) - safe haven flows
 * - US Dollar (UUP) - risk-off indicator
 */

import https from 'https';
import { execSync } from 'child_process';

// ============================================================================
// YAHOO FINANCE DATA FETCHER
// ============================================================================

async function fetchYahooFinance(symbol, period1, period2) {
    return new Promise((resolve, reject) => {
        // Yahoo Finance API endpoint
        const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?period1=${period1}&period2=${period2}&interval=1d`;

        const options = {
            hostname: 'query1.finance.yahoo.com',
            path: `/v8/finance/chart/${encodeURIComponent(symbol)}?period1=${period1}&period2=${period2}&interval=1d`,
            method: 'GET',
            headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            },
            rejectUnauthorized: false
        };

        const req = https.request(options, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                try {
                    const json = JSON.parse(data);
                    if (json.chart && json.chart.result && json.chart.result[0]) {
                        const result = json.chart.result[0];
                        const timestamps = result.timestamp || [];
                        const quotes = result.indicators.quote[0];

                        const prices = [];
                        for (let i = 0; i < timestamps.length; i++) {
                            if (quotes.close[i] !== null) {
                                prices.push({
                                    date: new Date(timestamps[i] * 1000).toISOString().split('T')[0],
                                    open: quotes.open[i],
                                    high: quotes.high[i],
                                    low: quotes.low[i],
                                    close: quotes.close[i],
                                    volume: quotes.volume[i]
                                });
                            }
                        }
                        resolve(prices);
                    } else {
                        reject(new Error(`No data for ${symbol}: ${JSON.stringify(json)}`));
                    }
                } catch (e) {
                    reject(e);
                }
            });
        });

        req.on('error', reject);
        req.end();
    });
}

// ============================================================================
// HARMONIC TENSION MODEL
// ============================================================================

class HarmonicTensionModel {
    constructor(config = {}) {
        this.config = {
            // Harmonic interval ratios (music theory)
            intervals: {
                unison: 1.0,        // Perfect consonance
                octave: 2.0,        // Perfect consonance
                fifth: 1.5,         // Perfect fifth (3:2)
                fourth: 1.333,      // Perfect fourth (4:3)
                majorThird: 1.25,   // Major third (5:4)
                minorThird: 1.2,    // Minor third (6:5)
                tritone: 1.414,     // Tritone - maximum dissonance (sqrt(2))
            },
            lookback: config.lookback || 20,  // Days for moving calculations
            ...config
        };
        this.history = [];
    }

    // Calculate "harmonic ratio" between price momentum and volatility
    calculateHarmonicRatio(priceData, vixData) {
        if (priceData.length < this.config.lookback) return null;

        const recent = priceData.slice(-this.config.lookback);
        const recentVix = vixData.slice(-this.config.lookback);

        // Price momentum (normalized)
        const startPrice = recent[0].close;
        const endPrice = recent[recent.length - 1].close;
        const momentum = (endPrice - startPrice) / startPrice;

        // Volatility level (normalized VIX)
        const avgVix = recentVix.reduce((s, v) => s + v.close, 0) / recentVix.length;
        const normalizedVix = avgVix / 30; // VIX 30 = "neutral"

        // The ratio between momentum direction and fear
        // When price up + low VIX = consonant (stable bull)
        // When price up + high VIX = dissonant (unstable rally)
        // When price down + high VIX = consonant (capitulation)
        // When price down + low VIX = dissonant (complacent decline)

        const ratio = Math.abs(momentum * 10 + 1) / (normalizedVix + 0.5);

        return {
            momentum,
            avgVix,
            normalizedVix,
            ratio,
            interval: this.classifyInterval(ratio),
            tension: this.calculateTension(momentum, normalizedVix)
        };
    }

    // Map ratio to musical interval
    classifyInterval(ratio) {
        const intervals = this.config.intervals;

        // Find closest interval
        let closest = 'unison';
        let minDist = Infinity;

        for (const [name, value] of Object.entries(intervals)) {
            const dist = Math.abs(ratio - value);
            if (dist < minDist) {
                minDist = dist;
                closest = name;
            }
        }

        return {
            name: closest,
            value: intervals[closest],
            distance: minDist,
            isConsonant: ['unison', 'octave', 'fifth', 'fourth'].includes(closest),
            isDissonant: ['tritone', 'minorThird'].includes(closest)
        };
    }

    // Calculate tension score (0-1)
    calculateTension(momentum, normalizedVix) {
        // Tension is high when:
        // 1. Price rising but VIX also rising (divergence)
        // 2. Price falling but VIX staying low (complacency)
        // 3. Extreme readings in either direction

        const momentumSign = momentum > 0 ? 1 : -1;
        const vixSignal = normalizedVix > 0.8 ? 1 : normalizedVix < 0.5 ? -1 : 0;

        // Divergence detection
        const divergence = (momentumSign > 0 && vixSignal > 0) ||  // Rally with fear
                          (momentumSign < 0 && vixSignal < 0);      // Decline with complacency

        let tension = 0.3; // Base tension

        if (divergence) {
            tension += 0.4;
        }

        // Add tension for extreme VIX
        if (normalizedVix > 1.2) tension += 0.2;  // VIX > 36
        if (normalizedVix > 1.5) tension += 0.1;  // VIX > 45

        // Add tension for extreme momentum
        if (Math.abs(momentum) > 0.1) tension += 0.1;  // >10% move in lookback

        return Math.min(1, tension);
    }

    // Analyze a single point
    analyze(date, spx, vix, gold, dollar) {
        this.history.push({ date, spx, vix, gold, dollar });

        if (this.history.length < this.config.lookback) {
            return null;
        }

        const spxData = this.history.map(h => ({ close: h.spx }));
        const vixData = this.history.map(h => ({ close: h.vix }));
        const goldData = this.history.map(h => ({ close: h.gold }));
        const dollarData = this.history.map(h => ({ close: h.dollar }));

        const harmonic = this.calculateHarmonicRatio(spxData, vixData);

        // Cross-asset analysis
        const goldMomentum = this.getMomentum(goldData);
        const dollarMomentum = this.getMomentum(dollarData);

        // Safe haven flow: gold up + dollar up = risk-off
        const safeHavenFlow = (goldMomentum > 0.02 && dollarMomentum > 0.01) ? 'RISK_OFF' :
                             (goldMomentum < -0.02 && dollarMomentum < -0.01) ? 'RISK_ON' : 'NEUTRAL';

        // Regime classification
        let regime;
        if (harmonic.tension > 0.7) {
            regime = 'HIGH_TENSION';
        } else if (harmonic.tension > 0.5) {
            regime = harmonic.momentum > 0 ? 'UNSTABLE_BULL' : 'UNSTABLE_BEAR';
        } else {
            regime = harmonic.momentum > 0 ? 'STABLE_BULL' : 'STABLE_BEAR';
        }

        // Generate signal
        let signal = 'HOLD';
        let confidence = 0.5;

        if (harmonic.tension > 0.7 && harmonic.interval.isDissonant) {
            signal = 'REDUCE_RISK';
            confidence = harmonic.tension;
        } else if (harmonic.tension < 0.4 && harmonic.interval.isConsonant && harmonic.momentum > 0) {
            signal = 'STAY_LONG';
            confidence = 1 - harmonic.tension;
        } else if (safeHavenFlow === 'RISK_OFF' && harmonic.tension > 0.5) {
            signal = 'DEFENSIVE';
            confidence = 0.7;
        }

        return {
            date,
            spx,
            vix,
            harmonic,
            safeHavenFlow,
            regime,
            signal,
            confidence
        };
    }

    getMomentum(data) {
        const lookback = Math.min(this.config.lookback, data.length);
        const recent = data.slice(-lookback);
        return (recent[recent.length - 1].close - recent[0].close) / recent[0].close;
    }
}

// ============================================================================
// BACKTEST ENGINE
// ============================================================================

class BacktestEngine {
    constructor(initialCapital = 100000) {
        this.initialCapital = initialCapital;
        this.capital = initialCapital;
        this.position = 1.0; // 100% invested initially
        this.trades = [];
        this.equity = [];
        this.maxDrawdown = 0;
        this.peak = initialCapital;
    }

    processSignal(date, signal, confidence, spxPrice, spxReturn) {
        const prevCapital = this.capital;

        // Apply market return to current position
        this.capital = this.capital * (1 + spxReturn * this.position);

        // Adjust position based on signal
        let trade = null;

        if (signal === 'REDUCE_RISK' && this.position > 0.5) {
            const targetPosition = Math.max(0.3, 1 - confidence);
            trade = { date, action: 'REDUCE', from: this.position, to: targetPosition };
            this.position = targetPosition;
        } else if (signal === 'DEFENSIVE' && this.position > 0.5) {
            trade = { date, action: 'DEFENSIVE', from: this.position, to: 0.5 };
            this.position = 0.5;
        } else if (signal === 'STAY_LONG' && this.position < 1.0) {
            trade = { date, action: 'INCREASE', from: this.position, to: 1.0 };
            this.position = 1.0;
        }

        if (trade) this.trades.push(trade);

        // Track equity and drawdown
        this.equity.push({ date, capital: this.capital, position: this.position });

        if (this.capital > this.peak) {
            this.peak = this.capital;
        }
        const drawdown = (this.peak - this.capital) / this.peak;
        if (drawdown > this.maxDrawdown) {
            this.maxDrawdown = drawdown;
        }

        return trade;
    }

    getResults() {
        const totalReturn = (this.capital - this.initialCapital) / this.initialCapital;
        return {
            initialCapital: this.initialCapital,
            finalCapital: this.capital,
            totalReturn,
            maxDrawdown: this.maxDrawdown,
            trades: this.trades.length,
            tradeLog: this.trades
        };
    }
}

// ============================================================================
// MAIN RESEARCH FUNCTION
// ============================================================================

async function runResearch() {
    console.log('='.repeat(70));
    console.log('  HARMONIC ALPHA RESEARCH - STOCK INDEX ANALYSIS');
    console.log('  Using Real Yahoo Finance Data');
    console.log('='.repeat(70));

    // Date range: 2022-01-01 to 2024-12-31 (3 years)
    const period1 = Math.floor(new Date('2022-01-01').getTime() / 1000);
    const period2 = Math.floor(new Date('2024-12-31').getTime() / 1000);

    console.log('\n[1] FETCHING DATA FROM YAHOO FINANCE...\n');

    try {
        // Fetch all data
        console.log('    Fetching S&P 500 (^GSPC)...');
        const spxData = await fetchYahooFinance('^GSPC', period1, period2);
        console.log(`    Got ${spxData.length} days of SPX data`);

        console.log('    Fetching VIX (^VIX)...');
        const vixData = await fetchYahooFinance('^VIX', period1, period2);
        console.log(`    Got ${vixData.length} days of VIX data`);

        console.log('    Fetching Gold (GLD)...');
        const goldData = await fetchYahooFinance('GLD', period1, period2);
        console.log(`    Got ${goldData.length} days of Gold data`);

        console.log('    Fetching Dollar (UUP)...');
        const dollarData = await fetchYahooFinance('UUP', period1, period2);
        console.log(`    Got ${dollarData.length} days of Dollar data`);

        // Merge by date
        console.log('\n[2] MERGING DATA BY DATE...\n');

        const spxByDate = Object.fromEntries(spxData.map(d => [d.date, d.close]));
        const vixByDate = Object.fromEntries(vixData.map(d => [d.date, d.close]));
        const goldByDate = Object.fromEntries(goldData.map(d => [d.date, d.close]));
        const dollarByDate = Object.fromEntries(dollarData.map(d => [d.date, d.close]));

        const allDates = [...new Set([...Object.keys(spxByDate)])].sort();

        const mergedData = [];
        for (const date of allDates) {
            if (spxByDate[date] && vixByDate[date] && goldByDate[date] && dollarByDate[date]) {
                mergedData.push({
                    date,
                    spx: spxByDate[date],
                    vix: vixByDate[date],
                    gold: goldByDate[date],
                    dollar: dollarByDate[date]
                });
            }
        }

        console.log(`    Merged dataset: ${mergedData.length} trading days`);
        console.log(`    Date range: ${mergedData[0].date} to ${mergedData[mergedData.length-1].date}`);

        // Split into train (2022) and test (2023-2024)
        const trainData = mergedData.filter(d => d.date < '2023-01-01');
        const testData = mergedData.filter(d => d.date >= '2023-01-01');

        console.log(`    Training set: ${trainData.length} days (2022)`);
        console.log(`    Test set: ${testData.length} days (2023-2024)`);

        // Run model on test set
        console.log('\n[3] RUNNING HARMONIC TENSION MODEL...\n');

        const model = new HarmonicTensionModel({ lookback: 20 });
        const backtest = new BacktestEngine(100000);
        const buyHold = new BacktestEngine(100000);

        // Warm up model with training data
        for (const d of trainData) {
            model.analyze(d.date, d.spx, d.vix, d.gold, d.dollar);
        }

        // Test on out-of-sample data
        const analyses = [];
        let prevSpx = testData[0].spx;

        for (const d of testData) {
            const spxReturn = (d.spx - prevSpx) / prevSpx;

            const analysis = model.analyze(d.date, d.spx, d.vix, d.gold, d.dollar);

            if (analysis) {
                analyses.push(analysis);
                backtest.processSignal(d.date, analysis.signal, analysis.confidence, d.spx, spxReturn);
                buyHold.processSignal(d.date, 'HOLD', 0.5, d.spx, spxReturn);
            }

            prevSpx = d.spx;
        }

        // Results
        console.log('[4] KEY SIGNALS GENERATED:\n');

        const significantSignals = analyses.filter(a => a.signal !== 'HOLD' && a.confidence > 0.6);
        console.log(`    Total signals: ${significantSignals.length}`);
        console.log('\n    Recent high-confidence signals:');

        for (const sig of significantSignals.slice(-15)) {
            console.log(`    ${sig.date}: ${sig.signal.padEnd(12)} (${(sig.confidence*100).toFixed(0)}%) | SPX: $${sig.spx.toFixed(0)} | VIX: ${sig.vix.toFixed(1)} | Regime: ${sig.regime}`);
        }

        // Regime distribution
        console.log('\n[5] REGIME DISTRIBUTION:\n');

        const regimeCounts = {};
        for (const a of analyses) {
            regimeCounts[a.regime] = (regimeCounts[a.regime] || 0) + 1;
        }

        for (const [regime, count] of Object.entries(regimeCounts).sort((a,b) => b[1] - a[1])) {
            const pct = (count / analyses.length * 100).toFixed(1);
            console.log(`    ${regime.padEnd(20)} ${count.toString().padStart(4)} days (${pct}%)`);
        }

        // Performance comparison
        console.log('\n[6] PERFORMANCE COMPARISON (2023-2024 TEST PERIOD):\n');

        const modelResults = backtest.getResults();
        const holdResults = buyHold.getResults();

        console.log('    Strategy              Final Value    Return    Max DD    Trades');
        console.log('    ' + '-'.repeat(60));
        console.log(`    Harmonic Alpha        $${modelResults.finalCapital.toFixed(0).padStart(9)}    ${(modelResults.totalReturn*100).toFixed(1).padStart(6)}%    ${(modelResults.maxDrawdown*100).toFixed(1).padStart(5)}%    ${modelResults.trades}`);
        console.log(`    Buy & Hold            $${holdResults.finalCapital.toFixed(0).padStart(9)}    ${(holdResults.totalReturn*100).toFixed(1).padStart(6)}%    ${(holdResults.maxDrawdown*100).toFixed(1).padStart(5)}%    ${holdResults.trades}`);

        // Did we beat buy & hold?
        const outperformance = modelResults.totalReturn - holdResults.totalReturn;
        console.log(`\n    Outperformance: ${outperformance > 0 ? '+' : ''}${(outperformance*100).toFixed(1)}%`);

        // Key events analysis
        console.log('\n[7] KEY MARKET EVENTS DETECTION:\n');

        // Find high tension periods
        const highTension = analyses.filter(a => a.harmonic && a.harmonic.tension > 0.65);
        console.log(`    High tension periods detected: ${highTension.length} days`);

        if (highTension.length > 0) {
            console.log('\n    Sample high-tension days:');
            for (const ht of highTension.slice(0, 10)) {
                console.log(`      ${ht.date}: Tension ${(ht.harmonic.tension*100).toFixed(0)}%, Interval: ${ht.harmonic.interval.name}, VIX: ${ht.vix.toFixed(1)}`);
            }
        }

        // Trade log
        if (modelResults.tradeLog.length > 0) {
            console.log('\n[8] TRADE LOG:\n');
            for (const trade of modelResults.tradeLog.slice(-20)) {
                console.log(`    ${trade.date}: ${trade.action} position ${(trade.from*100).toFixed(0)}% -> ${(trade.to*100).toFixed(0)}%`);
            }
        }

        console.log('\n' + '='.repeat(70));
        console.log('  RESEARCH COMPLETE - REAL YAHOO FINANCE DATA');
        console.log('='.repeat(70));

        return { analyses, modelResults, holdResults };

    } catch (error) {
        console.error('Error:', error.message);
        console.error(error.stack);
        process.exit(1);
    }
}

// Run
runResearch();
