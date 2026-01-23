#!/usr/bin/env node
/**
 * Run Harmonic Research on saved Yahoo Finance data
 */

import fs from 'fs';

// Load saved data
function loadYahooData(filepath) {
    const raw = fs.readFileSync(filepath, 'utf8');
    const json = JSON.parse(raw);

    if (!json.chart || !json.chart.result || !json.chart.result[0]) {
        throw new Error(`Invalid data in ${filepath}`);
    }

    const result = json.chart.result[0];
    const timestamps = result.timestamp || [];
    const quotes = result.indicators.quote[0];

    const prices = [];
    for (let i = 0; i < timestamps.length; i++) {
        if (quotes.close[i] !== null) {
            prices.push({
                date: new Date(timestamps[i] * 1000).toISOString().split('T')[0],
                close: quotes.close[i]
            });
        }
    }
    return prices;
}

// Harmonic model
class HarmonicModel {
    constructor(lookback = 20) {
        this.lookback = lookback;
        this.history = [];
    }

    analyze(date, spx, vix, gold, dollar) {
        this.history.push({ date, spx, vix, gold, dollar });

        if (this.history.length < this.lookback) return null;

        const recent = this.history.slice(-this.lookback);

        // Price momentum
        const spxMomentum = (recent[recent.length-1].spx - recent[0].spx) / recent[0].spx;
        const goldMomentum = (recent[recent.length-1].gold - recent[0].gold) / recent[0].gold;
        const dollarMomentum = (recent[recent.length-1].dollar - recent[0].dollar) / recent[0].dollar;

        // Average VIX
        const avgVix = recent.reduce((s, r) => s + r.vix, 0) / recent.length;
        const normVix = avgVix / 20; // 20 = baseline

        // Tension calculation
        // High tension when: price up + VIX up, or price down + VIX low
        const priceSign = spxMomentum > 0 ? 1 : -1;
        const vixHigh = normVix > 1.2;
        const vixLow = normVix < 0.8;

        let tension = 0.3;

        // Divergence: rally with fear or decline with complacency
        if ((priceSign > 0 && vixHigh) || (priceSign < 0 && vixLow)) {
            tension += 0.35;
        }

        // Safe haven flows
        if (goldMomentum > 0.03 && dollarMomentum > 0.02) {
            tension += 0.15; // Risk-off signal
        }

        // Extreme VIX
        if (avgVix > 30) tension += 0.15;
        if (avgVix > 40) tension += 0.1;

        tension = Math.min(1, tension);

        // Regime
        let regime;
        if (tension > 0.65) {
            regime = 'HIGH_TENSION';
        } else if (spxMomentum > 0.02) {
            regime = tension > 0.45 ? 'UNSTABLE_BULL' : 'STABLE_BULL';
        } else if (spxMomentum < -0.02) {
            regime = tension > 0.45 ? 'UNSTABLE_BEAR' : 'STABLE_BEAR';
        } else {
            regime = 'CONSOLIDATION';
        }

        // Signal
        let signal = 'HOLD';
        let confidence = 0.5;

        if (tension > 0.65) {
            signal = 'REDUCE_RISK';
            confidence = tension;
        } else if (tension < 0.4 && spxMomentum > 0.02) {
            signal = 'STAY_LONG';
            confidence = 1 - tension;
        } else if (goldMomentum > 0.03 && dollarMomentum > 0.02 && tension > 0.5) {
            signal = 'DEFENSIVE';
            confidence = 0.7;
        }

        return {
            date, spx, vix,
            spxMomentum, avgVix, tension,
            regime, signal, confidence,
            goldMomentum, dollarMomentum
        };
    }
}

// Backtest
class Backtest {
    constructor(initial = 100000) {
        this.initial = initial;
        this.capital = initial;
        this.position = 1.0;
        this.trades = [];
        this.peak = initial;
        this.maxDD = 0;
    }

    step(date, signal, confidence, spxReturn) {
        // Apply return
        this.capital *= (1 + spxReturn * this.position);

        // Track drawdown
        if (this.capital > this.peak) this.peak = this.capital;
        const dd = (this.peak - this.capital) / this.peak;
        if (dd > this.maxDD) this.maxDD = dd;

        // Adjust position
        if (signal === 'REDUCE_RISK' && this.position > 0.3) {
            const newPos = Math.max(0.3, 1 - confidence);
            this.trades.push({ date, action: 'REDUCE', from: this.position, to: newPos });
            this.position = newPos;
        } else if (signal === 'DEFENSIVE' && this.position > 0.5) {
            this.trades.push({ date, action: 'DEFENSIVE', from: this.position, to: 0.5 });
            this.position = 0.5;
        } else if (signal === 'STAY_LONG' && this.position < 1.0) {
            this.trades.push({ date, action: 'INCREASE', from: this.position, to: 1.0 });
            this.position = 1.0;
        }
    }

    results() {
        return {
            finalCapital: this.capital,
            totalReturn: (this.capital - this.initial) / this.initial,
            maxDrawdown: this.maxDD,
            trades: this.trades.length
        };
    }
}

// Main
async function main() {
    console.log('='.repeat(70));
    console.log('  HARMONIC ALPHA - S&P 500 INDEX RESEARCH');
    console.log('  Real Yahoo Finance Data (2022-2024)');
    console.log('='.repeat(70));

    // Load data
    console.log('\n[1] LOADING DATA...\n');

    const spxData = loadYahooData('/tmp/spx_data.json');
    const vixData = loadYahooData('/tmp/vix_data.json');
    const goldData = loadYahooData('/tmp/gold_data.json');
    const dollarData = loadYahooData('/tmp/dollar_data.json');

    console.log(`    S&P 500: ${spxData.length} days`);
    console.log(`    VIX:     ${vixData.length} days`);
    console.log(`    Gold:    ${goldData.length} days`);
    console.log(`    Dollar:  ${dollarData.length} days`);

    // Merge by date
    const spxByDate = Object.fromEntries(spxData.map(d => [d.date, d.close]));
    const vixByDate = Object.fromEntries(vixData.map(d => [d.date, d.close]));
    const goldByDate = Object.fromEntries(goldData.map(d => [d.date, d.close]));
    const dollarByDate = Object.fromEntries(dollarData.map(d => [d.date, d.close]));

    const merged = [];
    for (const date of Object.keys(spxByDate).sort()) {
        if (spxByDate[date] && vixByDate[date] && goldByDate[date] && dollarByDate[date]) {
            merged.push({
                date,
                spx: spxByDate[date],
                vix: vixByDate[date],
                gold: goldByDate[date],
                dollar: dollarByDate[date]
            });
        }
    }

    console.log(`\n    Merged: ${merged.length} trading days`);
    console.log(`    Range: ${merged[0].date} to ${merged[merged.length-1].date}`);

    // Split: train on 2022, test on 2023-2024
    const train = merged.filter(d => d.date < '2023-01-01');
    const test = merged.filter(d => d.date >= '2023-01-01');

    console.log(`    Train (2022): ${train.length} days`);
    console.log(`    Test (2023-24): ${test.length} days`);

    // Run model
    console.log('\n[2] RUNNING HARMONIC MODEL...\n');

    const model = new HarmonicModel(20);
    const backtest = new Backtest(100000);
    const buyhold = new Backtest(100000);

    // Warm up on training data
    for (const d of train) {
        model.analyze(d.date, d.spx, d.vix, d.gold, d.dollar);
    }

    // Test
    const analyses = [];
    let prevSpx = test[0].spx;

    for (const d of test) {
        const ret = (d.spx - prevSpx) / prevSpx;

        const analysis = model.analyze(d.date, d.spx, d.vix, d.gold, d.dollar);
        if (analysis) {
            analyses.push(analysis);
            backtest.step(d.date, analysis.signal, analysis.confidence, ret);
            buyhold.step(d.date, 'HOLD', 0.5, ret);
        }

        prevSpx = d.spx;
    }

    // Results
    console.log('[3] REGIME DISTRIBUTION (Test Period):\n');

    const regimes = {};
    for (const a of analyses) {
        regimes[a.regime] = (regimes[a.regime] || 0) + 1;
    }

    for (const [regime, count] of Object.entries(regimes).sort((a,b) => b[1] - a[1])) {
        console.log(`    ${regime.padEnd(18)} ${count.toString().padStart(4)} days (${(count/analyses.length*100).toFixed(1)}%)`);
    }

    console.log('\n[4] HIGH TENSION PERIODS:\n');

    const highTension = analyses.filter(a => a.tension > 0.6);
    console.log(`    Found ${highTension.length} high-tension days\n`);

    // Group consecutive high tension days
    let currentStreak = [];
    const streaks = [];

    for (let i = 0; i < highTension.length; i++) {
        if (currentStreak.length === 0) {
            currentStreak.push(highTension[i]);
        } else {
            const prevDate = new Date(currentStreak[currentStreak.length-1].date);
            const currDate = new Date(highTension[i].date);
            const daysDiff = (currDate - prevDate) / (1000 * 60 * 60 * 24);

            if (daysDiff <= 5) {
                currentStreak.push(highTension[i]);
            } else {
                if (currentStreak.length >= 3) streaks.push([...currentStreak]);
                currentStreak = [highTension[i]];
            }
        }
    }
    if (currentStreak.length >= 3) streaks.push(currentStreak);

    console.log(`    Significant tension clusters: ${streaks.length}\n`);

    for (const streak of streaks.slice(0, 5)) {
        const start = streak[0];
        const end = streak[streak.length-1];
        const avgTension = streak.reduce((s,a) => s + a.tension, 0) / streak.length;
        const spxChange = ((end.spx - start.spx) / start.spx * 100).toFixed(1);

        console.log(`    ${start.date} to ${end.date} (${streak.length} days)`);
        console.log(`      Avg tension: ${(avgTension*100).toFixed(0)}%, SPX move: ${spxChange}%, Avg VIX: ${(streak.reduce((s,a)=>s+a.avgVix,0)/streak.length).toFixed(1)}`);
    }

    console.log('\n[5] PERFORMANCE COMPARISON:\n');

    const modelRes = backtest.results();
    const holdRes = buyhold.results();

    console.log('    Strategy           Final Value    Return    Max DD    Trades');
    console.log('    ' + '-'.repeat(58));
    console.log(`    Harmonic Alpha     $${modelRes.finalCapital.toFixed(0).padStart(9)}    ${(modelRes.totalReturn*100).toFixed(1).padStart(5)}%    ${(modelRes.maxDrawdown*100).toFixed(1).padStart(5)}%    ${modelRes.trades.toString().padStart(3)}`);
    console.log(`    Buy & Hold         $${holdRes.finalCapital.toFixed(0).padStart(9)}    ${(holdRes.totalReturn*100).toFixed(1).padStart(5)}%    ${(holdRes.maxDrawdown*100).toFixed(1).padStart(5)}%    ${holdRes.trades.toString().padStart(3)}`);

    const outperf = modelRes.totalReturn - holdRes.totalReturn;
    console.log(`\n    Outperformance: ${outperf > 0 ? '+' : ''}${(outperf*100).toFixed(1)}%`);

    // Trade log
    if (modelRes.trades > 0) {
        console.log('\n[6] TRADE LOG:\n');
        for (const t of backtest.trades.slice(-15)) {
            console.log(`    ${t.date}: ${t.action.padEnd(10)} ${(t.from*100).toFixed(0)}% -> ${(t.to*100).toFixed(0)}%`);
        }
    }

    // Check if high tension predicted drawdowns
    console.log('\n[7] PREDICTIVE ANALYSIS:\n');

    let correctWarnings = 0;
    let totalWarnings = 0;

    for (let i = 0; i < analyses.length - 5; i++) {
        if (analyses[i].signal === 'REDUCE_RISK' || analyses[i].signal === 'DEFENSIVE') {
            totalWarnings++;
            // Check if SPX dropped in next 5 days
            const future = analyses.slice(i+1, i+6);
            const anyDrop = future.some(f => f.spx < analyses[i].spx * 0.98);
            if (anyDrop) correctWarnings++;
        }
    }

    if (totalWarnings > 0) {
        console.log(`    Risk warnings issued: ${totalWarnings}`);
        console.log(`    Followed by 2%+ drop within 5 days: ${correctWarnings}`);
        console.log(`    Accuracy: ${(correctWarnings/totalWarnings*100).toFixed(1)}%`);
    } else {
        console.log('    No risk warnings issued in test period');
    }

    console.log('\n' + '='.repeat(70));
    console.log('  COMPLETE - REAL DATA FROM YAHOO FINANCE');
    console.log('='.repeat(70));
}

main().catch(console.error);
