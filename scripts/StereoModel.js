#!/usr/bin/env node
/**
 * Stereoscopic Market Model
 *
 * Combines:
 * - Geometric tension → MAGNITUDE (how big will the move be?)
 * - Legacy indicators → DIRECTION (which way?)
 *
 * The geometric model doesn't compete with momentum/trend - it complements them.
 */

import fs from 'fs';

function loadData() {
    const spxRaw = JSON.parse(fs.readFileSync('/tmp/spx_data.json', 'utf8'));
    const vixRaw = JSON.parse(fs.readFileSync('/tmp/vix_data.json', 'utf8'));
    const goldRaw = JSON.parse(fs.readFileSync('/tmp/gold_data.json', 'utf8'));
    const dollarRaw = JSON.parse(fs.readFileSync('/tmp/dollar_data.json', 'utf8'));

    const spx = spxRaw.chart.result[0];
    const vix = vixRaw.chart.result[0];
    const gold = goldRaw.chart.result[0];
    const dollar = dollarRaw.chart.result[0];

    const data = [];
    for (let i = 0; i < spx.timestamp.length; i++) {
        const date = new Date(spx.timestamp[i] * 1000).toISOString().split('T')[0];
        const spxClose = spx.indicators.quote[0].close[i];
        const vixClose = vix.indicators.quote[0].close[i];
        const goldClose = gold.indicators.quote[0].close[i];
        const dollarClose = dollar.indicators.quote[0].close[i];
        if (spxClose && vixClose && goldClose && dollarClose) {
            data.push({ date, spx: spxClose, vix: vixClose, gold: goldClose, dollar: dollarClose });
        }
    }
    return data;
}

// ============================================================================
// LEGACY DIRECTION INDICATORS
// ============================================================================

class DirectionIndicators {
    constructor() {
        this.prices = [];
    }

    addPrice(price) {
        this.prices.push(price);
        if (this.prices.length > 200) this.prices.shift();
    }

    // Simple Moving Average crossover
    smaSignal() {
        if (this.prices.length < 50) return { signal: 0, name: 'SMA', confidence: 0 };

        const sma20 = this.prices.slice(-20).reduce((a,b) => a+b, 0) / 20;
        const sma50 = this.prices.slice(-50).reduce((a,b) => a+b, 0) / 50;

        const signal = sma20 > sma50 ? 1 : -1;
        const strength = Math.abs(sma20 - sma50) / sma50;
        const confidence = Math.min(1, strength * 20);

        return { signal, name: 'SMA_20_50', confidence };
    }

    // Momentum (rate of change)
    momentumSignal(period = 20) {
        if (this.prices.length < period) return { signal: 0, name: 'MOM', confidence: 0 };

        const current = this.prices[this.prices.length - 1];
        const past = this.prices[this.prices.length - period];
        const momentum = (current - past) / past;

        const signal = momentum > 0.01 ? 1 : momentum < -0.01 ? -1 : 0;
        const confidence = Math.min(1, Math.abs(momentum) * 10);

        return { signal, name: `MOM_${period}`, confidence };
    }

    // RSI
    rsiSignal(period = 14) {
        if (this.prices.length < period + 1) return { signal: 0, name: 'RSI', confidence: 0 };

        const changes = [];
        for (let i = this.prices.length - period; i < this.prices.length; i++) {
            changes.push(this.prices[i] - this.prices[i-1]);
        }

        const gains = changes.filter(c => c > 0);
        const losses = changes.filter(c => c < 0).map(c => Math.abs(c));

        const avgGain = gains.length ? gains.reduce((a,b) => a+b, 0) / period : 0;
        const avgLoss = losses.length ? losses.reduce((a,b) => a+b, 0) / period : 0;

        const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
        const rsi = 100 - (100 / (1 + rs));

        // RSI signals: oversold < 30 (buy), overbought > 70 (sell)
        let signal = 0;
        let confidence = 0;

        if (rsi < 30) {
            signal = 1; // Oversold, expect bounce
            confidence = (30 - rsi) / 30;
        } else if (rsi > 70) {
            signal = -1; // Overbought, expect pullback
            confidence = (rsi - 70) / 30;
        }

        return { signal, name: 'RSI', confidence, value: rsi };
    }

    // MACD
    macdSignal() {
        if (this.prices.length < 35) return { signal: 0, name: 'MACD', confidence: 0 };

        const ema = (prices, period) => {
            const k = 2 / (period + 1);
            let ema = prices.slice(0, period).reduce((a,b) => a+b, 0) / period;
            for (let i = period; i < prices.length; i++) {
                ema = prices[i] * k + ema * (1 - k);
            }
            return ema;
        };

        const ema12 = ema(this.prices, 12);
        const ema26 = ema(this.prices, 26);
        const macd = ema12 - ema26;

        // Simplified: positive MACD = bullish
        const signal = macd > 0 ? 1 : -1;
        const confidence = Math.min(1, Math.abs(macd) / this.prices[this.prices.length-1] * 100);

        return { signal, name: 'MACD', confidence };
    }

    // Combine all direction signals
    getCombinedDirection() {
        const signals = [
            this.smaSignal(),
            this.momentumSignal(10),
            this.momentumSignal(20),
            this.rsiSignal(),
            this.macdSignal()
        ];

        // Weighted vote
        let weightedSum = 0;
        let totalWeight = 0;

        for (const s of signals) {
            weightedSum += s.signal * s.confidence;
            totalWeight += s.confidence;
        }

        const consensus = totalWeight > 0 ? weightedSum / totalWeight : 0;

        // Agreement level (do signals agree?)
        const bullish = signals.filter(s => s.signal > 0).length;
        const bearish = signals.filter(s => s.signal < 0).length;
        const agreement = Math.max(bullish, bearish) / signals.length;

        return {
            direction: consensus > 0.2 ? 'BULLISH' : consensus < -0.2 ? 'BEARISH' : 'NEUTRAL',
            consensus,
            agreement,
            signals
        };
    }
}

// ============================================================================
// GEOMETRIC MAGNITUDE MODEL (from previous work)
// ============================================================================

class GeometricMagnitude {
    constructor() {
        this.lookback = 20;
        this.history = [];
    }

    addPoint(d) {
        this.history.push(d);
        if (this.history.length > 60) this.history.shift();
    }

    getTension() {
        if (this.history.length < this.lookback) return null;

        const recent = this.history.slice(-this.lookback);

        const priceMom = (recent[recent.length-1].spx - recent[0].spx) / recent[0].spx;
        const avgVix = recent.reduce((s, r) => s + r.vix, 0) / recent.length;
        const goldMom = (recent[recent.length-1].gold - recent[0].gold) / recent[0].gold;
        const dollarMom = (recent[recent.length-1].dollar - recent[0].dollar) / recent[0].dollar;

        let tension = 30;

        if (avgVix > 30) tension += 30;
        else if (avgVix > 25) tension += 20;
        else if (avgVix > 20) tension += 10;
        else if (avgVix < 13) tension += 15;

        if ((priceMom > 0.03 && avgVix > 20) || (priceMom < -0.03 && avgVix < 15)) {
            tension += 25;
        }

        if ((goldMom > 0.02 && dollarMom > 0.01) || (goldMom < -0.02 && dollarMom < -0.01)) {
            tension += 15;
        }

        tension = Math.min(100, tension);

        // Convert tension to expected magnitude
        let expectedMagnitude;
        if (tension >= 80) expectedMagnitude = { min: 5, max: 10, category: 'VERY_LARGE' };
        else if (tension >= 60) expectedMagnitude = { min: 3, max: 6, category: 'LARGE' };
        else if (tension >= 45) expectedMagnitude = { min: 2, max: 4, category: 'MODERATE' };
        else if (tension >= 30) expectedMagnitude = { min: 1, max: 3, category: 'SMALL' };
        else expectedMagnitude = { min: 0, max: 2, category: 'MINIMAL' };

        // Harmonic quality
        const normPrice = (priceMom + 0.2) / 0.4;
        const normVix = (avgVix - 10) / 40;
        const ratio = (normPrice + 0.1) / (normVix + 0.1);

        let harmonic;
        if (Math.abs(ratio - 1.414) < 0.1) harmonic = 'TRITONE'; // Most unstable
        else if (Math.abs(ratio - 1.5) < 0.1) harmonic = 'FIFTH'; // Most stable
        else if (ratio > 1.8) harmonic = 'SEVENTH'; // Seeking resolution
        else harmonic = 'MIXED';

        return {
            tension,
            expectedMagnitude,
            harmonic,
            avgVix,
            priceMom: priceMom * 100
        };
    }
}

// ============================================================================
// STEREOSCOPIC COMBINER
// ============================================================================

class StereoModel {
    constructor() {
        this.direction = new DirectionIndicators();
        this.magnitude = new GeometricMagnitude();
    }

    analyze(d) {
        this.direction.addPrice(d.spx);
        this.magnitude.addPoint(d);

        const dir = this.direction.getCombinedDirection();
        const mag = this.magnitude.getTension();

        if (!mag) return null;

        // Combine into actionable signal
        let action, positionSize, confidence;

        if (mag.tension >= 60) {
            // High tension: expect big move
            if (dir.direction === 'BULLISH' && dir.agreement > 0.6) {
                action = 'LONG_LEVERAGED';
                positionSize = 1.5; // 150% of normal
                confidence = dir.agreement * (mag.tension / 100);
            } else if (dir.direction === 'BEARISH' && dir.agreement > 0.6) {
                action = 'SHORT_OR_HEDGE';
                positionSize = 0.5;
                confidence = dir.agreement * (mag.tension / 100);
            } else {
                action = 'STRADDLE'; // High vol, unclear direction
                positionSize = 1.0;
                confidence = mag.tension / 100;
            }
        } else if (mag.tension < 35) {
            // Low tension: expect small move
            if (dir.direction === 'BULLISH') {
                action = 'LONG_NORMAL';
                positionSize = 1.0;
                confidence = dir.agreement * 0.5;
            } else if (dir.direction === 'BEARISH') {
                action = 'REDUCE_EXPOSURE';
                positionSize = 0.7;
                confidence = dir.agreement * 0.5;
            } else {
                action = 'HOLD';
                positionSize = 1.0;
                confidence = 0.3;
            }
        } else {
            // Moderate tension
            if (dir.direction === 'BULLISH') {
                action = 'LONG_CAUTIOUS';
                positionSize = 0.8;
                confidence = dir.agreement * 0.7;
            } else if (dir.direction === 'BEARISH') {
                action = 'REDUCE_OR_HEDGE';
                positionSize = 0.6;
                confidence = dir.agreement * 0.7;
            } else {
                action = 'HOLD_WATCH';
                positionSize = 0.8;
                confidence = 0.4;
            }
        }

        return {
            date: d.date,
            price: d.spx,
            direction: dir,
            magnitude: mag,
            combined: {
                action,
                positionSize,
                confidence,
                expectedMove: `${dir.direction === 'BEARISH' ? '-' : '+'}${mag.expectedMagnitude.min}-${mag.expectedMagnitude.max}%`
            }
        };
    }
}

// ============================================================================
// TEST THE STEREO MODEL
// ============================================================================

function runTest() {
    console.log('='.repeat(70));
    console.log('  STEREOSCOPIC MODEL TEST');
    console.log('  Direction (Legacy) + Magnitude (Geometric)');
    console.log('='.repeat(70));

    const data = loadData();
    const model = new StereoModel();
    const forwardDays = 20;
    const results = [];

    for (let i = 0; i < data.length - forwardDays; i++) {
        const analysis = model.analyze(data[i]);
        if (!analysis) continue;

        // Get actual future results
        const future = data.slice(i, i + forwardDays + 1);
        const actualReturn = (future[future.length-1].spx - future[0].spx) / future[0].spx * 100;
        const maxMove = Math.max(
            Math.abs((Math.max(...future.map(f => f.spx)) - future[0].spx) / future[0].spx * 100),
            Math.abs((Math.min(...future.map(f => f.spx)) - future[0].spx) / future[0].spx * 100)
        );

        results.push({
            ...analysis,
            actualReturn,
            maxMove
        });
    }

    console.log(`\nTested ${results.length} predictions\n`);

    // Evaluate direction accuracy
    console.log('[1] DIRECTION ACCURACY (from legacy indicators):\n');

    let dirCorrect = 0;
    for (const r of results) {
        const predicted = r.direction.direction;
        const actual = r.actualReturn > 1 ? 'BULLISH' : r.actualReturn < -1 ? 'BEARISH' : 'NEUTRAL';
        if (predicted === actual || (predicted === 'NEUTRAL' && Math.abs(r.actualReturn) < 2)) {
            dirCorrect++;
        }
    }
    console.log(`  Direction accuracy: ${(dirCorrect / results.length * 100).toFixed(1)}%`);

    // Evaluate magnitude accuracy
    console.log('\n[2] MAGNITUDE ACCURACY (from geometric model):\n');

    let magCorrect = 0;
    for (const r of results) {
        const pred = r.magnitude.expectedMagnitude;
        if (r.maxMove >= pred.min * 0.5 && r.maxMove <= pred.max * 1.5) {
            magCorrect++;
        }
    }
    console.log(`  Magnitude accuracy: ${(magCorrect / results.length * 100).toFixed(1)}%`);

    // Evaluate combined signal
    console.log('\n[3] COMBINED SIGNAL ACCURACY:\n');

    const actionResults = {};
    for (const r of results) {
        const action = r.combined.action;
        if (!actionResults[action]) {
            actionResults[action] = { count: 0, totalReturn: 0, wins: 0 };
        }
        actionResults[action].count++;
        actionResults[action].totalReturn += r.actualReturn;
        if ((action.includes('LONG') && r.actualReturn > 0) ||
            (action.includes('SHORT') && r.actualReturn < 0) ||
            (action.includes('HEDGE') && r.actualReturn < 0) ||
            (action.includes('REDUCE') && r.actualReturn < 2)) {
            actionResults[action].wins++;
        }
    }

    console.log('Action             Count   Avg Return   Win Rate');
    console.log('-'.repeat(55));

    for (const [action, stats] of Object.entries(actionResults).sort((a,b) => b[1].count - a[1].count)) {
        const avgRet = (stats.totalReturn / stats.count).toFixed(2);
        const winRate = (stats.wins / stats.count * 100).toFixed(1);
        console.log(`${action.padEnd(18)} ${stats.count.toString().padStart(4)}    ${avgRet.padStart(7)}%    ${winRate.padStart(6)}%`);
    }

    // Sample recent predictions
    console.log('\n[4] RECENT PREDICTIONS:\n');
    console.log('Date         Action              Expected    Actual   Correct?');
    console.log('-'.repeat(65));

    for (const r of results.slice(-15)) {
        const expected = r.combined.expectedMove;
        const actual = `${r.actualReturn >= 0 ? '+' : ''}${r.actualReturn.toFixed(1)}%`;
        const dirOk = (r.direction.direction === 'BULLISH' && r.actualReturn > 0) ||
                      (r.direction.direction === 'BEARISH' && r.actualReturn < 0) ||
                      r.direction.direction === 'NEUTRAL';
        const magOk = r.maxMove >= r.magnitude.expectedMagnitude.min * 0.5 &&
                      r.maxMove <= r.magnitude.expectedMagnitude.max * 1.5;

        const result = dirOk && magOk ? '✓ BOTH' : dirOk ? '◐ DIR' : magOk ? '◐ MAG' : '✗';

        console.log(
            `${r.date}   ${r.combined.action.padEnd(18)} ${expected.padEnd(10)}  ${actual.padEnd(7)}  ${result}`
        );
    }

    console.log('\n' + '='.repeat(70));
}

runTest();
