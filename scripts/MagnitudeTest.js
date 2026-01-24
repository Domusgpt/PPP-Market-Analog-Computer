#!/usr/bin/env node
/**
 * Magnitude Prediction Test
 *
 * Hypothesis: The model predicts SCALE of moves, not direction
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

class MagnitudePredictor {
    constructor() {
        this.lookback = 20;
        this.history = [];
    }

    addPoint(d) {
        this.history.push(d);
        if (this.history.length > 60) this.history.shift();
    }

    analyze(d) {
        this.addPoint(d);
        if (this.history.length < this.lookback) return null;

        const recent = this.history.slice(-this.lookback);

        // Calculate inputs
        const priceMom = (recent[recent.length-1].spx - recent[0].spx) / recent[0].spx;
        const avgVix = recent.reduce((s, r) => s + r.vix, 0) / recent.length;
        const goldMom = (recent[recent.length-1].gold - recent[0].gold) / recent[0].gold;
        const dollarMom = (recent[recent.length-1].dollar - recent[0].dollar) / recent[0].dollar;

        // Tension score (0-100)
        let tension = 30;

        // VIX contribution
        if (avgVix > 30) tension += 30;
        else if (avgVix > 25) tension += 20;
        else if (avgVix > 20) tension += 10;
        else if (avgVix < 13) tension += 15; // Complacency adds tension too

        // Divergence contribution
        if ((priceMom > 0.03 && avgVix > 20) || (priceMom < -0.03 && avgVix < 15)) {
            tension += 25;
        }

        // Safe haven contribution
        if ((goldMom > 0.02 && dollarMom > 0.01) || (goldMom < -0.02 && dollarMom < -0.01)) {
            tension += 15;
        }

        // Recent volatility
        const returns = [];
        for (let i = 1; i < recent.length; i++) {
            returns.push(Math.abs((recent[i].spx - recent[i-1].spx) / recent[i-1].spx));
        }
        const avgDailyVol = returns.reduce((a,b) => a+b, 0) / returns.length;
        if (avgDailyVol > 0.015) tension += 15;

        tension = Math.min(100, tension);

        // MAGNITUDE PREDICTION based on tension
        // Higher tension = expect larger moves
        let predictedMagnitude;
        let magnitudeCategory;

        if (tension >= 80) {
            predictedMagnitude = { min: 5, max: 10 };
            magnitudeCategory = 'VERY_LARGE';
        } else if (tension >= 60) {
            predictedMagnitude = { min: 3, max: 6 };
            magnitudeCategory = 'LARGE';
        } else if (tension >= 45) {
            predictedMagnitude = { min: 2, max: 4 };
            magnitudeCategory = 'MODERATE';
        } else if (tension >= 30) {
            predictedMagnitude = { min: 1, max: 3 };
            magnitudeCategory = 'SMALL';
        } else {
            predictedMagnitude = { min: 0, max: 2 };
            magnitudeCategory = 'MINIMAL';
        }

        return {
            date: d.date,
            price: d.spx,
            vix: d.vix,
            tension,
            predictedMagnitude,
            magnitudeCategory,
            avgVix,
            priceMom: priceMom * 100,
            avgDailyVol: avgDailyVol * 100
        };
    }
}

function runTest() {
    console.log('='.repeat(70));
    console.log('  MAGNITUDE PREDICTION TEST');
    console.log('  Can the model predict SIZE of moves (ignoring direction)?');
    console.log('='.repeat(70));

    const data = loadData();
    console.log(`\nData: ${data.length} days (${data[0].date} to ${data[data.length-1].date})\n`);

    const predictor = new MagnitudePredictor();
    const forwardDays = 20;
    const results = [];

    // Analyze each point and compare to future
    for (let i = 0; i < data.length - forwardDays; i++) {
        const analysis = predictor.analyze(data[i]);
        if (!analysis) continue;

        // Calculate actual magnitude over next 20 days
        const futureSlice = data.slice(i, i + forwardDays + 1);
        const startPrice = futureSlice[0].spx;
        const endPrice = futureSlice[futureSlice.length - 1].spx;
        const minPrice = Math.min(...futureSlice.map(d => d.spx));
        const maxPrice = Math.max(...futureSlice.map(d => d.spx));

        const actualReturn = Math.abs((endPrice - startPrice) / startPrice) * 100;
        const maxDrawdown = Math.abs((minPrice - startPrice) / startPrice) * 100;
        const maxGain = ((maxPrice - startPrice) / startPrice) * 100;
        const totalRange = ((maxPrice - minPrice) / startPrice) * 100;

        results.push({
            ...analysis,
            actualReturn,
            maxDrawdown,
            maxGain,
            totalRange,
            actualMagnitude: Math.max(actualReturn, maxDrawdown, Math.abs(maxGain))
        });
    }

    // Group by predicted magnitude category
    const categories = ['MINIMAL', 'SMALL', 'MODERATE', 'LARGE', 'VERY_LARGE'];

    console.log('[1] MAGNITUDE PREDICTION BY CATEGORY:\n');
    console.log('Category      Count   Pred Range   Actual Avg   Actual Med   Hit Rate');
    console.log('-'.repeat(75));

    const categoryStats = {};

    for (const cat of categories) {
        const group = results.filter(r => r.magnitudeCategory === cat);
        if (group.length === 0) continue;

        const actualMags = group.map(r => r.actualMagnitude).sort((a,b) => a-b);
        const avgActual = actualMags.reduce((a,b) => a+b, 0) / actualMags.length;
        const medActual = actualMags[Math.floor(actualMags.length / 2)];

        // Get predicted range for this category
        const predMin = group[0].predictedMagnitude.min;
        const predMax = group[0].predictedMagnitude.max;

        // Hit rate: how often was actual within predicted range (with some tolerance)
        const hits = group.filter(r =>
            r.actualMagnitude >= predMin * 0.5 && r.actualMagnitude <= predMax * 1.5
        ).length;
        const hitRate = (hits / group.length * 100).toFixed(1);

        categoryStats[cat] = { count: group.length, avgActual, medActual, hitRate, predMin, predMax };

        console.log(
            `${cat.padEnd(12)}  ${group.length.toString().padStart(4)}   ` +
            `${predMin}-${predMax}%`.padEnd(10) + `  ` +
            `${avgActual.toFixed(1)}%`.padStart(8) + `     ` +
            `${medActual.toFixed(1)}%`.padStart(6) + `      ` +
            `${hitRate}%`
        );
    }

    // Correlation analysis
    console.log('\n[2] TENSION vs ACTUAL MAGNITUDE CORRELATION:\n');

    const tensions = results.map(r => r.tension);
    const magnitudes = results.map(r => r.actualMagnitude);

    // Calculate Pearson correlation
    const n = tensions.length;
    const sumT = tensions.reduce((a,b) => a+b, 0);
    const sumM = magnitudes.reduce((a,b) => a+b, 0);
    const sumTM = tensions.reduce((s, t, i) => s + t * magnitudes[i], 0);
    const sumT2 = tensions.reduce((s, t) => s + t*t, 0);
    const sumM2 = magnitudes.reduce((s, m) => s + m*m, 0);

    const correlation = (n * sumTM - sumT * sumM) /
        Math.sqrt((n * sumT2 - sumT*sumT) * (n * sumM2 - sumM*sumM));

    console.log(`  Pearson correlation (tension vs magnitude): ${correlation.toFixed(3)}`);
    console.log(`  Interpretation: ${
        correlation > 0.5 ? 'STRONG positive' :
        correlation > 0.3 ? 'MODERATE positive' :
        correlation > 0.1 ? 'WEAK positive' :
        correlation > -0.1 ? 'NO correlation' :
        correlation > -0.3 ? 'WEAK negative' :
        'MODERATE negative'
    }`);

    // Bucket analysis
    console.log('\n[3] TENSION BUCKETS vs ACTUAL MAGNITUDE:\n');

    const buckets = [
        { name: '0-30 (Low)', min: 0, max: 30 },
        { name: '31-45 (Med-Low)', min: 31, max: 45 },
        { name: '46-60 (Medium)', min: 46, max: 60 },
        { name: '61-80 (High)', min: 61, max: 80 },
        { name: '81-100 (Extreme)', min: 81, max: 100 }
    ];

    console.log('Tension Bucket     Count    Avg Magnitude    Avg VIX    Avg Range');
    console.log('-'.repeat(70));

    for (const bucket of buckets) {
        const group = results.filter(r => r.tension >= bucket.min && r.tension <= bucket.max);
        if (group.length === 0) continue;

        const avgMag = group.reduce((s, r) => s + r.actualMagnitude, 0) / group.length;
        const avgVix = group.reduce((s, r) => s + r.avgVix, 0) / group.length;
        const avgRange = group.reduce((s, r) => s + r.totalRange, 0) / group.length;

        console.log(
            `${bucket.name.padEnd(18)} ${group.length.toString().padStart(4)}    ` +
            `${avgMag.toFixed(2)}%`.padStart(10) + `       ` +
            `${avgVix.toFixed(1)}`.padStart(5) + `      ` +
            `${avgRange.toFixed(2)}%`
        );
    }

    // Show specific examples where high tension preceded big moves
    console.log('\n[4] HIGH TENSION EXAMPLES (Tension > 60):\n');

    const highTension = results.filter(r => r.tension > 60).slice(0, 15);
    console.log('Date         Tension  Pred Range   Actual Mag   VIX    Result');
    console.log('-'.repeat(65));

    for (const r of highTension) {
        const inRange = r.actualMagnitude >= r.predictedMagnitude.min * 0.5 &&
                        r.actualMagnitude <= r.predictedMagnitude.max * 1.5;
        const result = inRange ? '✓ HIT' : '✗ MISS';

        console.log(
            `${r.date}   ${r.tension.toString().padStart(3)}      ` +
            `${r.predictedMagnitude.min}-${r.predictedMagnitude.max}%`.padEnd(8) + `   ` +
            `${r.actualMagnitude.toFixed(1)}%`.padStart(7) + `      ` +
            `${r.avgVix.toFixed(1)}`.padStart(4) + `   ${result}`
        );
    }

    // Compare to VIX as baseline
    console.log('\n[5] COMPARISON: TENSION vs VIX as MAGNITUDE PREDICTOR:\n');

    // VIX correlation with magnitude
    const vixValues = results.map(r => r.avgVix);
    const sumV = vixValues.reduce((a,b) => a+b, 0);
    const sumVM = vixValues.reduce((s, v, i) => s + v * magnitudes[i], 0);
    const sumV2 = vixValues.reduce((s, v) => s + v*v, 0);

    const vixCorrelation = (n * sumVM - sumV * sumM) /
        Math.sqrt((n * sumV2 - sumV*sumV) * (n * sumM2 - sumM*sumM));

    console.log(`  Tension correlation with magnitude: ${correlation.toFixed(3)}`);
    console.log(`  VIX correlation with magnitude:     ${vixCorrelation.toFixed(3)}`);
    console.log(`  Difference: ${(correlation - vixCorrelation).toFixed(3)} (${correlation > vixCorrelation ? 'Tension better' : 'VIX better'})`);

    // Summary
    console.log('\n' + '='.repeat(70));
    console.log('  SUMMARY');
    console.log('='.repeat(70));

    const overallHitRate = results.filter(r =>
        r.actualMagnitude >= r.predictedMagnitude.min * 0.5 &&
        r.actualMagnitude <= r.predictedMagnitude.max * 1.5
    ).length / results.length * 100;

    console.log(`\n  Total predictions: ${results.length}`);
    console.log(`  Overall magnitude hit rate: ${overallHitRate.toFixed(1)}%`);
    console.log(`  Tension-Magnitude correlation: ${correlation.toFixed(3)}`);
    console.log(`  vs VIX-Magnitude correlation: ${vixCorrelation.toFixed(3)}`);

    if (correlation > 0.3 && overallHitRate > 50) {
        console.log('\n  CONCLUSION: Model shows MEANINGFUL magnitude prediction ability');
    } else if (correlation > 0.15 || overallHitRate > 40) {
        console.log('\n  CONCLUSION: Model shows WEAK magnitude prediction ability');
    } else {
        console.log('\n  CONCLUSION: Model does NOT reliably predict magnitude');
    }

    console.log('\n' + '='.repeat(70));
}

runTest();
