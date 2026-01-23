#!/usr/bin/env node
/**
 * Tuned Geometric Market Analysis
 *
 * Properly calibrated thresholds for real market data
 */

import fs from 'fs';

// ============================================================================
// LOAD AND PREPARE DATA
// ============================================================================

function loadData() {
    const spxRaw = JSON.parse(fs.readFileSync('/tmp/spx_data.json', 'utf8'));
    const vixRaw = JSON.parse(fs.readFileSync('/tmp/vix_data.json', 'utf8'));
    const goldRaw = JSON.parse(fs.readFileSync('/tmp/gold_data.json', 'utf8'));
    const dollarRaw = JSON.parse(fs.readFileSync('/tmp/dollar_data.json', 'utf8'));

    const spx = spxRaw.chart.result[0];
    const vix = vixRaw.chart.result[0];
    const gold = goldRaw.chart.result[0];
    const dollar = dollarRaw.chart.result[0];

    // Build date-indexed data
    const data = [];
    const n = spx.timestamp.length;

    for (let i = 0; i < n; i++) {
        const date = new Date(spx.timestamp[i] * 1000).toISOString().split('T')[0];
        const spxClose = spx.indicators.quote[0].close[i];
        const vixClose = vix.indicators.quote[0].close[i];
        const goldClose = gold.indicators.quote[0].close[i];
        const dollarClose = dollar.indicators.quote[0].close[i];

        if (spxClose && vixClose && goldClose && dollarClose) {
            data.push({
                date,
                spx: spxClose,
                vix: vixClose,
                gold: goldClose,
                dollar: dollarClose
            });
        }
    }

    return data;
}

// ============================================================================
// TUNED GEOMETRIC ANALYZER
// ============================================================================

class TunedGeometricAnalyzer {
    constructor(config = {}) {
        this.lookback = config.lookback || 20;
        this.history = [];

        // Tuned thresholds based on market data characteristics
        this.thresholds = {
            priceVolatility: 0.02,    // 2% price move is significant
            vixLow: 15,               // VIX below 15 = complacent
            vixHigh: 25,              // VIX above 25 = fearful
            vixExtreme: 35,           // VIX above 35 = panic
            momentumStrong: 0.05,     // 5% momentum is strong
            divergenceThreshold: 0.3  // Normalized divergence threshold
        };
    }

    addPoint(d) {
        this.history.push(d);
        if (this.history.length > 60) this.history.shift();
    }

    analyze(d) {
        this.addPoint(d);

        if (this.history.length < this.lookback) {
            return { status: 'warming', needed: this.lookback - this.history.length };
        }

        const recent = this.history.slice(-this.lookback);
        const current = this.history[this.history.length - 1];
        const prev = this.history[this.history.length - 2];

        // 1. PRICE ANALYSIS
        const priceStart = recent[0].spx;
        const priceEnd = recent[recent.length - 1].spx;
        const priceMomentum = (priceEnd - priceStart) / priceStart;
        const dailyReturn = prev ? (current.spx - prev.spx) / prev.spx : 0;

        // 2. VOLATILITY REGIME
        const avgVix = recent.reduce((s, r) => s + r.vix, 0) / recent.length;
        const currentVix = current.vix;

        let volRegime;
        if (currentVix < this.thresholds.vixLow) {
            volRegime = 'COMPLACENT';
        } else if (currentVix < this.thresholds.vixHigh) {
            volRegime = 'NORMAL';
        } else if (currentVix < this.thresholds.vixExtreme) {
            volRegime = 'ELEVATED';
        } else {
            volRegime = 'PANIC';
        }

        // 3. SAFE HAVEN FLOWS
        const goldStart = recent[0].gold;
        const goldEnd = recent[recent.length - 1].gold;
        const goldMomentum = (goldEnd - goldStart) / goldStart;

        const dollarStart = recent[0].dollar;
        const dollarEnd = recent[recent.length - 1].dollar;
        const dollarMomentum = (dollarEnd - dollarStart) / dollarStart;

        let safeHavenFlow;
        if (goldMomentum > 0.02 && dollarMomentum > 0.01) {
            safeHavenFlow = 'STRONG_RISK_OFF';
        } else if (goldMomentum > 0.01 || dollarMomentum > 0.01) {
            safeHavenFlow = 'MILD_RISK_OFF';
        } else if (goldMomentum < -0.02 && dollarMomentum < -0.01) {
            safeHavenFlow = 'STRONG_RISK_ON';
        } else if (goldMomentum < -0.01 || dollarMomentum < -0.01) {
            safeHavenFlow = 'MILD_RISK_ON';
        } else {
            safeHavenFlow = 'NEUTRAL';
        }

        // 4. HARMONIC INTERVAL (price vs volatility relationship)
        // Normalize: price momentum [-0.2, 0.2] -> [0, 1]
        // VIX [10, 50] -> [0, 1]
        const normPrice = (priceMomentum + 0.2) / 0.4;
        const normVix = (avgVix - 10) / 40;

        // The "interval" is the ratio relationship
        const intervalRatio = (normPrice + 0.1) / (normVix + 0.1);

        const intervals = [
            { name: 'Unison', ratio: 1.0, quality: 'aligned' },
            { name: 'Minor Second', ratio: 1.067, quality: 'tense' },
            { name: 'Major Second', ratio: 1.125, quality: 'moving' },
            { name: 'Minor Third', ratio: 1.2, quality: 'soft' },
            { name: 'Major Third', ratio: 1.25, quality: 'bright' },
            { name: 'Perfect Fourth', ratio: 1.333, quality: 'suspended' },
            { name: 'Tritone', ratio: 1.414, quality: 'unstable' },
            { name: 'Perfect Fifth', ratio: 1.5, quality: 'stable' },
            { name: 'Minor Sixth', ratio: 1.6, quality: 'bittersweet' },
            { name: 'Major Sixth', ratio: 1.667, quality: 'warm' },
            { name: 'Minor Seventh', ratio: 1.778, quality: 'dominant' },
            { name: 'Major Seventh', ratio: 1.875, quality: 'leading' },
            { name: 'Octave', ratio: 2.0, quality: 'complete' },
        ];

        // Normalize ratio to [0.5, 2] range
        let ratio = Math.max(0.5, Math.min(2, intervalRatio));

        let closestInterval = intervals[0];
        let minDist = Infinity;
        for (const int of intervals) {
            const dist = Math.abs(ratio - int.ratio);
            if (dist < minDist) {
                minDist = dist;
                closestInterval = int;
            }
        }

        // 5. TENSION SCORE (0-100)
        let tension = 30; // Base

        // Price-vol divergence adds tension
        if ((priceMomentum > 0.03 && volRegime === 'ELEVATED') ||
            (priceMomentum < -0.03 && volRegime === 'COMPLACENT')) {
            tension += 30;
        }

        // Safe haven divergence
        if (safeHavenFlow.includes('RISK_OFF') && priceMomentum > 0) {
            tension += 20;
        }
        if (safeHavenFlow.includes('RISK_ON') && priceMomentum < 0) {
            tension += 20;
        }

        // Extreme VIX
        if (volRegime === 'PANIC') tension += 25;
        if (volRegime === 'ELEVATED') tension += 10;

        // Tritone interval
        if (closestInterval.name === 'Tritone') tension += 15;

        tension = Math.min(100, tension);

        // 6. DIALECTICAL POSITION
        const shortRecent = this.history.slice(-5);
        const shortMomentum = (shortRecent[shortRecent.length-1].spx - shortRecent[0].spx) / shortRecent[0].spx;

        let dialectic;
        if (Math.sign(priceMomentum) === Math.sign(shortMomentum)) {
            if (Math.abs(priceMomentum) > this.thresholds.momentumStrong) {
                dialectic = 'THESIS_STRONG';
            } else {
                dialectic = 'THESIS';
            }
        } else {
            if (Math.abs(shortMomentum) > Math.abs(priceMomentum) * 0.5) {
                dialectic = 'ANTITHESIS';
            } else {
                dialectic = 'ANTITHESIS_EMERGING';
            }
        }

        // Check for synthesis (consolidation)
        const priceRange = Math.max(...recent.map(r => r.spx)) - Math.min(...recent.map(r => r.spx));
        const avgPrice = recent.reduce((s, r) => s + r.spx, 0) / recent.length;
        if (priceRange / avgPrice < 0.03) {
            dialectic = 'SYNTHESIS';
        }

        // 7. REGIME CLASSIFICATION
        let regime;
        if (tension > 70) {
            regime = 'HIGH_STRESS';
        } else if (tension > 50) {
            regime = 'ELEVATED_TENSION';
        } else if (priceMomentum > 0.05 && volRegime === 'COMPLACENT') {
            regime = 'EUPHORIC_BULL';
        } else if (priceMomentum > 0.02) {
            regime = 'STEADY_BULL';
        } else if (priceMomentum < -0.05) {
            regime = 'CORRECTION';
        } else if (priceMomentum < -0.02) {
            regime = 'MILD_BEAR';
        } else {
            regime = 'CONSOLIDATION';
        }

        // 8. GEODESIC DEVIATION
        // Expected path based on momentum
        const expectedPrice = prev.spx * (1 + priceMomentum / this.lookback);
        const geodesicDeviation = (current.spx - expectedPrice) / expectedPrice;

        return {
            date: d.date,
            price: d.spx,
            vix: d.vix,

            metrics: {
                priceMomentum: (priceMomentum * 100).toFixed(2) + '%',
                dailyReturn: (dailyReturn * 100).toFixed(2) + '%',
                avgVix: avgVix.toFixed(1),
                goldMomentum: (goldMomentum * 100).toFixed(2) + '%',
                dollarMomentum: (dollarMomentum * 100).toFixed(2) + '%'
            },

            volRegime,
            safeHavenFlow,

            harmonic: {
                interval: closestInterval.name,
                quality: closestInterval.quality,
                ratio: ratio.toFixed(3)
            },

            tension,
            dialectic,
            regime,

            geodesicDeviation: (geodesicDeviation * 100).toFixed(2) + '%'
        };
    }
}

// ============================================================================
// RUN ANALYSIS
// ============================================================================

function main() {
    console.log('='.repeat(70));
    console.log('  TUNED GEOMETRIC MARKET ANALYSIS');
    console.log('  S&P 500 | 2022-2024 | Real Yahoo Finance Data');
    console.log('='.repeat(70));

    const data = loadData();
    console.log(`\nLoaded ${data.length} trading days`);
    console.log(`Range: ${data[0].date} to ${data[data.length-1].date}\n`);

    const analyzer = new TunedGeometricAnalyzer({ lookback: 20 });
    const results = [];

    for (const d of data) {
        const analysis = analyzer.analyze(d);
        if (analysis.status !== 'warming') {
            results.push(analysis);
        }
    }

    // Show detailed analysis for key periods
    console.log('[1] SAMPLE DAILY ANALYSIS (Last 15 days):\n');
    console.log('Date        Price    VIX   Momentum  Tension  Regime            Harmonic        Dialectic');
    console.log('-'.repeat(100));

    for (const r of results.slice(-15)) {
        console.log(
            `${r.date}  $${r.price.toFixed(0).padStart(5)}  ${r.vix.toFixed(1).padStart(5)}  ` +
            `${r.metrics.priceMomentum.padStart(7)}  ${r.tension.toString().padStart(3)}      ` +
            `${r.regime.padEnd(16)}  ${r.harmonic.interval.padEnd(14)}  ${r.dialectic}`
        );
    }

    // Aggregate by regime
    console.log('\n[2] REGIME DISTRIBUTION:\n');
    const regimeCounts = {};
    for (const r of results) {
        regimeCounts[r.regime] = (regimeCounts[r.regime] || 0) + 1;
    }
    for (const [regime, count] of Object.entries(regimeCounts).sort((a,b) => b[1] - a[1])) {
        const pct = (count / results.length * 100).toFixed(1);
        const bar = '█'.repeat(Math.round(pct / 2));
        console.log(`  ${regime.padEnd(18)} ${count.toString().padStart(4)} days (${pct.padStart(5)}%) ${bar}`);
    }

    // Aggregate by harmonic interval
    console.log('\n[3] HARMONIC INTERVAL DISTRIBUTION:\n');
    const harmonicCounts = {};
    for (const r of results) {
        harmonicCounts[r.harmonic.interval] = (harmonicCounts[r.harmonic.interval] || 0) + 1;
    }
    for (const [interval, count] of Object.entries(harmonicCounts).sort((a,b) => b[1] - a[1])) {
        const pct = (count / results.length * 100).toFixed(1);
        console.log(`  ${interval.padEnd(15)} ${count.toString().padStart(4)} days (${pct.padStart(5)}%)`);
    }

    // Aggregate by dialectical phase
    console.log('\n[4] DIALECTICAL PHASE DISTRIBUTION:\n');
    const dialectCounts = {};
    for (const r of results) {
        dialectCounts[r.dialectic] = (dialectCounts[r.dialectic] || 0) + 1;
    }
    for (const [phase, count] of Object.entries(dialectCounts).sort((a,b) => b[1] - a[1])) {
        const pct = (count / results.length * 100).toFixed(1);
        console.log(`  ${phase.padEnd(20)} ${count.toString().padStart(4)} days (${pct.padStart(5)}%)`);
    }

    // Tension distribution
    console.log('\n[5] TENSION DISTRIBUTION:\n');
    const tensionBuckets = { '0-30': 0, '31-50': 0, '51-70': 0, '71-100': 0 };
    for (const r of results) {
        if (r.tension <= 30) tensionBuckets['0-30']++;
        else if (r.tension <= 50) tensionBuckets['31-50']++;
        else if (r.tension <= 70) tensionBuckets['51-70']++;
        else tensionBuckets['71-100']++;
    }
    for (const [bucket, count] of Object.entries(tensionBuckets)) {
        const pct = (count / results.length * 100).toFixed(1);
        const bar = '█'.repeat(Math.round(pct / 2));
        console.log(`  Tension ${bucket.padEnd(6)} ${count.toString().padStart(4)} days (${pct.padStart(5)}%) ${bar}`);
    }

    // Find HIGH STRESS periods
    console.log('\n[6] HIGH STRESS PERIODS (Tension > 70):\n');
    const highStress = results.filter(r => r.tension > 70);

    // Group consecutive days
    const stressPeriods = [];
    let currentPeriod = [];

    for (let i = 0; i < highStress.length; i++) {
        if (currentPeriod.length === 0) {
            currentPeriod.push(highStress[i]);
        } else {
            const prevDate = new Date(currentPeriod[currentPeriod.length-1].date);
            const currDate = new Date(highStress[i].date);
            const daysDiff = (currDate - prevDate) / (1000 * 60 * 60 * 24);

            if (daysDiff <= 5) {
                currentPeriod.push(highStress[i]);
            } else {
                if (currentPeriod.length >= 2) stressPeriods.push([...currentPeriod]);
                currentPeriod = [highStress[i]];
            }
        }
    }
    if (currentPeriod.length >= 2) stressPeriods.push(currentPeriod);

    console.log(`  Found ${stressPeriods.length} distinct high-stress periods:\n`);

    for (const period of stressPeriods) {
        const start = period[0];
        const end = period[period.length - 1];
        const avgTension = period.reduce((s, p) => s + p.tension, 0) / period.length;
        const priceChange = ((end.price - start.price) / start.price * 100).toFixed(1);
        const maxVix = Math.max(...period.map(p => p.vix));

        console.log(`  ${start.date} to ${end.date} (${period.length} days)`);
        console.log(`    Avg Tension: ${avgTension.toFixed(0)} | Max VIX: ${maxVix.toFixed(1)} | Price: ${priceChange}%`);
        console.log(`    Regimes: ${[...new Set(period.map(p => p.regime))].join(', ')}`);
        console.log('');
    }

    // Regime transitions
    console.log('[7] REGIME TRANSITIONS:\n');
    const transitions = [];
    for (let i = 1; i < results.length; i++) {
        if (results[i].regime !== results[i-1].regime) {
            transitions.push({
                date: results[i].date,
                from: results[i-1].regime,
                to: results[i].regime,
                price: results[i].price,
                tension: results[i].tension
            });
        }
    }

    console.log(`  Total transitions: ${transitions.length}`);
    console.log(`  Average days per regime: ${(results.length / (transitions.length + 1)).toFixed(1)}\n`);

    console.log('  Last 20 transitions:');
    for (const t of transitions.slice(-20)) {
        console.log(`    ${t.date}: ${t.from.padEnd(16)} → ${t.to.padEnd(16)} @ $${t.price.toFixed(0)}`);
    }

    // Validate against known events
    console.log('\n[8] VALIDATION AGAINST KNOWN EVENTS:\n');

    const knownEvents = [
        { date: '2022-01-03', name: 'Market Peak Before 2022 Bear' },
        { date: '2022-06-16', name: '2022 Bear Market Low' },
        { date: '2022-10-12', name: '2022 Final Low' },
        { date: '2023-03-13', name: 'SVB Crisis' },
        { date: '2023-10-27', name: '2023 Correction Low' },
        { date: '2024-04-19', name: 'April 2024 Pullback' },
        { date: '2024-08-05', name: 'August Flash Crash' },
        { date: '2024-12-18', name: 'December Fed Meeting Drop' }
    ];

    for (const event of knownEvents) {
        // Find closest date in results
        const match = results.find(r => r.date === event.date) ||
                      results.find(r => r.date > event.date);

        if (match) {
            console.log(`  ${event.date}: ${event.name}`);
            console.log(`    Regime: ${match.regime} | Tension: ${match.tension} | VIX: ${match.vix.toFixed(1)}`);
            console.log(`    Harmonic: ${match.harmonic.interval} | Dialectic: ${match.dialectic}`);
            console.log('');
        }
    }

    console.log('='.repeat(70));
    console.log('  ANALYSIS COMPLETE');
    console.log('='.repeat(70));
}

main();
