#!/usr/bin/env node
/**
 * Historical Prediction Test
 *
 * Pick a random past date, make predictions, then verify against what actually happened.
 */

import fs from 'fs';

// Load data
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
    const n = spx.timestamp.length;

    for (let i = 0; i < n; i++) {
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

// Geometric analyzer (same as tuned version)
class GeometricAnalyzer {
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
        const current = this.history[this.history.length - 1];

        // Price momentum
        const priceStart = recent[0].spx;
        const priceEnd = recent[recent.length - 1].spx;
        const priceMomentum = (priceEnd - priceStart) / priceStart;

        // VIX
        const avgVix = recent.reduce((s, r) => s + r.vix, 0) / recent.length;
        let volRegime = avgVix < 15 ? 'COMPLACENT' : avgVix < 25 ? 'NORMAL' : avgVix < 35 ? 'ELEVATED' : 'PANIC';

        // Gold/Dollar momentum
        const goldMom = (recent[recent.length-1].gold - recent[0].gold) / recent[0].gold;
        const dollarMom = (recent[recent.length-1].dollar - recent[0].dollar) / recent[0].dollar;

        let safeHaven = 'NEUTRAL';
        if (goldMom > 0.02 && dollarMom > 0.01) safeHaven = 'RISK_OFF';
        else if (goldMom < -0.02 && dollarMom < -0.01) safeHaven = 'RISK_ON';

        // Tension
        let tension = 30;
        if ((priceMomentum > 0.03 && volRegime === 'ELEVATED') || (priceMomentum < -0.03 && volRegime === 'COMPLACENT')) tension += 30;
        if (safeHaven === 'RISK_OFF' && priceMomentum > 0) tension += 20;
        if (volRegime === 'PANIC') tension += 25;
        if (volRegime === 'ELEVATED') tension += 10;
        tension = Math.min(100, tension);

        // Regime
        let regime;
        if (tension > 70) regime = 'HIGH_STRESS';
        else if (tension > 50) regime = 'ELEVATED_TENSION';
        else if (priceMomentum > 0.05 && volRegime === 'COMPLACENT') regime = 'EUPHORIC_BULL';
        else if (priceMomentum > 0.02) regime = 'STEADY_BULL';
        else if (priceMomentum < -0.05) regime = 'CORRECTION';
        else if (priceMomentum < -0.02) regime = 'MILD_BEAR';
        else regime = 'CONSOLIDATION';

        // Dialectic
        const shortRecent = this.history.slice(-5);
        const shortMom = (shortRecent[shortRecent.length-1].spx - shortRecent[0].spx) / shortRecent[0].spx;
        let dialectic;
        if (Math.sign(priceMomentum) === Math.sign(shortMom)) {
            dialectic = Math.abs(priceMomentum) > 0.05 ? 'THESIS_STRONG' : 'THESIS';
        } else {
            dialectic = Math.abs(shortMom) > Math.abs(priceMomentum) * 0.5 ? 'ANTITHESIS' : 'ANTITHESIS_EMERGING';
        }

        return {
            date: d.date,
            price: d.spx,
            vix: d.vix,
            priceMomentum,
            avgVix,
            volRegime,
            safeHaven,
            tension,
            regime,
            dialectic,
            goldMom,
            dollarMom
        };
    }

    // Make prediction based on current state
    predict(analysis) {
        const predictions = {
            direction: null,
            confidence: null,
            reasoning: [],
            expectedMove: null,
            timeframe: '20 trading days'
        };

        // Rule-based predictions from geometric state
        if (analysis.regime === 'HIGH_STRESS') {
            predictions.direction = 'VOLATILE';
            predictions.confidence = 'HIGH';
            predictions.expectedMove = '±5% or more';
            predictions.reasoning.push('High stress regime = expect large moves');
            if (analysis.dialectic === 'ANTITHESIS') {
                predictions.direction = 'REVERSAL_LIKELY';
                predictions.reasoning.push('Antithesis phase suggests trend exhaustion');
            }
        } else if (analysis.regime === 'ELEVATED_TENSION') {
            predictions.direction = 'CAUTIOUS';
            predictions.confidence = 'MEDIUM';
            predictions.expectedMove = '±3-5%';
            predictions.reasoning.push('Elevated tension = increased risk of regime change');
        } else if (analysis.regime === 'EUPHORIC_BULL') {
            predictions.direction = 'UP_BUT_RISKY';
            predictions.confidence = 'MEDIUM';
            predictions.expectedMove = '+2-5% with correction risk';
            predictions.reasoning.push('Euphoria often precedes pullbacks');
            predictions.reasoning.push('Low VIX + strong momentum = complacency');
        } else if (analysis.regime === 'STEADY_BULL') {
            if (analysis.dialectic === 'THESIS_STRONG') {
                predictions.direction = 'UP';
                predictions.confidence = 'HIGH';
                predictions.expectedMove = '+2-4%';
                predictions.reasoning.push('Strong thesis = trend continuation likely');
            } else if (analysis.dialectic === 'ANTITHESIS_EMERGING') {
                predictions.direction = 'UP_SLOWING';
                predictions.confidence = 'MEDIUM';
                predictions.expectedMove = '+0-2%';
                predictions.reasoning.push('Counter-forces building, gains may slow');
            } else {
                predictions.direction = 'UP';
                predictions.confidence = 'MEDIUM';
                predictions.expectedMove = '+1-3%';
                predictions.reasoning.push('Steady bull continues');
            }
        } else if (analysis.regime === 'CORRECTION') {
            if (analysis.dialectic === 'THESIS_STRONG') {
                predictions.direction = 'DOWN';
                predictions.confidence = 'HIGH';
                predictions.expectedMove = '-3-5%';
                predictions.reasoning.push('Strong downtrend continues');
            } else if (analysis.dialectic === 'ANTITHESIS') {
                predictions.direction = 'BOTTOM_FORMING';
                predictions.confidence = 'MEDIUM';
                predictions.expectedMove = '-2% to +2%';
                predictions.reasoning.push('Counter-trend suggests possible reversal');
            } else {
                predictions.direction = 'DOWN';
                predictions.confidence = 'MEDIUM';
                predictions.expectedMove = '-2-4%';
                predictions.reasoning.push('Correction in progress');
            }
        } else if (analysis.regime === 'MILD_BEAR') {
            predictions.direction = 'DOWN_MILD';
            predictions.confidence = 'LOW';
            predictions.expectedMove = '-1-3%';
            predictions.reasoning.push('Mild bearish bias');
        } else {
            predictions.direction = 'FLAT';
            predictions.confidence = 'LOW';
            predictions.expectedMove = '±2%';
            predictions.reasoning.push('Consolidation = directionless');
        }

        // Add safe haven signal
        if (analysis.safeHaven === 'RISK_OFF') {
            predictions.reasoning.push('Gold + Dollar rising = defensive positioning');
            if (predictions.direction === 'UP') {
                predictions.direction = 'UP_CAUTIOUS';
            }
        }

        return predictions;
    }
}

// Run blind test
function runBlindTest(data, predictionDate, lookForwardDays = 20) {
    // Find the index of prediction date
    const predIdx = data.findIndex(d => d.date === predictionDate);
    if (predIdx === -1) {
        console.log(`Date ${predictionDate} not found, finding closest...`);
        for (let i = 0; i < data.length; i++) {
            if (data[i].date >= predictionDate) {
                return runBlindTest(data, data[i].date, lookForwardDays);
            }
        }
    }

    // Need at least 20 days before for warmup, and lookForwardDays after
    if (predIdx < 25 || predIdx + lookForwardDays >= data.length) {
        console.log('Not enough data around this date');
        return;
    }

    console.log('='.repeat(70));
    console.log('  BLIND HISTORICAL PREDICTION TEST');
    console.log('='.repeat(70));

    // Build analyzer with data UP TO prediction date only
    const analyzer = new GeometricAnalyzer();

    // Feed data up to prediction date
    for (let i = 0; i <= predIdx; i++) {
        analyzer.analyze(data[i]);
    }

    // Get analysis AT prediction date
    const analysis = analyzer.analyze(data[predIdx]);

    console.log(`\n[PREDICTION DATE: ${predictionDate}]\n`);
    console.log('CURRENT STATE (what the model sees):');
    console.log(`  Price:         $${analysis.price.toFixed(2)}`);
    console.log(`  VIX:           ${analysis.vix.toFixed(1)}`);
    console.log(`  20-day momentum: ${(analysis.priceMomentum * 100).toFixed(2)}%`);
    console.log(`  Vol Regime:    ${analysis.volRegime}`);
    console.log(`  Safe Haven:    ${analysis.safeHaven}`);
    console.log(`  Tension:       ${analysis.tension}`);
    console.log(`  Regime:        ${analysis.regime}`);
    console.log(`  Dialectic:     ${analysis.dialectic}`);

    // Make prediction
    const prediction = analyzer.predict(analysis);

    console.log('\nPREDICTION (for next 20 trading days):');
    console.log(`  Direction:     ${prediction.direction}`);
    console.log(`  Confidence:    ${prediction.confidence}`);
    console.log(`  Expected Move: ${prediction.expectedMove}`);
    console.log(`  Reasoning:`);
    for (const reason of prediction.reasoning) {
        console.log(`    - ${reason}`);
    }

    // Now reveal what ACTUALLY happened
    console.log('\n' + '='.repeat(70));
    console.log('[ACTUAL RESULTS - WHAT REALLY HAPPENED]');
    console.log('='.repeat(70));

    const futureData = data.slice(predIdx, predIdx + lookForwardDays + 1);
    const startPrice = futureData[0].spx;
    const endPrice = futureData[futureData.length - 1].spx;
    const actualReturn = (endPrice - startPrice) / startPrice;

    // Find min and max in the period
    const minPrice = Math.min(...futureData.map(d => d.spx));
    const maxPrice = Math.max(...futureData.map(d => d.spx));
    const maxDrawdown = (minPrice - startPrice) / startPrice;
    const maxGain = (maxPrice - startPrice) / startPrice;

    const minDate = futureData.find(d => d.spx === minPrice).date;
    const maxDate = futureData.find(d => d.spx === maxPrice).date;

    console.log(`\n  Period: ${futureData[0].date} to ${futureData[futureData.length-1].date}`);
    console.log(`  Start Price:   $${startPrice.toFixed(2)}`);
    console.log(`  End Price:     $${endPrice.toFixed(2)}`);
    console.log(`  Actual Return: ${(actualReturn * 100).toFixed(2)}%`);
    console.log(`  Max Drawdown:  ${(maxDrawdown * 100).toFixed(2)}% (on ${minDate})`);
    console.log(`  Max Gain:      ${(maxGain * 100).toFixed(2)}% (on ${maxDate})`);

    // VIX evolution
    const endVix = futureData[futureData.length - 1].vix;
    const maxVix = Math.max(...futureData.map(d => d.vix));
    console.log(`  VIX: ${analysis.vix.toFixed(1)} → ${endVix.toFixed(1)} (max: ${maxVix.toFixed(1)})`);

    // Grade the prediction
    console.log('\n' + '='.repeat(70));
    console.log('[PREDICTION ACCURACY]');
    console.log('='.repeat(70));

    let score = 0;
    let maxScore = 0;
    const grades = [];

    // Direction accuracy
    maxScore += 3;
    if (prediction.direction.includes('UP') && actualReturn > 0.01) {
        score += 3;
        grades.push('✓ Direction correct (predicted UP, got +' + (actualReturn*100).toFixed(1) + '%)');
    } else if (prediction.direction.includes('DOWN') && actualReturn < -0.01) {
        score += 3;
        grades.push('✓ Direction correct (predicted DOWN, got ' + (actualReturn*100).toFixed(1) + '%)');
    } else if (prediction.direction === 'FLAT' && Math.abs(actualReturn) < 0.02) {
        score += 3;
        grades.push('✓ Direction correct (predicted FLAT, got ' + (actualReturn*100).toFixed(1) + '%)');
    } else if (prediction.direction === 'VOLATILE' && (maxGain > 0.05 || maxDrawdown < -0.05)) {
        score += 3;
        grades.push('✓ Volatility correct (predicted VOLATILE, got ' + (maxGain*100).toFixed(1) + '% gain / ' + (maxDrawdown*100).toFixed(1) + '% drawdown)');
    } else if (prediction.direction.includes('CAUTIOUS') || prediction.direction.includes('SLOWING')) {
        if (Math.abs(actualReturn) < 0.03) {
            score += 2;
            grades.push('◐ Partially correct (predicted caution, move was modest)');
        } else {
            score += 1;
            grades.push('○ Partially wrong (predicted caution, but move was ' + (actualReturn*100).toFixed(1) + '%)');
        }
    } else {
        grades.push('✗ Direction wrong');
    }

    // Magnitude accuracy
    maxScore += 2;
    const expectedMatch = prediction.expectedMove.match(/([+-]?\d+)-?(\d+)?%/);
    if (expectedMatch) {
        const low = parseFloat(expectedMatch[1]) / 100;
        const high = expectedMatch[2] ? parseFloat(expectedMatch[2]) / 100 : Math.abs(low);
        const absReturn = Math.abs(actualReturn);

        if (absReturn >= Math.abs(low) * 0.5 && absReturn <= high * 1.5) {
            score += 2;
            grades.push('✓ Magnitude roughly correct');
        } else if (absReturn >= Math.abs(low) * 0.25 && absReturn <= high * 2) {
            score += 1;
            grades.push('◐ Magnitude in ballpark');
        } else {
            grades.push('✗ Magnitude off');
        }
    }

    console.log(`\n  Score: ${score}/${maxScore}\n`);
    for (const g of grades) {
        console.log(`  ${g}`);
    }

    console.log('\n' + '='.repeat(70));

    return { prediction, actual: { return: actualReturn, maxDrawdown, maxGain } };
}

// Main
function main() {
    const data = loadData();
    console.log(`Loaded ${data.length} days: ${data[0].date} to ${data[data.length-1].date}\n`);

    // Pick several random test dates across the range
    const testDates = [
        '2022-06-01',  // Before 2022 bear market bottom
        '2022-10-01',  // Near 2022 bottom
        '2023-03-01',  // Before SVB crisis
        '2023-07-01',  // Mid-2023 rally
        '2024-01-15',  // Early 2024
        '2024-04-01',  // Before April pullback
        '2024-07-15',  // Before August flash crash
    ];

    const results = [];

    for (const testDate of testDates) {
        const result = runBlindTest(data, testDate, 20);
        if (result) {
            results.push({ date: testDate, ...result });
        }
        console.log('\n');
    }

    // Summary
    console.log('='.repeat(70));
    console.log('  SUMMARY OF ALL PREDICTIONS');
    console.log('='.repeat(70));
    console.log('\nDate         Predicted        Actual    Result');
    console.log('-'.repeat(60));

    for (const r of results) {
        const actualPct = (r.actual.return * 100).toFixed(1) + '%';
        const match = r.prediction.direction.includes('UP') && r.actual.return > 0 ||
                      r.prediction.direction.includes('DOWN') && r.actual.return < 0 ||
                      r.prediction.direction === 'FLAT' && Math.abs(r.actual.return) < 0.02;
        const result = match ? '✓' : '✗';

        console.log(`${r.date}   ${r.prediction.direction.padEnd(15)}  ${actualPct.padStart(7)}   ${result}`);
    }
}

main();
