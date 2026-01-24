#!/usr/bin/env node
/**
 * Geometric Market Analysis
 *
 * Based on Geometric Information Theory - NOT crash prediction.
 *
 * This system asks different questions:
 * 1. What is the market's TOPOLOGICAL state? (Betti numbers)
 * 2. Where is the GEODESIC? (path of least resistance)
 * 3. What is the HARMONIC INTERVAL? (quality of relationship)
 * 4. What is the DIALECTICAL position? (thesis/antithesis/synthesis)
 *
 * Output: Characterization of market geometry, not probability of crash.
 */

import fs from 'fs';

// ============================================================================
// GEOMETRIC EMBEDDING
// Embed market data into 4D space for topological analysis
// ============================================================================

class GeometricEmbedding {
    constructor(config = {}) {
        this.dimensions = {
            price: 0,      // x: normalized price
            sentiment: 1,  // y: fear/greed normalized
            volatility: 2, // z: VIX normalized
            momentum: 3    // w: rate of change
        };
        this.history = [];
        this.pointCloud = [];
        this.lookback = config.lookback || 20;
    }

    // Embed a single observation into 4D space
    embed(price, sentiment, volatility, prevPrice = null) {
        // Normalize to [0, 1] range
        const priceNorm = this.normalizePrice(price);
        const sentNorm = sentiment / 100;
        const volNorm = this.normalizeVol(volatility);
        const momentum = prevPrice ? (price - prevPrice) / prevPrice : 0;
        const momNorm = this.normalizeMomentum(momentum);

        return [priceNorm, sentNorm, volNorm, momNorm];
    }

    normalizePrice(price) {
        // Dynamic normalization based on history
        if (this.history.length < 2) return 0.5;
        const prices = this.history.map(h => h.price);
        const min = Math.min(...prices) * 0.9;
        const max = Math.max(...prices) * 1.1;
        return (price - min) / (max - min);
    }

    normalizeVol(vol) {
        // VIX: 10 = very low, 30 = high, 50+ = extreme
        return Math.min(1, (vol - 10) / 40);
    }

    normalizeMomentum(mom) {
        // Momentum: -10% to +10% range
        return (mom + 0.1) / 0.2;
    }

    addObservation(date, price, sentiment, volatility) {
        const prevPrice = this.history.length > 0 ?
            this.history[this.history.length - 1].price : price;

        const point = this.embed(price, sentiment, volatility, prevPrice);

        this.history.push({ date, price, sentiment, volatility, point });
        this.pointCloud.push(point);

        // Keep rolling window
        if (this.pointCloud.length > this.lookback * 3) {
            this.pointCloud.shift();
        }

        return point;
    }

    getPointCloud() {
        return this.pointCloud;
    }
}

// ============================================================================
// TOPOLOGICAL ANALYZER
// Compute Betti numbers and detect topological features
// ============================================================================

class TopologicalAnalyzer {
    constructor(config = {}) {
        this.threshold = config.threshold || 0.3;
    }

    // Compute distance between two 4D points
    distance(p1, p2) {
        let sum = 0;
        for (let i = 0; i < 4; i++) {
            sum += (p1[i] - p2[i]) ** 2;
        }
        return Math.sqrt(sum);
    }

    // Count connected components (β₀) using union-find
    computeB0(points) {
        const n = points.length;
        if (n === 0) return 0;

        const parent = Array.from({ length: n }, (_, i) => i);

        const find = (i) => {
            if (parent[i] !== i) parent[i] = find(parent[i]);
            return parent[i];
        };

        const union = (i, j) => {
            const pi = find(i), pj = find(j);
            if (pi !== pj) parent[pi] = pj;
        };

        // Connect nearby points
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                if (this.distance(points[i], points[j]) < this.threshold) {
                    union(i, j);
                }
            }
        }

        // Count unique roots
        const roots = new Set();
        for (let i = 0; i < n; i++) roots.add(find(i));
        return roots.size;
    }

    // Estimate loops (β₁) from graph cycles
    computeB1(points) {
        const n = points.length;
        if (n < 3) return 0;

        let edges = 0;
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                if (this.distance(points[i], points[j]) < this.threshold) {
                    edges++;
                }
            }
        }

        // β₁ ≈ E - V + components (Euler formula)
        const components = this.computeB0(points);
        return Math.max(0, edges - n + components);
    }

    // Estimate voids (β₂) - shell-like structures
    computeB2(points) {
        const n = points.length;
        if (n < 8) return 0;

        // Compute centroid
        const centroid = [0, 0, 0, 0];
        for (const p of points) {
            for (let i = 0; i < 4; i++) centroid[i] += p[i];
        }
        for (let i = 0; i < 4; i++) centroid[i] /= n;

        // Compute distances to centroid
        const distances = points.map(p => this.distance(p, centroid));
        const avgDist = distances.reduce((a, b) => a + b, 0) / n;
        const variance = distances.reduce((s, d) => s + (d - avgDist) ** 2, 0) / n;

        // Low variance = shell-like structure = void
        if (variance < 0.05 * avgDist * avgDist) return 1;
        return 0;
    }

    // Full Betti number computation
    computeBetti(points) {
        return {
            b0: this.computeB0(points),  // Connected components (regimes)
            b1: this.computeB1(points),  // Loops (cycles/feedback)
            b2: this.computeB2(points),  // Voids (forbidden zones)
        };
    }

    // Interpret Betti numbers
    interpretTopology(betti) {
        const interpretations = [];

        // β₀: Connected components
        if (betti.b0 === 1) {
            interpretations.push('SINGLE_REGIME: Market in coherent state');
        } else if (betti.b0 === 2) {
            interpretations.push('BIFURCATED: Two distinct market regimes present');
        } else if (betti.b0 > 2) {
            interpretations.push(`FRAGMENTED: ${betti.b0} disconnected regimes`);
        }

        // β₁: Loops
        if (betti.b1 > 0) {
            interpretations.push(`CYCLES_DETECTED: ${betti.b1} feedback loop(s) in state space`);
        }

        // β₂: Voids
        if (betti.b2 > 0) {
            interpretations.push('VOID_PRESENT: Forbidden states detected (shell structure)');
        }

        return interpretations;
    }
}

// ============================================================================
// GEODESIC CALCULATOR
// Compute the "natural" path through state space
// ============================================================================

class GeodesicCalculator {
    constructor(config = {}) {
        this.smoothing = config.smoothing || 5;
    }

    // Compute geodesic direction at current point
    // Geodesic = path of least resistance = weighted average of recent trajectory
    computeGeodesic(pointCloud) {
        if (pointCloud.length < this.smoothing + 1) return null;

        const recent = pointCloud.slice(-this.smoothing);
        const current = pointCloud[pointCloud.length - 1];

        // Compute velocity vectors
        const velocities = [];
        for (let i = 1; i < recent.length; i++) {
            const v = [];
            for (let d = 0; d < 4; d++) {
                v.push(recent[i][d] - recent[i - 1][d]);
            }
            velocities.push(v);
        }

        // Average velocity = geodesic direction
        const geodesicDir = [0, 0, 0, 0];
        for (const v of velocities) {
            for (let d = 0; d < 4; d++) {
                geodesicDir[d] += v[d] / velocities.length;
            }
        }

        // Project where geodesic would put us
        const geodesicTarget = [];
        for (let d = 0; d < 4; d++) {
            geodesicTarget.push(current[d] + geodesicDir[d]);
        }

        return {
            direction: geodesicDir,
            target: geodesicTarget,
            current: current
        };
    }

    // Compute deviation from geodesic
    computeDeviation(pointCloud, actualNext) {
        const geodesic = this.computeGeodesic(pointCloud);
        if (!geodesic) return null;

        // Distance between actual position and geodesic prediction
        let deviation = 0;
        for (let d = 0; d < 4; d++) {
            deviation += (actualNext[d] - geodesic.target[d]) ** 2;
        }
        deviation = Math.sqrt(deviation);

        // Direction of deviation
        const deviationVector = [];
        for (let d = 0; d < 4; d++) {
            deviationVector.push(actualNext[d] - geodesic.target[d]);
        }

        return {
            magnitude: deviation,
            vector: deviationVector,
            interpretation: this.interpretDeviation(deviation, deviationVector)
        };
    }

    interpretDeviation(magnitude, vector) {
        if (magnitude < 0.05) {
            return 'ON_GEODESIC: Price following natural path';
        } else if (magnitude < 0.15) {
            return 'SLIGHT_DEVIATION: Minor perturbation from geodesic';
        } else if (magnitude < 0.3) {
            return 'SIGNIFICANT_DEVIATION: Strong forces acting on price';
        } else {
            return 'MAJOR_DEVIATION: Extreme dislocation from natural path';
        }
    }
}

// ============================================================================
// HARMONIC INTERVAL ANALYZER
// Characterize the QUALITY of relationships, not just consonance/dissonance
// ============================================================================

class HarmonicAnalyzer {
    constructor() {
        // Musical interval ratios and their qualities
        this.intervals = [
            { name: 'Unison', ratio: 1.0, quality: 'perfect_consonance', meaning: 'Complete alignment' },
            { name: 'Minor Second', ratio: 16/15, quality: 'sharp_dissonance', meaning: 'Tension seeking resolution' },
            { name: 'Major Second', ratio: 9/8, quality: 'mild_dissonance', meaning: 'Movement, instability' },
            { name: 'Minor Third', ratio: 6/5, quality: 'soft_consonance', meaning: 'Melancholy stability' },
            { name: 'Major Third', ratio: 5/4, quality: 'soft_consonance', meaning: 'Optimistic stability' },
            { name: 'Perfect Fourth', ratio: 4/3, quality: 'open_consonance', meaning: 'Suspension, waiting' },
            { name: 'Tritone', ratio: Math.sqrt(2), quality: 'extreme_dissonance', meaning: 'Maximum instability, must resolve' },
            { name: 'Perfect Fifth', ratio: 3/2, quality: 'perfect_consonance', meaning: 'Strong stability, foundation' },
            { name: 'Minor Sixth', ratio: 8/5, quality: 'soft_consonance', meaning: 'Bittersweet resolution' },
            { name: 'Major Sixth', ratio: 5/3, quality: 'soft_consonance', meaning: 'Warm resolution' },
            { name: 'Minor Seventh', ratio: 16/9, quality: 'mild_dissonance', meaning: 'Dominant tension' },
            { name: 'Major Seventh', ratio: 15/8, quality: 'sharp_dissonance', meaning: 'Leading tone, strong pull' },
            { name: 'Octave', ratio: 2.0, quality: 'perfect_consonance', meaning: 'Cycle complete, return' },
        ];
    }

    // Compute the ratio between price and sentiment dimensions
    computeInterval(point) {
        const price = point[0] + 0.001;  // Avoid division by zero
        const sentiment = point[1] + 0.001;
        const volatility = point[2] + 0.001;

        // Primary interval: price/sentiment relationship
        const priceSentRatio = Math.max(price, sentiment) / Math.min(price, sentiment);

        // Secondary interval: sentiment/volatility relationship
        const sentVolRatio = Math.max(sentiment, volatility) / Math.min(sentiment, volatility);

        return {
            priceSentiment: this.classifyInterval(priceSentRatio),
            sentimentVolatility: this.classifyInterval(sentVolRatio),
            overall: this.computeOverallHarmony(priceSentRatio, sentVolRatio)
        };
    }

    classifyInterval(ratio) {
        // Normalize ratio to [1, 2] range (one octave)
        while (ratio > 2) ratio /= 2;
        while (ratio < 1) ratio *= 2;

        // Find closest interval
        let closest = this.intervals[0];
        let minDist = Infinity;

        for (const interval of this.intervals) {
            const dist = Math.abs(ratio - interval.ratio);
            if (dist < minDist) {
                minDist = dist;
                closest = interval;
            }
        }

        return {
            ...closest,
            actualRatio: ratio,
            accuracy: 1 - minDist  // How close to pure interval
        };
    }

    computeOverallHarmony(ratio1, ratio2) {
        const int1 = this.classifyInterval(ratio1);
        const int2 = this.classifyInterval(ratio2);

        // Combine qualities
        const qualities = {
            perfect_consonance: 3,
            open_consonance: 2,
            soft_consonance: 1,
            mild_dissonance: -1,
            sharp_dissonance: -2,
            extreme_dissonance: -3
        };

        const score = (qualities[int1.quality] + qualities[int2.quality]) / 2;

        if (score > 1.5) return { state: 'HARMONIC', description: 'Stable, aligned state' };
        if (score > 0) return { state: 'CONSONANT', description: 'Generally stable with some tension' };
        if (score > -1.5) return { state: 'TENSE', description: 'Building tension, seeking resolution' };
        return { state: 'DISSONANT', description: 'Unstable, resolution imminent' };
    }
}

// ============================================================================
// DIALECTICAL ANALYZER
// Thesis (trend) / Antithesis (counter) / Synthesis (equilibrium)
// ============================================================================

class DialecticalAnalyzer {
    constructor(config = {}) {
        this.trendWindow = config.trendWindow || 10;
        this.counterWindow = config.counterWindow || 5;
    }

    analyze(history) {
        if (history.length < this.trendWindow) return null;

        const recent = history.slice(-this.trendWindow);
        const veryRecent = history.slice(-this.counterWindow);

        // Thesis: dominant trend (longer window)
        const thesisTrend = this.computeTrend(recent);

        // Antithesis: counter-movement (shorter window)
        const antithesisTrend = this.computeTrend(veryRecent);

        // Determine dialectical position
        const position = this.classifyPosition(thesisTrend, antithesisTrend);

        return {
            thesis: thesisTrend,
            antithesis: antithesisTrend,
            position,
            synthesis: this.projectSynthesis(thesisTrend, antithesisTrend)
        };
    }

    computeTrend(data) {
        const prices = data.map(d => d.price);
        const sentiments = data.map(d => d.sentiment);

        // Linear regression slope
        const priceSlope = this.slope(prices);
        const sentSlope = this.slope(sentiments);

        return {
            priceDirection: priceSlope > 0.001 ? 'UP' : priceSlope < -0.001 ? 'DOWN' : 'FLAT',
            sentimentDirection: sentSlope > 0.5 ? 'IMPROVING' : sentSlope < -0.5 ? 'DETERIORATING' : 'STABLE',
            priceSlope,
            sentSlope,
            alignment: Math.sign(priceSlope) === Math.sign(sentSlope) ? 'ALIGNED' : 'DIVERGING'
        };
    }

    slope(values) {
        const n = values.length;
        const xMean = (n - 1) / 2;
        const yMean = values.reduce((a, b) => a + b, 0) / n;

        let num = 0, den = 0;
        for (let i = 0; i < n; i++) {
            num += (i - xMean) * (values[i] - yMean);
            den += (i - xMean) ** 2;
        }

        return den !== 0 ? num / den : 0;
    }

    classifyPosition(thesis, antithesis) {
        // Thesis dominant: trend continuing
        if (thesis.alignment === 'ALIGNED' && antithesis.alignment === 'ALIGNED') {
            return {
                phase: 'THESIS',
                description: 'Dominant trend intact, no significant counter-force',
                stability: 'STABLE'
            };
        }

        // Antithesis emerging: counter-trend building
        if (thesis.alignment === 'ALIGNED' && antithesis.alignment === 'DIVERGING') {
            return {
                phase: 'ANTITHESIS_EMERGING',
                description: 'Counter-force building against dominant trend',
                stability: 'TRANSITIONING'
            };
        }

        // Antithesis dominant: reversal in progress
        if (thesis.alignment === 'DIVERGING' && antithesis.alignment !== thesis.alignment) {
            return {
                phase: 'ANTITHESIS',
                description: 'Counter-force dominant, trend challenged',
                stability: 'UNSTABLE'
            };
        }

        // Synthesis: new equilibrium forming
        if (thesis.priceDirection === 'FLAT' || antithesis.priceDirection === 'FLAT') {
            return {
                phase: 'SYNTHESIS',
                description: 'New equilibrium forming, forces balanced',
                stability: 'CONSOLIDATING'
            };
        }

        return {
            phase: 'UNDEFINED',
            description: 'Complex state, multiple forces interacting',
            stability: 'UNCERTAIN'
        };
    }

    projectSynthesis(thesis, antithesis) {
        // Where will the synthesis emerge?
        if (thesis.priceDirection === antithesis.priceDirection) {
            return { direction: thesis.priceDirection, confidence: 'HIGH' };
        }
        if (Math.abs(thesis.priceSlope) > Math.abs(antithesis.priceSlope) * 2) {
            return { direction: thesis.priceDirection, confidence: 'MEDIUM' };
        }
        return { direction: 'UNCERTAIN', confidence: 'LOW' };
    }
}

// ============================================================================
// MAIN: GEOMETRIC MARKET ANALYZER
// ============================================================================

class GeometricMarketAnalyzer {
    constructor(config = {}) {
        this.embedding = new GeometricEmbedding(config);
        this.topology = new TopologicalAnalyzer(config);
        this.geodesic = new GeodesicCalculator(config);
        this.harmonic = new HarmonicAnalyzer();
        this.dialectic = new DialecticalAnalyzer(config);
    }

    analyze(date, price, sentiment, volatility) {
        // Add to embedding
        const point = this.embedding.addObservation(date, price, sentiment, volatility);
        const cloud = this.embedding.getPointCloud();
        const history = this.embedding.history;

        if (cloud.length < 10) {
            return { status: 'WARMING_UP', pointsNeeded: 10 - cloud.length };
        }

        // 1. Topological analysis
        const betti = this.topology.computeBetti(cloud);
        const topoInterpretation = this.topology.interpretTopology(betti);

        // 2. Geodesic analysis
        const geodesicAnalysis = this.geodesic.computeGeodesic(cloud);

        // 3. Harmonic analysis
        const harmonicAnalysis = this.harmonic.computeInterval(point);

        // 4. Dialectical analysis
        const dialecticalAnalysis = this.dialectic.analyze(history);

        return {
            date,
            price,
            point,

            topology: {
                betti,
                interpretation: topoInterpretation
            },

            geodesic: geodesicAnalysis,

            harmonic: {
                priceSentiment: harmonicAnalysis.priceSentiment.name,
                priceSentimentQuality: harmonicAnalysis.priceSentiment.quality,
                priceSentimentMeaning: harmonicAnalysis.priceSentiment.meaning,
                sentimentVolatility: harmonicAnalysis.sentimentVolatility.name,
                overall: harmonicAnalysis.overall
            },

            dialectic: dialecticalAnalysis,

            // Summary characterization
            characterization: this.summarize(betti, harmonicAnalysis, dialecticalAnalysis)
        };
    }

    summarize(betti, harmonic, dialectic) {
        const parts = [];

        // Topological state
        if (betti.b0 === 1) parts.push('coherent');
        else parts.push('fragmented');

        // Harmonic state
        parts.push(harmonic.overall.state.toLowerCase());

        // Dialectical phase
        if (dialectic) {
            parts.push(dialectic.position.phase.toLowerCase().replace('_', '-'));
        }

        return parts.join(' / ');
    }
}

// ============================================================================
// RUN ON REAL DATA
// ============================================================================

async function main() {
    console.log('='.repeat(70));
    console.log('  GEOMETRIC MARKET ANALYSIS');
    console.log('  Topology | Geodesics | Harmonics | Dialectics');
    console.log('='.repeat(70));

    // Load saved Yahoo Finance data
    const spxRaw = JSON.parse(fs.readFileSync('/tmp/spx_data.json', 'utf8'));
    const vixRaw = JSON.parse(fs.readFileSync('/tmp/vix_data.json', 'utf8'));

    const spxData = spxRaw.chart.result[0];
    const vixData = vixRaw.chart.result[0];

    // Merge by index (same trading days)
    const data = [];
    const n = Math.min(spxData.timestamp.length, vixData.timestamp.length);

    for (let i = 0; i < n; i++) {
        const date = new Date(spxData.timestamp[i] * 1000).toISOString().split('T')[0];
        const price = spxData.indicators.quote[0].close[i];
        const vix = vixData.indicators.quote[0].close[i];

        if (price && vix) {
            // Use VIX as inverse sentiment proxy (high VIX = low sentiment)
            const sentiment = Math.max(0, Math.min(100, 100 - vix * 2));
            data.push({ date, price, sentiment, vix });
        }
    }

    console.log(`\nLoaded ${data.length} trading days`);
    console.log(`Range: ${data[0].date} to ${data[data.length-1].date}\n`);

    // Run analyzer
    const analyzer = new GeometricMarketAnalyzer({ lookback: 20 });

    const results = [];
    for (const d of data) {
        const analysis = analyzer.analyze(d.date, d.price, d.sentiment, d.vix);
        if (analysis.status !== 'WARMING_UP') {
            results.push(analysis);
        }
    }

    // Show sample analyses
    console.log('[1] SAMPLE GEOMETRIC ANALYSES:\n');

    for (const r of results.slice(-10)) {
        console.log(`${r.date} | SPX: $${r.price.toFixed(0)}`);
        console.log(`  Topology: β₀=${r.topology.betti.b0} β₁=${r.topology.betti.b1} β₂=${r.topology.betti.b2}`);
        console.log(`  Harmonic: ${r.harmonic.priceSentiment} (${r.harmonic.priceSentimentMeaning})`);
        console.log(`  Overall:  ${r.harmonic.overall.state} - ${r.harmonic.overall.description}`);
        if (r.dialectic) {
            console.log(`  Dialectic: ${r.dialectic.position.phase} - ${r.dialectic.position.description}`);
        }
        console.log(`  → ${r.characterization}`);
        console.log('');
    }

    // Aggregate statistics
    console.log('[2] TOPOLOGICAL DISTRIBUTION:\n');

    const b0Counts = {};
    const b1Counts = {};
    for (const r of results) {
        b0Counts[r.topology.betti.b0] = (b0Counts[r.topology.betti.b0] || 0) + 1;
        b1Counts[r.topology.betti.b1] = (b1Counts[r.topology.betti.b1] || 0) + 1;
    }

    console.log('  Connected components (β₀ = market regimes):');
    for (const [k, v] of Object.entries(b0Counts).sort((a,b) => b[1] - a[1])) {
        console.log(`    β₀=${k}: ${v} days (${(v/results.length*100).toFixed(1)}%)`);
    }

    console.log('\n  Loops (β₁ = feedback cycles):');
    for (const [k, v] of Object.entries(b1Counts).sort((a,b) => b[1] - a[1])) {
        console.log(`    β₁=${k}: ${v} days (${(v/results.length*100).toFixed(1)}%)`);
    }

    // Harmonic distribution
    console.log('\n[3] HARMONIC INTERVAL DISTRIBUTION:\n');

    const harmonicCounts = {};
    for (const r of results) {
        const h = r.harmonic.overall.state;
        harmonicCounts[h] = (harmonicCounts[h] || 0) + 1;
    }

    for (const [k, v] of Object.entries(harmonicCounts).sort((a,b) => b[1] - a[1])) {
        console.log(`  ${k.padEnd(12)} ${v.toString().padStart(4)} days (${(v/results.length*100).toFixed(1)}%)`);
    }

    // Dialectical phases
    console.log('\n[4] DIALECTICAL PHASE DISTRIBUTION:\n');

    const phaseCounts = {};
    for (const r of results) {
        if (r.dialectic) {
            const p = r.dialectic.position.phase;
            phaseCounts[p] = (phaseCounts[p] || 0) + 1;
        }
    }

    for (const [k, v] of Object.entries(phaseCounts).sort((a,b) => b[1] - a[1])) {
        console.log(`  ${k.padEnd(20)} ${v.toString().padStart(4)} days (${(v/results.length*100).toFixed(1)}%)`);
    }

    // Find interesting transitions
    console.log('\n[5] TOPOLOGICAL TRANSITIONS:\n');

    let transitions = [];
    for (let i = 1; i < results.length; i++) {
        const prev = results[i-1];
        const curr = results[i];

        if (prev.topology.betti.b0 !== curr.topology.betti.b0) {
            transitions.push({
                date: curr.date,
                from: `β₀=${prev.topology.betti.b0}`,
                to: `β₀=${curr.topology.betti.b0}`,
                price: curr.price,
                type: 'REGIME_CHANGE'
            });
        }
    }

    console.log(`  Found ${transitions.length} topological transitions\n`);
    for (const t of transitions.slice(-10)) {
        console.log(`  ${t.date}: ${t.from} → ${t.to} @ $${t.price.toFixed(0)}`);
    }

    console.log('\n' + '='.repeat(70));
    console.log('  GEOMETRIC ANALYSIS COMPLETE');
    console.log('='.repeat(70));
}

main().catch(console.error);
