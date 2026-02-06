const clamp = (value, min, max) => {
    if (!Number.isFinite(value)) {
        return min;
    }
    if (value < min) {
        return min;
    }
    if (value > max) {
        return max;
    }
    return value;
};

const stats = (values = []) => {
    const filtered = Array.isArray(values)
        ? values.filter((value) => Number.isFinite(value))
        : [];
    if (!filtered.length) {
        return {
            mean: null,
            min: null,
            max: null,
            stdDev: null,
            count: 0
        };
    }
    const mean = filtered.reduce((accumulator, value) => accumulator + value, 0) / filtered.length;
    let variance = 0;
    for (let index = 0; index < filtered.length; index += 1) {
        const delta = filtered[index] - mean;
        variance += delta * delta;
    }
    variance /= filtered.length;
    return {
        mean,
        min: Math.min(...filtered),
        max: Math.max(...filtered),
        stdDev: Math.sqrt(variance),
        count: filtered.length
    };
};

const normalizeMetric = (metric) => {
    if (!metric || typeof metric !== 'object') {
        return {
            average: null,
            min: null,
            max: null,
            count: 0
        };
    }
    return {
        average: Number.isFinite(metric.average) ? metric.average : null,
        min: Number.isFinite(metric.min) ? metric.min : null,
        max: Number.isFinite(metric.max) ? metric.max : null,
        count: Number.isFinite(metric.count) ? metric.count : 0
    };
};

const structuredCloneSafe = (value) => {
    if (typeof structuredClone === 'function') {
        try {
            return structuredClone(value);
        } catch (error) {
            console.warn('CalibrationInsightEngine structuredClone fallback triggered', error);
        }
    }
    try {
        return JSON.parse(JSON.stringify(value));
    } catch (error) {
        console.warn('CalibrationInsightEngine JSON clone fallback failed', error);
    }
    return null;
};

const defaultThresholds = {
    parityDeltaWarning: 0.1,
    parityDeltaCritical: 0.25,
    spinorCoherenceWarning: 0.65,
    spinorCoherenceCritical: 0.45,
    carrierGateWarning: 0.5,
    carrierGateCritical: 0.35,
    sampleScoreWarning: 0.7,
    sampleScoreCritical: 0.5
};

export class CalibrationInsightEngine {
    constructor({ thresholds = {}, maxOutliers = 8 } = {}) {
        this.thresholds = { ...defaultThresholds, ...thresholds };
        this.maxOutliers = Math.max(1, Math.floor(maxOutliers));
    }

    analyzeManifest(manifest = {}) {
        const totals = manifest?.totals || {};
        const parity = manifest?.parity || {};
        const samples = Array.isArray(manifest?.samples) ? manifest.samples : [];
        const sequences = Array.isArray(manifest?.sequences) ? manifest.sequences : [];
        const sampleScores = samples
            .map((sample) => (Number.isFinite(sample?.score) ? sample.score : null))
            .filter((value) => Number.isFinite(value));
        const scoreStats = stats(sampleScores);
        const aggregated = {
            visualContinuumDelta: normalizeMetric(parity.visualContinuumDelta),
            continuumAlignmentMean: normalizeMetric(parity.continuumAlignmentMean),
            carrierGateRatio: normalizeMetric(parity.carrierGateRatio),
            spinorCoherence: normalizeMetric(parity.spinorCoherence),
            envelopeResonance: normalizeMetric(parity.envelopeResonance)
        };
        const sequenceCoverage = sequences.map((sequence) => ({
            id: sequence.id || null,
            label: sequence.label || sequence.id || null,
            completed: Boolean(sequence.completed),
            frames: Number.isFinite(sequence.frames) ? sequence.frames : 0,
            sampleRate: Number.isFinite(sequence.sampleRate) ? sequence.sampleRate : null,
            durationSeconds: Number.isFinite(sequence.durationSeconds) ? sequence.durationSeconds : null,
            status: sequence.status || (sequence.completed ? 'complete' : 'unknown')
        }));
        const outliers = this.identifyOutliers(samples);
        const recommendations = this.buildRecommendations({
            aggregated,
            scoreStats,
            totals,
            sequenceCoverage,
            outliers,
            datasetScore: Number.isFinite(manifest?.score) ? manifest.score : null
        });
        return {
            summary: {
                datasetScore: Number.isFinite(manifest?.score) ? manifest.score : null,
                sampleScore: scoreStats,
                totals: {
                    sequences: Number.isFinite(totals.sequenceCount) ? totals.sequenceCount : sequences.length,
                    frames: Number.isFinite(totals.sampleCount) ? totals.sampleCount : samples.length
                }
            },
            parity: aggregated,
            sequenceCoverage,
            samples: {
                outliers,
                scoreStats
            },
            recommendations
        };
    }

    identifyOutliers(samples = []) {
        if (!Array.isArray(samples) || !samples.length) {
            return [];
        }
        const { parityDeltaCritical, spinorCoherenceCritical, sampleScoreCritical } = this.thresholds;
        const scored = [];
        samples.forEach((sample, index) => {
            const score = Number.isFinite(sample?.score) ? sample.score : null;
            const parity = sample?.parity || {};
            const delta = Number.isFinite(parity.visualContinuumDelta)
                ? Math.abs(parity.visualContinuumDelta)
                : null;
            const coherence = Number.isFinite(parity.spinorCoherence)
                ? parity.spinorCoherence
                : null;
            const issues = [];
            if (score !== null && score < sampleScoreCritical) {
                issues.push({
                    type: 'score',
                    severity: clamp(1 - score, 0, 1),
                    message: `Sample score ${(score * 100).toFixed(1)}% below target.`
                });
            }
            if (delta !== null && delta > parityDeltaCritical) {
                issues.push({
                    type: 'parity',
                    severity: clamp(delta / 2, 0, 1),
                    message: `Visual/continuum delta ${delta.toFixed(3)} exceeds threshold.`
                });
            }
            if (coherence !== null && coherence < spinorCoherenceCritical) {
                issues.push({
                    type: 'coherence',
                    severity: clamp(1 - coherence, 0, 1),
                    message: `Spinor coherence ${coherence.toFixed(3)} below threshold.`
                });
            }
            if (!issues.length) {
                return;
            }
            issues.sort((a, b) => b.severity - a.severity);
            const worst = issues[0];
            scored.push({
                index,
                score,
                delta,
                coherence,
                severity: worst.severity,
                primaryIssue: worst.type,
                messages: issues.map((issue) => issue.message)
            });
        });
        scored.sort((a, b) => b.severity - a.severity);
        return scored.slice(0, this.maxOutliers);
    }

    buildRecommendations({ aggregated, scoreStats, totals, sequenceCoverage, outliers, datasetScore }) {
        const suggestions = [];
        const { parityDeltaWarning, spinorCoherenceWarning, carrierGateWarning, sampleScoreWarning } = this.thresholds;
        const averageDelta = aggregated.visualContinuumDelta.average;
        if (Number.isFinite(averageDelta) && Math.abs(averageDelta) > parityDeltaWarning) {
            suggestions.push('Revisit mapping weights – average visual/sonic delta is above the target envelope.');
        }
        const averageCoherence = aggregated.spinorCoherence.average;
        if (Number.isFinite(averageCoherence) && averageCoherence < spinorCoherenceWarning) {
            suggestions.push('Tune quaternion bridge modulation – spinor coherence is trending low.');
        }
        const averageGates = aggregated.carrierGateRatio.average;
        if (Number.isFinite(averageGates) && averageGates < carrierGateWarning) {
            suggestions.push('Increase carrier activation or envelope drive – gate ratio is under the recommended coverage.');
        }
        if (Number.isFinite(scoreStats.mean) && scoreStats.mean < sampleScoreWarning) {
            suggestions.push('Average sample fidelity is lagging – consider additional calibration passes.');
        }
        const incompleteSequences = sequenceCoverage.filter((sequence) => !sequence.completed);
        if (incompleteSequences.length) {
            suggestions.push(`Replay ${incompleteSequences.length} incomplete sequence(s) to close dataset gaps.`);
        }
        if (!suggestions.length && Number.isFinite(datasetScore) && datasetScore > 0.9 && outliers.length === 0) {
            suggestions.push('Dataset meets high-fidelity targets – proceed to multimodal export.');
        }
        return suggestions;
    }

    generateNarrative(insights, { maxItems = 4 } = {}) {
        if (!insights || typeof insights !== 'object') {
            return [];
        }
        const items = [];
        const totals = insights.summary?.totals || {};
        const fidelity = Number.isFinite(insights.summary?.datasetScore)
            ? `${(insights.summary.datasetScore * 100).toFixed(1)}%`
            : 'n/a';
        const frames = Number.isFinite(totals.frames) ? totals.frames : 0;
        const sequences = Number.isFinite(totals.sequences) ? totals.sequences : 0;
        items.push(`Dataset fidelity ${fidelity} across ${frames} frame(s) and ${sequences} sequence(s).`);
        const delta = insights.parity?.visualContinuumDelta;
        if (delta && Number.isFinite(delta.average)) {
            items.push(`Visual↔continuum delta avg ${delta.average.toFixed(3)} (min ${Number.isFinite(delta.min) ? delta.min.toFixed(3) : 'n/a'}, max ${Number.isFinite(delta.max) ? delta.max.toFixed(3) : 'n/a'}).`);
        }
        const coherence = insights.parity?.spinorCoherence;
        if (coherence && Number.isFinite(coherence.average)) {
            items.push(`Spinor coherence avg ${(coherence.average * 100).toFixed(1)}% with peak ${(Number.isFinite(coherence.max) ? (coherence.max * 100).toFixed(1) : 'n/a')}%.`);
        }
        const outlierCount = Array.isArray(insights.samples?.outliers) ? insights.samples.outliers.length : 0;
        if (outlierCount) {
            const worst = insights.samples.outliers[0];
            const scoreText = Number.isFinite(worst.score) ? `${(worst.score * 100).toFixed(1)}%` : 'n/a';
            const deltaText = Number.isFinite(worst.delta) ? worst.delta.toFixed(3) : 'n/a';
            items.push(`Flagged ${outlierCount} frame(s) for review – worst score ${scoreText}, delta ${deltaText}.`);
        } else {
            items.push('No critical outliers detected in sampled frames.');
        }
        const recommendations = Array.isArray(insights.recommendations)
            ? insights.recommendations.slice(0, Math.max(0, maxItems - items.length))
            : [];
        recommendations.forEach((recommendation) => {
            items.push(recommendation);
        });
        return items.slice(0, maxItems);
    }

    cloneInsights(insights) {
        if (!insights) {
            return null;
        }
        return structuredCloneSafe(insights) || null;
    }
}

export const calibrationInsightEngine = new CalibrationInsightEngine();
