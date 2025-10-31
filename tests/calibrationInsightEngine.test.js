import assert from 'node:assert/strict';
import test from 'node:test';
import { CalibrationInsightEngine } from '../scripts/CalibrationInsightEngine.js';

test('analyzeManifest summarises dataset fidelity and outliers', () => {
    const engine = new CalibrationInsightEngine({
        thresholds: {
            parityDeltaWarning: 0.12,
            parityDeltaCritical: 0.2,
            spinorCoherenceWarning: 0.7,
            spinorCoherenceCritical: 0.5,
            carrierGateWarning: 0.55,
            carrierGateCritical: 0.4,
            sampleScoreWarning: 0.75,
            sampleScoreCritical: 0.55
        },
        maxOutliers: 5
    });

    const manifest = {
        score: 0.88,
        totals: {
            sequenceCount: 2,
            sampleCount: 3
        },
        sequences: [
            { id: 'hopf', label: 'Hopf Orbit', completed: true, frames: 2, sampleRate: 30, durationSeconds: 4 },
            { id: 'flux', label: 'Flux Ramp', completed: false, frames: 1, sampleRate: 24, durationSeconds: 3 }
        ],
        parity: {
            visualContinuumDelta: { average: 0.18, min: 0.05, max: 0.32, count: 3 },
            carrierGateRatio: { average: 0.42, min: 0.3, max: 0.5, count: 3 },
            spinorCoherence: { average: 0.58, min: 0.4, max: 0.9, count: 3 },
            continuumAlignmentMean: { average: 0.76, min: 0.62, max: 0.88, count: 3 },
            envelopeResonance: { average: 0.92, min: 0.82, max: 1.12, count: 3 }
        },
        samples: [
            { score: 0.95, parity: { visualContinuumDelta: 0.05, spinorCoherence: 0.9 } },
            { score: 0.48, parity: { visualContinuumDelta: 0.32, spinorCoherence: 0.4 } },
            { score: 0.6, parity: { visualContinuumDelta: 0.21, spinorCoherence: 0.55 } }
        ]
    };

    const insights = engine.analyzeManifest(manifest);
    assert.equal(insights.summary.totals.frames, 3);
    assert.equal(insights.summary.totals.sequences, 2);
    assert.equal(insights.summary.datasetScore, 0.88);
    assert.ok(insights.parity.visualContinuumDelta.average > 0.15);
    assert.ok(insights.samples.outliers.length >= 1);
    assert.equal(insights.samples.outliers[0].index, 1);
    assert.ok(insights.samples.outliers[0].messages.some((message) => message.includes('delta')));
    assert.ok(insights.recommendations.some((text) => text.includes('mapping weights')));
    assert.ok(insights.recommendations.some((text) => text.includes('Replay')));

    const clone = engine.cloneInsights(insights);
    clone.summary.datasetScore = 0;
    assert.equal(insights.summary.datasetScore, 0.88);

    const narrative = engine.generateNarrative(insights, { maxItems: 6 });
    assert.ok(narrative.length >= 3);
    assert.ok(narrative[0].includes('Dataset fidelity'));

    assert.deepEqual(engine.generateNarrative(null), []);
});

test('identifyOutliers respects maxOutliers and severity ordering', () => {
    const engine = new CalibrationInsightEngine({
        maxOutliers: 2,
        thresholds: {
            parityDeltaCritical: 0.15,
            sampleScoreCritical: 0.6,
            spinorCoherenceCritical: 0.55
        }
    });
    const samples = [
        { score: 0.55, parity: { visualContinuumDelta: 0.14, spinorCoherence: 0.52 } },
        { score: 0.62, parity: { visualContinuumDelta: 0.4, spinorCoherence: 0.6 } },
        { score: 0.9, parity: { visualContinuumDelta: 0.05, spinorCoherence: 0.92 } },
        { score: 0.45, parity: { visualContinuumDelta: 0.22, spinorCoherence: 0.4 } }
    ];
    const outliers = engine.identifyOutliers(samples);
    assert.equal(outliers.length, 2);
    assert.equal(outliers[0].index, 3);
    assert.ok(outliers[0].severity >= outliers[1].severity);
});
