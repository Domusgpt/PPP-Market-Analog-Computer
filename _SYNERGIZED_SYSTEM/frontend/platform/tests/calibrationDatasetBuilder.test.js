import { test } from 'node:test';
import assert from 'node:assert/strict';

import { CalibrationToolkit } from '../scripts/CalibrationToolkit.js';
import { CalibrationDatasetBuilder } from '../scripts/CalibrationDatasetBuilder.js';

const createToolkit = () => {
    let analysisCallCount = 0;
    return new CalibrationToolkit({
        applyDataArray: () => {},
        captureFrame: () => null,
        getSonicAnalysis: () => {
            analysisCallCount += 1;
            return {
                summary: `stub-${analysisCallCount}`,
                signal: {
                    carrierMatrix: [
                        [{ energy: 0.6, gate: 1, frequency: 440 }],
                        [{ energy: 0.4, gate: 0, frequency: 660 }]
                    ],
                    bitstream: { density: 0.5 },
                    envelope: { resonance: 0.4, centroid: 400, spread: 120 }
                },
                continuum: {
                    voices: [
                        {
                            gridEnergy: 0.5,
                            carrierEnergy: 0.6,
                            continuumAlignment: 0.7,
                            bitEntropy: 0.45
                        }
                    ]
                },
                resonance: {
                    aggregate: {
                        magnitude: 0.9,
                        spinorCoherence: 0.75
                    }
                },
                manifold: {},
                topology: {},
                lattice: {},
                constellation: {}
            };
        },
        sequences: [
            {
                id: 'unit-seq',
                label: 'Unit Sequence',
                durationSeconds: 0.1,
                sampleRate: 1,
                frame: () => ({
                    values: [0.2, 0.8, 0.4],
                    uniforms: { u_rotXY: 0.1, u_rotXZ: -0.3 }
                })
            }
        ]
    });
};

test('CalibrationDatasetBuilder aggregates parity metrics into manifest', async () => {
    const toolkit = createToolkit();
    const builder = new CalibrationDatasetBuilder({ toolkit });
    const manifest = await builder.runPlan([
        { sequenceId: 'unit-seq', sampleRate: 1, durationSeconds: 0.1 }
    ]);

    assert.ok(manifest, 'manifest should be generated');
    assert.strictEqual(manifest.totals.sequenceCount, 1);
    assert.strictEqual(manifest.totals.sampleCount, 1);

    const delta = manifest.parity.visualContinuumDelta;
    assert.ok(delta && Number.isFinite(delta.average), 'visualContinuumDelta average should be numeric');
    const gate = manifest.parity.carrierGateRatio;
    assert.ok(gate && gate.average >= 0 && gate.average <= 1, 'carrier gate ratio should be normalized');

    assert.ok(Array.isArray(manifest.samples) && manifest.samples.length === 1, 'manifest should contain the captured sample');
    const sample = manifest.samples[0];
    assert.ok(sample.parity, 'sample parity metrics available');
    assert.ok(Number.isFinite(sample.score), 'sample fidelity score recorded');

    const summary = builder.getLastManifest({ includeSamples: false });
    assert.ok(summary && !Object.prototype.hasOwnProperty.call(summary, 'samples'), 'summary manifest omits samples when requested');
});
