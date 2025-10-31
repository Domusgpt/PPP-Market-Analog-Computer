import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const projectRoot = resolve(__dirname, '..');
const summaryPath = resolve(projectRoot, 'samples/calibration/ppp-calibration-dataset-summary.json');

const loadSummary = async () => {
    const raw = await readFile(summaryPath, 'utf8');
    return JSON.parse(raw);
};

test('calibration dataset summary stays within calibrated thresholds', async () => {
    const summary = await loadSummary();
    assert.equal(summary?.metadata?.release, 'ppp-calibration-reference-1');
    assert.equal(summary?.totals?.sequenceCount, 3);
    assert.equal(summary?.totals?.sampleCount, 584);

    const artifact = summary?.metadata?.artifact;
    assert.ok(artifact, 'summary should reference the generated manifest artifact.');
    assert.equal(artifact?.path, 'dist/calibration/ppp-calibration-dataset.json');
    assert.equal(artifact?.format, 'json');
    assert.equal(artifact?.includesSamples, true);
    assert.ok(Number.isFinite(artifact?.bytes) && artifact.bytes > 0, 'manifest byte size should be recorded.');
    assert.ok(typeof artifact?.sha256 === 'string' && artifact.sha256.length === 64, 'manifest sha256 digest should be recorded.');

    assert.ok(summary.score >= 0.6 && summary.score <= 0.7, 'dataset score should remain in the 0.6–0.7 fidelity window.');
    assert.ok(summary.sampleScoreAverage >= 0.6, 'average sample fidelity should stay above 0.60.');

    const delta = summary?.parity?.visualContinuumDelta?.average;
    assert.ok(Number.isFinite(delta) && delta < 0.5, 'visual/continuum delta average should remain under 0.50.');

    const coherence = summary?.parity?.spinorCoherence?.average;
    assert.ok(Number.isFinite(coherence) && coherence >= 0.55, 'spinor coherence average should remain at or above 0.55.');
});
