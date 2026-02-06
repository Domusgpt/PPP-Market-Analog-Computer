import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { SonicGeometryEngine } from '../scripts/SonicGeometryEngine.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const FIXTURE_PATH = join(__dirname, 'fixtures', 'sonic-fixture.json');
const EPSILON = 1e-6;
const RELATIVE_EPSILON = 1e-6;

const approximateEqual = (actual, expected, path) => {
    assert.ok(Number.isFinite(expected), `${path} expected is not finite`);
    assert.ok(Number.isFinite(actual), `${path} actual is not finite`);
    const diff = Math.abs(actual - expected);
    const tolerance = EPSILON + Math.abs(expected) * RELATIVE_EPSILON;
    assert.ok(diff <= tolerance, `${path} expected ${expected} but received ${actual}`);
};

const compareTelemetry = (actual, expected, path = 'root') => {
    if (expected === null || expected === undefined) {
        assert.strictEqual(actual, expected, `${path} mismatch`);
        return;
    }
    if (typeof expected === 'number') {
        approximateEqual(actual, expected, path);
        return;
    }
    if (typeof expected === 'string' || typeof expected === 'boolean') {
        assert.strictEqual(actual, expected, `${path} mismatch`);
        return;
    }
    if (Array.isArray(expected)) {
        assert.ok(Array.isArray(actual), `${path} expected array`);
        assert.strictEqual(actual.length, expected.length, `${path} length mismatch`);
        expected.forEach((value, index) => {
            compareTelemetry(actual[index], value, `${path}[${index}]`);
        });
        return;
    }
    assert.ok(actual && typeof actual === 'object', `${path} expected object`);
    Object.keys(expected).forEach((key) => {
        if (key === 'timestamp' || key === 'generatedAt') {
            return;
        }
        compareTelemetry(actual[key], expected[key], `${path}.${key}`);
    });
};

test('SonicGeometryEngine produces stable telemetry for the golden fixture', async () => {
    const fixture = JSON.parse(await readFile(FIXTURE_PATH, 'utf8'));
    const engine = new SonicGeometryEngine({ outputMode: 'analysis', contextFactory: null });
    await engine.enable();

    const analysis = engine.updateFromData(fixture.values, fixture.metadata);
    assert.ok(analysis, 'analysis should be returned');

    const expectedQuaternion = {
        left: fixture.bridge.leftQuaternion,
        right: fixture.bridge.rightQuaternion,
        leftAngle: fixture.bridge.leftAngle,
        rightAngle: fixture.bridge.rightAngle,
        dot: fixture.bridge.dot,
        bridgeMagnitude: fixture.bridge.bridgeMagnitude,
        bridgeVector: fixture.bridge.bridgeVector,
        normalizedBridge: fixture.bridge.normalizedBridge,
        hopfFiber: fixture.bridge.hopfFiber
    };
    compareTelemetry(analysis.quaternion, expectedQuaternion, 'analysis.quaternion/bridge');
    compareTelemetry(analysis.spinor, fixture.coupler, 'analysis.spinor/coupler');
    compareTelemetry(analysis.resonance, fixture.resonance, 'analysis.resonance');
    compareTelemetry(analysis.signal, fixture.analysis.signal, 'analysis.signal');
    compareTelemetry(analysis.transduction, fixture.analysis.transduction, 'analysis.transduction');
    compareTelemetry(analysis.manifold, fixture.analysis.manifold, 'analysis.manifold');
    compareTelemetry(analysis.topology, fixture.analysis.topology, 'analysis.topology');
    compareTelemetry(analysis.continuum, fixture.analysis.continuum, 'analysis.continuum');
    compareTelemetry(analysis.lattice, fixture.analysis.lattice, 'analysis.lattice');
    compareTelemetry(analysis.constellation, fixture.analysis.constellation, 'analysis.constellation');
    compareTelemetry(analysis.transport, fixture.analysis.transport, 'analysis.transport');
    assert.strictEqual(analysis.voiceCount, fixture.analysis.voiceCount, 'voiceCount mismatch');
    assert.strictEqual(analysis.outputMode, 'analysis');

    const latestAnalysis = engine.getLastAnalysis();
    compareTelemetry(latestAnalysis, fixture.analysis, 'engine.getLastAnalysis');
    compareTelemetry(engine.getLastSignal(), fixture.analysis.signal, 'engine.getLastSignal');
    compareTelemetry(engine.getLastTransduction(), fixture.analysis.transduction, 'engine.getLastTransduction');
    compareTelemetry(engine.getLastManifold(), fixture.analysis.manifold, 'engine.getLastManifold');
    compareTelemetry(engine.getLastTopology(), fixture.analysis.topology, 'engine.getLastTopology');
    compareTelemetry(engine.getLastContinuum(), fixture.analysis.continuum, 'engine.getLastContinuum');
    compareTelemetry(engine.getLastLattice(), fixture.analysis.lattice, 'engine.getLastLattice');
    compareTelemetry(engine.getLastConstellation(), fixture.analysis.constellation, 'engine.getLastConstellation');

    const performance = engine.getPerformanceMetrics();
    assert.ok(performance.samples >= 1, 'performance samples should be recorded');
    assert.ok(performance.averageFrameMs >= 0, 'average frame time should be non-negative');
});
