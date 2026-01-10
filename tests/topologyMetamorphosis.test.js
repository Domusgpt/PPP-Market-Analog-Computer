import assert from 'node:assert/strict';
import test from 'node:test';

import { simplex5 } from '../lib/topology/Simplex5.ts';
import { hypercube8 } from '../lib/topology/Hypercube8.ts';
import { TopologyController } from '../lib/topology/TopologyController.ts';

test('Simplex5 provides five vertices and validates the centroid', () => {
    assert.equal(simplex5.vertices.length, 5);
    const centroid = [0, 0, 0, 0];
    const result = simplex5.checkConvexity(centroid);
    assert.equal(result.isValid, true);
});

test('Hypercube8 provides sixteen vertices and validates the origin', () => {
    assert.equal(hypercube8.vertices.length, 16);
    const origin = [0, 0, 0, 0];
    const result = hypercube8.checkConvexity(origin);
    assert.equal(result.isValid, true);
});

test('TopologyController inflates based on tension score', () => {
    const controller = new TopologyController({
        deflateThreshold: 0.2,
        hypercubeThreshold: 0.5,
        cell24Threshold: 0.8,
        minStageDurationMs: 0
    });

    const transition1 = controller.updateTension(0.6);
    assert.equal(controller.stage, 'HYPERCUBE');
    assert.equal(transition1?.to, 'HYPERCUBE');

    const transition2 = controller.updateTension(0.9);
    assert.equal(controller.stage, 'CELL24');
    assert.equal(transition2?.to, 'CELL24');

    const transition3 = controller.updateTension(0.1);
    assert.equal(controller.stage, 'SIMPLEX');
    assert.equal(transition3?.to, 'SIMPLEX');
});
