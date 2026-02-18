import assert from 'node:assert/strict';
import test from 'node:test';

import {
    clampValue,
    lerp,
    parseDataInput,
    formatDataArray,
    cloneMappingDefinition
} from '../scripts/utils.js';

test('clampValue clamps value between min and max', () => {
    assert.equal(clampValue(5, 0, 10), 5);
    assert.equal(clampValue(-5, 0, 10), 0);
    assert.equal(clampValue(15, 0, 10), 10);
    assert.equal(clampValue(0, 0, 10), 0);
    assert.equal(clampValue(10, 0, 10), 10);
});

test('clampValue handles swapped min and max', () => {
    assert.equal(clampValue(5, 10, 0), 5);
    assert.equal(clampValue(-5, 10, 0), 0);
    assert.equal(clampValue(15, 10, 0), 10);
});

test('lerp performs linear interpolation', () => {
    assert.equal(lerp(0, 10, 0), 0);
    assert.equal(lerp(0, 10, 1), 10);
    assert.equal(lerp(0, 10, 0.5), 5);
    assert.equal(lerp(10, 20, 0.1), 11);
});

test('lerp performs extrapolation when t is outside [0, 1]', () => {
    assert.equal(lerp(0, 10, 1.5), 15);
    assert.equal(lerp(0, 10, -0.5), -5);
});

test('parseDataInput parses numeric strings with various separators', () => {
    assert.deepEqual(parseDataInput('1, 2, 3'), [1, 2, 3]);
    assert.deepEqual(parseDataInput('4 5 6'), [4, 5, 6]);
    assert.deepEqual(parseDataInput('7,  8 9'), [7, 8, 9]);
    assert.deepEqual(parseDataInput(' 10, 11 '), [10, 11]);
});

test('parseDataInput filters out non-numeric tokens', () => {
    assert.deepEqual(parseDataInput('1, foo, 2, NaN, Infinity, 3'), [1, 2, 3]);
});

test('parseDataInput handles empty or invalid input', () => {
    assert.deepEqual(parseDataInput(''), []);
    assert.deepEqual(parseDataInput(null), []);
    assert.deepEqual(parseDataInput(undefined), []);
    assert.deepEqual(parseDataInput({}), []);
});

test('formatDataArray formats numbers to 3 decimal places', () => {
    assert.equal(formatDataArray([1.2344, 2, 0.1]), '1.234, 2.000, 0.100');
});

test('formatDataArray handles non-finite values as 0.000', () => {
    assert.equal(formatDataArray([NaN, Infinity, -Infinity]), '0.000, 0.000, 0.000');
});

test('formatDataArray handles empty array', () => {
    assert.equal(formatDataArray([]), '');
});

test('cloneMappingDefinition clones simple objects', () => {
    const original = { a: 1, b: 'test' };
    const clone = cloneMappingDefinition(original);
    assert.deepEqual(clone, original);
    assert.notEqual(clone, original);
});

test('cloneMappingDefinition deep clones special properties (indices, fallbackArray, weights)', () => {
    const original = {
        map1: {
            indices: [1, 2, 3],
            fallbackArray: [0.1, 0.2],
            weights: [0.5, 0.5],
            other: 'keep'
        }
    };
    const clone = cloneMappingDefinition(original);
    assert.deepEqual(clone, original);
    assert.notEqual(clone.map1, original.map1);
    assert.notEqual(clone.map1.indices, original.map1.indices);
    assert.notEqual(clone.map1.fallbackArray, original.map1.fallbackArray);
    assert.notEqual(clone.map1.weights, original.map1.weights);
});

test('cloneMappingDefinition handles TypedArrays for special properties', () => {
    const indices = new Uint16Array([1, 2, 3]);
    const weights = new Float32Array([0.5, 0.5]);
    const original = {
        map1: {
            indices: indices,
            weights: weights
        }
    };
    const clone = cloneMappingDefinition(original);

    // The implementation converts TypedArrays to regular arrays in some cases or slices them
    // Looking at the code:
    // ArrayBuffer.isView(value.indices) -> clone.indices = Array.from(value.indices);
    // ArrayBuffer.isView(value.weights) -> clone.weights = Array.from(value.weights);

    assert.deepEqual(clone.map1.indices, [1, 2, 3]);
    assert.deepEqual(clone.map1.weights, [0.5, 0.5]);
    assert.ok(Array.isArray(clone.map1.indices));
    assert.ok(Array.isArray(clone.map1.weights));
});

test('cloneMappingDefinition handles null/non-object input', () => {
    assert.deepEqual(cloneMappingDefinition(null), {});
    assert.deepEqual(cloneMappingDefinition(undefined), {});
    assert.deepEqual(cloneMappingDefinition(42), {});
});
