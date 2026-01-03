import assert from 'node:assert/strict';
import test from 'node:test';

import {
    createLiveStreamState,
    registerLiveFrameMetrics,
    mergeLiveAdapterMetrics,
    formatLiveTelemetrySummary,
    appendLiveStatusLog
} from '../scripts/liveTelemetry.js';
import { evaluateChecksumStatus } from '../scripts/LiveQuaternionAdapters.js';

test('registerLiveFrameMetrics updates latency, saturation, and checksum counters', () => {
    const originalNow = Date.now;
    const fixedNow = 1_700_000_000_000;
    Date.now = () => fixedNow;
    try {
        const state = createLiveStreamState();
        const data = Array.from({ length: 8 }, (_, idx) => idx / 8);
        const frame = {
            source: 'live-websocket',
            timestamp: fixedNow - 25,
            dataArray: data.slice(),
            raw: { data: data.slice() }
        };
        frame.raw.checksum = '00000000';
        const mismatch = evaluateChecksumStatus(frame);
        frame.raw.checksum = mismatch.computed;
        frame.checksumStatus = evaluateChecksumStatus(frame);
        registerLiveFrameMetrics(state, frame, { channelLimit: 16 });

        assert.equal(state.frames, 1);
        assert.equal(state.connected, true);
        assert.equal(state.channelCount, 8);
        assert.equal(state.channelLimit, 16);
        assert.equal(state.channelSaturation.latest, 0.5);
        assert.equal(state.channelSaturation.peak, 0.5);
        assert.ok(state.lastLatency >= 25);
        assert.equal(state.checksum.status, 'valid');
        assert.equal(state.checksum.validated, 1);
        assert.equal(state.checksum.failures, 0);
        assert.match(state.telemetrySummary, /frames 1/);
    } finally {
        Date.now = originalNow;
    }
});

test('mergeLiveAdapterMetrics merges counters and latency aggregates', () => {
    const state = createLiveStreamState();
    Object.assign(state, {
        frames: 5,
        drops: 1,
        parseErrors: 1,
        connectAttempts: 2,
        reconnectAttempts: 2,
        checksum: { status: 'valid', validated: 1, failures: 1 },
        latencySamples: 4,
        latencySum: 100,
        avgLatency: 25
    });
    mergeLiveAdapterMetrics(state, {
        frames: 10,
        drops: 2,
        parseErrors: 1,
        connectAttempts: 3,
        reconnectAttempts: 1,
        latency: { last: 40, min: 22, max: 75, avg: 47.5, samples: 6 },
        interFrameGap: { last: 110, peak: 240 },
        channelSaturation: { latest: 0.75, peak: 0.9 },
        checksum: { status: 'mismatch', validated: 4, failures: 1, last: 'abcd', computed: 'dcba' },
        lastErrorCode: 'serial-invalid-frame',
        connected: true
    });

    assert.equal(state.frames, 15);
    assert.equal(state.drops, 3);
    assert.equal(state.parseErrors, 2);
    assert.equal(state.connectAttempts, 5);
    assert.equal(state.reconnectAttempts, 3);
    assert.equal(state.lastLatency, 40);
    assert.equal(state.minLatency, 22);
    assert.equal(state.maxLatency, 75);
    assert.equal(state.latencySum, 385);
    assert.equal(state.latencySamples, 10);
    assert.equal(state.avgLatency, 38.5);
    assert.equal(state.interFrameGap, 110);
    assert.equal(state.peakInterFrameGap, 240);
    assert.equal(state.channelSaturation.latest, 0.75);
    assert.equal(state.channelSaturation.peak, 0.9);
    assert.equal(state.checksum.status, 'mismatch');
    assert.equal(state.checksum.validated, 5);
    assert.equal(state.checksum.failures, 2);
    assert.equal(state.checksum.lastReported, 'abcd');
    assert.equal(state.checksum.lastComputed, 'dcba');
    assert.equal(state.lastErrorCode, 'serial-invalid-frame');
    assert.equal(state.connected, true);
    assert.match(state.telemetrySummary, /frames 15/);
    assert.match(state.telemetrySummary, /latency 40ms/);
});

test('formatLiveTelemetrySummary produces readable banner text', () => {
    const state = createLiveStreamState();
    Object.assign(state, {
        connected: true,
        mode: 'websocket',
        frames: 12,
        lastLatency: 30.4,
        avgLatency: 28.2,
        drops: 1,
        reconnectAttempts: 2,
        interFrameGap: 55,
        channelCount: 24,
        channelLimit: 32,
        channelSaturation: { latest: 0.75, peak: 0.9 },
        checksum: { status: 'valid', validated: 3, failures: 0 },
        lastErrorCode: 'websocket-error',
        lastStatus: 'Streaming resumed',
        lastStatusLevel: 'info'
    });

    const summary = formatLiveTelemetrySummary(state);
    assert.match(summary, /connected \(websocket\)/);
    assert.match(summary, /frames 12/);
    assert.match(summary, /latency 30ms \(avg 28ms\)/);
    assert.match(summary, /reconnects 2/);
    assert.match(summary, /channels 24\/32 \(75% sat, peak 90%\)/);
    assert.match(summary, /checksum valid ✓3/);
    assert.match(summary, /last code websocket-error/);
    assert.match(summary, /status info: Streaming resumed/);
});

test('appendLiveStatusLog refreshes summary on connection toggles', () => {
    const state = createLiveStreamState();
    assert.match(state.telemetrySummary, /disconnected \(idle\)/);

    state.connected = true;
    state.mode = 'serial';
    appendLiveStatusLog(state, { message: 'Connected to adapter', level: 'info' });

    assert.match(state.telemetrySummary, /connected \(serial\)/);
    assert.match(state.telemetrySummary, /frames 0/);
    assert.equal(state.lastStatus, 'Connected to adapter');
    assert.equal(state.lastStatusLevel, 'info');
    assert.ok(Number.isFinite(state.lastStatusAt));
    assert.match(state.telemetrySummary, /status info: Connected to adapter/);
});

test('appendLiveStatusLog truncates long status entries in the summary', () => {
    const state = createLiveStreamState();
    const longMessage = 'A'.repeat(200);

    appendLiveStatusLog(state, { message: longMessage, level: 'warning' });

    assert.equal(state.lastStatus.startsWith('A'), true);
    assert.match(state.telemetrySummary, /status warning: AAA/);
    assert.equal(/…/.test(state.telemetrySummary), true);
});

test('mergeLiveAdapterMetrics consumes status payloads and refreshes summary', () => {
    const state = createLiveStreamState();
    state.mode = 'serial';
    mergeLiveAdapterMetrics(state, {
        status: { message: 'Adapter warming up', level: 'info', code: 'warming', at: 1700000100000 },
        connected: true
    });

    assert.equal(state.lastStatus, 'Adapter warming up');
    assert.equal(state.lastStatusLevel, 'info');
    assert.equal(state.lastErrorCode, 'warming');
    assert.ok(Number.isFinite(state.lastStatusAt));
    assert.equal(state.connected, true);
    assert.match(state.telemetrySummary, /connected \(serial\)/);
    assert.match(state.telemetrySummary, /status info: Adapter warming up/);
});

test('evaluateChecksumStatus detects mismatches and absence', () => {
    const payload = {
        raw: { data: [0.1, 0.2, 0.3], checksum: '00000000' },
        dataArray: [0.1, 0.2, 0.3]
    };
    const mismatch = evaluateChecksumStatus(payload);
    assert.equal(mismatch.status, 'mismatch');

    const absent = evaluateChecksumStatus({ dataArray: [0.1, 0.2, 0.3] });
    assert.equal(absent.status, 'absent');
});
