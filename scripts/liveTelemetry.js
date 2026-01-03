import { DATA_CHANNEL_COUNT } from './constants.js';
import { evaluateChecksumStatus } from './LiveQuaternionAdapters.js';

const clampRatio = (value) => {
    if (!Number.isFinite(value)) {
        return 0;
    }
    if (value < 0) {
        return 0;
    }
    if (value > 1) {
        return 1;
    }
    return value;
};

const formatLatency = (value) => {
    if (!Number.isFinite(value)) {
        return null;
    }
    return `${Math.round(value)}ms`;
};

const formatStatusSnippet = (message, level, maxLength = 120) => {
    if (typeof message !== 'string') {
        return null;
    }
    const normalizedMessage = message.trim().replace(/\s+/g, ' ');
    if (!normalizedMessage.length) {
        return null;
    }
    const prefix = typeof level === 'string' && level.length ? `${level}: ` : '';
    const combined = `${prefix}${normalizedMessage}`;
    if (combined.length > maxLength) {
        return `${combined.slice(0, maxLength - 1)}…`;
    }
    return combined;
};

export const formatLiveTelemetrySummary = (state) => {
    if (!state || typeof state !== 'object') {
        return 'No live telemetry.';
    }
    const segments = [];
    const modeLabel = typeof state.mode === 'string' && state.mode.length ? state.mode : null;
    if (state.connected === true) {
        segments.push(modeLabel ? `connected (${modeLabel})` : 'connected');
    } else if (state.connected === false) {
        segments.push(modeLabel ? `disconnected (${modeLabel})` : 'disconnected');
    } else if (modeLabel) {
        segments.push(modeLabel);
    }
    segments.push(`frames ${state.frames}`);
    const latencyText = formatLatency(state.lastLatency);
    if (latencyText) {
        const avgText = formatLatency(state.avgLatency);
        segments.push(`latency ${latencyText}${avgText ? ` (avg ${avgText})` : ''}`);
    }
    if (Number.isFinite(state.drops) && state.drops > 0) {
        segments.push(`drops ${state.drops}`);
    }
    if (Number.isFinite(state.parseErrors) && state.parseErrors > 0) {
        segments.push(`parse ${state.parseErrors}`);
    }
    if (Number.isFinite(state.reconnectAttempts) && state.reconnectAttempts > 0) {
        segments.push(`reconnects ${state.reconnectAttempts}`);
    }
    if (Number.isFinite(state.interFrameGap)) {
        segments.push(`gap ${formatLatency(state.interFrameGap)}`);
    }
    if (state.channelSaturation) {
        const latest = Number.isFinite(state.channelSaturation.latest)
            ? Math.round(state.channelSaturation.latest * 100)
            : null;
        const peak = Number.isFinite(state.channelSaturation.peak)
            ? Math.round(state.channelSaturation.peak * 100)
            : null;
        if (latest !== null) {
            segments.push(`channels ${state.channelCount}/${state.channelLimit} (${latest}% sat${
                peak !== null && peak !== latest ? `, peak ${peak}%` : ''
            })`);
        }
    }
    if (state.checksum && state.checksum.status && state.checksum.status !== 'absent') {
        const checksumSegment = [`checksum ${state.checksum.status}`];
        if (state.checksum.validated) {
            checksumSegment.push(`✓${state.checksum.validated}`);
        }
        if (state.checksum.failures) {
            checksumSegment.push(`⚠︎${state.checksum.failures}`);
        }
        segments.push(checksumSegment.join(' '));
    }
    const statusSnippet = formatStatusSnippet(state.lastStatus, state.lastStatusLevel);
    if (statusSnippet) {
        segments.push(`status ${statusSnippet}`);
    }
    if (state.lastErrorCode) {
        segments.push(`last code ${state.lastErrorCode}`);
    }
    if (!segments.length) {
        return 'No live telemetry yet.';
    }
    return segments.join(' • ');
};

const refreshTelemetrySummary = (state) => {
    if (!state || typeof state !== 'object') {
        return state;
    }
    state.telemetrySummary = formatLiveTelemetrySummary(state);
    return state;
};

export const createLiveStreamState = () => ({
    mode: 'idle',
    connected: false,
    frames: 0,
    lastSource: null,
    lastTimestamp: null,
    lastLatency: null,
    minLatency: null,
    maxLatency: null,
    avgLatency: null,
    latencySamples: 0,
    latencySum: 0,
    interFrameGap: null,
    peakInterFrameGap: null,
    lastReceiveAt: null,
    channelCount: 0,
    channelLimit: DATA_CHANNEL_COUNT,
    channelSaturation: { latest: 0, peak: 0 },
    drops: 0,
    parseErrors: 0,
    connectAttempts: 0,
    reconnectAttempts: 0,
    checksum: {
        status: 'absent',
        validated: 0,
        failures: 0,
        lastReported: null,
        lastComputed: null
    },
    lastStatus: null,
    lastStatusLevel: null,
    lastStatusAt: null,
    lastErrorCode: null,
    statusLog: [],
    telemetrySummary: formatLiveTelemetrySummary({ mode: 'idle', connected: false, frames: 0 })
});

export const mergeLiveAdapterMetrics = (state, metrics = {}) => {
    if (!state || typeof state !== 'object' || !metrics || typeof metrics !== 'object') {
        return state;
    }
    if (typeof metrics.connected === 'boolean') {
        state.connected = metrics.connected;
    }
    if (Number.isFinite(metrics.frames)) {
        state.frames += metrics.frames;
    }
    if (Number.isFinite(metrics.drops)) {
        state.drops += metrics.drops;
    }
    if (Number.isFinite(metrics.parseErrors)) {
        state.parseErrors += metrics.parseErrors;
    }
    if (Number.isFinite(metrics.connectAttempts)) {
        state.connectAttempts += metrics.connectAttempts;
    }
    if (Number.isFinite(metrics.reconnectAttempts)) {
        state.reconnectAttempts += metrics.reconnectAttempts;
    }
    if (metrics.latency) {
        const { last, min, max, avg, samples } = metrics.latency;
        if (Number.isFinite(last)) {
            state.lastLatency = last;
        }
        if (Number.isFinite(min)) {
            state.minLatency = state.minLatency == null ? min : Math.min(state.minLatency, min);
        }
        if (Number.isFinite(max)) {
            state.maxLatency = state.maxLatency == null ? max : Math.max(state.maxLatency, max);
        }
        const hasSamples = Number.isFinite(samples);
        const metricsLatencySum = Number.isFinite(avg) && hasSamples ? avg * samples : null;
        if (hasSamples) {
            const baseSamples = Number.isFinite(state.latencySamples) ? state.latencySamples : 0;
            const baseSum = Number.isFinite(state.latencySum) ? state.latencySum : 0;
            state.latencySamples = baseSamples + samples;
            if (metricsLatencySum !== null) {
                state.latencySum = baseSum + metricsLatencySum;
                state.avgLatency = state.latencySamples > 0
                    ? state.latencySum / state.latencySamples
                    : state.avgLatency ?? null;
            } else if (Number.isFinite(avg)) {
                state.avgLatency = avg;
            }
        } else if (Number.isFinite(avg)) {
            state.avgLatency = avg;
        }
    } else {
        if (Number.isFinite(metrics.lastLatency)) {
            state.lastLatency = metrics.lastLatency;
        }
        if (Number.isFinite(metrics.minLatency)) {
            state.minLatency = state.minLatency == null ? metrics.minLatency : Math.min(
                state.minLatency,
                metrics.minLatency
            );
        }
        if (Number.isFinite(metrics.maxLatency)) {
            state.maxLatency = state.maxLatency == null ? metrics.maxLatency : Math.max(
                state.maxLatency,
                metrics.maxLatency
            );
        }
        const hasSamples = Number.isFinite(metrics.latencySamples);
        const metricsLatencySum = Number.isFinite(metrics.avgLatency) && hasSamples
            ? metrics.avgLatency * metrics.latencySamples
            : null;
        if (hasSamples) {
            const baseSamples = Number.isFinite(state.latencySamples) ? state.latencySamples : 0;
            const baseSum = Number.isFinite(state.latencySum) ? state.latencySum : 0;
            state.latencySamples = baseSamples + metrics.latencySamples;
            if (metricsLatencySum !== null) {
                state.latencySum = baseSum + metricsLatencySum;
                state.avgLatency = state.latencySamples > 0
                    ? state.latencySum / state.latencySamples
                    : state.avgLatency ?? null;
            } else if (Number.isFinite(metrics.avgLatency)) {
                state.avgLatency = metrics.avgLatency;
            }
        } else if (Number.isFinite(metrics.avgLatency)) {
            state.avgLatency = metrics.avgLatency;
        }
    }
    const channelMetrics = metrics.channelSaturation;
    if (channelMetrics) {
        const latest = Number.isFinite(channelMetrics.latest)
            ? clampRatio(channelMetrics.latest)
            : null;
        const peak = Number.isFinite(channelMetrics.peak)
            ? clampRatio(channelMetrics.peak)
            : null;
        if (latest !== null) {
            state.channelSaturation.latest = latest;
        }
        if (peak !== null) {
            state.channelSaturation.peak = Math.max(state.channelSaturation.peak, peak);
        }
    } else if (Number.isFinite(metrics.channelSaturation)) {
        const saturation = clampRatio(metrics.channelSaturation);
        state.channelSaturation.latest = saturation;
        state.channelSaturation.peak = Math.max(state.channelSaturation.peak, saturation);
    }
    if (metrics.interFrameGap) {
        const { last, peak } = metrics.interFrameGap;
        if (Number.isFinite(last)) {
            state.interFrameGap = last;
        }
        if (Number.isFinite(peak)) {
            state.peakInterFrameGap = Math.max(state.peakInterFrameGap || 0, peak);
        }
    }
    if (metrics.checksum) {
        state.checksum = {
            status: metrics.checksum.status || state.checksum.status,
            validated: Number.isFinite(metrics.checksum.validated)
                ? state.checksum.validated + metrics.checksum.validated
                : state.checksum.validated,
            failures: Number.isFinite(metrics.checksum.failures)
                ? state.checksum.failures + metrics.checksum.failures
                : state.checksum.failures,
            lastReported: metrics.checksum.last ?? state.checksum.lastReported,
            lastComputed: metrics.checksum.computed ?? state.checksum.lastComputed
        };
    }
    if (typeof metrics.lastErrorCode === 'string') {
        state.lastErrorCode = metrics.lastErrorCode;
    }
    if (metrics.status && (typeof metrics.status.message === 'string' || typeof metrics.status.level === 'string')) {
        appendLiveStatusLog(state, metrics.status);
        return state;
    }
    return refreshTelemetrySummary(state);
};

export const registerLiveFrameMetrics = (
    state,
    frame,
    { channelLimit = DATA_CHANNEL_COUNT } = {}
) => {
    if (!state || typeof state !== 'object' || !frame) {
        return state;
    }
    const now = Date.now();
    state.frames += 1;
    const source = typeof frame.source === 'string' ? frame.source : 'live';
    state.lastSource = source;
    const frameTimestamp = Number.isFinite(frame.timestamp) ? frame.timestamp : null;
    if (frameTimestamp !== null) {
        const latency = Math.max(0, now - frameTimestamp);
        state.lastLatency = latency;
        state.minLatency = state.minLatency == null ? latency : Math.min(state.minLatency, latency);
        state.maxLatency = state.maxLatency == null ? latency : Math.max(state.maxLatency, latency);
        state.latencySamples += 1;
        state.latencySum += latency;
        state.avgLatency = state.latencySamples > 0 ? state.latencySum / state.latencySamples : null;
        state.lastTimestamp = frameTimestamp;
    } else {
        state.lastLatency = null;
        state.lastTimestamp = now;
    }
    state.connected = true;
    if (typeof state.lastReceiveAt === 'number') {
        const gap = Math.max(0, now - state.lastReceiveAt);
        state.interFrameGap = gap;
        state.peakInterFrameGap = Math.max(state.peakInterFrameGap || 0, gap);
    } else {
        state.interFrameGap = null;
    }
    state.lastReceiveAt = now;
    const length = Array.isArray(frame.dataArray) ? frame.dataArray.length : 0;
    state.channelCount = length;
    state.channelLimit = Number.isFinite(channelLimit) && channelLimit > 0 ? channelLimit : DATA_CHANNEL_COUNT;
    const saturation = state.channelLimit > 0 ? clampRatio(length / state.channelLimit) : 0;
    state.channelSaturation.latest = saturation;
    state.channelSaturation.peak = Math.max(state.channelSaturation.peak, saturation);
    const checksumStatus = frame.checksumStatus || evaluateChecksumStatus(frame);
    state.checksum = {
        status: checksumStatus.status,
        validated: state.checksum?.validated || 0,
        failures: state.checksum?.failures || 0,
        lastReported: checksumStatus.reported ?? null,
        lastComputed: checksumStatus.computed ?? null
    };
    if (checksumStatus.status === 'valid') {
        state.checksum.validated += 1;
    } else if (checksumStatus.status === 'mismatch') {
        state.checksum.failures += 1;
    }
    return refreshTelemetrySummary(state);
};

export const appendLiveStatusLog = (state, entry, { limit = 10 } = {}) => {
    if (!state || typeof state !== 'object' || !entry) {
        return state;
    }
    const timestamp = Number.isFinite(entry.at) ? entry.at : Date.now();
    const message = typeof entry.message === 'string' ? entry.message : null;
    if (message) {
        state.lastStatus = message;
        state.lastStatusAt = timestamp;
    }
    if (typeof entry.level === 'string') {
        state.lastStatusLevel = entry.level;
    }
    if (typeof entry.code === 'string') {
        state.lastErrorCode = entry.code;
    }
    state.statusLog = Array.isArray(state.statusLog) ? state.statusLog.slice() : [];
    state.statusLog.push({ ...entry, at: timestamp });
    while (state.statusLog.length > limit) {
        state.statusLog.shift();
    }
    return refreshTelemetrySummary(state);
};
