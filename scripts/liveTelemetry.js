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

const formatDuration = (milliseconds) => {
    if (!Number.isFinite(milliseconds)) {
        return null;
    }
    const totalSeconds = Math.max(0, Math.floor(milliseconds / 1000));
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    const parts = [];
    if (hours) {
        parts.push(`${hours}h`);
    }
    if (minutes) {
        parts.push(`${minutes}m`);
    }
    parts.push(`${seconds}s`);
    return parts.join(' ');
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

const formatStatusTime = (timestamp) => {
    const value = Number.isFinite(timestamp) ? timestamp : null;
    if (value === null) {
        return null;
    }
    const date = new Date(value);
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');
    const seconds = date.getSeconds().toString().padStart(2, '0');
    return `@${hours}:${minutes}:${seconds}`;
};

export const formatLiveTelemetrySummary = (state) => {
    if (!state || typeof state !== 'object') {
        return 'No live telemetry.';
    }
    const segments = [];
    const health = evaluateLiveHealth(state);
    segments.push(`health ${health.level}`);
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
    const lastReceiveAt = Number.isFinite(state.lastReceiveAt) ? state.lastReceiveAt : null;
    const staleAfterMs = Number.isFinite(state.staleAfterMs) ? state.staleAfterMs : null;
    const staleCriticalAfterMs = Number.isFinite(state.staleCriticalAfterMs)
        ? state.staleCriticalAfterMs
        : staleAfterMs;
    if (lastReceiveAt !== null) {
        const ageMs = Date.now() - lastReceiveAt;
        const ageText = formatDuration(ageMs);
        if (ageText) {
            if (staleCriticalAfterMs !== null && ageMs >= staleCriticalAfterMs) {
                segments.push(`stale ${ageText}`);
            } else if (staleAfterMs !== null && ageMs >= staleAfterMs) {
                segments.push(`stale ${ageText}`);
            } else if (state.frames > 0 || state.connected) {
                segments.push(`last ${ageText} ago`);
            }
        }
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
    const statusTime = formatStatusTime(state.lastStatusAt);
    if (statusSnippet) {
        segments.push(`status ${statusSnippet}${statusTime ? ` ${statusTime}` : ''}`);
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
    const health = evaluateLiveHealth(state);
    state.health = health;
    state.telemetrySummary = formatLiveTelemetrySummary(state);
    return state;
};

export const createLiveStreamState = () => ({
    mode: 'idle',
    connected: false,
    health: { level: 'ok', reasons: [] },
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
    staleAfterMs: 5_000,
    staleCriticalAfterMs: 15_000,
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
    statusTtlMs: null,
    statusLogLimit: 10,
    statusLog: [],
    telemetrySummary: formatLiveTelemetrySummary({ mode: 'idle', connected: false, frames: 0 })
});

export function evaluateLiveHealth(state) {
    if (!state || typeof state !== 'object') {
        return { level: 'unknown', reasons: ['missing-state'] };
    }
    const reasons = [];
    let level = 'ok';

    const now = Date.now();
    const lastReceiveAt = Number.isFinite(state.lastReceiveAt) ? state.lastReceiveAt : null;
    const staleAfterMs = Number.isFinite(state.staleAfterMs) ? state.staleAfterMs : null;
    const staleCriticalAfterMs = Number.isFinite(state.staleCriticalAfterMs)
        ? state.staleCriticalAfterMs
        : staleAfterMs;
    const staleness = lastReceiveAt !== null && staleAfterMs !== null ? Math.max(0, now - lastReceiveAt) : null;
    const hasCriticalStaleness = staleness !== null && staleCriticalAfterMs !== null
        ? staleness >= staleCriticalAfterMs
        : false;
    const hasStaleness = staleness !== null ? staleness >= (staleAfterMs ?? Number.POSITIVE_INFINITY) : false;

    const hasErrors = state.lastStatusLevel === 'error'
        || state.checksum?.status === 'mismatch'
        || (Number.isFinite(state.parseErrors) && state.parseErrors > 0)
        || (Number.isFinite(state.drops) && state.drops > 0 && Number.isFinite(state.frames)
            && state.drops > state.frames / 2)
        || hasCriticalStaleness;
    if (hasErrors) {
        level = 'error';
    } else {
        const warnings = [];
        if (state.lastStatusLevel === 'warning') warnings.push('status-warning');
        if (state.connected === false) warnings.push('disconnected');
        if (state.checksum?.status === 'absent') warnings.push('checksum-absent');
        if (Number.isFinite(state.drops) && state.drops > 0) warnings.push('drops');
        if (Number.isFinite(state.parseErrors) && state.parseErrors > 0) warnings.push('parse-errors');
        if (Number.isFinite(state.interFrameGap) && state.interFrameGap > 2_000) warnings.push('slow-stream');
        if (hasStaleness) warnings.push('stale');

        if (warnings.length) {
            level = 'warning';
            reasons.push(...warnings);
        }
    }

    if (level === 'error') {
        if (state.lastStatusLevel === 'error') reasons.push('status-error');
        if (state.checksum?.status === 'mismatch') reasons.push('checksum-mismatch');
        if (Number.isFinite(state.parseErrors) && state.parseErrors > 0) reasons.push('parse-errors');
        if (Number.isFinite(state.drops) && state.drops > 0 && Number.isFinite(state.frames)
            && state.drops > state.frames / 2) {
            reasons.push('excessive-drops');
        }
        if (hasCriticalStaleness) {
            reasons.push('stale-critical');
        } else if (hasStaleness && !reasons.includes('stale')) {
            reasons.push('stale');
        }
    }

    return { level, reasons };
}

export const mergeLiveAdapterMetrics = (state, metrics = {}) => {
    if (!state || typeof state !== 'object' || !metrics || typeof metrics !== 'object') {
        return state;
    }
    if (typeof metrics.connected === 'boolean') {
        state.connected = metrics.connected;
    }
    if (Number.isFinite(metrics.lastReceiveAt)) {
        state.lastReceiveAt = metrics.lastReceiveAt;
    }
    if (Number.isFinite(metrics.staleAfterMs)) {
        state.staleAfterMs = metrics.staleAfterMs;
    }
    if (Number.isFinite(metrics.staleCriticalAfterMs)) {
        state.staleCriticalAfterMs = metrics.staleCriticalAfterMs;
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
        const statusLimit = Number.isFinite(metrics.status.limit)
            ? metrics.status.limit
            : Number.isFinite(metrics.statusLogLimit)
                ? metrics.statusLogLimit
                : undefined;
        const statusTtl = Number.isFinite(metrics.status.ttlMs)
            ? metrics.status.ttlMs
            : Number.isFinite(metrics.statusTtlMs)
                ? metrics.statusTtlMs
                : undefined;
        appendLiveStatusLog(state, metrics.status, { limit: statusLimit, ttlMs: statusTtl });
        return state;
    }
    return refreshTelemetrySummary(state);
};

export const registerLiveFrameMetrics = (
    state,
    frame,
    { channelLimit = DATA_CHANNEL_COUNT, staleAfterMs, staleCriticalAfterMs } = {}
) => {
    if (!state || typeof state !== 'object' || !frame) {
        return state;
    }
    if (Number.isFinite(staleAfterMs)) {
        state.staleAfterMs = staleAfterMs;
    }
    if (Number.isFinite(staleCriticalAfterMs)) {
        state.staleCriticalAfterMs = staleCriticalAfterMs;
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

export const appendLiveStatusLog = (state, entry, options = {}) => {
    if (!state || typeof state !== 'object' || !entry) {
        return state;
    }
    const normalizedLimit = Number.isFinite(options.limit) && options.limit > 0 ? options.limit : null;
    if (normalizedLimit !== null) {
        state.statusLogLimit = normalizedLimit;
    }
    const normalizedTtl = Number.isFinite(options.ttlMs) && options.ttlMs > 0 ? options.ttlMs : null;
    if (normalizedTtl !== null) {
        state.statusTtlMs = normalizedTtl;
    }
    const resolvedLimit = Number.isFinite(state.statusLogLimit) && state.statusLogLimit > 0
        ? state.statusLogLimit
        : 10;
    const resolvedTtl = Number.isFinite(state.statusTtlMs) && state.statusTtlMs > 0
        ? state.statusTtlMs
        : null;
    const now = Date.now();
    const timestamp = Number.isFinite(entry.at) ? entry.at : Date.now();
    state.statusLog = Array.isArray(state.statusLog) ? state.statusLog.slice() : [];
    state.statusLog.push({ ...entry, at: timestamp });
    if (resolvedTtl !== null) {
        const cutoff = now - resolvedTtl;
        state.statusLog = state.statusLog.filter((item) => Number.isFinite(item.at) && item.at >= cutoff);
    }
    while (state.statusLog.length > resolvedLimit) {
        state.statusLog.shift();
    }
    const latestEntry = state.statusLog[state.statusLog.length - 1];
    const message = typeof latestEntry?.message === 'string' ? latestEntry.message : null;
    state.lastStatus = message || null;
    state.lastStatusAt = Number.isFinite(latestEntry?.at) ? latestEntry.at : null;
    state.lastStatusLevel = typeof latestEntry?.level === 'string' ? latestEntry.level : null;
    state.lastErrorCode = typeof latestEntry?.code === 'string' ? latestEntry.code : null;
    return refreshTelemetrySummary(state);
};
