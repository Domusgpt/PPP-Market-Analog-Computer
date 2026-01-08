const buildAttributes = (base, extras) => ({
    ...(base || {}),
    ...(extras || {})
});

const clampDelta = (value) => (Number.isFinite(value) && value > 0 ? value : 0);

const createNoopAdapter = () => ({
    recordLiveTelemetry: () => {},
    recordAuditEvidence: () => {},
    recordAuditBatch: () => {}
});

export const createTelemetryOtelAdapter = (config = {}) => {
    if (!config || config.enabled === false) {
        return createNoopAdapter();
    }
    const metricWriter = typeof config.metricWriter === 'function' ? config.metricWriter : null;
    const logWriter = typeof config.logWriter === 'function' ? config.logWriter : null;
    const baseAttributes = typeof config.attributes === 'object' && config.attributes !== null
        ? config.attributes
        : {};
    const namespace = typeof config.namespace === 'string' && config.namespace.length ? config.namespace : 'ppp';
    const lastCounters = {
        frames: 0,
        drops: 0,
        parseErrors: 0,
        reconnectAttempts: 0
    };

    const emitMetric = (name, value, kind, attributes) => {
        if (!metricWriter) {
            return;
        }
        metricWriter({
            name: `${namespace}.${name}`,
            value,
            kind,
            attributes: buildAttributes(baseAttributes, attributes)
        });
    };

    const emitLog = (name, body, attributes) => {
        if (!logWriter) {
            return;
        }
        logWriter({
            name: `${namespace}.${name}`,
            body,
            attributes: buildAttributes(baseAttributes, attributes)
        });
    };

    const recordLiveTelemetry = (state) => {
        if (!state || typeof state !== 'object') {
            return;
        }
        const attributes = {
            mode: state.mode || 'unknown',
            connected: typeof state.connected === 'boolean' ? String(state.connected) : 'unknown'
        };
        const nextCounters = {
            frames: Number.isFinite(state.frames) ? state.frames : lastCounters.frames,
            drops: Number.isFinite(state.drops) ? state.drops : lastCounters.drops,
            parseErrors: Number.isFinite(state.parseErrors) ? state.parseErrors : lastCounters.parseErrors,
            reconnectAttempts: Number.isFinite(state.reconnectAttempts)
                ? state.reconnectAttempts
                : lastCounters.reconnectAttempts
        };
        emitMetric('live.frames', clampDelta(nextCounters.frames - lastCounters.frames), 'counter', attributes);
        emitMetric('live.drops', clampDelta(nextCounters.drops - lastCounters.drops), 'counter', attributes);
        emitMetric('live.parse_errors', clampDelta(nextCounters.parseErrors - lastCounters.parseErrors), 'counter', attributes);
        emitMetric(
            'live.reconnects',
            clampDelta(nextCounters.reconnectAttempts - lastCounters.reconnectAttempts),
            'counter',
            attributes
        );
        lastCounters.frames = nextCounters.frames;
        lastCounters.drops = nextCounters.drops;
        lastCounters.parseErrors = nextCounters.parseErrors;
        lastCounters.reconnectAttempts = nextCounters.reconnectAttempts;

        if (Number.isFinite(state.lastLatency)) {
            emitMetric('live.latency_ms', state.lastLatency, 'gauge', attributes);
        }
        if (Number.isFinite(state.avgLatency)) {
            emitMetric('live.latency_avg_ms', state.avgLatency, 'gauge', attributes);
        }
        if (state.health && typeof state.health.level === 'string') {
            emitLog('live.health', { level: state.health.level, reasons: state.health.reasons || [] }, attributes);
        }
    };

    const recordAuditEvidence = (evidence) => {
        if (!evidence || typeof evidence !== 'object') {
            return;
        }
        emitLog('audit.evidence', evidence, {
            eventType: evidence.eventType || 'unknown'
        });
    };

    const recordAuditBatch = (batch, stage) => {
        if (!batch || typeof batch !== 'object') {
            return;
        }
        emitLog('audit.batch', {
            stage,
            index: batch.index,
            root: batch.root,
            summary: batch.summary,
            anchored: batch.anchored || null
        });
    };

    return {
        recordLiveTelemetry,
        recordAuditEvidence,
        recordAuditBatch
    };
};
