import { DATA_CHANNEL_COUNT } from './constants.js';

const now = () => Date.now();

const createAdapterMetrics = () => ({
    connectAttempts: 0,
    reconnectAttempts: 0,
    frames: 0,
    drops: 0,
    parseErrors: 0,
    checksumValidated: 0,
    checksumFailures: 0,
    channelSaturation: 0,
    peakChannelSaturation: 0,
    lastLatency: null,
    minLatency: null,
    maxLatency: null,
    avgLatency: null,
    latencySum: 0,
    latencySamples: 0,
    lastReceiveTimestamp: null,
    lastInterFrameGap: null,
    peakInterFrameGap: null,
    lastChecksum: null,
    lastChecksumComputed: null,
    lastChecksumStatus: 'absent',
    lastErrorCode: null,
    connected: false
});

const snapshotAdapterMetrics = (metrics) => ({
    connectAttempts: metrics.connectAttempts,
    reconnectAttempts: metrics.reconnectAttempts,
    frames: metrics.frames,
    drops: metrics.drops,
    parseErrors: metrics.parseErrors,
    checksum: {
        validated: metrics.checksumValidated,
        failures: metrics.checksumFailures,
        last: metrics.lastChecksum,
        computed: metrics.lastChecksumComputed,
        status: metrics.lastChecksumStatus
    },
    channelSaturation: {
        latest: metrics.channelSaturation,
        peak: metrics.peakChannelSaturation
    },
    latency: {
        last: metrics.lastLatency,
        min: metrics.minLatency,
        max: metrics.maxLatency,
        avg: metrics.avgLatency,
        samples: metrics.latencySamples
    },
    interFrameGap: {
        last: metrics.lastInterFrameGap,
        peak: metrics.peakInterFrameGap
    },
    connected: metrics.connected,
    lastErrorCode: metrics.lastErrorCode
});

const computeSimpleChecksum = (payload) => {
    if (!Array.isArray(payload) || payload.length === 0) {
        return null;
    }
    let hash = 0;
    const limit = Math.min(payload.length, 512);
    for (let idx = 0; idx < limit; idx += 1) {
        const value = Number(payload[idx]);
        if (!Number.isFinite(value)) {
            continue;
        }
        const scaled = Math.round(value * 1e6);
        hash = (hash + scaled * (idx + 31)) >>> 0;
        hash = ((hash << 5) - hash) >>> 0;
    }
    return hash.toString(16).padStart(8, '0');
};

const normalizeChecksumValue = (value) => {
    if (value == null) {
        return null;
    }
    if (typeof value === 'number') {
        return (value >>> 0).toString(16).padStart(8, '0');
    }
    if (typeof value === 'string') {
        return value.trim().toLowerCase();
    }
    return null;
};

export const evaluateChecksumStatus = (frame) => {
    if (!frame || typeof frame !== 'object') {
        return { status: 'absent', reported: null, computed: null };
    }
    const raw = typeof frame.raw === 'object' && frame.raw !== null ? frame.raw : frame;
    const payload = Array.isArray(raw.channels)
        ? raw.channels
        : Array.isArray(raw.values)
            ? raw.values
            : Array.isArray(raw.data)
                ? raw.data
                : Array.isArray(frame.dataArray)
                    ? frame.dataArray
                    : null;
    const reported = normalizeChecksumValue(raw.checksum ?? raw.crc ?? raw.signature ?? raw.hash);
    if (!reported) {
        return { status: 'absent', reported: null, computed: null };
    }
    const computed = computeSimpleChecksum(payload);
    if (!computed) {
        return { status: 'skipped', reported, computed: null };
    }
    if (reported === computed) {
        return { status: 'valid', reported, computed };
    }
    return { status: 'mismatch', reported, computed };
};

const applyChecksumMetrics = (metrics, status) => {
    metrics.lastChecksum = status.reported || null;
    metrics.lastChecksumComputed = status.computed || null;
    metrics.lastChecksumStatus = status.status;
    if (status.status === 'valid') {
        metrics.checksumValidated += 1;
    } else if (status.status === 'mismatch') {
        metrics.checksumFailures += 1;
    }
};

const updateLatencyMetrics = (metrics, frameTimestamp) => {
    if (!Number.isFinite(frameTimestamp)) {
        metrics.lastLatency = null;
        return;
    }
    const timestamp = now();
    const latency = Math.max(0, timestamp - frameTimestamp);
    metrics.lastLatency = latency;
    metrics.minLatency = metrics.minLatency == null ? latency : Math.min(metrics.minLatency, latency);
    metrics.maxLatency = metrics.maxLatency == null ? latency : Math.max(metrics.maxLatency, latency);
    metrics.latencySamples += 1;
    metrics.latencySum += latency;
    metrics.avgLatency = metrics.latencySamples > 0 ? metrics.latencySum / metrics.latencySamples : null;
    if (Number.isFinite(metrics.lastReceiveTimestamp)) {
        const gap = Math.max(0, timestamp - metrics.lastReceiveTimestamp);
        metrics.lastInterFrameGap = gap;
        metrics.peakInterFrameGap = metrics.peakInterFrameGap == null
            ? gap
            : Math.max(metrics.peakInterFrameGap, gap);
    }
    metrics.lastReceiveTimestamp = timestamp;
};

const updateChannelSaturationMetrics = (metrics, dataArray, limit) => {
    const frameLength = Array.isArray(dataArray) ? dataArray.length : 0;
    const channelLimit = Number.isFinite(limit) && limit > 0 ? limit : DATA_CHANNEL_COUNT;
    const saturation = channelLimit > 0 ? frameLength / channelLimit : 0;
    metrics.channelSaturation = saturation;
    metrics.peakChannelSaturation = Math.max(metrics.peakChannelSaturation, saturation);
};

const emitStatus = (adapter, status) => {
    if (typeof adapter.onStatus !== 'function') {
        return;
    }
    const payload = typeof status === 'object'
        ? {
            ...status,
            metrics: status.metrics || snapshotAdapterMetrics(adapter.metrics)
        }
        : {
            message: status,
            metrics: snapshotAdapterMetrics(adapter.metrics)
        };
    if (payload.connected === undefined) {
        payload.connected = adapter.metrics.connected;
    }
    adapter.onStatus(payload);
};

const clamp = (value, min, max) => {
    if (!Number.isFinite(value)) {
        return min;
    }
    if (value < min) {
        return min;
    }
    if (value > max) {
        return max;
    }
    return value;
};

const extractPlaneAngle = (source, keys) => {
    if (!source) {
        return 0;
    }
    for (const key of keys) {
        if (key in source) {
            const value = Number(source[key]);
            if (Number.isFinite(value)) {
                return value;
            }
        }
    }
    return 0;
};

const normalizeChannelValue = (value, scale) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
        return 0;
    }
    switch (scale) {
        case 'signed':
            return clamp(0.5 + 0.5 * numeric, 0, 1);
        case 'radians':
            return clamp((numeric + Math.PI) / (2 * Math.PI), 0, 1);
        case 'degrees':
            return clamp(numeric / 360, 0, 1);
        default:
            return clamp(numeric, 0, 1);
    }
};

const resolveTransport = (payload) => {
    const source = typeof payload.transport === 'object' && payload.transport !== null
        ? payload.transport
        : payload;
    return {
        playing: source.playing !== false,
        loop: Boolean(source.loop),
        progress: Number.isFinite(source.progress) ? clamp(source.progress, 0, 1) : 0,
        frameIndex: Number.isFinite(source.frameIndex) ? source.frameIndex : -1,
        frameCount: Number.isFinite(source.frameCount) ? source.frameCount : 0,
        mode: typeof source.mode === 'string' ? source.mode : 'live'
    };
};

const convertMatrixArrayToPlanes = (values) => {
    if (!Array.isArray(values)) {
        return null;
    }
    if (values.length === 6) {
        return {
            xy: Number(values[0]) || 0,
            xz: Number(values[1]) || 0,
            xw: Number(values[2]) || 0,
            yz: Number(values[3]) || 0,
            yw: Number(values[4]) || 0,
            zw: Number(values[5]) || 0
        };
    }
    if (values.length >= 16) {
        // Treat flattened 4x4 matrix as radians in the primary planes.
        // Use off-diagonal components to approximate plane rotations.
        const matrix = values.slice(0, 16);
        const xy = Math.atan2(matrix[1], matrix[0]);
        const xz = Math.atan2(matrix[2], matrix[0]);
        const xw = Math.atan2(matrix[3], matrix[0]);
        const yz = Math.atan2(matrix[6], matrix[5]);
        const yw = Math.atan2(matrix[7], matrix[5]);
        const zw = Math.atan2(matrix[11], matrix[10]);
        return { xy, xz, xw, yz, yw, zw };
    }
    return null;
};

export const decodeQuaternionFrame = (payload, { channelLimit = DATA_CHANNEL_COUNT, diagnostics = null } = {}) => {
    if (payload == null) {
        return null;
    }
    let frame = payload;
    if (typeof payload === 'string') {
        try {
            frame = JSON.parse(payload);
        } catch (error) {
            console.warn('Quaternion frame parse failed', error);
            if (diagnostics && typeof diagnostics.onParseError === 'function') {
                diagnostics.onParseError(error, payload);
            }
            return null;
        }
    }
    if (!frame || typeof frame !== 'object') {
        if (diagnostics && typeof diagnostics.onInvalidFrame === 'function') {
            diagnostics.onInvalidFrame(frame);
        }
        return null;
    }

    const uniforms = { ...(frame.uniforms || {}) };
    const rotationSource = frame.matrix || frame.matrices || frame.planes || frame.rotation || frame.rotations;
    if (rotationSource) {
        const resolved = Array.isArray(rotationSource)
            ? convertMatrixArrayToPlanes(rotationSource)
            : typeof rotationSource === 'object'
                ? rotationSource
                : null;
        if (resolved) {
            uniforms.u_rotXY = extractPlaneAngle(resolved, ['xy', 'XY', 'rotXY', 'u_rotXY', 0]);
            uniforms.u_rotXZ = extractPlaneAngle(resolved, ['xz', 'XZ', 'rotXZ', 'u_rotXZ', 1]);
            uniforms.u_rotXW = extractPlaneAngle(resolved, ['xw', 'XW', 'rotXW', 'u_rotXW', 2]);
            uniforms.u_rotYZ = extractPlaneAngle(resolved, ['yz', 'YZ', 'rotYZ', 'u_rotYZ', 3]);
            uniforms.u_rotYW = extractPlaneAngle(resolved, ['yw', 'YW', 'rotYW', 'u_rotYW', 4]);
            uniforms.u_rotZW = extractPlaneAngle(resolved, ['zw', 'ZW', 'rotZW', 'u_rotZW', 5]);
        }
    }

    const rawChannels = Array.isArray(frame.channels)
        ? frame.channels
        : Array.isArray(frame.values)
            ? frame.values
            : Array.isArray(frame.data)
                ? frame.data
                : null;
    const channelScale = typeof frame.channelScale === 'string' ? frame.channelScale : null;
    const dataArray = [];
    if (rawChannels) {
        const limit = Math.min(rawChannels.length, channelLimit);
        for (let index = 0; index < limit; index += 1) {
            dataArray.push(normalizeChannelValue(rawChannels[index], channelScale));
        }
    }

    const transport = resolveTransport(frame);
    const progress = Number.isFinite(frame.progress)
        ? clamp(frame.progress, 0, 1)
        : transport.progress;

    return {
        source: typeof frame.source === 'string' ? frame.source : 'live',
        origin: frame.origin || null,
        index: Number.isFinite(frame.index) ? frame.index : null,
        timestamp: Number.isFinite(frame.timestamp) ? frame.timestamp : null,
        transport,
        progress,
        uniforms,
        dataArray,
        raw: frame
    };
};

export class WebSocketQuaternionAdapter {
    constructor({ url, decode = decodeQuaternionFrame, channelLimit = DATA_CHANNEL_COUNT, onFrame, onStatus } = {}) {
        this.url = url || null;
        this.decode = decode;
        this.channelLimit = channelLimit;
        this.onFrame = onFrame;
        this.onStatus = onStatus;
        this.socket = null;
        this.metrics = createAdapterMetrics();
        this.diagnostics = {
            onParseError: () => {
                this.metrics.parseErrors += 1;
                this.metrics.drops += 1;
                emitStatus(this, {
                    message: 'WebSocket frame parse failed. Dropping payload.',
                    level: 'warning',
                    code: 'websocket-parse-error',
                    mode: 'websocket',
                    connected: this.metrics.connected
                });
            },
            onInvalidFrame: () => {
                this.metrics.drops += 1;
                emitStatus(this, {
                    message: 'WebSocket delivered invalid frame payload.',
                    level: 'warning',
                    code: 'websocket-invalid-frame',
                    mode: 'websocket',
                    connected: this.metrics.connected
                });
            }
        };
    }

    connect(url) {
        const targetUrl = url || this.url;
        if (!targetUrl) {
            throw new Error('WebSocketQuaternionAdapter requires a URL.');
        }
        this.disconnect();
        this.url = targetUrl;
        if (this.metrics.connectAttempts > 0) {
            this.metrics.reconnectAttempts += 1;
        }
        this.metrics.connectAttempts += 1;
        this.metrics.connected = false;
        this.metrics.lastErrorCode = null;
        emitStatus(this, {
            message: 'Connecting to live WebSocket stream…',
            level: 'info',
            mode: 'websocket',
            connected: false
        });
        try {
            this.socket = new WebSocket(targetUrl);
        } catch (error) {
            this.metrics.lastErrorCode = error && error.code ? error.code : 'websocket-connect-failed';
            if (typeof this.onStatus === 'function') {
                emitStatus(this, {
                    message: `WebSocket connection failed: ${error.message}`,
                    level: 'error',
                    code: this.metrics.lastErrorCode,
                    mode: 'websocket',
                    connected: false
                });
            }
            throw error;
        }
        this.socket.addEventListener('open', () => {
            this.metrics.connected = true;
            this.metrics.lastReceiveTimestamp = null;
            emitStatus(this, {
                message: 'WebSocket live stream connected.',
                level: 'info',
                mode: 'websocket',
                connected: true
            });
        });
        this.socket.addEventListener('message', (event) => {
            const frame = this.decode(event.data, {
                channelLimit: this.channelLimit,
                diagnostics: this.diagnostics
            });
            if (!frame) {
                return;
            }
            this.metrics.frames += 1;
            if (!frame.source || frame.source === 'live') {
                frame.source = 'live-websocket';
            }
            const checksumStatus = evaluateChecksumStatus(frame);
            frame.checksumStatus = checksumStatus;
            applyChecksumMetrics(this.metrics, checksumStatus);
            updateLatencyMetrics(this.metrics, frame.timestamp);
            updateChannelSaturationMetrics(this.metrics, frame.dataArray, this.channelLimit);
            emitStatus(this, {
                mode: 'websocket',
                connected: true
            });
            if (typeof this.onFrame === 'function') {
                this.onFrame(frame);
            }
        });
        this.socket.addEventListener('close', () => {
            this.metrics.connected = false;
            emitStatus(this, {
                message: 'WebSocket live stream disconnected.',
                level: 'info',
                mode: 'websocket',
                connected: false
            });
            this.socket = null;
        });
        this.socket.addEventListener('error', (error) => {
            console.error('WebSocketQuaternionAdapter error', error);
            this.metrics.lastErrorCode = error && error.code ? error.code : 'websocket-error';
            emitStatus(this, {
                message: 'WebSocket live stream error. Check console for details.',
                level: 'error',
                code: this.metrics.lastErrorCode,
                mode: 'websocket',
                connected: this.metrics.connected
            });
        });
    }

    disconnect() {
        if (this.socket) {
            try {
                this.socket.close();
            } catch (error) {
                console.warn('WebSocketQuaternionAdapter disconnect warning', error);
            }
            this.socket = null;
        }
        this.metrics.connected = false;
    }

    isConnected() {
        return Boolean(this.socket && this.socket.readyState === WebSocket.OPEN);
    }

    getMetrics() {
        return snapshotAdapterMetrics(this.metrics);
    }
}

export class SerialQuaternionAdapter {
    constructor({ baudRate = 115200, decode = decodeQuaternionFrame, channelLimit = DATA_CHANNEL_COUNT, onFrame, onStatus } = {}) {
        this.baudRate = baudRate;
        this.decode = decode;
        this.channelLimit = channelLimit;
        this.onFrame = onFrame;
        this.onStatus = onStatus;
        this.port = null;
        this.reader = null;
        this.buffer = '';
        this.active = false;
        this.metrics = createAdapterMetrics();
        this.diagnostics = {
            onParseError: () => {
                this.metrics.parseErrors += 1;
                this.metrics.drops += 1;
                emitStatus(this, {
                    message: 'Serial frame parse failed. Dropping payload.',
                    level: 'warning',
                    code: 'serial-parse-error',
                    mode: 'serial',
                    connected: this.metrics.connected
                });
            },
            onInvalidFrame: () => {
                this.metrics.drops += 1;
                emitStatus(this, {
                    message: 'Serial stream delivered invalid frame payload.',
                    level: 'warning',
                    code: 'serial-invalid-frame',
                    mode: 'serial',
                    connected: this.metrics.connected
                });
            }
        };
    }

    async connect({ baudRate, filters } = {}) {
        if (typeof navigator === 'undefined' || !navigator.serial) {
            const message = 'Web Serial API is not available in this environment.';
            if (typeof this.onStatus === 'function') {
                this.onStatus(message);
            }
            throw new Error(message);
        }
        this.disconnect();
        try {
            this.port = await navigator.serial.requestPort(filters || []);
            await this.port.open({ baudRate: Number.isFinite(baudRate) ? baudRate : this.baudRate });
            this.baudRate = Number.isFinite(baudRate) ? baudRate : this.baudRate;
            const textDecoder = new TextDecoderStream();
            this.readableStreamClosed = this.port.readable.pipeTo(textDecoder.writable).catch(() => {});
            this.reader = textDecoder.readable.getReader();
            this.buffer = '';
            this.active = true;
            if (this.metrics.connectAttempts > 0) {
                this.metrics.reconnectAttempts += 1;
            }
            this.metrics.connectAttempts += 1;
            this.metrics.connected = true;
            this.metrics.lastReceiveTimestamp = null;
            this.metrics.lastErrorCode = null;
            if (typeof this.onStatus === 'function') {
                emitStatus(this, {
                    message: `Serial live stream armed @ ${this.baudRate} baud.`,
                    level: 'info',
                    mode: 'serial',
                    connected: true
                });
            }
            this.#readLoop();
        } catch (error) {
            console.error('SerialQuaternionAdapter connection error', error);
            this.metrics.lastErrorCode = error && error.code ? error.code : 'serial-connect-failed';
            if (typeof this.onStatus === 'function') {
                emitStatus(this, {
                    message: `Serial connection failed: ${error.message}`,
                    level: 'error',
                    code: this.metrics.lastErrorCode,
                    mode: 'serial',
                    connected: false
                });
            }
            throw error;
        }
    }

    async #readLoop() {
        while (this.active && this.reader) {
            let chunk;
            try {
                const { value, done } = await this.reader.read();
                if (done) {
                    break;
                }
                chunk = value;
            } catch (error) {
                console.warn('SerialQuaternionAdapter read error', error);
                break;
            }
            if (!chunk) {
                continue;
            }
            this.buffer += chunk;
            let newlineIndex = this.buffer.indexOf('\n');
            while (newlineIndex >= 0) {
                const line = this.buffer.slice(0, newlineIndex).trim();
                this.buffer = this.buffer.slice(newlineIndex + 1);
                if (line.length) {
                    const frame = this.decode(line, {
                        channelLimit: this.channelLimit,
                        diagnostics: this.diagnostics
                    });
                    if (frame) {
                        this.metrics.frames += 1;
                        if (!frame.source || frame.source === 'live') {
                            frame.source = 'live-serial';
                        }
                        const checksumStatus = evaluateChecksumStatus(frame);
                        frame.checksumStatus = checksumStatus;
                        applyChecksumMetrics(this.metrics, checksumStatus);
                        updateLatencyMetrics(this.metrics, frame.timestamp);
                        updateChannelSaturationMetrics(this.metrics, frame.dataArray, this.channelLimit);
                        emitStatus(this, {
                            mode: 'serial',
                            connected: this.metrics.connected
                        });
                        if (typeof this.onFrame === 'function') {
                            this.onFrame(frame);
                        }
                    }
                }
                newlineIndex = this.buffer.indexOf('\n');
            }
        }
        if (this.active && typeof this.onStatus === 'function') {
            this.metrics.connected = false;
            emitStatus(this, {
                message: 'Serial live stream disconnected.',
                level: 'info',
                mode: 'serial',
                connected: false
            });
        }
        this.active = false;
    }

    async disconnect() {
        this.active = false;
        if (this.reader) {
            try {
                await this.reader.cancel();
            } catch (error) {
                console.warn('SerialQuaternionAdapter reader cancel warning', error);
            }
            this.reader = null;
        }
        if (this.port) {
            try {
                await this.port.close();
            } catch (error) {
                console.warn('SerialQuaternionAdapter port close warning', error);
            }
            this.port = null;
        }
        if (this.readableStreamClosed) {
            try {
                await this.readableStreamClosed;
            } catch (error) {
                console.warn('SerialQuaternionAdapter stream close warning', error);
            }
            this.readableStreamClosed = null;
        }
        this.metrics.connected = false;
    }

    isConnected() {
        return this.active;
    }

    getMetrics() {
        return snapshotAdapterMetrics(this.metrics);
    }
}
