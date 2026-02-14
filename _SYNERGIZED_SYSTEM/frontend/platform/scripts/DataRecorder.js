const cloneArray = (value) => {
    if (Array.isArray(value)) {
        return value.slice();
    }
    if (ArrayBuffer.isView(value)) {
        return Array.from(value);
    }
    if (value && typeof value.length === 'number') {
        try {
            return Array.from(value);
        } catch (error) {
            return [];
        }
    }
    return [];
};

const cloneUniformSnapshot = (uniforms) => {
    if (!uniforms || typeof uniforms !== 'object') {
        return {};
    }
    return Object.fromEntries(
        Object.entries(uniforms).map(([key, value]) => {
            if (Array.isArray(value)) {
                return [key, value.slice()];
            }
            if (ArrayBuffer.isView(value)) {
                return [key, Array.from(value)];
            }
            if (value && typeof value === 'object') {
                return [key, cloneUniformSnapshot(value)];
            }
            return [key, value];
        })
    );
};

const sanitizeMaxEntries = (value) => {
    if (!Number.isFinite(value)) {
        return 512;
    }
    return Math.max(1, Math.floor(value));
};

const now = () => {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
        return performance.now();
    }
    return Date.now();
};

export class DataRecorder {
    constructor({ maxEntries = 512, includeUniforms = true } = {}) {
        this.maxEntries = sanitizeMaxEntries(maxEntries);
        this.includeUniforms = includeUniforms !== false;
        this.records = [];
        this.recording = false;
        this.startedAt = null;
        this.updatedAt = null;
    }

    setMaxEntries(value) {
        this.maxEntries = sanitizeMaxEntries(value);
        if (this.records.length > this.maxEntries) {
            this.records.splice(0, this.records.length - this.maxEntries);
        }
    }

    setIncludeUniforms(includeUniforms) {
        this.includeUniforms = includeUniforms !== false;
        if (!this.includeUniforms) {
            this.records = this.records.map((entry) => ({
                index: entry.index,
                timestamp: entry.timestamp,
                elapsed: entry.elapsed,
                data: cloneArray(entry.data)
            }));
        }
    }

    start({ clear = false } = {}) {
        if (clear) {
            this.clear();
        }
        this.recording = true;
        if (!this.startedAt) {
            this.startedAt = now();
        } else if (clear) {
            this.startedAt = now();
        }
    }

    stop() {
        this.recording = false;
        return this.getRecords();
    }

    clear() {
        this.records = [];
        this.startedAt = null;
        this.updatedAt = null;
    }

    capture(dataArray, uniformSnapshot) {
        if (!this.recording) {
            return null;
        }
        const timestamp = now();
        if (!this.startedAt) {
            this.startedAt = timestamp;
        }
        const entry = {
            index: this.records.length ? this.records[this.records.length - 1].index + 1 : 0,
            timestamp,
            elapsed: this.startedAt ? timestamp - this.startedAt : 0,
            data: cloneArray(dataArray)
        };
        if (this.includeUniforms && uniformSnapshot) {
            entry.uniforms = cloneUniformSnapshot(uniformSnapshot);
        }
        this.records.push(entry);
        if (this.records.length > this.maxEntries) {
            this.records.splice(0, this.records.length - this.maxEntries);
        }
        this.updatedAt = timestamp;
        return { ...entry, data: entry.data.slice(), uniforms: entry.uniforms ? cloneUniformSnapshot(entry.uniforms) : undefined };
    }

    isRecording() {
        return this.recording;
    }

    getRecords() {
        return this.records.map((entry) => ({
            index: entry.index,
            timestamp: entry.timestamp,
            elapsed: entry.elapsed,
            data: cloneArray(entry.data),
            ...(entry.uniforms ? { uniforms: cloneUniformSnapshot(entry.uniforms) } : {})
        }));
    }

    getStats() {
        return {
            frameCount: this.records.length,
            maxEntries: this.maxEntries,
            recording: this.recording,
            includeUniforms: this.includeUniforms,
            startedAt: this.startedAt,
            updatedAt: this.updatedAt
        };
    }

    toJSON() {
        return {
            meta: {
                frameCount: this.records.length,
                maxEntries: this.maxEntries,
                includeUniforms: this.includeUniforms,
                recording: this.recording,
                startedAt: this.startedAt,
                updatedAt: this.updatedAt
            },
            records: this.getRecords()
        };
    }
}
