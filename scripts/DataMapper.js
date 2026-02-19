import { clampValue, lerp, cloneMappingDefinition } from './utils.js';

/**
 * DataMapper maps arbitrary numeric arrays to shader uniform payloads.
 * Mapping entries support scalar or vector uniforms via index/indices definitions,
 * configurable ranges, easing curves, and per-channel smoothing.
 */
export class DataMapper {
    constructor({ mapping = {}, smoothing = 0.25 } = {}) {
        this.mapping = cloneMappingDefinition(mapping);
        this.mappingKeys = Object.keys(this.mapping);
        this.globalSmoothing = clampValue(smoothing, 0, 1);
        this.uniformState = {};
    }

    setMapping(mapping) {
        this.mapping = cloneMappingDefinition(mapping);
        this.mappingKeys = Object.keys(this.mapping);
        this.uniformState = {};
    }

    getMappingDefinition() {
        return cloneMappingDefinition(this.mapping);
    }

    setSmoothing(value) {
        this.globalSmoothing = clampValue(value, 0, 1);
    }

    updateData(dataArray = []) {
        const array = Array.isArray(dataArray)
            ? dataArray
            : dataArray && typeof dataArray.length === 'number'
                ? Array.from(dataArray)
                : [];

        for (const uniformName of this.mappingKeys) {
            const config = this.mapping[uniformName];
            if (!config) {
                continue;
            }

            const hasIndices = Array.isArray(config.indices) || ArrayBuffer.isView(config.indices);

            if (hasIndices) {
                const indices = Array.isArray(config.indices) ? config.indices : Array.from(config.indices);
                if (config.combine) {
                    const rawCombined = this.#combineIndexedValues(array, { ...config, indices });
                    const mappedCombined = this.#mapValue(rawCombined, config);
                    const previous = this.uniformState[uniformName];
                    const mix = this.#resolveSmoothing(config);
                    if (typeof previous === 'number' && Number.isFinite(previous)) {
                        this.uniformState[uniformName] = lerp(previous, mappedCombined, mix);
                    } else {
                        this.uniformState[uniformName] = mappedCombined;
                    }
                    return;
                }

                const values = indices.map((index, idx) => {
                    const fallbackValue = this.#resolveFallbackValue(config.fallbackArray, idx, config.fallback);
                    const raw = this.#resolveValue(array, index, fallbackValue, config.fallback);
                    return this.#mapValue(raw, config);
                });

                const previous = this.uniformState[uniformName];
                const mix = this.#resolveSmoothing(config);
                if (Array.isArray(previous) && previous.length === values.length) {
                    this.uniformState[uniformName] = values.map((value, idx) => lerp(previous[idx], value, mix));
                } else {
                    this.uniformState[uniformName] = values.slice();
                }
            } else {
                const index = Number.isInteger(config.index) ? config.index : 0;
                const raw = this.#resolveValue(array, index, config.fallback, 0.5);
                const mapped = this.#mapValue(raw, config);
                const previous = this.uniformState[uniformName];
                const mix = this.#resolveSmoothing(config);
                if (typeof previous === 'number' && Number.isFinite(previous)) {
                    this.uniformState[uniformName] = lerp(previous, mapped, mix);
                } else {
                    this.uniformState[uniformName] = mapped;
                }
            }
        });
    }

    getUniformSnapshot() {
        const snapshot = {};
        const stateKeys = Object.keys(this.uniformState);
        for (const key of stateKeys) {
            const value = this.uniformState[key];
            if (Array.isArray(value)) {
                snapshot[key] = value.slice();
            } else if (value instanceof Float32Array) {
                snapshot[key] = Array.from(value);
            } else {
                snapshot[key] = value;
            }
        }
        return snapshot;
    }

    #resolveValue(array, index, fallbackValue, defaultFallback) {
        if (index < array.length && Number.isFinite(array[index])) {
            return array[index];
        }
        if (Number.isFinite(fallbackValue)) {
            return fallbackValue;
        }
        return Number.isFinite(defaultFallback) ? defaultFallback : 0;
    }

    #resolveFallbackValue(fallbackArray, idx, fallbackValue) {
        if (Array.isArray(fallbackArray) || ArrayBuffer.isView(fallbackArray)) {
            const candidate = fallbackArray[idx];
            if (Number.isFinite(candidate)) {
                return candidate;
            }
        }
        return fallbackValue;
    }

    #resolveSmoothing(config) {
        const smoothing = typeof config.smoothing === 'number' ? config.smoothing : this.globalSmoothing;
        return clampValue(smoothing, 0, 1);
    }

    #combineIndexedValues(array, config) {
        const indices = Array.isArray(config.indices) ? config.indices : ArrayBuffer.isView(config.indices) ? Array.from(config.indices) : [];
        const values = indices.map((index, idx) => {
            const fallbackValue = this.#resolveFallbackValue(config.fallbackArray, idx, config.fallback);
            return this.#resolveValue(array, index, fallbackValue, config.fallback);
        });

        if (!values.length) {
            if (Number.isFinite(config.fallback)) {
                return config.fallback;
            }
            return 0;
        }

        const { combine } = config;
        if (typeof combine === 'function') {
            try {
                const result = combine(values.slice(), config);
                return Number.isFinite(result) ? result : values[0];
            } catch (error) {
                return values[0];
            }
        }

        if (typeof combine === 'string') {
            switch (combine) {
                case 'average':
                    return values.reduce((sum, value) => sum + value, 0) / values.length;
                case 'sum':
                    return values.reduce((sum, value) => sum + value, 0);
                case 'max':
                    return values.reduce((max, value) => Math.max(max, value), values[0]);
                case 'min':
                    return values.reduce((min, value) => Math.min(min, value), values[0]);
                case 'median': {
                    const sorted = values.slice().sort((a, b) => a - b);
                    const mid = Math.floor(sorted.length / 2);
                    if (sorted.length % 2 === 0) {
                        return (sorted[mid - 1] + sorted[mid]) * 0.5;
                    }
                    return sorted[mid];
                }
                case 'magnitude':
                    return Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));
                case 'rms': {
                    const total = values.reduce((sum, value) => sum + value * value, 0);
                    return Math.sqrt(total / values.length);
                }
                case 'weighted': {
                    const weights = Array.isArray(config.weights) ? config.weights : [];
                    let numerator = 0;
                    let denominator = 0;
                    for (let i = 0; i < values.length; i += 1) {
                        const weight = Number.isFinite(weights[i]) ? weights[i] : 0;
                        numerator += values[i] * weight;
                        denominator += weight;
                    }
                    return denominator !== 0 ? numerator / denominator : values[0];
                }
                default:
                    break;
            }
        }

        return values[0];
    }

    #mapValue(value, config) {
        const inputRange = Array.isArray(config.inputRange) && config.inputRange.length === 2 ? config.inputRange : [0, 1];
        const outputRange = Array.isArray(config.outputRange) && config.outputRange.length === 2 ? config.outputRange : [0, 1];
        const clamped = clampValue(value, inputRange[0], inputRange[1]);
        const denom = inputRange[1] - inputRange[0];
        const normalized = denom !== 0 ? (clamped - inputRange[0]) / denom : 0;
        let result = outputRange[0] + normalized * (outputRange[1] - outputRange[0]);

        if (config.curve === 'easeInOut') {
            result = result * result * (3.0 - 2.0 * result);
        } else if (config.curve === 'easeIn') {
            result = result * result;
        } else if (config.curve === 'easeOut') {
            result = 1.0 - (1.0 - result) * (1.0 - result);
        }

        if (typeof config.transform === 'function') {
            result = config.transform(result, value);
        }
        return result;
    }
}
