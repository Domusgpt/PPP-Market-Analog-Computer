export const clampValue = (value, min, max) => {
    const lower = Math.min(min, max);
    const upper = Math.max(min, max);
    return Math.min(Math.max(value, lower), upper);
};

export const lerp = (a, b, t) => a + (b - a) * t;

export const parseDataInput = (text) => {
    if (typeof text !== 'string') {
        return [];
    }
    return text
        .split(/[\,\s]+/)
        .map((token) => parseFloat(token.trim()))
        .filter((value) => Number.isFinite(value));
};

export const formatDataArray = (array) =>
    array.map((value) => (Number.isFinite(value) ? value.toFixed(3) : '0.000')).join(', ');

export const cloneMappingDefinition = (mapping) => {
    if (!mapping || typeof mapping !== 'object') {
        return {};
    }
    return Object.fromEntries(
        Object.entries(mapping).map(([key, value]) => {
            if (!value || typeof value !== 'object') {
                return [key, value];
            }
            const clone = { ...value };
            if (Array.isArray(value.indices)) {
                clone.indices = value.indices.slice();
            } else if (ArrayBuffer.isView(value.indices)) {
                clone.indices = Array.from(value.indices);
            }
            if (Array.isArray(value.fallbackArray)) {
                clone.fallbackArray = value.fallbackArray.slice();
            } else if (ArrayBuffer.isView(value.fallbackArray)) {
                clone.fallbackArray = value.fallbackArray.slice ? value.fallbackArray.slice() : Array.from(value.fallbackArray);
            }
            if (Array.isArray(value.weights)) {
                clone.weights = value.weights.slice();
            } else if (ArrayBuffer.isView(value.weights)) {
                clone.weights = Array.from(value.weights);
            }
            return [key, clone];
        })
    );
};
