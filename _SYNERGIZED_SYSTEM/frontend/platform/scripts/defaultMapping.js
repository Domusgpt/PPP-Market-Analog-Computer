import { DATA_CHANNEL_COUNT } from './constants.js';

export const defaultMapping = {
    u_dimension: { index: 15, inputRange: [0, 1], outputRange: [3.0, 5.0], fallback: 0.75 },
    u_morphFactor: { index: 0, inputRange: [0, 1], outputRange: [0.1, 1.4], fallback: 0.5, smoothing: 0.22 },
    u_colorShift: { index: 1, inputRange: [0, 1], outputRange: [-0.35, 0.55], fallback: 0.35 },
    u_patternIntensity: { index: 2, inputRange: [0, 1], outputRange: [1.0, 6.2], fallback: 0.6 },
    u_gridDensity: { index: 3, inputRange: [0, 1], outputRange: [2.0, 12.0], fallback: 0.5 },
    u_universeModifier: { index: 4, inputRange: [0, 1], outputRange: [0.4, 1.6], fallback: 0.5 },
    u_lineThickness: { index: 5, inputRange: [0, 1], outputRange: [0.01, 0.18], fallback: 0.5, curve: 'easeInOut' },
    u_shellWidth: { index: 6, inputRange: [0, 1], outputRange: [0.05, 0.45], fallback: 0.5 },
    u_tetraThickness: { index: 7, inputRange: [0, 1], outputRange: [0.02, 0.35], fallback: 0.5 },
    u_glitchIntensity: { index: 8, inputRange: [0, 1], outputRange: [0.0, 0.45], fallback: 0.2 },
    u_rotXY: { index: 9, inputRange: [0, 1], outputRange: [0.0, 6.28318], fallback: 0.2 },
    u_rotXZ: { index: 10, inputRange: [0, 1], outputRange: [0.0, 6.28318], fallback: 0.4 },
    u_rotXW: { index: 11, inputRange: [0, 1], outputRange: [0.0, 6.28318], fallback: 0.6 },
    u_rotYZ: { index: 12, inputRange: [0, 1], outputRange: [0.0, 6.28318], fallback: 0.3 },
    u_rotYW: { index: 13, inputRange: [0, 1], outputRange: [0.0, 6.28318], fallback: 0.5 },
    u_rotZW: { index: 14, inputRange: [0, 1], outputRange: [0.0, 6.28318], fallback: 0.7 },
    u_dataChannels: {
        indices: Array.from({ length: DATA_CHANNEL_COUNT }, (_, idx) => idx),
        inputRange: [0, 1],
        outputRange: [-1, 1],
        fallbackArray: new Array(DATA_CHANNEL_COUNT).fill(0.5)
    }
};
