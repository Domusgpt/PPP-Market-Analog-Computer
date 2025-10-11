import { defaultMapping } from './defaultMapping.js';
import { cloneMappingDefinition } from './utils.js';

const TWO_PI = 6.28318;

const createPreset = ({ id, label, description, mapping, source = 'built-in', sourceLabel }) => ({
    id,
    label,
    description,
    mapping,
    source,
    sourceLabel
});

const createRotationSynthesisMapping = () => {
    const mapping = cloneMappingDefinition(defaultMapping);
    mapping.u_morphFactor = {
        indices: [0, 16],
        combine: 'average',
        inputRange: [0, 1],
        outputRange: [0.15, 1.6],
        fallbackArray: [0.5, 0.55],
        smoothing: 0.2
    };
    mapping.u_patternIntensity = {
        indices: [2, 18],
        combine: 'sum',
        inputRange: [0, 2],
        outputRange: [1.0, 6.8],
        fallbackArray: [0.6, 0.45]
    };
    mapping.u_gridDensity = {
        indices: [3, 17, 23],
        combine: 'sum',
        inputRange: [0, 3],
        outputRange: [2.0, 14.0],
        fallbackArray: [0.45, 0.35, 0.3],
        curve: 'easeIn'
    };
    mapping.u_lineThickness = {
        indices: [5, 21],
        combine: 'min',
        inputRange: [0, 1],
        outputRange: [0.01, 0.16],
        fallbackArray: [0.5, 0.4],
        curve: 'easeInOut'
    };
    mapping.u_shellWidth = {
        indices: [6, 22],
        combine: 'average',
        inputRange: [0, 1],
        outputRange: [0.05, 0.5],
        fallbackArray: [0.5, 0.55]
    };
    mapping.u_tetraThickness = {
        indices: [7, 26],
        combine: 'average',
        inputRange: [0, 1],
        outputRange: [0.02, 0.36],
        fallbackArray: [0.5, 0.45]
    };
    mapping.u_glitchIntensity = {
        indices: [8, 19],
        combine: 'max',
        inputRange: [0, 1],
        outputRange: [0.0, 0.5],
        fallbackArray: [0.2, 0.3]
    };
    mapping.u_colorShift = {
        indices: [1, 18, 24],
        combine: 'median',
        inputRange: [0, 1],
        outputRange: [-0.4, 0.6],
        fallbackArray: [0.3, 0.45, 0.55]
    };
    mapping.u_universeModifier = {
        indices: [4, 20, 31],
        combine: 'weighted',
        weights: [0.6, 0.3, 0.1],
        inputRange: [0, 1],
        outputRange: [0.3, 1.8],
        fallbackArray: [0.5, 0.6, 0.7]
    };
    mapping.u_dimension = {
        indices: [15, 30],
        combine: 'weighted',
        weights: [0.7, 0.3],
        inputRange: [0, 1],
        outputRange: [3.0, 5.4],
        fallbackArray: [0.75, 0.65]
    };
    mapping.u_rotXY = {
        indices: [9, 21, 27],
        combine: 'rms',
        inputRange: [0, 1],
        outputRange: [0.0, TWO_PI],
        fallbackArray: [0.2, 0.45, 0.7]
    };
    mapping.u_rotXZ = {
        indices: [10, 22, 28],
        combine: 'rms',
        inputRange: [0, 1],
        outputRange: [0.0, TWO_PI],
        fallbackArray: [0.4, 0.55, 0.75]
    };
    mapping.u_rotXW = {
        indices: [11, 23, 29],
        combine: 'rms',
        inputRange: [0, 1],
        outputRange: [0.0, TWO_PI],
        fallbackArray: [0.6, 0.7, 0.8]
    };
    mapping.u_rotYZ = {
        indices: [12, 24, 30],
        combine: 'rms',
        inputRange: [0, 1],
        outputRange: [0.0, TWO_PI],
        fallbackArray: [0.3, 0.55, 0.7]
    };
    mapping.u_rotYW = {
        indices: [13, 25, 31],
        combine: 'rms',
        inputRange: [0, 1],
        outputRange: [0.0, TWO_PI],
        fallbackArray: [0.5, 0.65, 0.8]
    };
    mapping.u_rotZW = {
        indices: [14, 20, 27],
        combine: 'rms',
        inputRange: [0, 1],
        outputRange: [0.0, TWO_PI],
        fallbackArray: [0.7, 0.6, 0.5]
    };
    return mapping;
};

const createStructureReactiveMapping = () => {
    const mapping = cloneMappingDefinition(defaultMapping);
    mapping.u_morphFactor = {
        indices: [0, 7, 15],
        combine: 'weighted',
        weights: [0.55, 0.25, 0.2],
        inputRange: [0, 1],
        outputRange: [0.1, 1.3],
        fallbackArray: [0.5, 0.45, 0.7],
        smoothing: 0.18
    };
    mapping.u_patternIntensity = {
        indices: [2, 18, 19, 20],
        combine: 'sum',
        inputRange: [0, 4],
        outputRange: [1.2, 9.5],
        fallbackArray: [0.6, 0.4, 0.35, 0.3],
        curve: 'easeOut'
    };
    mapping.u_gridDensity = {
        indices: [3, 16, 17],
        combine: 'average',
        inputRange: [0, 1],
        outputRange: [3.0, 14.0],
        fallbackArray: [0.5, 0.45, 0.35]
    };
    mapping.u_lineThickness = {
        indices: [5, 21],
        combine: 'min',
        inputRange: [0, 1],
        outputRange: [0.01, 0.12],
        fallbackArray: [0.5, 0.42],
        curve: 'easeInOut'
    };
    mapping.u_shellWidth = {
        indices: [6, 22],
        combine: 'max',
        inputRange: [0, 1],
        outputRange: [0.04, 0.5],
        fallbackArray: [0.5, 0.55]
    };
    mapping.u_tetraThickness = {
        indices: [7, 23],
        combine: 'average',
        inputRange: [0, 1],
        outputRange: [0.02, 0.38],
        fallbackArray: [0.5, 0.45]
    };
    mapping.u_glitchIntensity = {
        indices: [8, 24, 25],
        combine: 'average',
        inputRange: [0, 1],
        outputRange: [0.0, 0.5],
        fallbackArray: [0.2, 0.22, 0.28]
    };
    mapping.u_colorShift = {
        indices: [1, 26],
        combine: 'average',
        inputRange: [0, 1],
        outputRange: [-0.45, 0.55],
        fallbackArray: [0.35, 0.45]
    };
    mapping.u_universeModifier = {
        indices: [4, 27],
        combine: 'weighted',
        weights: [0.7, 0.3],
        inputRange: [0, 1],
        outputRange: [0.35, 1.7],
        fallbackArray: [0.5, 0.55]
    };
    mapping.u_dimension = {
        indices: [15, 28, 29],
        combine: 'median',
        inputRange: [0, 1],
        outputRange: [3.0, 5.3],
        fallbackArray: [0.8, 0.6, 0.7]
    };
    return mapping;
};

export const builtInMappingPresets = [
    createPreset({
        id: 'default',
        label: 'Default – Balanced 4D Sweep',
        description: 'One-to-one channel bindings tuned for stable hypercube motion and smooth morphing.',
        mapping: cloneMappingDefinition(defaultMapping),
        source: 'built-in',
        sourceLabel: 'Built-in preset'
    }),
    createPreset({
        id: 'rotation-synthesis',
        label: 'Rotation Synthesis (RMS rotations)',
        description: 'Blends multiple channel groups per rotation plane using RMS aggregation for complex angular motion.',
        mapping: createRotationSynthesisMapping(),
        source: 'built-in',
        sourceLabel: 'Built-in preset'
    }),
    createPreset({
        id: 'structure-reactive',
        label: 'Structure Reactive Grid',
        description: 'Aggregates sensor groups into structural parameters, amplifying lattice and shell modulation.',
        mapping: createStructureReactiveMapping(),
        source: 'built-in',
        sourceLabel: 'Built-in preset'
    })
];
