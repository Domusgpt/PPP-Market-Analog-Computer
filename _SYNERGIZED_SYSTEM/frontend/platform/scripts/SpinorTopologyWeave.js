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

const safeNow = () => {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
        return performance.now();
    }
    return Date.now();
};

const cloneArray = (array) => (Array.isArray(array) ? array.slice() : []);

const dotProduct = (a, b) => {
    if (!Array.isArray(a) || !Array.isArray(b) || !a.length || !b.length) {
        return 0;
    }
    const size = Math.min(a.length, b.length);
    let sum = 0;
    for (let index = 0; index < size; index += 1) {
        const av = Number.isFinite(a[index]) ? a[index] : 0;
        const bv = Number.isFinite(b[index]) ? b[index] : 0;
        sum += av * bv;
    }
    return sum;
};

const magnitude = (vector) => {
    if (!Array.isArray(vector) || !vector.length) {
        return 0;
    }
    let sum = 0;
    vector.forEach((value) => {
        if (Number.isFinite(value)) {
            sum += value * value;
        }
    });
    return Math.sqrt(sum);
};

const normalizeVector = (vector) => {
    if (!Array.isArray(vector) || !vector.length) {
        return [];
    }
    const mag = magnitude(vector);
    if (mag <= 0) {
        return vector.map(() => 0);
    }
    return vector.map((value) => (Number.isFinite(value) ? value / mag : 0));
};

const computeMean = (values) => {
    if (!Array.isArray(values) || !values.length) {
        return 0;
    }
    const sum = values.reduce((total, value) => total + (Number.isFinite(value) ? value : 0), 0);
    return sum / values.length;
};

const computeVariance = (values, mean) => {
    if (!Array.isArray(values) || !values.length) {
        return 0;
    }
    const resolvedMean = Number.isFinite(mean) ? mean : computeMean(values);
    const variance = values.reduce((total, value) => {
        if (!Number.isFinite(value)) {
            return total;
        }
        const diff = value - resolvedMean;
        return total + diff * diff;
    }, 0) / values.length;
    return variance;
};

const correlation = (a, b) => {
    if (!Array.isArray(a) || !Array.isArray(b) || !a.length || !b.length) {
        return 0;
    }
    const size = Math.min(a.length, b.length);
    if (!size) {
        return 0;
    }
    const meanA = computeMean(a.slice(0, size));
    const meanB = computeMean(b.slice(0, size));
    let numerator = 0;
    let denomA = 0;
    let denomB = 0;
    for (let index = 0; index < size; index += 1) {
        const av = Number.isFinite(a[index]) ? a[index] : 0;
        const bv = Number.isFinite(b[index]) ? b[index] : 0;
        const diffA = av - meanA;
        const diffB = bv - meanB;
        numerator += diffA * diffB;
        denomA += diffA * diffA;
        denomB += diffB * diffB;
    }
    if (denomA <= 0 || denomB <= 0) {
        return 0;
    }
    return clamp(numerator / Math.sqrt(denomA * denomB), -1, 1);
};

const computeFlux = (source, target) => {
    if (!Array.isArray(source) || !Array.isArray(target) || !source.length || !target.length) {
        return 0;
    }
    const size = Math.min(source.length, target.length);
    let sum = 0;
    for (let index = 0; index < size; index += 1) {
        const s = Number.isFinite(source[index]) ? source[index] : 0;
        const t = Number.isFinite(target[index]) ? target[index] : 0;
        sum += s * t;
    }
    return sum / size;
};

const buildVoiceBindings = (voices = [], resonanceMap = new Map()) => {
    return voices.map((voice, order) => {
        const index = Number.isFinite(voice?.index) ? voice.index : order;
        const resonanceVoice = resonanceMap.get(index) || null;
        const resonanceVector = Array.isArray(resonanceVoice?.rotated)
            ? resonanceVoice.rotated
            : Array.isArray(resonanceVoice?.source)
                ? resonanceVoice.source
                : [];
        const spinor = voice?.spinor || {};
        const signal = voice?.signal || {};
        const transduction = voice?.transduction || {};
        return {
            index,
            gate: Number.isFinite(voice?.gate) ? clamp(voice.gate, 0, 1) : 0,
            ratio: Number.isFinite(spinor?.ratio) ? spinor.ratio : 0,
            coherence: Number.isFinite(spinor?.coherence) ? spinor.coherence : 0,
            braid: Number.isFinite(spinor?.braid) ? spinor.braid : 0,
            bitDensity: Number.isFinite(signal?.bitDensity) ? clamp(signal.bitDensity, 0, 1) : 0,
            bitEntropy: Number.isFinite(signal?.bitEntropy) ? clamp(signal.bitEntropy, 0, 1) : 0,
            gridEnergy: Number.isFinite(transduction?.gridEnergy) ? transduction.gridEnergy : 0,
            carrierEnergy: Number.isFinite(voice?.carrierEnergy) ? voice.carrierEnergy : 0,
            resonanceVector
        };
    });
};

const buildAxisTopology = ({
    axis,
    axisIndex,
    normalizedBridge,
    hopfFiber,
    voiceBindings
}) => {
    const orientation = Array.isArray(axis)
        ? axis.map((value) => (Number.isFinite(value) ? value : 0))
        : [];
    const normalized = normalizeVector(orientation);
    const bridgeCoupling = normalized.length && normalizedBridge.length
        ? clamp(dotProduct(normalized, normalizedBridge), -1, 1)
        : 0;
    const hopfCoupling = normalized.length && hopfFiber.length
        ? clamp(dotProduct(normalized, hopfFiber), -1, 1)
        : 0;

    const resonanceSamples = voiceBindings.map((binding) => {
        const vector = binding.resonanceVector || [];
        return Number.isFinite(vector[axisIndex]) ? vector[axisIndex] : 0;
    });
    if (!resonanceSamples.length) {
        return {
            index: axisIndex,
            orientation: normalized,
            magnitude: 0,
            bridgeCoupling,
            hopfCoupling,
            gateFlux: 0,
            ratioFlux: 0,
            coherenceFlux: 0,
            braidFlux: 0,
            bitFlux: 0,
            entropyFlux: 0,
            gridFlux: 0,
            carrierFlux: 0,
            gateCorrelation: 0,
            ratioCorrelation: 0,
            coherenceCorrelation: 0,
            braidCorrelation: 0,
            bitCorrelation: 0,
            entropyCorrelation: 0,
            gridCorrelation: 0,
            carrierCorrelation: 0
        };
    }

    const gates = voiceBindings.map((binding) => binding.gate);
    const ratios = voiceBindings.map((binding) => binding.ratio);
    const coherences = voiceBindings.map((binding) => binding.coherence);
    const braids = voiceBindings.map((binding) => binding.braid);
    const bitDensity = voiceBindings.map((binding) => binding.bitDensity);
    const bitEntropy = voiceBindings.map((binding) => binding.bitEntropy);
    const gridEnergy = voiceBindings.map((binding) => binding.gridEnergy);
    const carrierEnergy = voiceBindings.map((binding) => binding.carrierEnergy);

    const magnitudeMean = computeMean(resonanceSamples.map((value) => Math.abs(value)));

    return {
        index: axisIndex,
        orientation: normalized,
        magnitude: magnitudeMean,
        bridgeCoupling,
        hopfCoupling,
        gateFlux: computeFlux(resonanceSamples, gates),
        ratioFlux: computeFlux(resonanceSamples, ratios),
        coherenceFlux: computeFlux(resonanceSamples, coherences),
        braidFlux: computeFlux(resonanceSamples, braids),
        bitFlux: computeFlux(resonanceSamples, bitDensity),
        entropyFlux: computeFlux(resonanceSamples, bitEntropy),
        gridFlux: computeFlux(resonanceSamples, gridEnergy),
        carrierFlux: computeFlux(resonanceSamples, carrierEnergy),
        gateCorrelation: correlation(resonanceSamples, gates),
        ratioCorrelation: correlation(resonanceSamples, ratios),
        coherenceCorrelation: correlation(resonanceSamples, coherences),
        braidCorrelation: correlation(resonanceSamples, braids),
        bitCorrelation: correlation(resonanceSamples, bitDensity),
        entropyCorrelation: correlation(resonanceSamples, bitEntropy),
        gridCorrelation: correlation(resonanceSamples, gridEnergy),
        carrierCorrelation: correlation(resonanceSamples, carrierEnergy)
    };
};

const cloneAxisTopology = (axis) => ({
    index: axis.index,
    orientation: cloneArray(axis.orientation),
    magnitude: Number.isFinite(axis.magnitude) ? axis.magnitude : 0,
    bridgeCoupling: Number.isFinite(axis.bridgeCoupling) ? axis.bridgeCoupling : 0,
    hopfCoupling: Number.isFinite(axis.hopfCoupling) ? axis.hopfCoupling : 0,
    gateFlux: Number.isFinite(axis.gateFlux) ? axis.gateFlux : 0,
    ratioFlux: Number.isFinite(axis.ratioFlux) ? axis.ratioFlux : 0,
    coherenceFlux: Number.isFinite(axis.coherenceFlux) ? axis.coherenceFlux : 0,
    braidFlux: Number.isFinite(axis.braidFlux) ? axis.braidFlux : 0,
    bitFlux: Number.isFinite(axis.bitFlux) ? axis.bitFlux : 0,
    entropyFlux: Number.isFinite(axis.entropyFlux) ? axis.entropyFlux : 0,
    gridFlux: Number.isFinite(axis.gridFlux) ? axis.gridFlux : 0,
    carrierFlux: Number.isFinite(axis.carrierFlux) ? axis.carrierFlux : 0,
    gateCorrelation: Number.isFinite(axis.gateCorrelation) ? axis.gateCorrelation : 0,
    ratioCorrelation: Number.isFinite(axis.ratioCorrelation) ? axis.ratioCorrelation : 0,
    coherenceCorrelation: Number.isFinite(axis.coherenceCorrelation) ? axis.coherenceCorrelation : 0,
    braidCorrelation: Number.isFinite(axis.braidCorrelation) ? axis.braidCorrelation : 0,
    bitCorrelation: Number.isFinite(axis.bitCorrelation) ? axis.bitCorrelation : 0,
    entropyCorrelation: Number.isFinite(axis.entropyCorrelation) ? axis.entropyCorrelation : 0,
    gridCorrelation: Number.isFinite(axis.gridCorrelation) ? axis.gridCorrelation : 0,
    carrierCorrelation: Number.isFinite(axis.carrierCorrelation) ? axis.carrierCorrelation : 0
});

export const buildSpinorTopologyWeave = (payload = {}) => {
    const {
        quaternion = null,
        resonance = null,
        signal = null,
        manifold = null,
        transport = {},
        timelineProgress = 0
    } = payload;

    const timestamp = safeNow();
    const progress = clamp(Number.isFinite(timelineProgress) ? timelineProgress : 0, 0, 1);

    const normalizedBridge = normalizeVector(quaternion?.normalizedBridge || quaternion?.bridgeVector || []);
    const hopfFiber = normalizeVector(quaternion?.hopfFiber || []);

    const resonanceMatrix = Array.isArray(resonance?.matrix)
        ? resonance.matrix.map((row) => (Array.isArray(row) ? row.map((value) => (Number.isFinite(value) ? value : 0)) : []))
        : [];
    const resonanceAxes = Array.isArray(resonance?.axes) && resonance.axes.length
        ? resonance.axes
        : resonanceMatrix;
    const resonanceVoices = Array.isArray(resonance?.voices) ? resonance.voices : [];

    if (!resonanceAxes.length) {
        return {
            timestamp,
            progress,
            transport: {
                playing: Boolean(transport?.playing),
                mode: transport?.mode || 'idle'
            },
            matrix: [],
            bridge: {
                normalized: normalizedBridge,
                hopf: hopfFiber,
                magnitude: Number.isFinite(quaternion?.bridgeMagnitude) ? quaternion.bridgeMagnitude : 0
            },
            spectrum: {
                ratioMean: 0,
                ratioVariance: 0,
                ratioEntropy: Number.isFinite(manifold?.spinor?.ratioEntropy) ? manifold.spinor.ratioEntropy : 0,
                gateMean: Number.isFinite(manifold?.resonance?.gateMean) ? manifold.resonance.gateMean : 0,
                gateVariance: Number.isFinite(manifold?.resonance?.gateVariance) ? manifold.resonance.gateVariance : 0,
                carrierMagnitude: Number.isFinite(manifold?.resonance?.carrierMagnitude)
                    ? manifold.resonance.carrierMagnitude
                    : 0
            },
            axes: [],
            braiding: {
                bridgeToGateFlux: 0,
                bridgeToRatioFlux: 0,
                hopfToCarrierFlux: 0,
                gateToGridFlux: 0,
                ratioToBitFlux: 0
            }
        };
    }

    const resonanceMap = new Map();
    resonanceVoices.forEach((voice, index) => {
        const key = Number.isFinite(voice?.index) ? voice.index : index;
        resonanceMap.set(key, voice);
    });

    const manifoldVoices = Array.isArray(manifold?.voices) ? manifold.voices : [];
    const voiceBindings = buildVoiceBindings(manifoldVoices, resonanceMap);

    const axes = resonanceAxes.map((axis, axisIndex) => buildAxisTopology({
        axis,
        axisIndex,
        normalizedBridge,
        hopfFiber,
        voiceBindings
    }));

    const bridgeMagnitude = Number.isFinite(quaternion?.bridgeMagnitude) ? quaternion.bridgeMagnitude : 0;
    const spinorRatios = Array.isArray(manifold?.spinor?.ratios) ? manifold.spinor.ratios : [];
    const ratioMean = computeMean(spinorRatios);
    const ratioVariance = computeVariance(spinorRatios, ratioMean);

    const axisBridgeCouplings = axes.map((axis) => axis.bridgeCoupling);
    const axisGateFlux = axes.map((axis) => axis.gateFlux);
    const axisRatioFlux = axes.map((axis) => axis.ratioFlux);
    const axisHopfCoupling = axes.map((axis) => axis.hopfCoupling);
    const axisCarrierFlux = axes.map((axis) => axis.carrierFlux);
    const axisGridFlux = axes.map((axis) => axis.gridFlux);
    const axisBitFlux = axes.map((axis) => axis.bitFlux);

    const braiding = {
        bridgeToGateFlux: correlation(axisBridgeCouplings, axisGateFlux),
        bridgeToRatioFlux: correlation(axisBridgeCouplings, axisRatioFlux),
        hopfToCarrierFlux: correlation(axisHopfCoupling, axisCarrierFlux),
        gateToGridFlux: correlation(axisGateFlux, axisGridFlux),
        ratioToBitFlux: correlation(axisRatioFlux, axisBitFlux)
    };

    return {
        timestamp,
        progress,
        transport: {
            playing: Boolean(transport?.playing),
            mode: transport?.mode || 'idle'
        },
        matrix: resonanceMatrix,
        bridge: {
            normalized: normalizedBridge,
            hopf: hopfFiber,
            magnitude: bridgeMagnitude
        },
        spectrum: {
            ratioMean,
            ratioVariance,
            ratioEntropy: Number.isFinite(manifold?.spinor?.ratioEntropy) ? manifold.spinor.ratioEntropy : 0,
            gateMean: Number.isFinite(manifold?.resonance?.gateMean) ? manifold.resonance.gateMean : 0,
            gateVariance: Number.isFinite(manifold?.resonance?.gateVariance) ? manifold.resonance.gateVariance : 0,
            carrierMagnitude: Number.isFinite(manifold?.resonance?.carrierMagnitude)
                ? manifold.resonance.carrierMagnitude
                : 0
        },
        axes,
        braiding
    };
};

export const cloneSpinorTopologyWeave = (weave) => {
    if (!weave || typeof weave !== 'object') {
        return null;
    }
    return {
        timestamp: Number(weave.timestamp) || 0,
        progress: Number.isFinite(weave.progress) ? weave.progress : 0,
        transport: weave.transport
            ? {
                playing: Boolean(weave.transport.playing),
                mode: typeof weave.transport.mode === 'string' ? weave.transport.mode : 'idle'
            }
            : { playing: false, mode: 'idle' },
        matrix: Array.isArray(weave.matrix)
            ? weave.matrix.map((row) => (Array.isArray(row) ? row.map((value) => (Number.isFinite(value) ? value : 0)) : []))
            : [],
        bridge: weave.bridge
            ? {
                normalized: cloneArray(weave.bridge.normalized),
                hopf: cloneArray(weave.bridge.hopf),
                magnitude: Number.isFinite(weave.bridge.magnitude) ? weave.bridge.magnitude : 0
            }
            : { normalized: [], hopf: [], magnitude: 0 },
        spectrum: weave.spectrum
            ? {
                ratioMean: Number.isFinite(weave.spectrum.ratioMean) ? weave.spectrum.ratioMean : 0,
                ratioVariance: Number.isFinite(weave.spectrum.ratioVariance) ? weave.spectrum.ratioVariance : 0,
                ratioEntropy: Number.isFinite(weave.spectrum.ratioEntropy) ? weave.spectrum.ratioEntropy : 0,
                gateMean: Number.isFinite(weave.spectrum.gateMean) ? weave.spectrum.gateMean : 0,
                gateVariance: Number.isFinite(weave.spectrum.gateVariance) ? weave.spectrum.gateVariance : 0,
                carrierMagnitude: Number.isFinite(weave.spectrum.carrierMagnitude)
                    ? weave.spectrum.carrierMagnitude
                    : 0
            }
            : {
                ratioMean: 0,
                ratioVariance: 0,
                ratioEntropy: 0,
                gateMean: 0,
                gateVariance: 0,
                carrierMagnitude: 0
            },
        axes: Array.isArray(weave.axes)
            ? weave.axes.map((axis) => cloneAxisTopology(axis))
            : [],
        braiding: weave.braiding
            ? {
                bridgeToGateFlux: Number.isFinite(weave.braiding.bridgeToGateFlux) ? weave.braiding.bridgeToGateFlux : 0,
                bridgeToRatioFlux: Number.isFinite(weave.braiding.bridgeToRatioFlux) ? weave.braiding.bridgeToRatioFlux : 0,
                hopfToCarrierFlux: Number.isFinite(weave.braiding.hopfToCarrierFlux) ? weave.braiding.hopfToCarrierFlux : 0,
                gateToGridFlux: Number.isFinite(weave.braiding.gateToGridFlux) ? weave.braiding.gateToGridFlux : 0,
                ratioToBitFlux: Number.isFinite(weave.braiding.ratioToBitFlux) ? weave.braiding.ratioToBitFlux : 0
            }
            : {
                bridgeToGateFlux: 0,
                bridgeToRatioFlux: 0,
                hopfToCarrierFlux: 0,
                gateToGridFlux: 0,
                ratioToBitFlux: 0
            }
    };
};
