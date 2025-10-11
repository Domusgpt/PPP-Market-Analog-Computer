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

const vectorMagnitude = (vector) => {
    if (!Array.isArray(vector) || !vector.length) {
        return 0;
    }
    const sum = vector.reduce((total, value) => total + (Number.isFinite(value) ? value * value : 0), 0);
    return Math.sqrt(sum);
};

const normalizeVector = (vector) => {
    if (!Array.isArray(vector) || !vector.length) {
        return [];
    }
    const magnitude = vectorMagnitude(vector);
    if (magnitude <= 0) {
        return vector.map(() => 0);
    }
    return vector.map((value) => (Number.isFinite(value) ? value / magnitude : 0));
};

const computeAxisIntensity = (axis) => {
    const fluxes = [
        axis.gateFlux,
        axis.ratioFlux,
        axis.coherenceFlux,
        axis.braidFlux,
        axis.bitFlux,
        axis.entropyFlux,
        axis.gridFlux,
        axis.carrierFlux
    ];
    const sum = fluxes.reduce((total, value) => {
        if (!Number.isFinite(value)) {
            return total;
        }
        return total + value * value;
    }, 0);
    return Math.sqrt(sum);
};

const buildAxisContinuum = (axis = {}) => {
    const orientation = normalizeVector(Array.isArray(axis.orientation) ? axis.orientation : []);
    const entry = {
        index: Number.isFinite(axis.index) ? axis.index : 0,
        orientation,
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
        carrierCorrelation: Number.isFinite(axis.carrierCorrelation) ? axis.carrierCorrelation : 0,
        bridgeCoupling: Number.isFinite(axis.bridgeCoupling) ? axis.bridgeCoupling : 0,
        hopfCoupling: Number.isFinite(axis.hopfCoupling) ? axis.hopfCoupling : 0
    };
    entry.intensity = computeAxisIntensity(entry);
    return entry;
};

const buildOrientationContinuum = (axes) => {
    const dimension = axes.reduce((max, axis) => Math.max(max, axis.orientation.length), 0);
    if (!dimension) {
        return [];
    }
    const vector = new Array(dimension).fill(0);
    axes.forEach((axis) => {
        const intensity = Number.isFinite(axis.intensity) ? axis.intensity : 0;
        const bridgeCoupling = Number.isFinite(axis.bridgeCoupling) ? axis.bridgeCoupling : 0;
        axis.orientation.forEach((value, index) => {
            if (!Number.isFinite(value) || index >= vector.length) {
                return;
            }
            vector[index] += value * intensity * (0.5 + Math.abs(bridgeCoupling));
        });
    });
    return vector;
};

const buildVoiceContinuum = (voices, resonanceVoices) => {
    if (!Array.isArray(voices) || !voices.length || !Array.isArray(resonanceVoices)) {
        return { orientation: [], magnitude: 0 };
    }
    const dimension = resonanceVoices.reduce((max, voice) => {
        const rotated = Array.isArray(voice?.rotated) ? voice.rotated : Array.isArray(voice?.source) ? voice.source : [];
        return Math.max(max, rotated.length);
    }, 0);
    if (!dimension) {
        return { orientation: [], magnitude: 0 };
    }
    const vector = new Array(dimension).fill(0);
    voices.forEach((voice, index) => {
        const resonanceVoice = resonanceVoices[index] || {};
        const rotated = normalizeVector(Array.isArray(resonanceVoice.rotated)
            ? resonanceVoice.rotated
            : Array.isArray(resonanceVoice.source)
                ? resonanceVoice.source
                : []);
        if (!rotated.length) {
            return;
        }
        const gate = Number.isFinite(voice?.gate) ? voice.gate : 0;
        const ratio = Number.isFinite(voice?.spinor?.ratio) ? voice.spinor.ratio : 0;
        const coherence = Number.isFinite(voice?.spinor?.coherence) ? voice.spinor.coherence : 0;
        const braid = Number.isFinite(voice?.spinor?.braid) ? voice.spinor.braid : 0;
        const carrierEnergy = Number.isFinite(voice?.carrierEnergy) ? voice.carrierEnergy : 0;
        const gridEnergy = Number.isFinite(voice?.transduction?.gridEnergy) ? voice.transduction.gridEnergy : 0;
        const weight = Math.max(0, gate) + Math.abs(ratio) + Math.max(0, coherence) + Math.abs(braid) + carrierEnergy + gridEnergy;
        rotated.forEach((value, axisIndex) => {
            if (!Number.isFinite(value) || axisIndex >= vector.length) {
                return;
            }
            vector[axisIndex] += value * weight;
        });
    });
    const magnitude = vectorMagnitude(vector);
    if (magnitude <= 0) {
        return { orientation: vector.map(() => 0), magnitude: 0 };
    }
    return { orientation: vector.map((value) => value / magnitude), magnitude };
};

const buildBraidingSnapshot = (braiding = {}) => ({
    bridgeToGateFlux: Number.isFinite(braiding.bridgeToGateFlux) ? braiding.bridgeToGateFlux : 0,
    bridgeToRatioFlux: Number.isFinite(braiding.bridgeToRatioFlux) ? braiding.bridgeToRatioFlux : 0,
    hopfToCarrierFlux: Number.isFinite(braiding.hopfToCarrierFlux) ? braiding.hopfToCarrierFlux : 0,
    gateToGridFlux: Number.isFinite(braiding.gateToGridFlux) ? braiding.gateToGridFlux : 0,
    ratioToBitFlux: Number.isFinite(braiding.ratioToBitFlux) ? braiding.ratioToBitFlux : 0
});

export const buildSpinorFluxContinuum = (payload = {}) => {
    const {
        quaternion = null,
        resonance = null,
        signal = null,
        manifold = null,
        topology = null,
        transport = {},
        timelineProgress = 0
    } = payload;

    const timestamp = safeNow();
    const progress = clamp(Number.isFinite(timelineProgress) ? timelineProgress : 0, 0, 1);

    const topologyAxes = Array.isArray(topology?.axes) ? topology.axes : [];
    const axisContinuum = topologyAxes.map((axis) => buildAxisContinuum(axis));
    const axisIntensities = axisContinuum.map((axis) => axis.intensity);
    const fluxDensity = axisIntensities.length ? computeMean(axisIntensities) : 0;
    const fluxVariance = axisIntensities.length ? computeVariance(axisIntensities, fluxDensity) : 0;

    const normalizedBridge = normalizeVector(quaternion?.normalizedBridge || quaternion?.bridgeVector || []);
    const hopfFiber = normalizeVector(quaternion?.hopfFiber || []);

    const continuumVector = buildOrientationContinuum(axisContinuum);
    const continuumOrientation = normalizeVector(continuumVector);

    const resonanceVoices = Array.isArray(resonance?.voices) ? resonance.voices : [];
    const manifoldVoices = Array.isArray(manifold?.voices) ? manifold.voices : [];
    const voiceContinuum = buildVoiceContinuum(manifoldVoices, resonanceVoices);

    const bridgeAlignment = continuumOrientation.length && normalizedBridge.length
        ? clamp(dotProduct(continuumOrientation, normalizedBridge.slice(0, continuumOrientation.length)), -1, 1)
        : 0;
    const hopfAlignment = continuumOrientation.length && hopfFiber.length
        ? clamp(dotProduct(continuumOrientation, hopfFiber.slice(0, continuumOrientation.length)), -1, 1)
        : 0;
    const voiceAlignment = continuumOrientation.length && voiceContinuum.orientation.length
        ? clamp(dotProduct(continuumOrientation, voiceContinuum.orientation.slice(0, continuumOrientation.length)), -1, 1)
        : 0;

    const voiceCoherence = manifoldVoices.length
        ? computeMean(manifoldVoices.map((voice) => (Number.isFinite(voice?.spinor?.coherence) ? voice.spinor.coherence : 0)))
        : 0;
    const voiceGateMean = manifoldVoices.length
        ? computeMean(manifoldVoices.map((voice) => (Number.isFinite(voice?.gate) ? voice.gate : 0)))
        : 0;
    const voiceBraidMean = manifoldVoices.length
        ? computeMean(manifoldVoices.map((voice) => (Number.isFinite(voice?.spinor?.braid) ? voice.spinor.braid : 0)))
        : 0;
    const voiceBitDensity = manifoldVoices.length
        ? computeMean(manifoldVoices.map((voice) => (Number.isFinite(voice?.signal?.bitDensity) ? voice.signal.bitDensity : 0)))
        : 0;
    const voiceBitEntropy = manifoldVoices.length
        ? computeMean(manifoldVoices.map((voice) => (Number.isFinite(voice?.signal?.bitEntropy) ? voice.signal.bitEntropy : 0)))
        : 0;
    const voiceGridEnergy = manifoldVoices.length
        ? computeMean(manifoldVoices.map((voice) => (Number.isFinite(voice?.transduction?.gridEnergy) ? voice.transduction.gridEnergy : 0)))
        : 0;
    const voiceCarrierEnergy = manifoldVoices.length
        ? computeMean(manifoldVoices.map((voice) => (Number.isFinite(voice?.carrierEnergy) ? voice.carrierEnergy : 0)))
        : 0;

    const signalVoices = Array.isArray(signal?.voices) ? signal.voices : [];
    const sequence = signalVoices.length
        ? signalVoices.map((voice) => (typeof voice?.sequence === 'string' ? voice.sequence : '')).join('')
        : '';

    const voiceEntries = manifoldVoices.map((voice, index) => {
        const resonanceVoice = resonanceVoices[index] || {};
        const orientation = normalizeVector(Array.isArray(resonanceVoice.rotated)
            ? resonanceVoice.rotated
            : Array.isArray(resonanceVoice.source)
                ? resonanceVoice.source
                : []);
        const continuumCoupling = orientation.length && continuumOrientation.length
            ? clamp(dotProduct(orientation, continuumOrientation.slice(0, orientation.length)), -1, 1)
            : 0;
        return {
            index: Number.isFinite(voice?.index) ? voice.index : index,
            gate: Number.isFinite(voice?.gate) ? voice.gate : 0,
            ratio: Number.isFinite(voice?.spinor?.ratio) ? voice.spinor.ratio : 0,
            coherence: Number.isFinite(voice?.spinor?.coherence) ? voice.spinor.coherence : 0,
            braid: Number.isFinite(voice?.spinor?.braid) ? voice.spinor.braid : 0,
            bitDensity: Number.isFinite(voice?.signal?.bitDensity) ? voice.signal.bitDensity : 0,
            bitEntropy: Number.isFinite(voice?.signal?.bitEntropy) ? voice.signal.bitEntropy : 0,
            gridEnergy: Number.isFinite(voice?.transduction?.gridEnergy) ? voice.transduction.gridEnergy : 0,
            carrierEnergy: Number.isFinite(voice?.carrierEnergy) ? voice.carrierEnergy : 0,
            dominantFrequency: Number.isFinite(voice?.signal?.dominantFrequency) ? voice.signal.dominantFrequency : 0,
            continuumAlignment: continuumCoupling,
            orientation
        };
    });

    return {
        timestamp,
        progress,
        transport: {
            playing: Boolean(transport?.playing),
            mode: typeof transport?.mode === 'string' ? transport.mode : 'idle'
        },
        flux: {
            density: fluxDensity,
            variance: fluxVariance,
            bridgeAlignment,
            hopfAlignment,
            voiceAlignment,
            gateMean: voiceGateMean,
            coherenceMean: voiceCoherence,
            braidMean: voiceBraidMean,
            bitDensity: voiceBitDensity,
            bitEntropy: voiceBitEntropy,
            gridEnergy: voiceGridEnergy,
            carrierEnergy: voiceCarrierEnergy
        },
        continuum: {
            orientation: continuumOrientation,
            magnitude: vectorMagnitude(continuumVector),
            voiceOrientation: voiceContinuum.orientation,
            voiceMagnitude: voiceContinuum.magnitude,
            sequence
        },
        axes: axisContinuum,
        voices: voiceEntries,
        braiding: buildBraidingSnapshot(topology?.braiding),
        quaternion: {
            bridge: normalizedBridge,
            hopf: hopfFiber
        }
    };
};

export const cloneSpinorFluxContinuum = (continuum) => {
    if (!continuum || typeof continuum !== 'object') {
        return null;
    }
    return {
        timestamp: Number(continuum.timestamp) || 0,
        progress: Number.isFinite(continuum.progress) ? continuum.progress : 0,
        transport: continuum.transport
            ? {
                playing: Boolean(continuum.transport.playing),
                mode: typeof continuum.transport.mode === 'string' ? continuum.transport.mode : 'idle'
            }
            : { playing: false, mode: 'idle' },
        flux: continuum.flux
            ? {
                density: Number.isFinite(continuum.flux.density) ? continuum.flux.density : 0,
                variance: Number.isFinite(continuum.flux.variance) ? continuum.flux.variance : 0,
                bridgeAlignment: Number.isFinite(continuum.flux.bridgeAlignment) ? continuum.flux.bridgeAlignment : 0,
                hopfAlignment: Number.isFinite(continuum.flux.hopfAlignment) ? continuum.flux.hopfAlignment : 0,
                voiceAlignment: Number.isFinite(continuum.flux.voiceAlignment) ? continuum.flux.voiceAlignment : 0,
                gateMean: Number.isFinite(continuum.flux.gateMean) ? continuum.flux.gateMean : 0,
                coherenceMean: Number.isFinite(continuum.flux.coherenceMean) ? continuum.flux.coherenceMean : 0,
                braidMean: Number.isFinite(continuum.flux.braidMean) ? continuum.flux.braidMean : 0,
                bitDensity: Number.isFinite(continuum.flux.bitDensity) ? continuum.flux.bitDensity : 0,
                bitEntropy: Number.isFinite(continuum.flux.bitEntropy) ? continuum.flux.bitEntropy : 0,
                gridEnergy: Number.isFinite(continuum.flux.gridEnergy) ? continuum.flux.gridEnergy : 0,
                carrierEnergy: Number.isFinite(continuum.flux.carrierEnergy) ? continuum.flux.carrierEnergy : 0
            }
            : {
                density: 0,
                variance: 0,
                bridgeAlignment: 0,
                hopfAlignment: 0,
                voiceAlignment: 0,
                gateMean: 0,
                coherenceMean: 0,
                braidMean: 0,
                bitDensity: 0,
                bitEntropy: 0,
                gridEnergy: 0,
                carrierEnergy: 0
            },
        continuum: continuum.continuum
            ? {
                orientation: cloneArray(continuum.continuum.orientation),
                magnitude: Number.isFinite(continuum.continuum.magnitude) ? continuum.continuum.magnitude : 0,
                voiceOrientation: cloneArray(continuum.continuum.voiceOrientation),
                voiceMagnitude: Number.isFinite(continuum.continuum.voiceMagnitude) ? continuum.continuum.voiceMagnitude : 0,
                sequence: typeof continuum.continuum.sequence === 'string' ? continuum.continuum.sequence : ''
            }
            : {
                orientation: [],
                magnitude: 0,
                voiceOrientation: [],
                voiceMagnitude: 0,
                sequence: ''
            },
        axes: Array.isArray(continuum.axes)
            ? continuum.axes.map((axis) => ({
                index: Number.isFinite(axis.index) ? axis.index : 0,
                orientation: cloneArray(axis.orientation),
                intensity: Number.isFinite(axis.intensity) ? axis.intensity : 0,
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
                carrierCorrelation: Number.isFinite(axis.carrierCorrelation) ? axis.carrierCorrelation : 0,
                bridgeCoupling: Number.isFinite(axis.bridgeCoupling) ? axis.bridgeCoupling : 0,
                hopfCoupling: Number.isFinite(axis.hopfCoupling) ? axis.hopfCoupling : 0
            }))
            : [],
        voices: Array.isArray(continuum.voices)
            ? continuum.voices.map((voice) => ({
                index: Number.isFinite(voice.index) ? voice.index : 0,
                gate: Number.isFinite(voice.gate) ? voice.gate : 0,
                ratio: Number.isFinite(voice.ratio) ? voice.ratio : 0,
                coherence: Number.isFinite(voice.coherence) ? voice.coherence : 0,
                braid: Number.isFinite(voice.braid) ? voice.braid : 0,
                bitDensity: Number.isFinite(voice.bitDensity) ? voice.bitDensity : 0,
                bitEntropy: Number.isFinite(voice.bitEntropy) ? voice.bitEntropy : 0,
                gridEnergy: Number.isFinite(voice.gridEnergy) ? voice.gridEnergy : 0,
                carrierEnergy: Number.isFinite(voice.carrierEnergy) ? voice.carrierEnergy : 0,
                dominantFrequency: Number.isFinite(voice.dominantFrequency) ? voice.dominantFrequency : 0,
                continuumAlignment: Number.isFinite(voice.continuumAlignment) ? voice.continuumAlignment : 0,
                orientation: cloneArray(voice.orientation)
            }))
            : [],
        braiding: buildBraidingSnapshot(continuum.braiding),
        quaternion: continuum.quaternion
            ? {
                bridge: cloneArray(continuum.quaternion.bridge),
                hopf: cloneArray(continuum.quaternion.hopf)
            }
            : { bridge: [], hopf: [] }
    };
};
