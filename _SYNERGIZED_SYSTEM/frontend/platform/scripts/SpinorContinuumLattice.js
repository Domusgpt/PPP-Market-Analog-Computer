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

const padVector = (vector, size) => {
    const result = new Array(size).fill(0);
    if (!Array.isArray(vector)) {
        return result;
    }
    const limit = Math.min(size, vector.length);
    for (let index = 0; index < limit; index += 1) {
        const value = vector[index];
        result[index] = Number.isFinite(value) ? value : 0;
    }
    return result;
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

const computeWeightedMean = (entries, selector, weightSelector) => {
    if (!Array.isArray(entries) || !entries.length) {
        return 0;
    }
    let weightedSum = 0;
    let totalWeight = 0;
    entries.forEach((entry) => {
        const weight = Number.isFinite(weightSelector(entry)) ? weightSelector(entry) : 0;
        if (weight <= 0) {
            return;
        }
        const value = Number.isFinite(selector(entry)) ? selector(entry) : 0;
        weightedSum += value * weight;
        totalWeight += weight;
    });
    if (totalWeight <= 0) {
        return 0;
    }
    return weightedSum / totalWeight;
};

const computeSequenceDensity = (voices) => {
    if (!Array.isArray(voices) || !voices.length) {
        return 0;
    }
    let active = 0;
    let total = 0;
    voices.forEach((voice) => {
        const bits = Array.isArray(voice.sequenceBits) ? voice.sequenceBits : [];
        bits.forEach((bit) => {
            total += 1;
            active += bit ? 1 : 0;
        });
    });
    if (!total) {
        return 0;
    }
    return active / total;
};

const buildAxisEntry = (axis, continuumOrientation, voiceOrientation, bridge, hopf) => {
    const orientation = normalizeVector(Array.isArray(axis?.orientation) ? axis.orientation : []);
    const continuumProjection = orientation.length && continuumOrientation.length
        ? clamp(dotProduct(orientation, continuumOrientation.slice(0, orientation.length)), -1, 1)
        : 0;
    const voiceProjection = orientation.length && voiceOrientation.length
        ? clamp(dotProduct(orientation, voiceOrientation.slice(0, orientation.length)), -1, 1)
        : 0;
    const bridgeProjection = orientation.length && bridge.length
        ? clamp(dotProduct(orientation, bridge.slice(0, orientation.length)), -1, 1)
        : 0;
    const hopfProjection = orientation.length && hopf.length
        ? clamp(dotProduct(orientation, hopf.slice(0, orientation.length)), -1, 1)
        : 0;
    const intensity = Number.isFinite(axis?.intensity) ? axis.intensity : 0;
    return {
        index: Number.isFinite(axis?.index) ? axis.index : 0,
        intensity,
        continuumProjection,
        voiceProjection,
        bridgeProjection,
        hopfProjection,
        gateFlux: Number.isFinite(axis?.gateFlux) ? axis.gateFlux : 0,
        ratioFlux: Number.isFinite(axis?.ratioFlux) ? axis.ratioFlux : 0,
        coherenceFlux: Number.isFinite(axis?.coherenceFlux) ? axis.coherenceFlux : 0,
        braidFlux: Number.isFinite(axis?.braidFlux) ? axis.braidFlux : 0,
        bitFlux: Number.isFinite(axis?.bitFlux) ? axis.bitFlux : 0,
        entropyFlux: Number.isFinite(axis?.entropyFlux) ? axis.entropyFlux : 0,
        gridFlux: Number.isFinite(axis?.gridFlux) ? axis.gridFlux : 0,
        carrierFlux: Number.isFinite(axis?.carrierFlux) ? axis.carrierFlux : 0,
        bridgeCoupling: Number.isFinite(axis?.bridgeCoupling) ? axis.bridgeCoupling : 0,
        hopfCoupling: Number.isFinite(axis?.hopfCoupling) ? axis.hopfCoupling : 0,
        orientation
    };
};

const buildVoiceEntry = (
    index,
    continuumVoice,
    signalVoice,
    manifoldVoice,
    continuumOrientation,
    bridge,
    hopf
) => {
    const continuum = continuumVoice || {};
    const signal = signalVoice || {};
    const manifold = manifoldVoice || {};
    const orientation = normalizeVector(Array.isArray(continuum.orientation) ? continuum.orientation : []);
    const continuumAlignment = Number.isFinite(continuum.continuumAlignment)
        ? continuum.continuumAlignment
        : orientation.length && continuumOrientation.length
            ? clamp(dotProduct(orientation, continuumOrientation.slice(0, orientation.length)), -1, 1)
            : 0;
    const bridgeProjection = orientation.length && bridge.length
        ? clamp(dotProduct(orientation, bridge.slice(0, orientation.length)), -1, 1)
        : 0;
    const hopfProjection = orientation.length && hopf.length
        ? clamp(dotProduct(orientation, hopf.slice(0, orientation.length)), -1, 1)
        : 0;
    const carriers = Array.isArray(signal.carriers)
        ? signal.carriers.map((carrier) => ({
            label: typeof carrier.label === 'string' ? carrier.label : null,
            frequency: Number.isFinite(carrier.frequency) ? carrier.frequency : 0,
            amplitude: Number.isFinite(carrier.amplitude) ? carrier.amplitude : 0,
            energy: clamp(Number.isFinite(carrier.energy) ? carrier.energy : 0, 0, 1),
            active: carrier.active ? 1 : 0
        }))
        : [];
    const carrierEnergy = carriers.reduce((sum, carrier) => sum + carrier.energy, 0);
    const gridEnergy = Number.isFinite(continuum.gridEnergy)
        ? continuum.gridEnergy
        : Number.isFinite(manifold?.transduction?.gridEnergy)
            ? manifold.transduction.gridEnergy
            : 0;
    const bits = Array.isArray(signal.bits)
        ? signal.bits.slice()
        : Array.isArray(manifold?.signal?.bits)
            ? manifold.signal.bits.slice()
            : [];
    const dominantFrequency = Number.isFinite(continuum.dominantFrequency)
        ? continuum.dominantFrequency
        : Number.isFinite(manifold?.signal?.dominantFrequency)
            ? manifold.signal.dominantFrequency
            : Number.isFinite(signal?.dominant?.frequency)
                ? signal.dominant.frequency
                : 0;
    const gate = Number.isFinite(continuum.gate)
        ? continuum.gate
        : Number.isFinite(manifold?.gate)
            ? manifold.gate
            : 0;
    const ratio = Number.isFinite(continuum.ratio)
        ? continuum.ratio
        : Number.isFinite(manifold?.spinor?.ratio)
            ? manifold.spinor.ratio
            : 0;
    const coherence = Number.isFinite(continuum.coherence)
        ? continuum.coherence
        : Number.isFinite(manifold?.spinor?.coherence)
            ? manifold.spinor.coherence
            : 0;
    const braid = Number.isFinite(continuum.braid)
        ? continuum.braid
        : Number.isFinite(manifold?.spinor?.braid)
            ? manifold.spinor.braid
            : 0;
    const sequence = typeof signal.sequence === 'string'
        ? signal.sequence
        : typeof continuum.sequence === 'string'
            ? continuum.sequence
            : '';
    const bitDensity = Number.isFinite(continuum.bitDensity)
        ? continuum.bitDensity
        : Number.isFinite(manifold?.signal?.bitDensity)
            ? manifold.signal.bitDensity
            : 0;
    const bitEntropy = Number.isFinite(continuum.bitEntropy)
        ? continuum.bitEntropy
        : Number.isFinite(manifold?.signal?.bitEntropy)
            ? manifold.signal.bitEntropy
            : 0;
    const sequenceBits = Array.isArray(signal.bits)
        ? signal.bits.slice()
        : Array.isArray(manifold?.signal?.bits)
            ? manifold.signal.bits.slice()
            : Array.isArray(continuum?.bits)
                ? continuum.bits.slice()
                : [];
    const spinorPhase = Number.isFinite(signal?.phase)
        ? signal.phase
        : Number.isFinite(manifold?.spinor?.phase)
            ? manifold.spinor.phase
            : 0;
    const spinorPan = Number.isFinite(signal?.pan)
        ? clamp(signal.pan, -1, 1)
        : Number.isFinite(manifold?.spinor?.pan)
            ? clamp(manifold.spinor.pan, -1, 1)
            : 0;
    const voiceWeight = gate + Math.abs(ratio) + Math.abs(braid) + Math.max(0, coherence) + carrierEnergy + Math.max(0, gridEnergy);
    return {
        index: Number.isFinite(continuum.index) ? continuum.index : Number.isFinite(manifold?.index) ? manifold.index : index,
        gate,
        ratio,
        coherence,
        braid,
        continuumAlignment,
        bridgeProjection,
        hopfProjection,
        bitDensity,
        bitEntropy,
        gridEnergy,
        carrierEnergy,
        dominantFrequency,
        sequence,
        sequenceBits,
        spinorPhase,
        spinorPan,
        voiceWeight,
        orientation,
        carriers
    };
};

const buildCarrierAggregate = (voices) => {
    const carriers = [];
    voices.forEach((voice) => {
        voice.carriers.forEach((carrier) => {
            carriers.push({
                voice: voice.index,
                label: carrier.label,
                frequency: carrier.frequency,
                amplitude: carrier.amplitude,
                energy: carrier.energy,
                active: carrier.active,
                alignment: voice.continuumAlignment,
                bridgeProjection: voice.bridgeProjection,
                hopfProjection: voice.hopfProjection
            });
        });
    });
    return carriers;
};

const computeSpectralMetrics = (carriers) => {
    if (!Array.isArray(carriers) || !carriers.length) {
        return {
            span: 0,
            mean: 0,
            variance: 0,
            active: 0
        };
    }
    const frequencies = carriers
        .map((carrier) => (Number.isFinite(carrier.frequency) ? carrier.frequency : null))
        .filter((value) => value !== null);
    if (!frequencies.length) {
        return {
            span: 0,
            mean: 0,
            variance: 0,
            active: carriers.reduce((sum, carrier) => sum + (carrier.active ? 1 : 0), 0)
        };
    }
    const min = Math.min(...frequencies);
    const max = Math.max(...frequencies);
    const mean = computeMean(frequencies);
    const variance = computeVariance(frequencies, mean);
    const active = carriers.reduce((sum, carrier) => sum + (carrier.active ? 1 : 0), 0);
    return {
        span: max - min,
        mean,
        variance,
        active
    };
};

export const buildSpinorContinuumLattice = (payload = {}) => {
    const {
        quaternion = null,
        signal = null,
        manifold = null,
        topology = null,
        continuum = null,
        transport = {},
        timelineProgress = 0
    } = payload;

    const timestamp = safeNow();
    const progress = clamp(Number.isFinite(timelineProgress) ? timelineProgress : 0, 0, 1);

    const continuumOrientation = normalizeVector(continuum?.continuum?.orientation || []);
    const voiceOrientation = normalizeVector(continuum?.continuum?.voiceOrientation || []);
    const bridge = normalizeVector(quaternion?.normalizedBridge || quaternion?.bridgeVector || []);
    const hopf = normalizeVector(quaternion?.hopfFiber || []);

    const dimension = Math.max(
        continuumOrientation.length,
        voiceOrientation.length,
        bridge.length,
        hopf.length
    );

    const paddedContinuum = padVector(continuumOrientation, dimension);
    const paddedVoice = padVector(voiceOrientation, dimension);
    const paddedBridge = padVector(bridge, dimension);
    const paddedHopf = padVector(hopf, dimension);

    const bridgeContinuum = dimension
        ? clamp(dotProduct(paddedBridge, paddedContinuum), -1, 1)
        : 0;
    const hopfContinuum = dimension
        ? clamp(dotProduct(paddedHopf, paddedContinuum), -1, 1)
        : 0;
    const voiceContinuum = dimension
        ? clamp(dotProduct(paddedVoice, paddedContinuum), -1, 1)
        : 0;
    const bridgeVoice = dimension
        ? clamp(dotProduct(paddedBridge, paddedVoice), -1, 1)
        : 0;

    const continuumResidualVector = paddedContinuum.map((value, index) => value - paddedBridge[index] * bridgeContinuum);
    const continuumResidualMagnitude = vectorMagnitude(continuumResidualVector);
    const continuumResidual = continuumResidualMagnitude > 0
        ? continuumResidualVector.map((value) => value / continuumResidualMagnitude)
        : continuumResidualVector;

    const voiceResidualVector = paddedVoice.map((value, index) => value - paddedBridge[index] * bridgeVoice);
    const voiceResidualMagnitude = vectorMagnitude(voiceResidualVector);
    const voiceResidual = voiceResidualMagnitude > 0
        ? voiceResidualVector.map((value) => value / voiceResidualMagnitude)
        : voiceResidualVector;

    const continuumAxes = Array.isArray(topology?.axes) ? topology.axes : Array.isArray(continuum?.axes) ? continuum.axes : [];
    const axisEntries = continuumAxes.map((axis) => buildAxisEntry(axis, paddedContinuum, paddedVoice, paddedBridge, paddedHopf));

    const continuumVoices = Array.isArray(continuum?.voices) ? continuum.voices : [];
    const signalVoices = Array.isArray(signal?.voices) ? signal.voices : [];
    const manifoldVoices = Array.isArray(manifold?.voices) ? manifold.voices : [];

    const voiceEntries = continuumVoices.map((continuumVoice, index) => buildVoiceEntry(
        index,
        continuumVoice,
        signalVoices[index],
        manifoldVoices[index],
        paddedContinuum,
        paddedBridge,
        paddedHopf
    ));

    const voiceWeights = voiceEntries.map((voice) => voice.voiceWeight);
    const weightedCoherence = computeWeightedMean(voiceEntries, (voice) => voice.coherence, (voice) => voice.voiceWeight);
    const weightedBraid = computeWeightedMean(voiceEntries, (voice) => voice.braid, (voice) => voice.voiceWeight);
    const weightedCarrierAlignment = computeWeightedMean(
        voiceEntries,
        (voice) => voice.bridgeProjection,
        (voice) => voice.carrierEnergy
    );

    const gateMean = computeMean(voiceEntries.map((voice) => voice.gate));
    const ratioMean = computeMean(voiceEntries.map((voice) => voice.ratio));
    const ratioVariance = computeVariance(voiceEntries.map((voice) => voice.ratio), ratioMean);

    const carriers = buildCarrierAggregate(voiceEntries);
    const spectral = computeSpectralMetrics(carriers);

    const totalCarrierEnergy = voiceEntries.reduce((sum, voice) => sum + voice.carrierEnergy, 0);
    const totalGridEnergy = voiceEntries.reduce((sum, voice) => sum + voice.gridEnergy, 0);
    const sequenceDensity = computeSequenceDensity(voiceEntries);

    const timelineSequence = typeof continuum?.continuum?.sequence === 'string'
        ? continuum.continuum.sequence
        : Array.isArray(signalVoices)
            ? signalVoices.map((voice) => (typeof voice?.sequence === 'string' ? voice.sequence : '')).join('')
            : '';

    return {
        timestamp,
        progress,
        transport: {
            playing: Boolean(transport?.playing),
            mode: typeof transport?.mode === 'string' ? transport.mode : 'idle'
        },
        orientation: {
            continuum: paddedContinuum,
            voice: paddedVoice,
            bridge: paddedBridge,
            hopf: paddedHopf,
            residual: continuumResidual,
            voiceResidual
        },
        synergy: {
            bridgeContinuum,
            hopfContinuum,
            voiceContinuum,
            bridgeVoice,
            continuumResidual: continuumResidualMagnitude,
            voiceResidual: voiceResidualMagnitude,
            coherence: weightedCoherence,
            braid: weightedBraid,
            carrier: weightedCarrierAlignment,
            gateMean,
            ratioMean,
            ratioVariance,
            sequenceDensity
        },
        flux: continuum?.flux
            ? {
                density: Number.isFinite(continuum.flux.density) ? continuum.flux.density : 0,
                variance: Number.isFinite(continuum.flux.variance) ? continuum.flux.variance : 0,
                bridgeAlignment: Number.isFinite(continuum.flux.bridgeAlignment) ? continuum.flux.bridgeAlignment : 0,
                hopfAlignment: Number.isFinite(continuum.flux.hopfAlignment) ? continuum.flux.hopfAlignment : 0,
                voiceAlignment: Number.isFinite(continuum.flux.voiceAlignment) ? continuum.flux.voiceAlignment : 0,
                carrierEnergy: Number.isFinite(continuum.flux.carrierEnergy) ? continuum.flux.carrierEnergy : 0,
                gridEnergy: Number.isFinite(continuum.flux.gridEnergy) ? continuum.flux.gridEnergy : 0
            }
            : {
                density: 0,
                variance: 0,
                bridgeAlignment: 0,
                hopfAlignment: 0,
                voiceAlignment: 0,
                carrierEnergy: 0,
                gridEnergy: 0
            },
        axes: axisEntries,
        voices: voiceEntries,
        carriers,
        spectral: {
            ...spectral,
            energy: totalCarrierEnergy,
            gridEnergy: totalGridEnergy
        },
        timeline: {
            sequence: timelineSequence,
            density: sequenceDensity,
            weights: voiceWeights
        }
    };
};

export const cloneSpinorContinuumLattice = (lattice) => {
    if (!lattice || typeof lattice !== 'object') {
        return null;
    }
    return {
        timestamp: Number(lattice.timestamp) || 0,
        progress: Number.isFinite(lattice.progress) ? lattice.progress : 0,
        transport: lattice.transport
            ? {
                playing: Boolean(lattice.transport.playing),
                mode: typeof lattice.transport.mode === 'string' ? lattice.transport.mode : 'idle'
            }
            : { playing: false, mode: 'idle' },
        orientation: lattice.orientation
            ? {
                continuum: cloneArray(lattice.orientation.continuum),
                voice: cloneArray(lattice.orientation.voice),
                bridge: cloneArray(lattice.orientation.bridge),
                hopf: cloneArray(lattice.orientation.hopf),
                residual: cloneArray(lattice.orientation.residual),
                voiceResidual: cloneArray(lattice.orientation.voiceResidual)
            }
            : {
                continuum: [],
                voice: [],
                bridge: [],
                hopf: [],
                residual: [],
                voiceResidual: []
            },
        synergy: lattice.synergy
            ? {
                bridgeContinuum: Number.isFinite(lattice.synergy.bridgeContinuum) ? lattice.synergy.bridgeContinuum : 0,
                hopfContinuum: Number.isFinite(lattice.synergy.hopfContinuum) ? lattice.synergy.hopfContinuum : 0,
                voiceContinuum: Number.isFinite(lattice.synergy.voiceContinuum) ? lattice.synergy.voiceContinuum : 0,
                bridgeVoice: Number.isFinite(lattice.synergy.bridgeVoice) ? lattice.synergy.bridgeVoice : 0,
                continuumResidual: Number.isFinite(lattice.synergy.continuumResidual) ? lattice.synergy.continuumResidual : 0,
                voiceResidual: Number.isFinite(lattice.synergy.voiceResidual) ? lattice.synergy.voiceResidual : 0,
                coherence: Number.isFinite(lattice.synergy.coherence) ? lattice.synergy.coherence : 0,
                braid: Number.isFinite(lattice.synergy.braid) ? lattice.synergy.braid : 0,
                carrier: Number.isFinite(lattice.synergy.carrier) ? lattice.synergy.carrier : 0,
                gateMean: Number.isFinite(lattice.synergy.gateMean) ? lattice.synergy.gateMean : 0,
                ratioMean: Number.isFinite(lattice.synergy.ratioMean) ? lattice.synergy.ratioMean : 0,
                ratioVariance: Number.isFinite(lattice.synergy.ratioVariance) ? lattice.synergy.ratioVariance : 0,
                sequenceDensity: Number.isFinite(lattice.synergy.sequenceDensity) ? lattice.synergy.sequenceDensity : 0
            }
            : {
                bridgeContinuum: 0,
                hopfContinuum: 0,
                voiceContinuum: 0,
                bridgeVoice: 0,
                continuumResidual: 0,
                voiceResidual: 0,
                coherence: 0,
                braid: 0,
                carrier: 0,
                gateMean: 0,
                ratioMean: 0,
                ratioVariance: 0,
                sequenceDensity: 0
            },
        flux: lattice.flux
            ? {
                density: Number.isFinite(lattice.flux.density) ? lattice.flux.density : 0,
                variance: Number.isFinite(lattice.flux.variance) ? lattice.flux.variance : 0,
                bridgeAlignment: Number.isFinite(lattice.flux.bridgeAlignment) ? lattice.flux.bridgeAlignment : 0,
                hopfAlignment: Number.isFinite(lattice.flux.hopfAlignment) ? lattice.flux.hopfAlignment : 0,
                voiceAlignment: Number.isFinite(lattice.flux.voiceAlignment) ? lattice.flux.voiceAlignment : 0,
                carrierEnergy: Number.isFinite(lattice.flux.carrierEnergy) ? lattice.flux.carrierEnergy : 0,
                gridEnergy: Number.isFinite(lattice.flux.gridEnergy) ? lattice.flux.gridEnergy : 0
            }
            : {
                density: 0,
                variance: 0,
                bridgeAlignment: 0,
                hopfAlignment: 0,
                voiceAlignment: 0,
                carrierEnergy: 0,
                gridEnergy: 0
            },
        axes: Array.isArray(lattice.axes)
            ? lattice.axes.map((axis) => ({
                index: Number.isFinite(axis.index) ? axis.index : 0,
                intensity: Number.isFinite(axis.intensity) ? axis.intensity : 0,
                continuumProjection: Number.isFinite(axis.continuumProjection) ? axis.continuumProjection : 0,
                voiceProjection: Number.isFinite(axis.voiceProjection) ? axis.voiceProjection : 0,
                bridgeProjection: Number.isFinite(axis.bridgeProjection) ? axis.bridgeProjection : 0,
                hopfProjection: Number.isFinite(axis.hopfProjection) ? axis.hopfProjection : 0,
                gateFlux: Number.isFinite(axis.gateFlux) ? axis.gateFlux : 0,
                ratioFlux: Number.isFinite(axis.ratioFlux) ? axis.ratioFlux : 0,
                coherenceFlux: Number.isFinite(axis.coherenceFlux) ? axis.coherenceFlux : 0,
                braidFlux: Number.isFinite(axis.braidFlux) ? axis.braidFlux : 0,
                bitFlux: Number.isFinite(axis.bitFlux) ? axis.bitFlux : 0,
                entropyFlux: Number.isFinite(axis.entropyFlux) ? axis.entropyFlux : 0,
                gridFlux: Number.isFinite(axis.gridFlux) ? axis.gridFlux : 0,
                carrierFlux: Number.isFinite(axis.carrierFlux) ? axis.carrierFlux : 0,
                bridgeCoupling: Number.isFinite(axis.bridgeCoupling) ? axis.bridgeCoupling : 0,
                hopfCoupling: Number.isFinite(axis.hopfCoupling) ? axis.hopfCoupling : 0,
                orientation: cloneArray(axis.orientation)
            }))
            : [],
        voices: Array.isArray(lattice.voices)
            ? lattice.voices.map((voice) => ({
                index: Number.isFinite(voice.index) ? voice.index : 0,
                gate: Number.isFinite(voice.gate) ? voice.gate : 0,
                ratio: Number.isFinite(voice.ratio) ? voice.ratio : 0,
                coherence: Number.isFinite(voice.coherence) ? voice.coherence : 0,
                braid: Number.isFinite(voice.braid) ? voice.braid : 0,
                continuumAlignment: Number.isFinite(voice.continuumAlignment) ? voice.continuumAlignment : 0,
                bridgeProjection: Number.isFinite(voice.bridgeProjection) ? voice.bridgeProjection : 0,
                hopfProjection: Number.isFinite(voice.hopfProjection) ? voice.hopfProjection : 0,
                bitDensity: Number.isFinite(voice.bitDensity) ? voice.bitDensity : 0,
                bitEntropy: Number.isFinite(voice.bitEntropy) ? voice.bitEntropy : 0,
                gridEnergy: Number.isFinite(voice.gridEnergy) ? voice.gridEnergy : 0,
                carrierEnergy: Number.isFinite(voice.carrierEnergy) ? voice.carrierEnergy : 0,
                dominantFrequency: Number.isFinite(voice.dominantFrequency) ? voice.dominantFrequency : 0,
                sequence: typeof voice.sequence === 'string' ? voice.sequence : '',
                sequenceBits: cloneArray(voice.sequenceBits),
                spinorPhase: Number.isFinite(voice.spinorPhase) ? voice.spinorPhase : 0,
                spinorPan: Number.isFinite(voice.spinorPan) ? voice.spinorPan : 0,
                voiceWeight: Number.isFinite(voice.voiceWeight) ? voice.voiceWeight : 0,
                orientation: cloneArray(voice.orientation),
                carriers: Array.isArray(voice.carriers)
                    ? voice.carriers.map((carrier) => ({
                        label: typeof carrier.label === 'string' ? carrier.label : null,
                        frequency: Number.isFinite(carrier.frequency) ? carrier.frequency : 0,
                        amplitude: Number.isFinite(carrier.amplitude) ? carrier.amplitude : 0,
                        energy: Number.isFinite(carrier.energy) ? carrier.energy : 0,
                        active: carrier.active ? 1 : 0
                    }))
                    : []
            }))
            : [],
        carriers: Array.isArray(lattice.carriers)
            ? lattice.carriers.map((carrier) => ({
                voice: Number.isFinite(carrier.voice) ? carrier.voice : 0,
                label: typeof carrier.label === 'string' ? carrier.label : null,
                frequency: Number.isFinite(carrier.frequency) ? carrier.frequency : 0,
                amplitude: Number.isFinite(carrier.amplitude) ? carrier.amplitude : 0,
                energy: Number.isFinite(carrier.energy) ? carrier.energy : 0,
                active: carrier.active ? 1 : 0,
                alignment: Number.isFinite(carrier.alignment) ? carrier.alignment : 0,
                bridgeProjection: Number.isFinite(carrier.bridgeProjection) ? carrier.bridgeProjection : 0,
                hopfProjection: Number.isFinite(carrier.hopfProjection) ? carrier.hopfProjection : 0
            }))
            : [],
        spectral: lattice.spectral
            ? {
                span: Number.isFinite(lattice.spectral.span) ? lattice.spectral.span : 0,
                mean: Number.isFinite(lattice.spectral.mean) ? lattice.spectral.mean : 0,
                variance: Number.isFinite(lattice.spectral.variance) ? lattice.spectral.variance : 0,
                active: Number.isFinite(lattice.spectral.active) ? lattice.spectral.active : 0,
                energy: Number.isFinite(lattice.spectral.energy) ? lattice.spectral.energy : 0,
                gridEnergy: Number.isFinite(lattice.spectral.gridEnergy) ? lattice.spectral.gridEnergy : 0
            }
            : {
                span: 0,
                mean: 0,
                variance: 0,
                active: 0,
                energy: 0,
                gridEnergy: 0
            },
        timeline: lattice.timeline
            ? {
                sequence: typeof lattice.timeline.sequence === 'string' ? lattice.timeline.sequence : '',
                density: Number.isFinite(lattice.timeline.density) ? lattice.timeline.density : 0,
                weights: cloneArray(lattice.timeline.weights)
            }
            : {
                sequence: '',
                density: 0,
                weights: []
            }
    };
};
