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

const normalizeVector = (vector) => {
    if (!Array.isArray(vector) || !vector.length) {
        return [];
    }
    const magnitude = Math.sqrt(vector.reduce((sum, value) => sum + (Number.isFinite(value) ? value * value : 0), 0));
    if (magnitude <= 0) {
        return vector.map(() => 0);
    }
    return vector.map((value) => (Number.isFinite(value) ? value / magnitude : 0));
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

const computeWeightedCentroid = (nodes) => {
    if (!Array.isArray(nodes) || !nodes.length) {
        return [];
    }
    const dimension = Math.max(...nodes.map((node) => Array.isArray(node.orientation) ? node.orientation.length : 0));
    if (!dimension) {
        return [];
    }
    const accumulator = new Array(dimension).fill(0);
    let totalWeight = 0;
    nodes.forEach((node) => {
        const orientation = Array.isArray(node.orientation) ? node.orientation : [];
        const weight = Number.isFinite(node.voiceWeight) ? Math.max(0, node.voiceWeight) : 0;
        if (!orientation.length || weight <= 0) {
            return;
        }
        const padded = padVector(orientation, dimension);
        for (let index = 0; index < dimension; index += 1) {
            accumulator[index] += padded[index] * weight;
        }
        totalWeight += weight;
    });
    if (totalWeight <= 0) {
        return accumulator.map(() => 0);
    }
    return normalizeVector(accumulator.map((value) => value / totalWeight));
};

const computeDispersion = (nodes, centroid) => {
    if (!Array.isArray(nodes) || !nodes.length) {
        return {
            mean: 0,
            max: 0
        };
    }
    const dimension = Array.isArray(centroid) ? centroid.length : 0;
    if (!dimension) {
        return {
            mean: 0,
            max: 0
        };
    }
    let total = 0;
    let max = 0;
    let count = 0;
    nodes.forEach((node) => {
        if (!Array.isArray(node.orientation) || !node.orientation.length) {
            return;
        }
        const padded = padVector(node.orientation, dimension);
        let distanceSquared = 0;
        for (let index = 0; index < dimension; index += 1) {
            const diff = padded[index] - centroid[index];
            distanceSquared += diff * diff;
        }
        total += distanceSquared;
        if (distanceSquared > max) {
            max = distanceSquared;
        }
        count += 1;
    });
    if (!count) {
        return {
            mean: 0,
            max: 0
        };
    }
    return {
        mean: total / count,
        max
    };
};

const computeFrequencyStats = (nodes) => {
    if (!Array.isArray(nodes) || !nodes.length) {
        return {
            min: 0,
            max: 0,
            mean: 0,
            variance: 0
        };
    }
    const frequencies = nodes
        .map((node) => (Number.isFinite(node.dominantFrequency) ? node.dominantFrequency : null))
        .filter((value) => value !== null);
    if (!frequencies.length) {
        return {
            min: 0,
            max: 0,
            mean: 0,
            variance: 0
        };
    }
    const min = Math.min(...frequencies);
    const max = Math.max(...frequencies);
    const mean = frequencies.reduce((sum, value) => sum + value, 0) / frequencies.length;
    const variance = frequencies.reduce((sum, value) => {
        const diff = value - mean;
        return sum + diff * diff;
    }, 0) / frequencies.length;
    return {
        min,
        max,
        mean,
        variance
    };
};

const computeGateStats = (nodes) => {
    if (!Array.isArray(nodes) || !nodes.length) {
        return {
            mean: 0,
            variance: 0
        };
    }
    const gates = nodes.map((node) => (Number.isFinite(node.gate) ? clamp(node.gate, 0, 1) : 0));
    const mean = gates.reduce((sum, value) => sum + value, 0) / gates.length;
    const variance = gates.reduce((sum, value) => {
        const diff = value - mean;
        return sum + diff * diff;
    }, 0) / gates.length;
    return {
        mean,
        variance
    };
};

const computeSpinorStats = (nodes) => {
    if (!Array.isArray(nodes) || !nodes.length) {
        return {
            panMean: 0,
            panVariance: 0,
            phaseMean: 0,
            phaseVariance: 0
        };
    }
    const pans = nodes.map((node) => (Number.isFinite(node.spinorPan) ? clamp(node.spinorPan, -1, 1) : 0));
    const phases = nodes.map((node) => (Number.isFinite(node.spinorPhase) ? node.spinorPhase : 0));
    const panMean = pans.reduce((sum, value) => sum + value, 0) / pans.length;
    const phaseMean = phases.reduce((sum, value) => sum + value, 0) / phases.length;
    const panVariance = pans.reduce((sum, value) => {
        const diff = value - panMean;
        return sum + diff * diff;
    }, 0) / pans.length;
    const phaseVariance = phases.reduce((sum, value) => {
        const diff = value - phaseMean;
        return sum + diff * diff;
    }, 0) / phases.length;
    return {
        panMean,
        panVariance,
        phaseMean,
        phaseVariance
    };
};

const computeSequenceStats = (nodes) => {
    if (!Array.isArray(nodes) || !nodes.length) {
        return {
            bitDensity: 0,
            bitEntropy: 0
        };
    }
    const densities = nodes.map((node) => (Number.isFinite(node.bitDensity) ? clamp(node.bitDensity, 0, 1) : 0));
    const entropies = nodes.map((node) => (Number.isFinite(node.bitEntropy) ? Math.max(0, node.bitEntropy) : 0));
    const bitDensity = densities.reduce((sum, value) => sum + value, 0) / densities.length;
    const bitEntropy = entropies.reduce((sum, value) => sum + value, 0) / entropies.length;
    return {
        bitDensity,
        bitEntropy
    };
};

const buildNodeEntry = (voice, index) => {
    const orientation = normalizeVector(Array.isArray(voice?.orientation) ? voice.orientation : []);
    const carriers = Array.isArray(voice?.carriers)
        ? voice.carriers.map((carrier) => ({
            label: typeof carrier.label === 'string' ? carrier.label : null,
            frequency: Number.isFinite(carrier.frequency) ? carrier.frequency : 0,
            amplitude: Number.isFinite(carrier.amplitude) ? carrier.amplitude : 0,
            energy: Number.isFinite(carrier.energy) ? carrier.energy : 0,
            active: carrier.active ? 1 : 0
        }))
        : [];
    return {
        index: Number.isFinite(voice?.index) ? voice.index : index,
        orientation,
        gate: Number.isFinite(voice?.gate) ? clamp(voice.gate, 0, 1) : 0,
        ratio: Number.isFinite(voice?.ratio) ? voice.ratio : 0,
        coherence: Number.isFinite(voice?.coherence) ? voice.coherence : 0,
        braid: Number.isFinite(voice?.braid) ? voice.braid : 0,
        continuumAlignment: Number.isFinite(voice?.continuumAlignment) ? voice.continuumAlignment : 0,
        bridgeProjection: Number.isFinite(voice?.bridgeProjection) ? voice.bridgeProjection : 0,
        hopfProjection: Number.isFinite(voice?.hopfProjection) ? voice.hopfProjection : 0,
        bitDensity: Number.isFinite(voice?.bitDensity) ? voice.bitDensity : 0,
        bitEntropy: Number.isFinite(voice?.bitEntropy) ? voice.bitEntropy : 0,
        gridEnergy: Number.isFinite(voice?.gridEnergy) ? voice.gridEnergy : 0,
        carrierEnergy: Number.isFinite(voice?.carrierEnergy) ? voice.carrierEnergy : 0,
        dominantFrequency: Number.isFinite(voice?.dominantFrequency) ? voice.dominantFrequency : 0,
        sequence: typeof voice?.sequence === 'string' ? voice.sequence : '',
        sequenceBits: Array.isArray(voice?.sequenceBits) ? voice.sequenceBits.slice() : [],
        spinorPhase: Number.isFinite(voice?.spinorPhase) ? voice.spinorPhase : 0,
        spinorPan: Number.isFinite(voice?.spinorPan) ? voice.spinorPan : 0,
        voiceWeight: Number.isFinite(voice?.voiceWeight) ? Math.max(0, voice.voiceWeight) : 0,
        carriers
    };
};

const summarizeCarriers = (carriers) => {
    if (!Array.isArray(carriers) || !carriers.length) {
        return {
            active: 0,
            energy: 0,
            meanFrequency: 0,
            span: 0
        };
    }
    let active = 0;
    let energy = 0;
    const frequencies = [];
    carriers.forEach((carrier) => {
        if (carrier.active) {
            active += 1;
        }
        if (Number.isFinite(carrier.energy)) {
            energy += carrier.energy;
        }
        if (Number.isFinite(carrier.frequency)) {
            frequencies.push(carrier.frequency);
        }
    });
    const meanFrequency = frequencies.length
        ? frequencies.reduce((sum, value) => sum + value, 0) / frequencies.length
        : 0;
    const span = frequencies.length ? Math.max(...frequencies) - Math.min(...frequencies) : 0;
    return {
        active,
        energy,
        meanFrequency,
        span
    };
};

const cloneNodes = (nodes) => {
    if (!Array.isArray(nodes)) {
        return [];
    }
    return nodes.map((node) => ({
        ...node,
        orientation: cloneArray(node.orientation),
        sequenceBits: cloneArray(node.sequenceBits),
        carriers: Array.isArray(node.carriers)
            ? node.carriers.map((carrier) => ({ ...carrier }))
            : []
    }));
};

export const buildSpinorContinuumConstellation = (payload = {}) => {
    const {
        quaternion = null,
        continuum = null,
        lattice = null,
        signal = null,
        manifold = null,
        transport = {},
        timelineProgress = 0
    } = payload;

    const timestamp = safeNow();
    const progress = clamp(Number.isFinite(timelineProgress) ? timelineProgress : 0, 0, 1);

    const latticeOrientation = lattice?.orientation || {};
    const continuumVector = normalizeVector(latticeOrientation.continuum || continuum?.continuum?.orientation || []);
    const voiceVector = normalizeVector(latticeOrientation.voice || []);
    const residualVector = normalizeVector(latticeOrientation.residual || []);
    const bridgeVector = normalizeVector(quaternion?.normalizedBridge || quaternion?.bridgeVector || []);
    const hopfVector = normalizeVector(quaternion?.hopfFiber || []);

    const nodes = Array.isArray(lattice?.voices)
        ? lattice.voices.map((voice, index) => buildNodeEntry(voice, index))
        : [];
    const centroid = computeWeightedCentroid(nodes);
    const dispersion = computeDispersion(nodes, centroid);
    const frequency = computeFrequencyStats(nodes);
    const gating = computeGateStats(nodes);
    const spinorStats = computeSpinorStats(nodes);
    const sequenceStats = computeSequenceStats(nodes);

    const paddedCentroid = padVector(centroid, Math.max(centroid.length, bridgeVector.length, hopfVector.length));
    const paddedBridge = padVector(bridgeVector, paddedCentroid.length);
    const paddedHopf = padVector(hopfVector, paddedCentroid.length);

    const centroidBridge = paddedCentroid.length ? clamp(dotProduct(paddedCentroid, paddedBridge), -1, 1) : 0;
    const centroidHopf = paddedCentroid.length ? clamp(dotProduct(paddedCentroid, paddedHopf), -1, 1) : 0;

    const constellationFlux = continuum?.flux || {};
    const synergy = lattice?.synergy || {};
    const spectral = lattice?.spectral || {};

    const carriers = Array.isArray(lattice?.carriers)
        ? lattice.carriers.map((carrier) => ({
            ...carrier,
            voice: Number.isFinite(carrier.voice) ? carrier.voice : null,
            label: typeof carrier.label === 'string' ? carrier.label : null
        }))
        : [];
    const carrierSummary = summarizeCarriers(carriers);

    return {
        timestamp,
        progress,
        transport: {
            playing: Boolean(transport?.playing),
            mode: typeof transport?.mode === 'string' ? transport.mode : 'idle'
        },
        orientation: {
            continuum: continuumVector,
            voice: voiceVector,
            residual: residualVector,
            bridge: bridgeVector,
            hopf: hopfVector,
            centroid
        },
        centroid: {
            bridgeProjection: centroidBridge,
            hopfProjection: centroidHopf,
            dispersion
        },
        nodes,
        sequence: {
            sequence: typeof lattice?.timeline?.sequence === 'string'
                ? lattice.timeline.sequence
                : Array.isArray(signal?.voices)
                    ? signal.voices.map((voice) => (typeof voice.sequence === 'string' ? voice.sequence : '')).join('')
                    : '',
            density: Number.isFinite(lattice?.timeline?.density) ? lattice.timeline.density : 0,
            weights: cloneArray(lattice?.timeline?.weights),
            bitDensity: sequenceStats.bitDensity,
            bitEntropy: sequenceStats.bitEntropy
        },
        gating,
        spinor: {
            coherence: Number.isFinite(synergy?.coherence) ? synergy.coherence : 0,
            braid: Number.isFinite(synergy?.braid) ? synergy.braid : 0,
            ratioMean: Number.isFinite(synergy?.ratioMean) ? synergy.ratioMean : 0,
            ratioVariance: Number.isFinite(synergy?.ratioVariance) ? synergy.ratioVariance : 0,
            panMean: spinorStats.panMean,
            panVariance: spinorStats.panVariance,
            phaseMean: spinorStats.phaseMean,
            phaseVariance: spinorStats.phaseVariance
        },
        energy: {
            carrier: Number.isFinite(synergy?.carrier) ? synergy.carrier : carrierSummary.energy,
            grid: Number.isFinite(lattice?.spectral?.gridEnergy) ? lattice.spectral.gridEnergy : 0,
            continuum: Number.isFinite(constellationFlux?.carrierEnergy) ? constellationFlux.carrierEnergy : 0,
            signal: Number.isFinite(signal?.envelope?.energy) ? signal.envelope.energy : 0
        },
        frequency: {
            ...frequency,
            centroid: Number.isFinite(lattice?.spectral?.mean) ? lattice.spectral.mean : frequency.mean,
            span: Number.isFinite(lattice?.spectral?.span) ? lattice.spectral.span : frequency.max - frequency.min,
            variance: Number.isFinite(lattice?.spectral?.variance) ? lattice.spectral.variance : frequency.variance
        },
        flux: {
            density: Number.isFinite(constellationFlux?.density) ? constellationFlux.density : 0,
            variance: Number.isFinite(constellationFlux?.variance) ? constellationFlux.variance : 0,
            bridgeAlignment: Number.isFinite(constellationFlux?.bridgeAlignment) ? constellationFlux.bridgeAlignment : 0,
            hopfAlignment: Number.isFinite(constellationFlux?.hopfAlignment) ? constellationFlux.hopfAlignment : 0,
            voiceAlignment: Number.isFinite(constellationFlux?.voiceAlignment) ? constellationFlux.voiceAlignment : 0
        },
        synergy: {
            bridgeContinuum: Number.isFinite(synergy?.bridgeContinuum) ? synergy.bridgeContinuum : 0,
            hopfContinuum: Number.isFinite(synergy?.hopfContinuum) ? synergy.hopfContinuum : 0,
            voiceContinuum: Number.isFinite(synergy?.voiceContinuum) ? synergy.voiceContinuum : 0,
            bridgeVoice: Number.isFinite(synergy?.bridgeVoice) ? synergy.bridgeVoice : 0,
            gateMean: Number.isFinite(synergy?.gateMean) ? synergy.gateMean : gating.mean,
            sequenceDensity: Number.isFinite(synergy?.sequenceDensity) ? synergy.sequenceDensity : 0
        },
        carriers: {
            summary: carrierSummary,
            spectral: {
                span: Number.isFinite(spectral?.span) ? spectral.span : carrierSummary.span,
                variance: Number.isFinite(spectral?.variance) ? spectral.variance : 0,
                mean: Number.isFinite(spectral?.mean) ? spectral.mean : carrierSummary.meanFrequency,
                active: Number.isFinite(spectral?.active) ? spectral.active : carrierSummary.active
            },
            entries: carriers
        }
    };
};

export const cloneSpinorContinuumConstellation = (constellation) => {
    if (!constellation || typeof constellation !== 'object') {
        return null;
    }
    return {
        timestamp: Number(constellation.timestamp) || 0,
        progress: Number.isFinite(constellation.progress) ? constellation.progress : 0,
        transport: {
            playing: Boolean(constellation.transport?.playing),
            mode: typeof constellation.transport?.mode === 'string' ? constellation.transport.mode : 'idle'
        },
        orientation: {
            continuum: cloneArray(constellation.orientation?.continuum),
            voice: cloneArray(constellation.orientation?.voice),
            residual: cloneArray(constellation.orientation?.residual),
            bridge: cloneArray(constellation.orientation?.bridge),
            hopf: cloneArray(constellation.orientation?.hopf),
            centroid: cloneArray(constellation.orientation?.centroid)
        },
        centroid: {
            bridgeProjection: Number.isFinite(constellation.centroid?.bridgeProjection)
                ? constellation.centroid.bridgeProjection
                : 0,
            hopfProjection: Number.isFinite(constellation.centroid?.hopfProjection)
                ? constellation.centroid.hopfProjection
                : 0,
            dispersion: {
                mean: Number.isFinite(constellation.centroid?.dispersion?.mean)
                    ? constellation.centroid.dispersion.mean
                    : 0,
                max: Number.isFinite(constellation.centroid?.dispersion?.max)
                    ? constellation.centroid.dispersion.max
                    : 0
            }
        },
        nodes: cloneNodes(constellation.nodes),
        sequence: {
            sequence: typeof constellation.sequence?.sequence === 'string' ? constellation.sequence.sequence : '',
            density: Number.isFinite(constellation.sequence?.density) ? constellation.sequence.density : 0,
            weights: cloneArray(constellation.sequence?.weights),
            bitDensity: Number.isFinite(constellation.sequence?.bitDensity) ? constellation.sequence.bitDensity : 0,
            bitEntropy: Number.isFinite(constellation.sequence?.bitEntropy) ? constellation.sequence.bitEntropy : 0
        },
        gating: {
            mean: Number.isFinite(constellation.gating?.mean) ? constellation.gating.mean : 0,
            variance: Number.isFinite(constellation.gating?.variance) ? constellation.gating.variance : 0
        },
        spinor: {
            coherence: Number.isFinite(constellation.spinor?.coherence) ? constellation.spinor.coherence : 0,
            braid: Number.isFinite(constellation.spinor?.braid) ? constellation.spinor.braid : 0,
            ratioMean: Number.isFinite(constellation.spinor?.ratioMean) ? constellation.spinor.ratioMean : 0,
            ratioVariance: Number.isFinite(constellation.spinor?.ratioVariance)
                ? constellation.spinor.ratioVariance
                : 0,
            panMean: Number.isFinite(constellation.spinor?.panMean) ? constellation.spinor.panMean : 0,
            panVariance: Number.isFinite(constellation.spinor?.panVariance) ? constellation.spinor.panVariance : 0,
            phaseMean: Number.isFinite(constellation.spinor?.phaseMean) ? constellation.spinor.phaseMean : 0,
            phaseVariance: Number.isFinite(constellation.spinor?.phaseVariance)
                ? constellation.spinor.phaseVariance
                : 0
        },
        energy: {
            carrier: Number.isFinite(constellation.energy?.carrier) ? constellation.energy.carrier : 0,
            grid: Number.isFinite(constellation.energy?.grid) ? constellation.energy.grid : 0,
            continuum: Number.isFinite(constellation.energy?.continuum) ? constellation.energy.continuum : 0,
            signal: Number.isFinite(constellation.energy?.signal) ? constellation.energy.signal : 0
        },
        frequency: {
            min: Number.isFinite(constellation.frequency?.min) ? constellation.frequency.min : 0,
            max: Number.isFinite(constellation.frequency?.max) ? constellation.frequency.max : 0,
            mean: Number.isFinite(constellation.frequency?.mean) ? constellation.frequency.mean : 0,
            centroid: Number.isFinite(constellation.frequency?.centroid) ? constellation.frequency.centroid : 0,
            span: Number.isFinite(constellation.frequency?.span) ? constellation.frequency.span : 0,
            variance: Number.isFinite(constellation.frequency?.variance) ? constellation.frequency.variance : 0
        },
        flux: {
            density: Number.isFinite(constellation.flux?.density) ? constellation.flux.density : 0,
            variance: Number.isFinite(constellation.flux?.variance) ? constellation.flux.variance : 0,
            bridgeAlignment: Number.isFinite(constellation.flux?.bridgeAlignment)
                ? constellation.flux.bridgeAlignment
                : 0,
            hopfAlignment: Number.isFinite(constellation.flux?.hopfAlignment)
                ? constellation.flux.hopfAlignment
                : 0,
            voiceAlignment: Number.isFinite(constellation.flux?.voiceAlignment)
                ? constellation.flux.voiceAlignment
                : 0
        },
        synergy: {
            bridgeContinuum: Number.isFinite(constellation.synergy?.bridgeContinuum)
                ? constellation.synergy.bridgeContinuum
                : 0,
            hopfContinuum: Number.isFinite(constellation.synergy?.hopfContinuum)
                ? constellation.synergy.hopfContinuum
                : 0,
            voiceContinuum: Number.isFinite(constellation.synergy?.voiceContinuum)
                ? constellation.synergy.voiceContinuum
                : 0,
            bridgeVoice: Number.isFinite(constellation.synergy?.bridgeVoice) ? constellation.synergy.bridgeVoice : 0,
            gateMean: Number.isFinite(constellation.synergy?.gateMean) ? constellation.synergy.gateMean : 0,
            sequenceDensity: Number.isFinite(constellation.synergy?.sequenceDensity)
                ? constellation.synergy.sequenceDensity
                : 0
        },
        carriers: {
            summary: {
                active: Number.isFinite(constellation.carriers?.summary?.active)
                    ? constellation.carriers.summary.active
                    : 0,
                energy: Number.isFinite(constellation.carriers?.summary?.energy)
                    ? constellation.carriers.summary.energy
                    : 0,
                meanFrequency: Number.isFinite(constellation.carriers?.summary?.meanFrequency)
                    ? constellation.carriers.summary.meanFrequency
                    : 0,
                span: Number.isFinite(constellation.carriers?.summary?.span)
                    ? constellation.carriers.summary.span
                    : 0
            },
            spectral: {
                span: Number.isFinite(constellation.carriers?.spectral?.span)
                    ? constellation.carriers.spectral.span
                    : 0,
                variance: Number.isFinite(constellation.carriers?.spectral?.variance)
                    ? constellation.carriers.spectral.variance
                    : 0,
                mean: Number.isFinite(constellation.carriers?.spectral?.mean)
                    ? constellation.carriers.spectral.mean
                    : 0,
                active: Number.isFinite(constellation.carriers?.spectral?.active)
                    ? constellation.carriers.spectral.active
                    : 0
            },
            entries: Array.isArray(constellation.carriers?.entries)
                ? constellation.carriers.entries.map((entry) => ({ ...entry }))
                : []
        }
    };
};
