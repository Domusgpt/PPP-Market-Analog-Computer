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

const correlation = (a, b) => {
    if (!Array.isArray(a) || !Array.isArray(b) || !a.length || !b.length) {
        return 0;
    }
    const normalizedA = normalizeVector(a);
    const normalizedB = normalizeVector(b);
    if (!normalizedA.length || !normalizedB.length) {
        return 0;
    }
    return clamp(dotProduct(normalizedA, normalizedB), -1, 1);
};

const computeEntropy = (samples) => {
    if (!Array.isArray(samples) || !samples.length) {
        return 0;
    }
    const normalized = samples
        .map((value) => (Number.isFinite(value) && value > 0 ? value : 0))
        .filter((value) => value > 0);
    if (!normalized.length) {
        return 0;
    }
    const total = normalized.reduce((sum, value) => sum + value, 0);
    if (total <= 0) {
        return 0;
    }
    const probabilities = normalized.map((value) => value / total);
    const entropy = probabilities.reduce((sum, probability) => {
        if (probability <= 0) {
            return sum;
        }
        return sum - probability * Math.log2(probability);
    }, 0);
    const maxEntropy = Math.log2(probabilities.length);
    if (maxEntropy <= 0) {
        return 0;
    }
    return clamp(entropy / maxEntropy, 0, 1);
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

const deriveBitDensity = (bits) => {
    if (!Array.isArray(bits) || !bits.length) {
        return 0;
    }
    const sum = bits.reduce((total, bit) => total + (bit ? 1 : 0), 0);
    return sum / bits.length;
};

const computeGridEnergy = (carriers) => {
    if (!Array.isArray(carriers) || !carriers.length) {
        return 0;
    }
    const total = carriers.reduce((sum, carrier) => {
        const projection = Number.isFinite(carrier?.projection) ? Math.abs(carrier.projection) : 0;
        const energy = Number.isFinite(carrier?.energy) ? carrier.energy : 0;
        return sum + projection * energy;
    }, 0);
    return total / carriers.length;
};

const computeProjectionMetrics = (carriers) => {
    if (!Array.isArray(carriers) || !carriers.length) {
        return { mean: 0, deviation: 0 };
    }
    const projections = carriers.map((carrier) => (Number.isFinite(carrier?.projection) ? carrier.projection : 0));
    const mean = computeMean(projections);
    const deviation = Math.sqrt(computeVariance(projections, mean));
    return { mean, deviation };
};

const normalizeProgress = (value) => {
    if (!Number.isFinite(value)) {
        return 0;
    }
    return clamp(value, 0, 1);
};

export const buildSpinorMetricManifold = (payload = {}) => {
    const {
        quaternion = null,
        spinor = null,
        resonance = null,
        signal = null,
        transduction = null,
        voices: sourceVoices = [],
        timelineProgress = 0
    } = payload;

    const progress = normalizeProgress(timelineProgress);
    const timestamp = safeNow();

    const normalizedBridge = normalizeVector(quaternion?.normalizedBridge || quaternion?.bridgeVector || []);
    const hopfFiber = normalizeVector(quaternion?.hopfFiber || []);
    const quaternionTrace = Number.isFinite(transduction?.invariants?.trace)
        ? transduction.invariants.trace
        : 0;
    const quaternionDeterminant = Number.isFinite(transduction?.invariants?.determinant)
        ? transduction.invariants.determinant
        : 0;
    const quaternionFrobenius = Number.isFinite(transduction?.invariants?.frobenius)
        ? transduction.invariants.frobenius
        : 0;
    const hopfAlignment = Number.isFinite(transduction?.invariants?.hopfAlignment)
        ? transduction.invariants.hopfAlignment
        : correlation(normalizedBridge, hopfFiber);

    const spinorRatios = Array.isArray(spinor?.ratios) ? spinor.ratios : [];
    const spinorPanOrbit = Array.isArray(spinor?.panOrbit) ? spinor.panOrbit : [];
    const spinorPhaseOrbit = Array.isArray(spinor?.phaseOrbit) ? spinor.phaseOrbit : [];
    const spinorPanVariance = computeVariance(spinorPanOrbit, computeMean(spinorPanOrbit));
    const spinorPhaseVariance = computeVariance(spinorPhaseOrbit, computeMean(spinorPhaseOrbit));
    const spinorRatioEntropy = computeEntropy(spinorRatios.map((ratio) => (ratio > 0 ? ratio : 0)));

    const resonanceAggregate = resonance?.aggregate || null;
    const resonanceCentroid = cloneArray(resonanceAggregate?.centroid);
    const resonanceCarrierCentroid = cloneArray(resonanceAggregate?.carrierCentroid);
    const resonanceMagnitude = Number.isFinite(resonanceAggregate?.magnitude) ? resonanceAggregate.magnitude : 0;
    const resonanceBridgeProjection = Number.isFinite(resonanceAggregate?.bridgeProjection)
        ? resonanceAggregate.bridgeProjection
        : 0;
    const resonanceHopfProjection = Number.isFinite(resonanceAggregate?.hopfProjection)
        ? resonanceAggregate.hopfProjection
        : 0;
    const resonanceGateMean = Number.isFinite(resonanceAggregate?.gateMean) ? resonanceAggregate.gateMean : 0;
    const resonanceGateVariance = Number.isFinite(resonanceAggregate?.gateVariance)
        ? resonanceAggregate.gateVariance
        : 0;
    const resonanceCarrierMagnitude = Number.isFinite(resonanceAggregate?.carrierMagnitude)
        ? resonanceAggregate.carrierMagnitude
        : 0;

    const signalDensity = Number.isFinite(signal?.bitstream?.density) ? signal.bitstream.density : 0;
    const signalCentroid = Number.isFinite(signal?.envelope?.centroid) ? signal.envelope.centroid : 0;
    const signalSpread = Number.isFinite(signal?.envelope?.spread) ? signal.envelope.spread : 0;
    const signalResonance = Number.isFinite(signal?.envelope?.resonance) ? signal.envelope.resonance : 0;

    const transductionGrid = Array.isArray(transduction?.grid) ? transduction.grid : [];
    const transductionGateCoherence = transductionGrid.length
        ? transductionGrid.reduce((sum, entry) => sum + (entry.gateBit ? 1 : 0), 0) / transductionGrid.length
        : 0;
    const transductionProjection = computeProjectionMetrics(transductionGrid);
    const transductionEnergy = computeGridEnergy(transductionGrid);

    const resonanceVoices = Array.isArray(resonance?.voices) ? resonance.voices : [];
    const signalVoices = Array.isArray(signal?.voices) ? signal.voices : [];
    const transductionVoices = Array.isArray(transduction?.voices) ? transduction.voices : [];

    const voices = Array.isArray(sourceVoices)
        ? sourceVoices.map((voice, index) => {
            const modulation = voice?.modulation || {};
            const gate = modulation.binary ? 1 : clamp(Number.isFinite(modulation.gate) ? modulation.gate : 0, 0, 1);
            const resonanceEntry = resonanceVoices[index] || {};
            const signalEntry = signalVoices[index] || {};
            const transductionEntry = transductionVoices[index] || {};
            const carriers = Array.isArray(voice?.carriers) ? voice.carriers : [];
            const carrierEnergy = carriers.reduce((sum, carrier) => {
                return sum + (Number.isFinite(carrier?.energy) ? carrier.energy : 0);
            }, 0);
            const dominantFrequency = Number.isFinite(signalEntry?.dominant?.frequency)
                ? signalEntry.dominant.frequency
                : Number.isFinite(voice?.frequency)
                    ? voice.frequency
                    : 0;
            const transductionCarriers = Array.isArray(transductionEntry?.carriers) ? transductionEntry.carriers : [];
            const gridEnergy = computeGridEnergy(transductionCarriers);
            const projectionMetrics = computeProjectionMetrics(transductionCarriers);
            const bits = Array.isArray(signalEntry?.bits) ? signalEntry.bits : Array.isArray(transductionEntry?.bits)
                ? transductionEntry.bits
                : [];
            const bitDensity = Number.isFinite(transductionEntry?.bitDensity)
                ? transductionEntry.bitDensity
                : deriveBitDensity(bits);
            const bitEntropy = computeEntropy(bits);

            return {
                index: voice?.index ?? index,
                gate,
                frequency: Number.isFinite(voice?.frequency) ? voice.frequency : 0,
                pan: Number.isFinite(voice?.pan) ? clamp(voice.pan, -1, 1) : 0,
                drift: Number.isFinite(voice?.drift) ? voice.drift : 0,
                spinor: voice?.spinor
                    ? {
                        ratio: Number.isFinite(voice.spinor.ratio) ? voice.spinor.ratio : 0,
                        coherence: Number.isFinite(voice.spinor.coherence) ? voice.spinor.coherence : 0,
                        braid: Number.isFinite(voice.spinor.braid) ? voice.spinor.braid : 0,
                        pan: Number.isFinite(voice.spinor.pan) ? voice.spinor.pan : 0,
                        phase: Number.isFinite(voice.spinor.phase) ? voice.spinor.phase : 0
                    }
                    : null,
                quaternion: voice?.quaternion
                    ? {
                        weight: Number.isFinite(voice.quaternion.weight) ? voice.quaternion.weight : 0,
                        bridge: Number.isFinite(voice.quaternion.bridge) ? voice.quaternion.bridge : 0,
                        hopf: Number.isFinite(voice.quaternion.hopf) ? voice.quaternion.hopf : 0
                    }
                    : null,
                resonance: resonanceEntry
                    ? {
                        magnitude: Number.isFinite(resonanceEntry.magnitude) ? resonanceEntry.magnitude : 0,
                        projection: Number.isFinite(resonanceEntry.projection) ? resonanceEntry.projection : 0,
                        hopfProjection: Number.isFinite(resonanceEntry.hopfProjection)
                            ? resonanceEntry.hopfProjection
                            : 0
                    }
                    : null,
                signal: {
                    dominantFrequency,
                    bitDensity,
                    bitEntropy,
                    bits: cloneArray(bits)
                },
                transduction: {
                    gridEnergy,
                    projectionMean: projectionMetrics.mean,
                    projectionDeviation: projectionMetrics.deviation
                },
                carrierEnergy
            };
        })
        : [];

    const voiceGateMean = voices.length ? computeMean(voices.map((voice) => voice.gate)) : 0;
    const voiceCarrierEnergy = voices.length ? computeMean(voices.map((voice) => voice.carrierEnergy)) : 0;
    const voiceSpinorCoherence = voices.length
        ? computeMean(voices.map((voice) => (voice.spinor ? voice.spinor.coherence : 0)))
        : 0;

    const alignment = {
        bridgeToResonance: correlation(normalizedBridge, resonanceCentroid),
        bridgeToCarrierCentroid: correlation(normalizedBridge, resonanceCarrierCentroid),
        hopfToCarrierCentroid: correlation(hopfFiber, resonanceCarrierCentroid),
        spinorToSignal: correlation(spinorPanOrbit, voices.map((voice) => voice.pan)),
        signalToGrid: correlation(
            voices.map((voice) => voice.signal.bitDensity),
            voices.map((voice) => voice.transduction.gridEnergy)
        )
    };

    return {
        timestamp,
        progress,
        quaternion: {
            bridgeMagnitude: Number.isFinite(quaternion?.bridgeMagnitude) ? quaternion.bridgeMagnitude : 0,
            hopfAlignment,
            trace: quaternionTrace,
            determinant: quaternionDeterminant,
            frobenius: quaternionFrobenius,
            normalizedBridge,
            hopfFiber
        },
        spinor: {
            coherence: Number.isFinite(spinor?.coherence) ? spinor.coherence : 0,
            braidDensity: Number.isFinite(spinor?.braidDensity) ? spinor.braidDensity : 0,
            ratioEntropy: spinorRatioEntropy,
            panVariance: spinorPanVariance,
            phaseVariance: spinorPhaseVariance,
            ratios: cloneArray(spinorRatios),
            panOrbit: cloneArray(spinorPanOrbit),
            phaseOrbit: cloneArray(spinorPhaseOrbit)
        },
        resonance: {
            magnitude: resonanceMagnitude,
            bridgeProjection: resonanceBridgeProjection,
            hopfProjection: resonanceHopfProjection,
            gateMean: resonanceGateMean,
            gateVariance: resonanceGateVariance,
            carrierMagnitude: resonanceCarrierMagnitude,
            centroid: cloneArray(resonanceCentroid),
            carrierCentroid: cloneArray(resonanceCarrierCentroid)
        },
        signal: {
            density: signalDensity,
            centroid: signalCentroid,
            spread: signalSpread,
            resonance: signalResonance,
            entropy: computeEntropy(signalVoices.flatMap((voice) => (Array.isArray(voice?.bits) ? voice.bits : [])))
        },
        transduction: {
            energy: transductionEnergy,
            gateCoherence: transductionGateCoherence,
            projectionMean: transductionProjection.mean,
            projectionDeviation: transductionProjection.deviation,
            trace: quaternionTrace,
            determinant: quaternionDeterminant,
            frobenius: quaternionFrobenius,
            hopfAlignment
        },
        summary: {
            voiceGateMean,
            voiceCarrierEnergy,
            voiceSpinorCoherence
        },
        alignment,
        voices
    };
};

export const cloneSpinorMetricManifold = (manifold) => {
    if (!manifold || typeof manifold !== 'object') {
        return null;
    }
    return {
        timestamp: Number(manifold.timestamp) || 0,
        progress: Number(manifold.progress) || 0,
        quaternion: manifold.quaternion
            ? {
                bridgeMagnitude: Number(manifold.quaternion.bridgeMagnitude) || 0,
                hopfAlignment: Number(manifold.quaternion.hopfAlignment) || 0,
                trace: Number(manifold.quaternion.trace) || 0,
                determinant: Number(manifold.quaternion.determinant) || 0,
                frobenius: Number(manifold.quaternion.frobenius) || 0,
                normalizedBridge: cloneArray(manifold.quaternion.normalizedBridge),
                hopfFiber: cloneArray(manifold.quaternion.hopfFiber)
            }
            : null,
        spinor: manifold.spinor
            ? {
                coherence: Number(manifold.spinor.coherence) || 0,
                braidDensity: Number(manifold.spinor.braidDensity) || 0,
                ratioEntropy: Number(manifold.spinor.ratioEntropy) || 0,
                panVariance: Number(manifold.spinor.panVariance) || 0,
                phaseVariance: Number(manifold.spinor.phaseVariance) || 0,
                ratios: cloneArray(manifold.spinor.ratios),
                panOrbit: cloneArray(manifold.spinor.panOrbit),
                phaseOrbit: cloneArray(manifold.spinor.phaseOrbit)
            }
            : null,
        resonance: manifold.resonance
            ? {
                magnitude: Number(manifold.resonance.magnitude) || 0,
                bridgeProjection: Number(manifold.resonance.bridgeProjection) || 0,
                hopfProjection: Number(manifold.resonance.hopfProjection) || 0,
                gateMean: Number(manifold.resonance.gateMean) || 0,
                gateVariance: Number(manifold.resonance.gateVariance) || 0,
                carrierMagnitude: Number(manifold.resonance.carrierMagnitude) || 0,
                centroid: cloneArray(manifold.resonance.centroid),
                carrierCentroid: cloneArray(manifold.resonance.carrierCentroid)
            }
            : null,
        signal: manifold.signal
            ? {
                density: Number(manifold.signal.density) || 0,
                centroid: Number(manifold.signal.centroid) || 0,
                spread: Number(manifold.signal.spread) || 0,
                resonance: Number(manifold.signal.resonance) || 0,
                entropy: Number(manifold.signal.entropy) || 0
            }
            : null,
        transduction: manifold.transduction
            ? {
                energy: Number(manifold.transduction.energy) || 0,
                gateCoherence: Number(manifold.transduction.gateCoherence) || 0,
                projectionMean: Number(manifold.transduction.projectionMean) || 0,
                projectionDeviation: Number(manifold.transduction.projectionDeviation) || 0,
                trace: Number(manifold.transduction.trace) || 0,
                determinant: Number(manifold.transduction.determinant) || 0,
                frobenius: Number(manifold.transduction.frobenius) || 0,
                hopfAlignment: Number(manifold.transduction.hopfAlignment) || 0
            }
            : null,
        summary: manifold.summary
            ? {
                voiceGateMean: Number(manifold.summary.voiceGateMean) || 0,
                voiceCarrierEnergy: Number(manifold.summary.voiceCarrierEnergy) || 0,
                voiceSpinorCoherence: Number(manifold.summary.voiceSpinorCoherence) || 0
            }
            : null,
        alignment: manifold.alignment
            ? {
                bridgeToResonance: Number(manifold.alignment.bridgeToResonance) || 0,
                bridgeToCarrierCentroid: Number(manifold.alignment.bridgeToCarrierCentroid) || 0,
                hopfToCarrierCentroid: Number(manifold.alignment.hopfToCarrierCentroid) || 0,
                spinorToSignal: Number(manifold.alignment.spinorToSignal) || 0,
                signalToGrid: Number(manifold.alignment.signalToGrid) || 0
            }
            : null,
        voices: Array.isArray(manifold.voices)
            ? manifold.voices.map((voice) => ({
                index: Number.isFinite(voice?.index) ? voice.index : 0,
                gate: Number(voice?.gate) || 0,
                frequency: Number(voice?.frequency) || 0,
                pan: Number(voice?.pan) || 0,
                drift: Number(voice?.drift) || 0,
                spinor: voice?.spinor
                    ? {
                        ratio: Number(voice.spinor.ratio) || 0,
                        coherence: Number(voice.spinor.coherence) || 0,
                        braid: Number(voice.spinor.braid) || 0,
                        pan: Number(voice.spinor.pan) || 0,
                        phase: Number(voice.spinor.phase) || 0
                    }
                    : null,
                quaternion: voice?.quaternion
                    ? {
                        weight: Number(voice.quaternion.weight) || 0,
                        bridge: Number(voice.quaternion.bridge) || 0,
                        hopf: Number(voice.quaternion.hopf) || 0
                    }
                    : null,
                resonance: voice?.resonance
                    ? {
                        magnitude: Number(voice.resonance.magnitude) || 0,
                        projection: Number(voice.resonance.projection) || 0,
                        hopfProjection: Number(voice.resonance.hopfProjection) || 0
                    }
                    : null,
                signal: voice?.signal
                    ? {
                        dominantFrequency: Number(voice.signal.dominantFrequency) || 0,
                        bitDensity: Number(voice.signal.bitDensity) || 0,
                        bitEntropy: Number(voice.signal.bitEntropy) || 0,
                        bits: cloneArray(voice.signal.bits)
                    }
                    : null,
                transduction: voice?.transduction
                    ? {
                        gridEnergy: Number(voice.transduction.gridEnergy) || 0,
                        projectionMean: Number(voice.transduction.projectionMean) || 0,
                        projectionDeviation: Number(voice.transduction.projectionDeviation) || 0
                    }
                    : null,
                carrierEnergy: Number(voice?.carrierEnergy) || 0
            }))
            : []
    };
};
