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

const normalizeSequenceBits = (sequence) => {
    const hex = typeof sequence === 'string' && sequence.trim()
        ? sequence.trim().slice(0, 2)
        : '0';
    let value = 0;
    try {
        value = parseInt(hex, 16);
    } catch (error) {
        value = 0;
    }
    if (!Number.isFinite(value) || value < 0) {
        value = 0;
    }
    const bits = value.toString(2).padStart(4, '0').slice(-4).split('').map((char) => (char === '1' ? 1 : 0));
    return { hex, bits };
};

const cloneArray = (array) => (Array.isArray(array) ? array.slice() : []);

const computeCarrierStatistics = (voices) => {
    const samples = [];
    voices.forEach((voice) => {
        const gateWeight = voice.gate > 0 ? voice.gate : 0.0001;
        voice.carriers.forEach((carrier) => {
            if (!Number.isFinite(carrier.frequency)) {
                return;
            }
            samples.push({
                frequency: carrier.frequency,
                weight: gateWeight * clamp(carrier.energy, 0, 1)
            });
        });
    });
    if (!samples.length) {
        return {
            centroid: 0,
            spread: 0
        };
    }
    const totalWeight = samples.reduce((sum, sample) => sum + sample.weight, 0) || 1;
    const centroid = samples.reduce((sum, sample) => sum + sample.frequency * sample.weight, 0) / totalWeight;
    const spread = Math.sqrt(samples.reduce((sum, sample) => {
        const diff = sample.frequency - centroid;
        return sum + (diff * diff * sample.weight);
    }, 0) / totalWeight);
    return {
        centroid,
        spread
    };
};

export const buildSpinorSignalFabric = (payload = {}) => {
    const {
        voices: sourceVoices = [],
        quaternion = null,
        spinor = null,
        resonance = null,
        transport = {},
        timelineProgress = 0
    } = payload;

    const normalizedProgress = clamp(Number.isFinite(timelineProgress) ? timelineProgress : 0, 0, 1);
    const voiceCount = Array.isArray(sourceVoices) ? sourceVoices.length : 0;
    if (!voiceCount) {
        return {
            progress: normalizedProgress,
            transport: {
                playing: Boolean(transport.playing),
                mode: transport.mode || 'idle'
            },
            voices: [],
            carrierMatrix: [],
            bitstream: { sequence: '', density: 0, segments: [] },
            envelope: { centroid: 0, spread: 0, resonance: 0, progress: normalizedProgress },
            quantum: quaternion ? { hopf: cloneArray(quaternion.hopfFiber) } : null,
            spinor: spinor ? { coherence: spinor.coherence || 0 } : null,
            resonance: resonance && resonance.aggregate ? { aggregate: { ...resonance.aggregate } } : null
        };
    }

    const resonanceVoices = resonance && Array.isArray(resonance.voices) ? resonance.voices : [];

    const voices = sourceVoices.map((voice, index) => {
        const modulation = voice && typeof voice === 'object' ? voice.modulation || {} : {};
        const carriers = Array.isArray(voice.carriers)
            ? voice.carriers.map((carrier) => ({
                label: typeof carrier.band === 'string' ? carrier.band : `band-${index + 1}`,
                frequency: Number.isFinite(carrier.frequency) ? carrier.frequency : 0,
                amplitude: Number.isFinite(carrier.amplitude) ? carrier.amplitude : 0,
                energy: clamp(Number.isFinite(carrier.energy) ? carrier.energy : 0, 0, 1),
                active: carrier.active ? 1 : 0
            }))
            : [];
        const gate = modulation.binary ? 1 : clamp(Number.isFinite(modulation.gate) ? modulation.gate : 0, 0, 1);
        const { hex, bits } = normalizeSequenceBits(modulation.sequence);
        const dominantCarrier = carriers.reduce((best, carrier) => {
            if (!best || carrier.energy > best.energy) {
                return carrier;
            }
            return best;
        }, null);
        const resonanceSource = resonanceVoices[index] || null;
        const resonancePayload = resonanceSource
            ? {
                vector: cloneArray(resonanceSource.rotated || resonanceSource.source || []),
                normalized: cloneArray(resonanceSource.normalized),
                carriers: Array.isArray(resonanceSource.carriers)
                    ? resonanceSource.carriers.map((carrier) => ({
                        label: carrier.label || null,
                        magnitude: Number.isFinite(carrier.magnitude) ? carrier.magnitude : 0,
                        phase: Number.isFinite(carrier.phase) ? carrier.phase : 0,
                        orientation: cloneArray(carrier.orientation || carrier.rotated)
                    }))
                    : []
            }
            : null;
        return {
            index,
            gate,
            sequence: hex,
            bits,
            phase: Number.isFinite(modulation.phase) ? modulation.phase : 0,
            baseFrequency: Number.isFinite(voice.frequency) ? voice.frequency : 0,
            pan: Number.isFinite(voice.pan) ? clamp(voice.pan, -1, 1) : 0,
            drift: Number.isFinite(voice.drift) ? voice.drift : 0,
            quaternion: voice.quaternion
                ? {
                    weight: Number.isFinite(voice.quaternion.weight) ? voice.quaternion.weight : 0,
                    bridge: Number.isFinite(voice.quaternion.bridge) ? voice.quaternion.bridge : 0,
                    hopf: Number.isFinite(voice.quaternion.hopf) ? voice.quaternion.hopf : 0
                }
                : null,
            spinor: voice.spinor
                ? {
                    ratio: Number.isFinite(voice.spinor.ratio) ? voice.spinor.ratio : 0,
                    pan: Number.isFinite(voice.spinor.pan) ? voice.spinor.pan : 0,
                    phase: Number.isFinite(voice.spinor.phase) ? voice.spinor.phase : 0,
                    coherence: Number.isFinite(voice.spinor.coherence) ? voice.spinor.coherence : 0,
                    braid: Number.isFinite(voice.spinor.braid) ? voice.spinor.braid : 0
                }
                : null,
            carriers,
            dominant: dominantCarrier ? { ...dominantCarrier } : null,
            resonance: resonancePayload
        };
    });

    const carrierStats = computeCarrierStatistics(voices);
    const bitSegments = voices.map((voice) => ({
        voice: voice.index,
        hex: voice.sequence,
        bits: voice.bits.slice(),
        gate: voice.gate
    }));
    const bitDensity = bitSegments.length
        ? bitSegments.reduce((sum, segment) => sum + segment.bits.reduce((acc, bit) => acc + bit, 0), 0)
            / (bitSegments.length * 4)
        : 0;

    return {
        progress: normalizedProgress,
        transport: {
            playing: Boolean(transport.playing),
            mode: transport.mode || 'idle'
        },
        quantum: quaternion
            ? {
                left: cloneArray(quaternion.leftQuaternion || quaternion.left),
                right: cloneArray(quaternion.rightQuaternion || quaternion.right),
                dot: Number.isFinite(quaternion.dot) ? quaternion.dot : 0,
                bridgeMagnitude: Number.isFinite(quaternion.bridgeMagnitude) ? quaternion.bridgeMagnitude : 0,
                hopf: cloneArray(quaternion.hopfFiber)
            }
            : null,
        spinor: spinor
            ? {
                coherence: Number.isFinite(spinor.coherence) ? spinor.coherence : 0,
                braidDensity: Number.isFinite(spinor.braidDensity) ? spinor.braidDensity : 0,
                ratios: cloneArray(spinor.ratios),
                panOrbit: cloneArray(spinor.panOrbit),
                phaseOrbit: cloneArray(spinor.phaseOrbit)
            }
            : null,
        resonance: resonance
            ? {
                aggregate: resonance.aggregate ? { ...resonance.aggregate } : null,
                axes: Array.isArray(resonance.axes) ? resonance.axes.map((axis) => cloneArray(axis)) : [],
                bridge: resonance.bridge
                    ? {
                        magnitude: Number.isFinite(resonance.bridge.magnitude) ? resonance.bridge.magnitude : 0,
                        coherence: Number.isFinite(resonance.bridge.coherence) ? resonance.bridge.coherence : 0,
                        braidDensity: Number.isFinite(resonance.bridge.braidDensity) ? resonance.bridge.braidDensity : 0
                    }
                    : null
            }
            : null,
        voices,
        carrierMatrix: voices.map((voice) => voice.carriers.map((carrier) => ({
            frequency: carrier.frequency,
            relative: carrierStats.centroid ? carrier.frequency / carrierStats.centroid : 0,
            gate: voice.gate,
            energy: carrier.energy
        }))),
        bitstream: {
            sequence: bitSegments.map((segment) => segment.hex).join(''),
            density: clamp(bitDensity, 0, 1),
            segments: bitSegments
        },
        envelope: {
            centroid: carrierStats.centroid,
            spread: carrierStats.spread,
            resonance: resonance && resonance.aggregate && Number.isFinite(resonance.aggregate.magnitude)
                ? resonance.aggregate.magnitude
                : 0,
            progress: normalizedProgress
        }
    };
};

export const cloneSpinorSignalFabric = (fabric) => {
    if (!fabric || typeof fabric !== 'object') {
        return null;
    }
    return {
        ...fabric,
        transport: fabric.transport ? { ...fabric.transport } : { playing: false, mode: 'idle' },
        quantum: fabric.quantum
            ? {
                ...fabric.quantum,
                left: cloneArray(fabric.quantum.left),
                right: cloneArray(fabric.quantum.right),
                hopf: cloneArray(fabric.quantum.hopf)
            }
            : null,
        spinor: fabric.spinor
            ? {
                ...fabric.spinor,
                ratios: cloneArray(fabric.spinor.ratios),
                panOrbit: cloneArray(fabric.spinor.panOrbit),
                phaseOrbit: cloneArray(fabric.spinor.phaseOrbit)
            }
            : null,
        resonance: fabric.resonance
            ? {
                aggregate: fabric.resonance.aggregate ? { ...fabric.resonance.aggregate } : null,
                axes: Array.isArray(fabric.resonance.axes)
                    ? fabric.resonance.axes.map((axis) => cloneArray(axis))
                    : [],
                bridge: fabric.resonance.bridge ? { ...fabric.resonance.bridge } : null
            }
            : null,
        voices: Array.isArray(fabric.voices)
            ? fabric.voices.map((voice) => ({
                ...voice,
                bits: cloneArray(voice.bits),
                carriers: Array.isArray(voice.carriers)
                    ? voice.carriers.map((carrier) => ({ ...carrier }))
                    : [],
                dominant: voice.dominant ? { ...voice.dominant } : null,
                resonance: voice.resonance
                    ? {
                        ...voice.resonance,
                        vector: cloneArray(voice.resonance.vector),
                        normalized: cloneArray(voice.resonance.normalized),
                        carriers: Array.isArray(voice.resonance.carriers)
                            ? voice.resonance.carriers.map((carrier) => ({
                                ...carrier,
                                orientation: cloneArray(carrier.orientation)
                            }))
                            : []
                    }
                    : null
            }))
            : [],
        carrierMatrix: Array.isArray(fabric.carrierMatrix)
            ? fabric.carrierMatrix.map((row) => Array.isArray(row)
                ? row.map((cell) => ({ ...cell }))
                : [])
            : [],
        bitstream: fabric.bitstream
            ? {
                ...fabric.bitstream,
                segments: Array.isArray(fabric.bitstream.segments)
                    ? fabric.bitstream.segments.map((segment) => ({
                        ...segment,
                        bits: cloneArray(segment.bits)
                    }))
                    : []
            }
            : { sequence: '', density: 0, segments: [] },
        envelope: fabric.envelope ? { ...fabric.envelope } : { centroid: 0, spread: 0, resonance: 0, progress: 0 }
    };
};
