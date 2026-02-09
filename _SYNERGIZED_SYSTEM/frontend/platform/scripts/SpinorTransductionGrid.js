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

const cloneArray = (array) => (Array.isArray(array) ? array.slice() : []);

const normalizeProgress = (value) => {
    if (!Number.isFinite(value)) {
        return 0;
    }
    return clamp(value, 0, 1);
};

const toMatrix = (matrix) => {
    if (!Array.isArray(matrix)) {
        return [];
    }
    return matrix
        .filter((row) => Array.isArray(row))
        .map((row) => row.map((value) => (Number.isFinite(value) ? value : 0)));
};

const computeTrace = (matrix) => {
    if (!matrix.length) {
        return 0;
    }
    const size = Math.min(matrix.length, matrix[0].length || 0);
    let sum = 0;
    for (let index = 0; index < size; index += 1) {
        const value = matrix[index] && Number.isFinite(matrix[index][index])
            ? matrix[index][index]
            : 0;
        sum += value;
    }
    return sum;
};

const computeFrobenius = (matrix) => {
    if (!matrix.length) {
        return 0;
    }
    let sum = 0;
    matrix.forEach((row) => {
        row.forEach((value) => {
            if (Number.isFinite(value)) {
                sum += value * value;
            }
        });
    });
    return Math.sqrt(sum);
};

const computeDeterminant = (matrix) => {
    const size = matrix.length;
    if (!size || size !== matrix[0].length) {
        return 0;
    }
    const working = matrix.map((row) => row.slice());
    let det = 1;
    for (let pivotIndex = 0; pivotIndex < size; pivotIndex += 1) {
        let pivotRow = pivotIndex;
        let pivotValue = working[pivotRow][pivotIndex];
        for (let row = pivotIndex + 1; row < size && Math.abs(pivotValue) < 1e-9; row += 1) {
            if (Math.abs(working[row][pivotIndex]) > Math.abs(pivotValue)) {
                pivotRow = row;
                pivotValue = working[row][pivotIndex];
            }
        }
        if (Math.abs(pivotValue) < 1e-12) {
            return 0;
        }
        if (pivotRow !== pivotIndex) {
            const temp = working[pivotIndex];
            working[pivotIndex] = working[pivotRow];
            working[pivotRow] = temp;
            det *= -1;
        }
        det *= pivotValue;
        const invPivot = 1 / pivotValue;
        for (let column = pivotIndex; column < size; column += 1) {
            working[pivotIndex][column] *= invPivot;
        }
        for (let row = pivotIndex + 1; row < size; row += 1) {
            const factor = working[row][pivotIndex];
            if (Math.abs(factor) < 1e-12) {
                continue;
            }
            for (let column = pivotIndex; column < size; column += 1) {
                working[row][column] -= factor * working[pivotIndex][column];
            }
        }
    }
    return det;
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

const wrapPhase = (value) => {
    if (!Number.isFinite(value)) {
        return 0;
    }
    let wrapped = value % 1;
    if (wrapped < 0) {
        wrapped += 1;
    }
    return wrapped;
};

const computeRowProjection = (row, bridge) => {
    if (!Array.isArray(row) || !row.length || !Array.isArray(bridge) || !bridge.length) {
        return 0;
    }
    return row.reduce((sum, value, index) => {
        const bridgeValue = bridge[index % bridge.length] || 0;
        return sum + (Number.isFinite(value) ? value * bridgeValue : 0);
    }, 0);
};

const computeCarrierProjection = (matrixRow, resonanceVector, carrierIndex) => {
    if (!Array.isArray(matrixRow) || !matrixRow.length) {
        return 0;
    }
    const resonanceValue = Array.isArray(resonanceVector) && resonanceVector.length
        ? resonanceVector[carrierIndex % resonanceVector.length]
        : 0;
    const base = matrixRow[carrierIndex % matrixRow.length] || 0;
    return Number.isFinite(base) ? base * resonanceValue : 0;
};

const deriveBitDensity = (bits) => {
    if (!Array.isArray(bits) || !bits.length) {
        return 0;
    }
    const sum = bits.reduce((total, bit) => total + (bit ? 1 : 0), 0);
    return sum / bits.length;
};

export const buildSpinorTransductionGrid = (payload = {}) => {
    const {
        quaternion = null,
        spinor = null,
        resonance = null,
        signal = null,
        voices: sourceVoices = [],
        timelineProgress = 0
    } = payload;

    const progress = normalizeProgress(timelineProgress);
    const matrix = toMatrix(quaternion?.rotationMatrix || []);
    const trace = computeTrace(matrix);
    const frobenius = computeFrobenius(matrix);
    const determinant = matrix.length ? computeDeterminant(matrix) : 0;
    const normalizedBridge = normalizeVector(quaternion?.normalizedBridge || []);
    const hopfFiber = normalizeVector(quaternion?.hopfFiber || []);
    const hopfAlignment = normalizedBridge.length && hopfFiber.length
        ? dotProduct(normalizedBridge, hopfFiber)
        : 0;

    const spinorRatios = Array.isArray(spinor?.ratios) ? spinor.ratios : [];
    const spinorPanOrbit = Array.isArray(spinor?.panOrbit) ? spinor.panOrbit : [];
    const spinorPhaseOrbit = Array.isArray(spinor?.phaseOrbit) ? spinor.phaseOrbit : [];
    const resonanceVoices = Array.isArray(resonance?.voices) ? resonance.voices : [];
    const signalVoices = Array.isArray(signal?.voices) ? signal.voices : [];

    const voices = sourceVoices.map((voice, index) => {
        const modulation = voice?.modulation || {};
        const gate = modulation.binary ? 1 : clamp(Number.isFinite(modulation.gate) ? modulation.gate : 0, 0, 1);
        const matrixRow = matrix[index % (matrix.length || 1)] || [];
        const resonanceEntry = resonanceVoices[index] || {};
        const resonanceVector = Array.isArray(resonanceEntry.normalized)
            ? resonanceEntry.normalized
            : Array.isArray(resonanceEntry.rotated)
                ? resonanceEntry.rotated
                : [];
        const signalEntry = signalVoices[index] || {};
        const bits = cloneArray(signalEntry.bits || []);
        const bitSequence = typeof signalEntry.sequence === 'string'
            ? signalEntry.sequence
            : (typeof modulation.sequence === 'string' ? modulation.sequence : '0');
        const bitDensity = deriveBitDensity(bits);
        const rowProjection = computeRowProjection(matrixRow, normalizedBridge);

        const carriers = Array.isArray(voice?.carriers)
            ? voice.carriers.map((carrier, carrierIndex) => {
                const ratio = Number.isFinite(spinorRatios[carrierIndex % (spinorRatios.length || 1)])
                    ? spinorRatios[carrierIndex % spinorRatios.length]
                    : 1;
                const cents = ratio > 0 ? Math.log2(ratio) * 1200 : 0;
                const pan = Number.isFinite(spinorPanOrbit[carrierIndex % (spinorPanOrbit.length || 1)])
                    ? spinorPanOrbit[carrierIndex % spinorPanOrbit.length]
                    : (Number.isFinite(voice.pan) ? voice.pan : 0);
                const phase = Number.isFinite(spinorPhaseOrbit[carrierIndex % (spinorPhaseOrbit.length || 1)])
                    ? wrapPhase(spinorPhaseOrbit[carrierIndex % spinorPhaseOrbit.length])
                    : wrapPhase(modulation.phase || 0);
                const projection = computeCarrierProjection(matrixRow, resonanceVector, carrierIndex);
                const hopf = hopfFiber.length ? hopfFiber[carrierIndex % hopfFiber.length] : 0;
                const gateBit = bits.length ? bits[carrierIndex % bits.length] : 0;
                return {
                    index: carrierIndex,
                    label: typeof carrier.label === 'string' ? carrier.label : `band-${carrierIndex + 1}`,
                    frequency: Number.isFinite(carrier.frequency) ? carrier.frequency : 0,
                    amplitude: Number.isFinite(carrier.amplitude) ? carrier.amplitude : 0,
                    energy: Number.isFinite(carrier.energy) ? clamp(carrier.energy, 0, 1) : 0,
                    active: carrier.active ? 1 : 0,
                    ratio,
                    cents,
                    pan,
                    phase,
                    projection,
                    hopf,
                    gateBit
                };
            })
            : [];

        return {
            index: voice.index ?? index,
            gate,
            quaternionWeight: Number.isFinite(voice?.quaternion?.weight) ? voice.quaternion.weight : 0,
            bridge: Number.isFinite(voice?.quaternion?.bridge) ? voice.quaternion.bridge : 0,
            hopf: Number.isFinite(voice?.quaternion?.hopf) ? voice.quaternion.hopf : 0,
            spinor: voice?.spinor
                ? {
                    ratio: Number.isFinite(voice.spinor.ratio) ? voice.spinor.ratio : 0,
                    pan: Number.isFinite(voice.spinor.pan) ? voice.spinor.pan : 0,
                    phase: Number.isFinite(voice.spinor.phase) ? voice.spinor.phase : 0,
                    coherence: Number.isFinite(voice.spinor.coherence) ? voice.spinor.coherence : 0
                }
                : null,
            bits,
            bitSequence,
            bitDensity,
            rowProjection,
            matrixRow: cloneArray(matrixRow),
            resonanceVector: cloneArray(resonanceVector),
            carriers
        };
    });

    const flattenedGrid = voices.reduce((grid, voice) => {
        voice.carriers.forEach((carrier) => {
            grid.push({
                voice: voice.index,
                carrier: carrier.index,
                ratio: carrier.ratio,
                frequency: carrier.frequency,
                projection: carrier.projection,
                gateBit: carrier.gateBit,
                energy: carrier.energy
            });
        });
        return grid;
    }, []);

    return {
        progress,
        invariants: {
            determinant,
            trace,
            frobenius,
            hopfAlignment,
            bridgeMagnitude: Number.isFinite(quaternion?.bridgeMagnitude) ? quaternion.bridgeMagnitude : 0
        },
        topology: {
            bridge: cloneArray(normalizedBridge),
            hopf: cloneArray(hopfFiber),
            spinorCoherence: Number.isFinite(spinor?.coherence) ? spinor.coherence : 0,
            braidDensity: Number.isFinite(spinor?.braidDensity) ? spinor.braidDensity : 0
        },
        matrix,
        signal: signal
            ? {
                density: Number.isFinite(signal.bitstream?.density) ? signal.bitstream.density : 0,
                centroid: Number.isFinite(signal.envelope?.centroid) ? signal.envelope.centroid : 0,
                spread: Number.isFinite(signal.envelope?.spread) ? signal.envelope.spread : 0
            }
            : null,
        resonance: resonance?.aggregate ? { ...resonance.aggregate } : null,
        voices,
        grid: flattenedGrid
    };
};

export const cloneSpinorTransductionGrid = (grid) => {
    if (!grid || typeof grid !== 'object') {
        return null;
    }
    return {
        ...grid,
        invariants: grid.invariants ? { ...grid.invariants } : null,
        topology: grid.topology
            ? {
                ...grid.topology,
                bridge: Array.isArray(grid.topology.bridge) ? grid.topology.bridge.slice() : [],
                hopf: Array.isArray(grid.topology.hopf) ? grid.topology.hopf.slice() : []
            }
            : null,
        matrix: Array.isArray(grid.matrix) ? grid.matrix.map((row) => row.slice()) : [],
        signal: grid.signal ? { ...grid.signal } : null,
        resonance: grid.resonance ? { ...grid.resonance } : null,
        voices: Array.isArray(grid.voices)
            ? grid.voices.map((voice) => ({
                ...voice,
                bits: Array.isArray(voice.bits) ? voice.bits.slice() : [],
                matrixRow: Array.isArray(voice.matrixRow) ? voice.matrixRow.slice() : [],
                resonanceVector: Array.isArray(voice.resonanceVector) ? voice.resonanceVector.slice() : [],
                carriers: Array.isArray(voice.carriers)
                    ? voice.carriers.map((carrier) => ({ ...carrier }))
                    : []
            }))
            : [],
        grid: Array.isArray(grid.grid)
            ? grid.grid.map((cell) => ({ ...cell }))
            : []
    };
};
