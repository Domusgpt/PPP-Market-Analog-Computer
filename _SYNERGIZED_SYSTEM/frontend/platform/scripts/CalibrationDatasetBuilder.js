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

const average = (values = []) => {
    const filtered = values.filter((value) => Number.isFinite(value));
    if (!filtered.length) {
        return null;
    }
    const sum = filtered.reduce((accumulator, value) => accumulator + value, 0);
    return sum / filtered.length;
};

const stats = (values = []) => {
    const filtered = values.filter((value) => Number.isFinite(value));
    if (!filtered.length) {
        return {
            mean: null,
            min: null,
            max: null,
            stdDev: null
        };
    }
    const mean = average(filtered);
    let variance = 0;
    for (let index = 0; index < filtered.length; index += 1) {
        const delta = filtered[index] - mean;
        variance += delta * delta;
    }
    variance /= filtered.length;
    return {
        mean,
        min: Math.min(...filtered),
        max: Math.max(...filtered),
        stdDev: Math.sqrt(variance)
    };
};

const deepClone = (value) => {
    if (value === null || value === undefined) {
        return value;
    }
    if (typeof structuredClone === 'function') {
        try {
            return structuredClone(value);
        } catch (error) {
            console.warn('CalibrationDatasetBuilder structuredClone fallback triggered', error);
        }
    }
    try {
        return JSON.parse(JSON.stringify(value));
    } catch (error) {
        console.warn('CalibrationDatasetBuilder JSON clone fallback failed', error);
    }
    if (Array.isArray(value)) {
        return value.slice();
    }
    if (typeof value === 'object') {
        return { ...value };
    }
    return value;
};

const gatherVoiceMetric = (voices = [], key) => {
    if (!Array.isArray(voices)) {
        return [];
    }
    return voices
        .map((voice) => (voice && Number.isFinite(voice[key]) ? voice[key] : null))
        .filter((value) => Number.isFinite(value));
};

const flattenCarrierEnergies = (carrierMatrix) => {
    if (!Array.isArray(carrierMatrix)) {
        return [];
    }
    const values = [];
    carrierMatrix.forEach((voice) => {
        if (!Array.isArray(voice)) {
            return;
        }
        voice.forEach((cell) => {
            if (cell && Number.isFinite(cell.energy)) {
                values.push(cell.energy);
            }
        });
    });
    return values;
};

const computeGateRatio = (carrierMatrix) => {
    if (!Array.isArray(carrierMatrix)) {
        return null;
    }
    let gateCount = 0;
    let total = 0;
    carrierMatrix.forEach((voice) => {
        if (!Array.isArray(voice)) {
            return;
        }
        voice.forEach((cell) => {
            if (!cell) {
                return;
            }
            if (cell.gate) {
                gateCount += 1;
            }
            total += 1;
        });
    });
    if (!total) {
        return null;
    }
    return clamp(gateCount / total, 0, 1);
};

const computeVisualContinuumParity = (sample) => {
    const values = Array.isArray(sample?.values) ? sample.values : [];
    const visualStats = stats(values);
    const continuumVoices = Array.isArray(sample?.continuum?.voices)
        ? sample.continuum.voices
        : [];
    const gridStats = stats(gatherVoiceMetric(continuumVoices, 'gridEnergy'));
    const carrierStats = stats(gatherVoiceMetric(continuumVoices, 'carrierEnergy'));
    return {
        visualMean: visualStats.mean,
        visualStd: visualStats.stdDev,
        continuumGridMean: gridStats.mean,
        continuumGridStd: gridStats.stdDev,
        continuumCarrierMean: carrierStats.mean,
        visualContinuumDelta: Number.isFinite(visualStats.mean) && Number.isFinite(gridStats.mean)
            ? Math.abs(visualStats.mean - gridStats.mean)
            : null
    };
};

const computeCarrierParity = (sample) => {
    const carrierMatrix = sample?.signal?.carrierMatrix;
    const carrierEnergyStats = stats(flattenCarrierEnergies(carrierMatrix));
    const bitDensity = Number.isFinite(sample?.signal?.bitstream?.density)
        ? sample.signal.bitstream.density
        : null;
    const envelope = sample?.signal?.envelope || {};
    return {
        carrierEnergyMean: carrierEnergyStats.mean,
        carrierEnergyStd: carrierEnergyStats.stdDev,
        carrierGateRatio: computeGateRatio(carrierMatrix),
        bitDensity,
        envelopeResonance: Number.isFinite(envelope.resonance) ? envelope.resonance : null,
        envelopeCentroid: Number.isFinite(envelope.centroid) ? envelope.centroid : null
    };
};

const computeSpinorParity = (sample) => {
    const continuumVoices = Array.isArray(sample?.continuum?.voices)
        ? sample.continuum.voices
        : [];
    const alignmentStats = stats(gatherVoiceMetric(continuumVoices, 'continuumAlignment'));
    const entropyStats = stats(gatherVoiceMetric(continuumVoices, 'bitEntropy'));
    const uniformValues = sample?.uniforms && typeof sample.uniforms === 'object'
        ? Object.values(sample.uniforms).filter((value) => Number.isFinite(value))
        : [];
    const uniformMagnitude = uniformValues.length
        ? Math.sqrt(uniformValues.reduce((accumulator, value) => accumulator + value * value, 0))
        : null;
    const resonanceAggregate = sample?.resonance?.aggregate || {};
    return {
        continuumAlignmentMean: alignmentStats.mean,
        continuumAlignmentStd: alignmentStats.stdDev,
        bitEntropyMean: entropyStats.mean,
        uniformVectorMagnitude: uniformMagnitude,
        resonanceAggregateEnergy: Number.isFinite(resonanceAggregate.magnitude)
            ? resonanceAggregate.magnitude
            : null,
        spinorCoherence: Number.isFinite(resonanceAggregate.spinorCoherence)
            ? resonanceAggregate.spinorCoherence
            : null
    };
};

export const DEFAULT_PARITY_EVALUATORS = [
    computeVisualContinuumParity,
    computeCarrierParity,
    computeSpinorParity
];

const defaultStatus = () => {};

export class CalibrationDatasetBuilder {
    constructor({
        toolkit,
        parityEvaluators = DEFAULT_PARITY_EVALUATORS,
        onStatus = defaultStatus,
        metadata = {}
    } = {}) {
        if (!toolkit) {
            throw new Error('CalibrationDatasetBuilder requires a CalibrationToolkit instance.');
        }
        this.toolkit = toolkit;
        this.onStatus = typeof onStatus === 'function' ? onStatus : defaultStatus;
        this.metadata = { ...metadata };
        this.parityEvaluators = Array.isArray(parityEvaluators) && parityEvaluators.length
            ? parityEvaluators
            : DEFAULT_PARITY_EVALUATORS;
        this.plan = [];
        this.sequenceSummaries = [];
        this.samples = [];
        this.manifest = null;
    }

    planSequence(sequenceId, options = {}) {
        if (!sequenceId) {
            return this;
        }
        this.plan.push({ sequenceId, ...options });
        return this;
    }

    clearPlan() {
        this.plan = [];
        return this;
    }

    getPlan() {
        return this.plan.map((entry) => ({ ...entry }));
    }

    createDefaultPlan() {
        return this.toolkit.listSequences().map((sequence) => ({
            sequenceId: sequence.id,
            label: sequence.label,
            sampleRate: sequence.sampleRate,
            durationSeconds: sequence.durationSeconds
        }));
    }

    async runPlan(planInput = null) {
        const availableSequences = new Map(
            this.toolkit.listSequences().map((sequence) => [sequence.id, sequence])
        );
        const plan = Array.isArray(planInput) && planInput.length
            ? planInput
            : (this.plan.length ? this.plan : this.createDefaultPlan());
        if (!plan.length) {
            this.onStatus('Dataset plan is empty. Register calibration sequences before running.');
            return null;
        }
        this.sequenceSummaries = [];
        this.samples = [];
        this.manifest = null;
        for (let index = 0; index < plan.length; index += 1) {
            const entry = plan[index] || {};
            const descriptor = availableSequences.get(entry.sequenceId) || {};
            const label = entry.label || descriptor.label || entry.sequenceId || `sequence-${index + 1}`;
            this.onStatus(`Running dataset sequence: ${label}`);
            const started = this.toolkit.runSequence(entry.sequenceId, {
                sampleRate: entry.sampleRate,
                durationSeconds: entry.durationSeconds,
                suppressCallbacks: entry.suppressCallbacks
            });
            if (!started) {
                this.sequenceSummaries.push({
                    id: entry.sequenceId,
                    label,
                    completed: false,
                    frames: 0,
                    sampleRate: entry.sampleRate || descriptor.sampleRate || null,
                    durationSeconds: entry.durationSeconds || descriptor.durationSeconds || null,
                    status: 'unavailable'
                });
                continue;
            }
            const result = await this.toolkit.waitForCompletion();
            const capturedSamples = Array.isArray(result.samples) ? result.samples : [];
            const enriched = capturedSamples.map((sample) => this.enrichSample(sample));
            this.samples.push(...enriched);
            this.sequenceSummaries.push({
                id: entry.sequenceId,
                label,
                completed: Boolean(result.completed),
                frames: enriched.length,
                sampleRate: entry.sampleRate || descriptor.sampleRate || null,
                durationSeconds: entry.durationSeconds || descriptor.durationSeconds || null,
                status: result.completed ? 'complete' : 'incomplete'
            });
        }
        this.manifest = this.buildManifest();
        this.onStatus(`Dataset plan complete – ${this.samples.length} frames captured.`);
        return this.manifest;
    }

    evaluateParity(sample) {
        const metrics = {};
        this.parityEvaluators.forEach((evaluator) => {
            try {
                const result = evaluator(sample) || {};
                Object.keys(result).forEach((key) => {
                    const value = result[key];
                    if (Number.isFinite(value)) {
                        metrics[key] = value;
                    }
                });
            } catch (error) {
                console.warn('CalibrationDatasetBuilder parity evaluator failed', error);
            }
        });
        return metrics;
    }

    computeSampleScore(parityMetrics) {
        const components = [];
        if (Number.isFinite(parityMetrics.visualContinuumDelta)) {
            components.push(clamp(1 - Math.min(Math.abs(parityMetrics.visualContinuumDelta), 1), 0, 1));
        }
        if (Number.isFinite(parityMetrics.continuumAlignmentMean)) {
            components.push(clamp(parityMetrics.continuumAlignmentMean, 0, 1));
        }
        if (Number.isFinite(parityMetrics.carrierGateRatio)) {
            components.push(clamp(parityMetrics.carrierGateRatio, 0, 1));
        }
        if (Number.isFinite(parityMetrics.spinorCoherence)) {
            components.push(clamp(parityMetrics.spinorCoherence, 0, 1));
        }
        if (Number.isFinite(parityMetrics.envelopeResonance)) {
            components.push(clamp(parityMetrics.envelopeResonance / 2, 0, 1));
        }
        if (!components.length) {
            return null;
        }
        return components.reduce((accumulator, value) => accumulator + value, 0) / components.length;
    }

    enrichSample(sample) {
        const clone = deepClone(sample) || {};
        const parity = this.evaluateParity(sample);
        const score = this.computeSampleScore(parity);
        clone.parity = parity;
        if (Number.isFinite(score)) {
            clone.score = score;
        } else {
            clone.score = null;
        }
        return clone;
    }

    aggregateParity() {
        const aggregated = {};
        this.samples.forEach((sample) => {
            const parity = sample.parity || {};
            Object.keys(parity).forEach((key) => {
                const value = parity[key];
                if (!Number.isFinite(value)) {
                    return;
                }
                if (!aggregated[key]) {
                    aggregated[key] = {
                        sum: 0,
                        count: 0,
                        min: value,
                        max: value
                    };
                }
                const bucket = aggregated[key];
                bucket.sum += value;
                bucket.count += 1;
                bucket.min = Math.min(bucket.min, value);
                bucket.max = Math.max(bucket.max, value);
            });
        });
        const metrics = {};
        Object.keys(aggregated).forEach((key) => {
            const bucket = aggregated[key];
            metrics[key] = {
                average: bucket.sum / bucket.count,
                min: bucket.min,
                max: bucket.max,
                count: bucket.count
            };
        });
        return metrics;
    }

    computeDatasetScore(parityMetrics) {
        const components = [];
        const delta = parityMetrics.visualContinuumDelta;
        if (delta && Number.isFinite(delta.average)) {
            components.push(clamp(1 - Math.min(Math.abs(delta.average), 1), 0, 1));
        }
        const alignment = parityMetrics.continuumAlignmentMean;
        if (alignment && Number.isFinite(alignment.average)) {
            components.push(clamp(alignment.average, 0, 1));
        }
        const gates = parityMetrics.carrierGateRatio;
        if (gates && Number.isFinite(gates.average)) {
            components.push(clamp(gates.average, 0, 1));
        }
        const coherence = parityMetrics.spinorCoherence;
        if (coherence && Number.isFinite(coherence.average)) {
            components.push(clamp(coherence.average, 0, 1));
        }
        const resonance = parityMetrics.envelopeResonance;
        if (resonance && Number.isFinite(resonance.average)) {
            components.push(clamp(resonance.average / 2, 0, 1));
        }
        if (!components.length) {
            return null;
        }
        return components.reduce((accumulator, value) => accumulator + value, 0) / components.length;
    }

    buildManifest({ includeSamples = true } = {}) {
        const parity = this.aggregateParity();
        const manifest = {
            generatedAt: new Date().toISOString(),
            metadata: { ...this.metadata },
            totals: {
                sequenceCount: this.sequenceSummaries.length,
                sampleCount: this.samples.length
            },
            sequences: this.sequenceSummaries.map((sequence) => ({ ...sequence })),
            parity,
            sampleScoreAverage: (() => {
                const scores = this.samples
                    .map((sample) => (Number.isFinite(sample.score) ? sample.score : null))
                    .filter((value) => Number.isFinite(value));
                return scores.length ? average(scores) : null;
            })()
        };
        manifest.score = this.computeDatasetScore(parity);
        if (includeSamples) {
            manifest.samples = this.samples.map((sample) => deepClone(sample));
        }
        return manifest;
    }

    getLastManifest({ includeSamples = true } = {}) {
        if (!this.manifest) {
            return null;
        }
        if (includeSamples) {
            return deepClone(this.manifest);
        }
        const clone = deepClone(this.manifest);
        if (clone) {
            delete clone.samples;
        }
        return clone;
    }
}
