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

const lerp = (a, b, t) => a + (b - a) * clamp(t, 0, 1);

const cloneValues = (values) => {
    if (!values) {
        return [];
    }
    if (Array.isArray(values)) {
        return values.slice();
    }
    if (typeof values.length === 'number') {
        return Array.from(values);
    }
    return [];
};

const cloneUniforms = (uniforms) => {
    if (!uniforms || typeof uniforms !== 'object') {
        return {};
    }
    return { ...uniforms };
};

const structuredCloneSafe = (value) => {
    if (typeof structuredClone === 'function') {
        try {
            return structuredClone(value);
        } catch (error) {
            console.warn('CalibrationToolkit structuredClone fallback triggered', error);
        }
    }
    try {
        return JSON.parse(JSON.stringify(value));
    } catch (error) {
        console.warn('CalibrationToolkit JSON clone fallback failed', error);
    }
    return null;
};

const cloneAnalysisPayload = (analysis) => {
    if (!analysis || typeof analysis !== 'object') {
        return null;
    }
    const payload = {
        summary: typeof analysis.summary === 'string' ? analysis.summary : null,
        transport: analysis.transport && typeof analysis.transport === 'object'
            ? analysis.transport
            : null,
        quaternion: analysis.quaternion && typeof analysis.quaternion === 'object'
            ? analysis.quaternion
            : null,
        spinor: analysis.spinor && typeof analysis.spinor === 'object'
            ? analysis.spinor
            : null,
        resonance: analysis.resonance && typeof analysis.resonance === 'object'
            ? analysis.resonance
            : null,
        signal: analysis.signal && typeof analysis.signal === 'object'
            ? analysis.signal
            : null,
        transduction: analysis.transduction && typeof analysis.transduction === 'object'
            ? analysis.transduction
            : null,
        manifold: analysis.manifold && typeof analysis.manifold === 'object'
            ? analysis.manifold
            : null,
        topology: analysis.topology && typeof analysis.topology === 'object'
            ? analysis.topology
            : null,
        continuum: analysis.continuum && typeof analysis.continuum === 'object'
            ? analysis.continuum
            : null,
        lattice: analysis.lattice && typeof analysis.lattice === 'object'
            ? analysis.lattice
            : null,
        constellation: analysis.constellation && typeof analysis.constellation === 'object'
            ? analysis.constellation
            : null
    };
    return structuredCloneSafe(payload);
};

const defaultSequenceFrames = ({ step, totalSteps, progress, baseFrequency = 0.2 }) => {
    const theta = progress * Math.PI * 2;
    const phi = progress * Math.PI * 4;
    const data = Array.from({ length: 32 }, (_, index) => {
        const ratio = index / 32;
        const spin = Math.sin(theta + ratio * Math.PI * 0.5);
        const weave = Math.cos(phi + ratio * Math.PI * 0.25);
        const modulation = Math.sin(theta * 0.5 + index * baseFrequency);
        return clamp(0.5 + 0.35 * spin + 0.25 * weave + 0.15 * modulation, 0, 1);
    });
    const uniforms = {
        u_rotXY: Math.sin(theta * 0.5) * Math.PI * 0.35,
        u_rotXZ: Math.cos(theta * 0.75) * Math.PI * 0.25,
        u_rotXW: Math.sin(phi * 0.5) * Math.PI * 0.4,
        u_rotYZ: Math.cos(phi * 0.25) * Math.PI * 0.3,
        u_rotYW: Math.sin(theta * 0.33 + phi * 0.12) * Math.PI * 0.28,
        u_rotZW: Math.cos(theta * 0.42 - phi * 0.18) * Math.PI * 0.32,
        u_morphFactor: lerp(0.35, 0.78, (Math.sin(theta) + 1) * 0.5),
        u_patternIntensity: lerp(0.2, 0.85, (Math.cos(phi) + 1) * 0.5),
        u_gridDensity: lerp(2.5, 7.5, progress),
        u_shellWidth: lerp(0.18, 0.35, (Math.sin(theta * 0.33) + 1) * 0.5),
        u_tetraThickness: lerp(0.12, 0.28, (Math.cos(phi * 0.42) + 1) * 0.5)
    };
    return { values: data, uniforms };
};

const DEFAULT_CALIBRATION_SEQUENCES = [
    {
        id: 'hopf-orbit',
        label: 'Hopf Orbit Sweep',
        description: 'Sweeps quaternion planes along a Hopf fiber to exercise sonic resonance tracking.',
        durationSeconds: 8,
        sampleRate: 30,
        frame: ({ step, totalSteps, progress }) => {
            const base = defaultSequenceFrames({ step, totalSteps, progress });
            const drift = Math.sin(progress * Math.PI * 6) * Math.PI * 0.2;
            return {
                values: base.values,
                uniforms: {
                    ...base.uniforms,
                    u_rotXY: base.uniforms.u_rotXY + drift,
                    u_rotXZ: base.uniforms.u_rotXZ - drift * 0.5,
                    u_rotYW: base.uniforms.u_rotYW + drift * 0.33
                }
            };
        }
    },
    {
        id: 'flux-ramp',
        label: 'Flux Continuum Ramp',
        description: 'Ramps carrier lattices and flux continua to validate manifold alignment metrics.',
        durationSeconds: 6,
        sampleRate: 24,
        frame: ({ progress }) => {
            const theta = progress * Math.PI * 2;
            const values = Array.from({ length: 32 }, (_, index) => {
                const harmonics = Math.sin(theta * (1 + index * 0.05));
                const envelope = Math.cos(theta * 0.5 + index * 0.1);
                return clamp(0.5 + 0.4 * harmonics + 0.1 * envelope, 0, 1);
            });
            return {
                values,
                uniforms: {
                    u_rotXY: Math.sin(theta) * Math.PI * 0.25,
                    u_rotXZ: Math.sin(theta * 0.5) * Math.PI * 0.18,
                    u_rotXW: Math.cos(theta * 0.75) * Math.PI * 0.28,
                    u_rotYZ: Math.sin(theta * 0.9) * Math.PI * 0.22,
                    u_rotYW: Math.cos(theta * 0.33) * Math.PI * 0.3,
                    u_rotZW: Math.sin(theta * 0.67) * Math.PI * 0.34,
                    u_morphFactor: lerp(0.25, 0.85, progress),
                    u_patternIntensity: lerp(0.15, 0.95, Math.pow(progress, 0.85)),
                    u_gridDensity: lerp(3.5, 8.5, Math.pow(progress, 0.65))
                }
            };
        }
    },
    {
        id: 'spinor-coherence',
        label: 'Spinor Coherence Pulse',
        description: 'Pulses quaternion bridges to stress-test spinor coherence telemetry.',
        durationSeconds: 10,
        sampleRate: 20,
        frame: ({ progress }) => {
            const pulses = Math.sin(progress * Math.PI * 10);
            const modulation = Math.sin(progress * Math.PI * 4);
            const values = Array.from({ length: 32 }, (_, index) => {
                const offset = index / 32;
                const gate = Math.sin(progress * Math.PI * 6 + offset * Math.PI);
                return clamp(0.5 + 0.45 * pulses * gate + 0.2 * modulation, 0, 1);
            });
            const pulseFactor = (Math.sin(progress * Math.PI * 8) + 1) * 0.5;
            return {
                values,
                uniforms: {
                    u_rotXY: Math.sin(progress * Math.PI * 6) * Math.PI * 0.38,
                    u_rotXZ: Math.cos(progress * Math.PI * 5) * Math.PI * 0.24,
                    u_rotXW: Math.sin(progress * Math.PI * 7) * Math.PI * 0.3,
                    u_rotYZ: Math.cos(progress * Math.PI * 3) * Math.PI * 0.26,
                    u_rotYW: Math.sin(progress * Math.PI * 4) * Math.PI * 0.22,
                    u_rotZW: Math.cos(progress * Math.PI * 2) * Math.PI * 0.18,
                    u_morphFactor: lerp(0.42, 0.92, pulseFactor),
                    u_patternIntensity: lerp(0.18, 0.88, pulseFactor),
                    u_gridDensity: lerp(2.8, 6.2, pulseFactor)
                }
            };
        }
    }
];

export class CalibrationToolkit {
    constructor({
        applyDataArray,
        captureFrame,
        getSonicAnalysis,
        onStatus,
        onSample,
        sequences = DEFAULT_CALIBRATION_SEQUENCES,
        defaultSampleRate = 30
    } = {}) {
        this.applyDataArray = typeof applyDataArray === 'function' ? applyDataArray : () => {};
        this.captureFrame = typeof captureFrame === 'function' ? captureFrame : () => null;
        this.getSonicAnalysis = typeof getSonicAnalysis === 'function' ? getSonicAnalysis : () => null;
        this.onStatus = typeof onStatus === 'function' ? onStatus : () => {};
        this.onSample = typeof onSample === 'function' ? onSample : () => {};
        this.defaultSampleRate = Number.isFinite(defaultSampleRate) ? defaultSampleRate : 30;
        this.sequences = new Map();
        sequences.forEach((sequence) => this.registerSequence(sequence));
        this.timer = null;
        this.active = null;
        this.samples = [];
        this.completionPromise = null;
        this.completionResolver = null;
    }

    registerSequence(sequence) {
        if (!sequence || typeof sequence !== 'object') {
            return false;
        }
        if (!sequence.id) {
            sequence.id = `sequence-${this.sequences.size + 1}`;
        }
        this.sequences.set(sequence.id, { ...sequence });
        return true;
    }

    listSequences() {
        return Array.from(this.sequences.values()).map((sequence) => ({
            id: sequence.id,
            label: sequence.label,
            description: sequence.description,
            durationSeconds: sequence.durationSeconds,
            sampleRate: sequence.sampleRate
        }));
    }

    getStatus() {
        if (!this.active) {
            return {
                running: false,
                sequenceId: null,
                progress: 0,
                step: 0,
                totalSteps: 0,
                sampleRate: this.defaultSampleRate,
                samplesCaptured: this.samples.length
            };
        }
        return {
            running: Boolean(this.timer),
            sequenceId: this.active.sequence.id,
            progress: clamp(this.active.step / Math.max(1, this.active.totalSteps - 1), 0, 1),
            step: this.active.step,
            totalSteps: this.active.totalSteps,
            sampleRate: this.active.sampleRate,
            samplesCaptured: this.samples.length
        };
    }

    clearResults() {
        this.samples = [];
    }

    getResults() {
        return this.samples.map((sample) => ({ ...sample }));
    }

    stop({ completed = false } = {}) {
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = null;
        }
        if (this.active) {
            const label = this.active.sequence.label || this.active.sequence.id;
            this.onStatus(`${label} calibration ${completed ? 'complete' : 'stopped'}. Captured ${this.samples.length} frames.`);
        }
        this.active = null;
        if (this.completionResolver) {
            const resolver = this.completionResolver;
            this.completionResolver = null;
            const payload = {
                completed,
                samples: this.getResults()
            };
            resolver(payload);
            this.completionPromise = null;
        }
    }

    runSequence(sequenceId, options = {}) {
        const sequence = this.sequences.get(sequenceId) || this.sequences.values().next().value;
        if (!sequence) {
            this.onStatus('No calibration sequences registered.');
            return false;
        }
        this.stop();
        if (this.completionResolver) {
            const resolver = this.completionResolver;
            this.completionResolver = null;
            resolver({ completed: false, samples: this.getResults() });
            this.completionPromise = null;
        }
        const sampleRate = Number.isFinite(options.sampleRate)
            ? options.sampleRate
            : Number.isFinite(sequence.sampleRate)
                ? sequence.sampleRate
                : this.defaultSampleRate;
        const durationSeconds = Number.isFinite(options.durationSeconds)
            ? options.durationSeconds
            : Number.isFinite(sequence.durationSeconds)
                ? sequence.durationSeconds
                : 5;
        const totalSteps = Math.max(1, Math.round(sampleRate * durationSeconds));
        this.active = {
            sequence,
            sampleRate,
            totalSteps,
            step: 0,
            startedAt: performance.now()
        };
        this.samples = [];
        this.completionPromise = new Promise((resolve) => {
            this.completionResolver = resolve;
        });
        const label = sequence.label || sequence.id;
        this.onStatus(`Running ${label} (${totalSteps} frames @ ${sampleRate.toFixed(1)} Hz)…`);

        const tick = () => {
            if (!this.active) {
                return;
            }
            const { step, totalSteps: steps } = this.active;
            const progress = clamp(steps > 1 ? step / (steps - 1) : 1, 0, 1);
            const frameFactory = typeof sequence.frame === 'function' ? sequence.frame : defaultSequenceFrames;
            const frame = frameFactory({
                step,
                totalSteps: steps,
                progress,
                sampleRate,
                startedAt: this.active.startedAt
            }) || {};
            const values = cloneValues(frame.values || frame.data);
            const uniforms = cloneUniforms(frame.uniforms);
            const transport = {
                playing: true,
                loop: false,
                progress,
                frameIndex: step,
                frameCount: steps,
                mode: 'calibration'
            };
            this.applyDataArray(values, {
                source: 'calibration',
                uniformOverride: uniforms,
                suppressCallbacks: options.suppressCallbacks !== false,
                playbackFrame: { progress },
                transportOverride: transport,
                analysisMetadata: {
                    calibration: {
                        sequenceId: sequence.id,
                        step,
                        totalSteps: steps,
                        progress
                    }
                }
            });
            const screenshot = frame.capture === false ? null : this.captureFrame();
            const analysis = this.getSonicAnalysis();
            const analysisClone = cloneAnalysisPayload(analysis);
            const sample = {
                sequenceId: sequence.id,
                label,
                step,
                totalSteps: steps,
                progress,
                timestamp: performance.now(),
                values,
                uniforms,
                sampleRate,
                transport,
                summary: analysisClone ? analysisClone.summary : (analysis ? analysis.summary : null),
                signal: analysisClone ? analysisClone.signal : (analysis && analysis.signal ? structuredCloneSafe(analysis.signal) : null),
                continuum: analysisClone ? analysisClone.continuum : (analysis && analysis.continuum ? structuredCloneSafe(analysis.continuum) : null),
                resonance: analysisClone ? analysisClone.resonance : (analysis && analysis.resonance ? structuredCloneSafe(analysis.resonance) : null),
                manifold: analysisClone ? analysisClone.manifold : (analysis && analysis.manifold ? structuredCloneSafe(analysis.manifold) : null),
                topology: analysisClone ? analysisClone.topology : (analysis && analysis.topology ? structuredCloneSafe(analysis.topology) : null),
                lattice: analysisClone ? analysisClone.lattice : (analysis && analysis.lattice ? structuredCloneSafe(analysis.lattice) : null),
                constellation: analysisClone ? analysisClone.constellation : (analysis && analysis.constellation ? structuredCloneSafe(analysis.constellation) : null),
                transduction: analysisClone ? analysisClone.transduction : (analysis && analysis.transduction ? structuredCloneSafe(analysis.transduction) : null),
                screenshot,
                analysis: analysisClone
            };
            this.samples.push(sample);
            this.onSample(sample);
            this.active.step += 1;
            if (this.active.step >= steps) {
                this.stop({ completed: true });
            }
        };

        this.timer = setInterval(tick, 1000 / sampleRate);
        tick();
        return true;
    }

    waitForCompletion() {
        if (this.completionPromise) {
            return this.completionPromise;
        }
        return Promise.resolve({
            completed: false,
            samples: this.getResults()
        });
    }
}

export { DEFAULT_CALIBRATION_SEQUENCES };
