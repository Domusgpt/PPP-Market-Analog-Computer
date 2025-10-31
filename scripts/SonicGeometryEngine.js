import { computeQuaternionBridge } from './QuaternionSonicBridge.js';
import { deriveSpinorHarmonics } from './QuaternionHarmonicCoupler.js';
import { buildSpinorResonanceAtlas, cloneSpinorResonanceAtlas } from './SpinorResonanceAtlas.js';
import { buildSpinorSignalFabric, cloneSpinorSignalFabric } from './SpinorSignalFabric.js';
import { buildSpinorTransductionGrid, cloneSpinorTransductionGrid } from './SpinorTransductionGrid.js';
import { buildSpinorMetricManifold, cloneSpinorMetricManifold } from './SpinorMetricManifold.js';
import { buildSpinorTopologyWeave, cloneSpinorTopologyWeave } from './SpinorTopologyWeave.js';
import { buildSpinorFluxContinuum, cloneSpinorFluxContinuum } from './SpinorFluxContinuum.js';
import { buildSpinorContinuumLattice, cloneSpinorContinuumLattice } from './SpinorContinuumLattice.js';
import { buildSpinorContinuumConstellation, cloneSpinorContinuumConstellation } from './SpinorContinuumConstellation.js';
import { DATA_CHANNEL_COUNT } from './constants.js';

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

const DEFAULT_GEOMETRY_PAIRS = [
    ['u_rotXY', 'u_rotXW'],
    ['u_rotYZ', 'u_rotYW'],
    ['u_rotXZ', 'u_rotZW'],
    ['u_morphFactor', 'u_patternIntensity']
];

const DEFAULT_CARRIER_SETTINGS = [
    { label: 'sub', multiple: 0.5, spread: 0.35 },
    { label: 'prime', multiple: 1, spread: 0.5 },
    { label: 'hyper', multiple: 3.25, spread: 0.85 }
];

const cloneVoiceAnalysis = (voice) => {
    if (!voice || typeof voice !== 'object') {
        return null;
    }
    return {
        ...voice,
        carriers: Array.isArray(voice.carriers)
            ? voice.carriers.map((carrier) => ({ ...carrier }))
            : [],
        modulation: voice.modulation ? { ...voice.modulation } : null,
        quaternion: voice.quaternion ? { ...voice.quaternion } : null
    };
};

const safeNow = () => {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
        return performance.now();
    }
    return Date.now();
};

export class SonicGeometryEngine {
    constructor(options = {}) {
        const {
            voiceCount = 4,
            baseFrequency = 110,
            octaveSpan = 4,
            masterLevel = 0.38,
            voiceGainBase = 0.025,
            voiceGainRange = 0.12,
            geometryBlend = 0.45,
            rotationNormalization = Math.PI,
            transitionTime = 0.18,
            driftFactor = 0.35,
            geometryPairs = DEFAULT_GEOMETRY_PAIRS,
            carrierSettings = DEFAULT_CARRIER_SETTINGS,
            carrierBlend = 0.62,
            carrierSpread = 0.85,
            harmonicWarp = 0.45,
            gateThreshold = 0.18,
            amRateBase = 1.75,
            amRateRange = 38,
            amOrbitRate = 5.5,
            amDepthBase = 0.12,
            amDepthRange = 0.9,
            fmRateBase = 7.5,
            fmRateRange = 92,
            fmRateGeometry = 23,
            fmDepthBase = 18,
            fmDepthRange = 180,
            pulseRateBase = 11,
            pulseRateRange = 64,
            sequenceResolution = 32,
            carrierEnergyFloor = 0.18,
            spectralFocus = 0.5,
            ultraHarmonicBoost = 0.6,
            quaternionBlend = 0.32,
            quaternionDrift = 0.28,
            spinorFrequencyGain = 0.55,
            spinorPanGain = 0.85,
            spinorPhaseGain = 0.4,
            contextFactory
        } = options;

        const audioFactory = typeof contextFactory === 'function'
            ? contextFactory
            : (() => {
                if (typeof window === 'undefined') {
                    return null;
                }
                const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
                if (!AudioContextCtor) {
                    return null;
                }
                return () => new AudioContextCtor();
            })();

        this.contextFactory = typeof audioFactory === 'function' ? audioFactory : null;
        this.audioSupported = Boolean(this.contextFactory);
        this.supported = true;
        this.voiceCount = Math.max(1, Math.floor(voiceCount));
        this.baseFrequency = Math.max(20, baseFrequency);
        this.octaveSpan = Math.max(1, octaveSpan);
        this.masterLevel = clamp(masterLevel, 0.01, 1);
        this.voiceGainBase = Math.max(0, voiceGainBase);
        this.voiceGainRange = Math.max(0, voiceGainRange);
        this.geometryBlend = clamp(geometryBlend, 0, 1);
        this.rotationNormalization = Math.max(0.001, rotationNormalization);
        this.transitionTime = Math.max(0.01, transitionTime);
        this.driftFactor = clamp(driftFactor, 0, 1);
        this.geometryPairs = Array.isArray(geometryPairs) && geometryPairs.length
            ? geometryPairs
            : DEFAULT_GEOMETRY_PAIRS;
        const normalizedCarrierSettings = Array.isArray(carrierSettings) && carrierSettings.length
            ? carrierSettings.map((carrier, index) => {
                const fallback = DEFAULT_CARRIER_SETTINGS[index % DEFAULT_CARRIER_SETTINGS.length];
                if (!carrier || typeof carrier !== 'object') {
                    return { ...fallback };
                }
                const multiple = Number.isFinite(carrier.multiple)
                    ? Math.max(0.0625, carrier.multiple)
                    : fallback.multiple;
                const spread = Number.isFinite(carrier.spread)
                    ? carrier.spread
                    : fallback.spread;
                const label = typeof carrier.label === 'string' && carrier.label.trim()
                    ? carrier.label.trim()
                    : fallback.label || `band-${index + 1}`;
                return { label, multiple, spread };
            })
            : DEFAULT_CARRIER_SETTINGS.map((carrier) => ({ ...carrier }));
        this.carrierSettings = normalizedCarrierSettings;
        this.carrierBlend = clamp(carrierBlend, 0, 1);
        this.carrierSpread = Math.max(0, carrierSpread);
        this.harmonicWarp = clamp(harmonicWarp, -4, 4);
        this.gateThreshold = clamp(gateThreshold, 0, 1);
        this.amRateBase = Math.max(0.01, amRateBase);
        this.amRateRange = Math.max(0, amRateRange);
        this.amOrbitRate = Math.max(0, amOrbitRate);
        this.amDepthBase = clamp(amDepthBase, 0, 1);
        this.amDepthRange = Math.max(0, amDepthRange);
        this.fmRateBase = Math.max(0.01, fmRateBase);
        this.fmRateRange = Math.max(0, fmRateRange);
        this.fmRateGeometry = Math.max(0, fmRateGeometry);
        this.fmDepthBase = fmDepthBase;
        this.fmDepthRange = Math.max(0, fmDepthRange);
        this.pulseRateBase = Math.max(0.01, pulseRateBase);
        this.pulseRateRange = Math.max(0, pulseRateRange);
        this.sequenceResolution = Math.max(4, Math.floor(sequenceResolution));
        this.carrierEnergyFloor = clamp(carrierEnergyFloor, 0, 1);
        this.spectralFocus = clamp(spectralFocus, 0, 1);
        this.ultraHarmonicBoost = clamp(ultraHarmonicBoost, 0, 4);
        this.quaternionBlend = clamp(quaternionBlend, 0, 1);
        this.quaternionDrift = clamp(quaternionDrift, 0, 1);
        this.spinorFrequencyGain = clamp(spinorFrequencyGain, 0, 2);
        this.spinorPanGain = clamp(spinorPanGain, 0, 2);
        this.spinorPhaseGain = clamp(spinorPhaseGain, 0, 2);

        this.audioContext = null;
        this.masterGain = null;
        this.voicePool = [];
        this.active = false;
        this.audioActive = false;
        const requestedMode = options.outputMode === 'analysis' ? 'analysis' : 'hybrid';
        this.outputMode = this.audioSupported ? requestedMode : 'analysis';
        this.transportState = {
            playing: false,
            progress: 0,
            mode: 'idle',
            loop: false,
            frameIndex: -1,
            frameCount: 0
        };
        this.lastFrame = null;
        this.lastSummary = '';
        this.lastAnalysis = null;
        this.lastQuaternion = null;
        this.lastSignal = null;
        this.lastTransduction = null;
        this.lastManifold = null;
        this.lastTopology = null;
        this.lastContinuum = null;
        this.lastLattice = null;
        this.lastConstellation = null;
        this.channelLimit = Math.max(1, Math.floor(options.channelLimit || DATA_CHANNEL_COUNT));
        this.normalizedScratch = new Float32Array(this.channelLimit);
        const scratchVoiceCount = Math.max(1, this.voiceCount);
        this.voiceScratch = {
            size: scratchVoiceCount,
            sums: new Float32Array(scratchVoiceCount),
            counts: new Float32Array(scratchVoiceCount),
            averages: new Float32Array(scratchVoiceCount),
            energies: new Float32Array(scratchVoiceCount)
        };
        this.performanceWindowSize = Number.isFinite(options.performanceWindowSize)
            ? Math.max(1, Math.floor(options.performanceWindowSize))
            : 240;
        this.performanceWindow = [];
        this.performanceAccumulator = 0;
        this.performanceMin = Infinity;
        this.performanceMax = 0;
        this.performanceAverage = 0;
        this.lastFrameDuration = 0;
        this.frameBudgetMs = Number.isFinite(options.frameBudgetMs) ? options.frameBudgetMs : 16.7;
    }

    isSupported() {
        return this.supported;
    }

    hasAudioSupport() {
        return this.audioSupported;
    }

    isActive() {
        return this.active;
    }

    getOutputMode() {
        return this.outputMode;
    }

    getState() {
        return {
            supported: this.supported,
            active: this.active,
            voiceCount: this.voicePool.length || this.voiceCount,
            outputMode: this.outputMode,
            audio: {
                supported: this.audioSupported,
                active: this.audioActive,
                contextState: this.audioContext ? this.audioContext.state : 'uninitialized'
            },
            performance: this.getPerformanceMetrics(),
            transport: { ...this.transportState },
            lastSummary: this.lastSummary,
            lastAnalysis: this.lastAnalysis
                ? {
                    ...this.lastAnalysis,
                    voices: this.lastAnalysis.voices.map((voice) => cloneVoiceAnalysis(voice)).filter(Boolean),
                    transmission: this.lastAnalysis.transmission
                        ? {
                            ...this.lastAnalysis.transmission,
                            carriers: Array.isArray(this.lastAnalysis.transmission.carriers)
                                ? this.lastAnalysis.transmission.carriers.map((carrier) => ({
                                    ...carrier,
                                    carriers: Array.isArray(carrier.carriers)
                                        ? carrier.carriers.map((band) => ({ ...band }))
                                        : []
                                }))
                                : [],
                            resonance: this.lastAnalysis.transmission.resonance
                                ? cloneSpinorResonanceAtlas(this.lastAnalysis.transmission.resonance)
                                : null,
                            signal: this.lastAnalysis.transmission.signal
                                ? cloneSpinorSignalFabric(this.lastAnalysis.transmission.signal)
                                : null,
                            manifold: this.lastAnalysis.transmission.manifold
                                ? cloneSpinorMetricManifold(this.lastAnalysis.transmission.manifold)
                                : null,
                            topology: this.lastAnalysis.transmission.topology
                                ? cloneSpinorTopologyWeave(this.lastAnalysis.transmission.topology)
                                : null,
                            continuum: this.lastAnalysis.transmission.continuum
                                ? cloneSpinorFluxContinuum(this.lastAnalysis.transmission.continuum)
                                : null,
                            lattice: this.lastAnalysis.transmission.lattice
                                ? cloneSpinorContinuumLattice(this.lastAnalysis.transmission.lattice)
                                : null,
                            constellation: this.lastAnalysis.transmission.constellation
                                ? cloneSpinorContinuumConstellation(this.lastAnalysis.transmission.constellation)
                                : null
                        }
                        : null,
                    quaternion: this.lastAnalysis.quaternion
                        ? {
                            ...this.lastAnalysis.quaternion,
                            left: Array.isArray(this.lastAnalysis.quaternion.left)
                                ? this.lastAnalysis.quaternion.left.slice()
                                : [],
                            right: Array.isArray(this.lastAnalysis.quaternion.right)
                                ? this.lastAnalysis.quaternion.right.slice()
                                : [],
                            bridgeVector: Array.isArray(this.lastAnalysis.quaternion.bridgeVector)
                                ? this.lastAnalysis.quaternion.bridgeVector.slice()
                                : [],
                            normalizedBridge: Array.isArray(this.lastAnalysis.quaternion.normalizedBridge)
                                ? this.lastAnalysis.quaternion.normalizedBridge.slice()
                                : [],
                            hopfFiber: Array.isArray(this.lastAnalysis.quaternion.hopfFiber)
                                ? this.lastAnalysis.quaternion.hopfFiber.slice()
                                : []
                        }
                        : null,
                resonance: this.lastAnalysis.resonance
                    ? cloneSpinorResonanceAtlas(this.lastAnalysis.resonance)
                    : null,
                signal: this.lastAnalysis.signal
                    ? cloneSpinorSignalFabric(this.lastAnalysis.signal)
                    : null,
                manifold: this.lastAnalysis.manifold
                    ? cloneSpinorMetricManifold(this.lastAnalysis.manifold)
                    : null,
                topology: this.lastAnalysis.topology
                    ? cloneSpinorTopologyWeave(this.lastAnalysis.topology)
                    : null,
                continuum: this.lastAnalysis.continuum
                    ? cloneSpinorFluxContinuum(this.lastAnalysis.continuum)
                    : null,
                lattice: this.lastAnalysis.lattice
                    ? cloneSpinorContinuumLattice(this.lastAnalysis.lattice)
                    : null,
                constellation: this.lastAnalysis.constellation
                    ? cloneSpinorContinuumConstellation(this.lastAnalysis.constellation)
                    : null
            }
            : null,
            lastSignal: this.lastSignal ? cloneSpinorSignalFabric(this.lastSignal) : null,
            lastTransduction: this.lastTransduction ? cloneSpinorTransductionGrid(this.lastTransduction) : null,
            lastManifold: this.lastManifold ? cloneSpinorMetricManifold(this.lastManifold) : null,
            lastTopology: this.lastTopology ? cloneSpinorTopologyWeave(this.lastTopology) : null,
            lastContinuum: this.lastContinuum ? cloneSpinorFluxContinuum(this.lastContinuum) : null,
            lastLattice: this.lastLattice ? cloneSpinorContinuumLattice(this.lastLattice) : null,
            lastConstellation: this.lastConstellation
                ? cloneSpinorContinuumConstellation(this.lastConstellation)
                : null
        };
    }

    getPerformanceMetrics() {
        const samples = this.performanceWindow.length;
        const average = samples ? this.performanceAverage : 0;
        const min = samples ? this.performanceMin : 0;
        const max = samples ? this.performanceMax : 0;
        const utilization = this.frameBudgetMs > 0 && Number.isFinite(average)
            ? average / this.frameBudgetMs
            : 0;
        return {
            lastFrameMs: this.lastFrameDuration,
            averageFrameMs: average,
            minFrameMs: Number.isFinite(min) ? min : 0,
            maxFrameMs: Number.isFinite(max) ? max : 0,
            samples,
            budgetMs: this.frameBudgetMs,
            budgetUtilization: utilization
        };
    }

    async enable() {
        if (!this.supported) {
            return false;
        }
        this.active = true;
        if (!this.#shouldUseAudio()) {
            this.audioActive = false;
            this.#updateMasterLevel();
            if (this.lastSummary) {
                return true;
            }
            this.lastSummary = 'Sonic geometry analysis active. Audio muted by mode selection.';
            return true;
        }
        const activated = await this.#activateAudio();
        if (!activated) {
            this.audioActive = false;
            if (this.outputMode !== 'analysis') {
                this.outputMode = 'analysis';
            }
            this.lastSummary = 'Sonic geometry analysis active. Audio unavailable in this environment.';
            return true;
        }
        this.#updateMasterLevel();
        if (this.lastAnalysis) {
            this.#renderAudio(this.lastAnalysis, { immediate: true });
        } else if (this.lastFrame) {
            const analysis = this.#analyzeFrame(this.lastFrame.values, this.lastFrame.metadata);
            if (analysis) {
                this.lastAnalysis = analysis;
                this.lastSummary = analysis.summary;
                this.#renderAudio(analysis, { immediate: true });
            }
        }
        return true;
    }

    disable({ immediate = false } = {}) {
        this.#muteAudio({ immediate });
        this.active = false;
        this.transportState = {
            ...this.transportState,
            playing: false,
            mode: 'muted'
        };
        this.lastSummary = 'Sonic geometry analysis muted.';
    }

    setTransportState(state = {}) {
        if (!state || typeof state !== 'object') {
            return;
        }
        const nextState = {
            ...this.transportState,
            ...state
        };
        nextState.playing = Boolean(state.playing);
        nextState.loop = Boolean(state.loop);
        if (Number.isFinite(state.progress)) {
            nextState.progress = clamp(state.progress, 0, 1);
        }
        if (!state.mode) {
            nextState.mode = this.transportState.mode;
        }
        if (!Number.isFinite(nextState.frameIndex)) {
            nextState.frameIndex = -1;
        }
        if (!Number.isFinite(nextState.frameCount)) {
            nextState.frameCount = 0;
        }
        this.transportState = nextState;
        if (this.lastAnalysis) {
            const updatedAnalysis = {
                ...this.lastAnalysis,
                transport: { ...nextState }
            };
            if (this.lastAnalysis.topology) {
                updatedAnalysis.topology = {
                    ...this.lastAnalysis.topology,
                    progress: Number.isFinite(nextState.progress)
                        ? clamp(nextState.progress, 0, 1)
                        : this.lastAnalysis.topology.progress,
                    transport: {
                        playing: Boolean(nextState.playing),
                        mode: typeof nextState.mode === 'string' ? nextState.mode : 'idle'
                    }
                };
            }
            if (this.lastAnalysis.continuum) {
                updatedAnalysis.continuum = {
                    ...this.lastAnalysis.continuum,
                    progress: Number.isFinite(nextState.progress)
                        ? clamp(nextState.progress, 0, 1)
                        : this.lastAnalysis.continuum.progress,
                    transport: {
                        playing: Boolean(nextState.playing),
                        mode: typeof nextState.mode === 'string' ? nextState.mode : 'idle'
                    }
                };
            }
            if (this.lastAnalysis.lattice) {
                updatedAnalysis.lattice = {
                    ...this.lastAnalysis.lattice,
                    progress: Number.isFinite(nextState.progress)
                        ? clamp(nextState.progress, 0, 1)
                        : this.lastAnalysis.lattice.progress,
                    transport: {
                        playing: Boolean(nextState.playing),
                        mode: typeof nextState.mode === 'string' ? nextState.mode : 'idle'
                    }
                };
            }
            if (this.lastAnalysis.constellation) {
                updatedAnalysis.constellation = {
                    ...this.lastAnalysis.constellation,
                    progress: Number.isFinite(nextState.progress)
                        ? clamp(nextState.progress, 0, 1)
                        : this.lastAnalysis.constellation.progress,
                    transport: {
                        playing: Boolean(nextState.playing),
                        mode: typeof nextState.mode === 'string' ? nextState.mode : 'idle'
                    }
                };
            }
            if (this.lastAnalysis.transmission) {
                const updatedTransmission = {
                    ...this.lastAnalysis.transmission
                };
                if (this.lastAnalysis.transmission.topology) {
                    updatedTransmission.topology = {
                        ...this.lastAnalysis.transmission.topology,
                        progress: Number.isFinite(nextState.progress)
                            ? clamp(nextState.progress, 0, 1)
                            : this.lastAnalysis.transmission.topology.progress,
                        transport: {
                            playing: Boolean(nextState.playing),
                            mode: typeof nextState.mode === 'string' ? nextState.mode : 'idle'
                        }
                    };
                }
                if (this.lastAnalysis.transmission.continuum) {
                    updatedTransmission.continuum = {
                        ...this.lastAnalysis.transmission.continuum,
                        progress: Number.isFinite(nextState.progress)
                            ? clamp(nextState.progress, 0, 1)
                            : this.lastAnalysis.transmission.continuum.progress,
                        transport: {
                            playing: Boolean(nextState.playing),
                            mode: typeof nextState.mode === 'string' ? nextState.mode : 'idle'
                        }
                    };
                }
                if (this.lastAnalysis.transmission.lattice) {
                    updatedTransmission.lattice = {
                        ...this.lastAnalysis.transmission.lattice,
                        progress: Number.isFinite(nextState.progress)
                            ? clamp(nextState.progress, 0, 1)
                            : this.lastAnalysis.transmission.lattice.progress,
                        transport: {
                            playing: Boolean(nextState.playing),
                            mode: typeof nextState.mode === 'string' ? nextState.mode : 'idle'
                        }
                    };
                }
                if (this.lastAnalysis.transmission.constellation) {
                    updatedTransmission.constellation = {
                        ...this.lastAnalysis.transmission.constellation,
                        progress: Number.isFinite(nextState.progress)
                            ? clamp(nextState.progress, 0, 1)
                            : this.lastAnalysis.transmission.constellation.progress,
                        transport: {
                            playing: Boolean(nextState.playing),
                            mode: typeof nextState.mode === 'string' ? nextState.mode : 'idle'
                        }
                    };
                }
                updatedAnalysis.transmission = updatedTransmission;
            }
            this.lastAnalysis = updatedAnalysis;
        }
        if (this.lastSignal) {
            this.lastSignal = {
                ...this.lastSignal,
                transport: { ...nextState }
            };
        }
        if (this.lastTransduction) {
            this.lastTransduction = {
                ...this.lastTransduction,
                progress: Number.isFinite(nextState.progress)
                    ? clamp(nextState.progress, 0, 1)
                    : this.lastTransduction.progress
            };
        }
        if (this.lastManifold) {
            this.lastManifold = {
                ...this.lastManifold,
                progress: Number.isFinite(nextState.progress)
                    ? clamp(nextState.progress, 0, 1)
                    : this.lastManifold.progress
            };
        }
        if (this.lastTopology) {
            this.lastTopology = {
                ...this.lastTopology,
                progress: Number.isFinite(nextState.progress)
                    ? clamp(nextState.progress, 0, 1)
                    : this.lastTopology.progress,
                transport: {
                    playing: Boolean(nextState.playing),
                    mode: typeof nextState.mode === 'string' ? nextState.mode : 'idle'
                }
            };
        }
        if (this.lastContinuum) {
            this.lastContinuum = {
                ...this.lastContinuum,
                progress: Number.isFinite(nextState.progress)
                    ? clamp(nextState.progress, 0, 1)
                    : this.lastContinuum.progress,
                transport: {
                    playing: Boolean(nextState.playing),
                    mode: typeof nextState.mode === 'string' ? nextState.mode : 'idle'
                }
            };
        }
        if (this.lastConstellation) {
            this.lastConstellation = {
                ...this.lastConstellation,
                progress: Number.isFinite(nextState.progress)
                    ? clamp(nextState.progress, 0, 1)
                    : this.lastConstellation.progress,
                transport: {
                    playing: Boolean(nextState.playing),
                    mode: typeof nextState.mode === 'string' ? nextState.mode : 'idle'
                }
            };
        }
        this.#updateMasterLevel();
    }

    updateFromData(values, metadata = {}) {
        const normalizedValues = this.#normalizeValues(values);
        const transport = metadata.transport || this.transportState;
        this.lastFrame = {
            values: normalizedValues.slice(),
            metadata: {
                ...metadata,
                transport
            },
            timestamp: safeNow()
        };
        if (!this.active) {
            return null;
        }
        const frameStart = safeNow();
        const analysis = this.#analyzeFrame(normalizedValues, { ...metadata, transport });
        const frameDuration = safeNow() - frameStart;
        this.#recordFrameDuration(frameDuration);
        if (!analysis) {
            return null;
        }
        this.lastAnalysis = analysis;
        this.lastSummary = analysis.summary;
        this.lastSignal = analysis.signal ? cloneSpinorSignalFabric(analysis.signal) : null;
        this.lastTransduction = analysis.transduction ? cloneSpinorTransductionGrid(analysis.transduction) : null;
        this.lastManifold = analysis.manifold ? cloneSpinorMetricManifold(analysis.manifold) : null;
        this.lastTopology = analysis.topology ? cloneSpinorTopologyWeave(analysis.topology) : null;
        this.lastContinuum = analysis.continuum ? cloneSpinorFluxContinuum(analysis.continuum) : null;
        this.lastLattice = analysis.lattice ? cloneSpinorContinuumLattice(analysis.lattice) : null;
        this.lastConstellation = analysis.constellation
            ? cloneSpinorContinuumConstellation(analysis.constellation)
            : null;
        if (this.audioActive) {
            this.#renderAudio(analysis);
        }
        return analysis;
    }

    getLastSummary() {
        return this.lastSummary;
    }

    getLastAnalysis() {
        if (!this.lastAnalysis) {
            return null;
        }
        const result = {
            ...this.lastAnalysis,
            voices: this.lastAnalysis.voices.map((voice) => cloneVoiceAnalysis(voice)).filter(Boolean)
        };
        if (this.lastAnalysis.transmission) {
            result.transmission = {
                ...this.lastAnalysis.transmission,
                carriers: Array.isArray(this.lastAnalysis.transmission.carriers)
                    ? this.lastAnalysis.transmission.carriers.map((carrier) => ({
                        ...carrier,
                        carriers: Array.isArray(carrier.carriers)
                            ? carrier.carriers.map((band) => ({ ...band }))
                            : []
                    }))
                    : []
            };
            if (this.lastAnalysis.transmission.resonance) {
                result.transmission.resonance = cloneSpinorResonanceAtlas(this.lastAnalysis.transmission.resonance);
            }
            if (this.lastAnalysis.transmission.signal) {
                result.transmission.signal = cloneSpinorSignalFabric(this.lastAnalysis.transmission.signal);
            }
            if (this.lastAnalysis.transmission.transduction) {
                result.transmission.transduction = cloneSpinorTransductionGrid(this.lastAnalysis.transmission.transduction);
            }
            if (this.lastAnalysis.transmission.manifold) {
                result.transmission.manifold = cloneSpinorMetricManifold(this.lastAnalysis.transmission.manifold);
            }
            if (this.lastAnalysis.transmission.topology) {
                result.transmission.topology = cloneSpinorTopologyWeave(this.lastAnalysis.transmission.topology);
            }
            if (this.lastAnalysis.transmission.continuum) {
                result.transmission.continuum = cloneSpinorFluxContinuum(this.lastAnalysis.transmission.continuum);
            }
            if (this.lastAnalysis.transmission.lattice) {
                result.transmission.lattice = cloneSpinorContinuumLattice(this.lastAnalysis.transmission.lattice);
            }
            if (this.lastAnalysis.transmission.constellation) {
                result.transmission.constellation = cloneSpinorContinuumConstellation(
                    this.lastAnalysis.transmission.constellation
                );
            }
        }
        if (this.lastAnalysis.quaternion) {
            result.quaternion = {
                ...this.lastAnalysis.quaternion,
                left: Array.isArray(this.lastAnalysis.quaternion.left)
                    ? this.lastAnalysis.quaternion.left.slice()
                    : [],
                right: Array.isArray(this.lastAnalysis.quaternion.right)
                    ? this.lastAnalysis.quaternion.right.slice()
                    : [],
                bridgeVector: Array.isArray(this.lastAnalysis.quaternion.bridgeVector)
                    ? this.lastAnalysis.quaternion.bridgeVector.slice()
                    : [],
                normalizedBridge: Array.isArray(this.lastAnalysis.quaternion.normalizedBridge)
                    ? this.lastAnalysis.quaternion.normalizedBridge.slice()
                    : [],
                hopfFiber: Array.isArray(this.lastAnalysis.quaternion.hopfFiber)
                    ? this.lastAnalysis.quaternion.hopfFiber.slice()
                    : []
            };
        }
        if (this.lastAnalysis.resonance) {
            result.resonance = cloneSpinorResonanceAtlas(this.lastAnalysis.resonance);
        }
        if (this.lastAnalysis.signal) {
            result.signal = cloneSpinorSignalFabric(this.lastAnalysis.signal);
        }
        if (this.lastAnalysis.transduction) {
            result.transduction = cloneSpinorTransductionGrid(this.lastAnalysis.transduction);
        }
        if (this.lastAnalysis.manifold) {
            result.manifold = cloneSpinorMetricManifold(this.lastAnalysis.manifold);
        }
        if (this.lastAnalysis.topology) {
            result.topology = cloneSpinorTopologyWeave(this.lastAnalysis.topology);
        }
        if (this.lastAnalysis.continuum) {
            result.continuum = cloneSpinorFluxContinuum(this.lastAnalysis.continuum);
        }
        if (this.lastAnalysis.lattice) {
            result.lattice = cloneSpinorContinuumLattice(this.lastAnalysis.lattice);
        }
        if (this.lastAnalysis.constellation) {
            result.constellation = cloneSpinorContinuumConstellation(this.lastAnalysis.constellation);
        }
        return result;
    }

    getLastSignal() {
        return this.lastSignal ? cloneSpinorSignalFabric(this.lastSignal) : null;
    }

    getLastTransduction() {
        return this.lastTransduction ? cloneSpinorTransductionGrid(this.lastTransduction) : null;
    }

    getLastManifold() {
        return this.lastManifold ? cloneSpinorMetricManifold(this.lastManifold) : null;
    }

    getLastTopology() {
        return this.lastTopology ? cloneSpinorTopologyWeave(this.lastTopology) : null;
    }

    getLastContinuum() {
        return this.lastContinuum ? cloneSpinorFluxContinuum(this.lastContinuum) : null;
    }

    getLastLattice() {
        return this.lastLattice ? cloneSpinorContinuumLattice(this.lastLattice) : null;
    }

    getLastConstellation() {
        return this.lastConstellation
            ? cloneSpinorContinuumConstellation(this.lastConstellation)
            : null;
    }

    #normalizeValues(values) {
        if (!values || typeof values.length !== 'number') {
            return this.normalizedScratch.subarray(0, 0);
        }
        const length = Math.min(this.channelLimit, Math.max(0, Number(values.length) || 0));
        if (this.normalizedScratch.length < length) {
            this.normalizedScratch = new Float32Array(length);
        }
        const scratch = this.normalizedScratch;
        for (let index = 0; index < length; index += 1) {
            const numeric = Number.isFinite(values[index]) ? values[index] : 0;
            scratch[index] = clamp(numeric, 0, 1);
        }
        return scratch.subarray(0, length);
    }

    #ensureVoiceScratch(count) {
        const required = Math.max(1, count);
        if (!this.voiceScratch || this.voiceScratch.size < required) {
            this.voiceScratch = {
                size: required,
                sums: new Float32Array(required),
                counts: new Float32Array(required),
                averages: new Float32Array(required),
                energies: new Float32Array(required)
            };
        }
        return this.voiceScratch;
    }

    #recordFrameDuration(duration) {
        if (!Number.isFinite(duration)) {
            return;
        }
        this.lastFrameDuration = duration;
        this.performanceWindow.push(duration);
        this.performanceAccumulator += duration;
        if (this.performanceWindow.length > this.performanceWindowSize) {
            const removed = this.performanceWindow.shift();
            this.performanceAccumulator -= removed;
            if (removed === this.performanceMin || removed === this.performanceMax) {
                if (this.performanceWindow.length) {
                    this.performanceMin = Math.min(...this.performanceWindow);
                    this.performanceMax = Math.max(...this.performanceWindow);
                } else {
                    this.performanceMin = Infinity;
                    this.performanceMax = 0;
                }
            }
        }
        if (duration < this.performanceMin) {
            this.performanceMin = duration;
        }
        if (duration > this.performanceMax) {
            this.performanceMax = duration;
        }
        if (this.performanceWindow.length) {
            this.performanceAverage = this.performanceAccumulator / this.performanceWindow.length;
        } else {
            this.performanceAverage = 0;
        }
    }

    async setOutputMode(mode) {
        const normalized = mode === 'analysis' ? 'analysis' : 'hybrid';
        const nextMode = normalized === 'hybrid' && !this.audioSupported ? 'analysis' : normalized;
        if (nextMode === this.outputMode) {
            return this.outputMode;
        }
        this.outputMode = nextMode;
        if (!this.active) {
            return this.outputMode;
        }
        if (this.#shouldUseAudio()) {
            const activated = await this.#activateAudio();
            if (!activated) {
                this.outputMode = 'analysis';
                this.audioActive = false;
                this.lastSummary = 'Sonic geometry analysis active. Audio unavailable in this environment.';
            } else if (this.lastAnalysis) {
                this.#renderAudio(this.lastAnalysis, { immediate: false });
            }
        } else {
            this.#muteAudio();
        }
        this.#updateMasterLevel();
        return this.outputMode;
    }

    #shouldUseAudio() {
        return this.outputMode !== 'analysis';
    }

    async #activateAudio() {
        if (!this.audioSupported || typeof this.contextFactory !== 'function') {
            return false;
        }
        if (!this.audioContext) {
            try {
                this.audioContext = this.contextFactory();
            } catch (error) {
                console.error('SonicGeometryEngine failed to create AudioContext.', error);
                this.audioSupported = false;
                return false;
            }
        }
        if (!this.audioContext) {
            return false;
        }
        if (this.audioContext.state === 'suspended') {
            try {
                await this.audioContext.resume();
            } catch (error) {
                console.warn('SonicGeometryEngine could not resume the AudioContext.', error);
                return false;
            }
        }
        if (this.audioContext.state === 'closed') {
            try {
                this.audioContext = this.contextFactory();
            } catch (error) {
                console.error('SonicGeometryEngine failed to recreate AudioContext after close.', error);
                return false;
            }
        }
        if (!this.audioContext) {
            return false;
        }
        this.#ensureGraph();
        this.audioActive = true;
        this.lastSummary = this.lastSummary || 'Sonic geometry resonance active.';
        return true;
    }

    #ensureGraph() {
        if (!this.audioContext) {
            return;
        }
        if (!this.masterGain) {
            this.masterGain = this.audioContext.createGain();
            this.masterGain.gain.value = 0;
            this.masterGain.connect(this.audioContext.destination);
        }
        if (this.voicePool.length >= this.voiceCount) {
            return;
        }
        const targetVoices = this.voiceCount;
        while (this.voicePool.length < targetVoices) {
            const oscillator = this.audioContext.createOscillator();
            oscillator.type = 'sine';
            const fmModulator = this.audioContext.createOscillator();
            fmModulator.type = 'sine';
            const fmDepth = this.audioContext.createGain();
            fmDepth.gain.value = 0;
            fmModulator.connect(fmDepth);
            fmDepth.connect(oscillator.frequency);
            const filter = this.audioContext.createBiquadFilter();
            filter.type = 'bandpass';
            const gain = this.audioContext.createGain();
            gain.gain.value = 0;
            const amModulator = this.audioContext.createOscillator();
            amModulator.type = 'triangle';
            const amDepth = this.audioContext.createGain();
            amDepth.gain.value = 0;
            amModulator.connect(amDepth);
            amDepth.connect(gain.gain);
            const gate = this.audioContext.createGain();
            gate.gain.value = 0.0001;
            const hasStereo = typeof this.audioContext.createStereoPanner === 'function';
            const panNode = hasStereo ? this.audioContext.createStereoPanner() : null;
            if (panNode) {
                oscillator.connect(filter);
                filter.connect(gain);
                gain.connect(gate);
                gate.connect(panNode);
                panNode.connect(this.masterGain);
            } else {
                oscillator.connect(filter);
                filter.connect(gain);
                gain.connect(gate);
                gate.connect(this.masterGain);
            }
            oscillator.start();
            fmModulator.start();
            amModulator.start();
            this.voicePool.push({
                oscillator,
                filter,
                gain,
                gate,
                pan: panNode,
                fmModulator,
                fmDepth,
                amModulator,
                amDepth,
                amShape: 'triangle'
            });
        }
    }

    #muteAudio({ immediate = false } = {}) {
        if (!this.audioContext || !this.masterGain) {
            this.audioActive = false;
            return;
        }
        const now = this.audioContext.currentTime;
        const timeConstant = immediate ? 0.01 : this.transitionTime * 2;
        this.masterGain.gain.setTargetAtTime(0.0001, now, timeConstant);
        for (const voice of this.voicePool) {
            voice.gain.gain.setTargetAtTime(0.0001, now, timeConstant);
            if (voice.gate) {
                voice.gate.gain.setTargetAtTime(0.0001, now, timeConstant);
            }
            if (voice.amDepth) {
                voice.amDepth.gain.setTargetAtTime(0, now, timeConstant);
            }
            if (voice.fmDepth) {
                voice.fmDepth.gain.setTargetAtTime(0, now, timeConstant);
            }
        }
        this.audioActive = false;
    }

    #updateMasterLevel() {
        if (!this.audioContext || !this.masterGain) {
            return;
        }
        const now = this.audioContext.currentTime;
        if (!this.audioActive || !this.active) {
            this.masterGain.gain.setTargetAtTime(0.0001, now, this.transitionTime * 2);
            return;
        }
        const target = this.active
            ? (this.transportState.playing ? this.masterLevel : this.masterLevel * 0.35)
            : 0.0001;
        this.masterGain.gain.setTargetAtTime(target, now, this.transitionTime * 2);
    }

    #resolveQuaternionTelemetry(visualUniforms, derivedUniforms) {
        const merged = {
            ...(visualUniforms || {}),
            ...(derivedUniforms || {})
        };
        let telemetry = null;
        try {
            telemetry = computeQuaternionBridge(merged);
        } catch (error) {
            telemetry = null;
        }
        if (telemetry) {
            const snapshot = {
                ...telemetry,
                leftQuaternion: Array.isArray(telemetry.leftQuaternion)
                    ? telemetry.leftQuaternion.slice()
                    : [],
                rightQuaternion: Array.isArray(telemetry.rightQuaternion)
                    ? telemetry.rightQuaternion.slice()
                    : [],
                bridgeVector: Array.isArray(telemetry.bridgeVector)
                    ? telemetry.bridgeVector.slice()
                    : [],
                normalizedBridge: Array.isArray(telemetry.normalizedBridge)
                    ? telemetry.normalizedBridge.slice()
                    : [],
                hopfFiber: Array.isArray(telemetry.hopfFiber)
                    ? telemetry.hopfFiber.slice()
                    : []
            };
            this.lastQuaternion = snapshot;
            return snapshot;
        }
        return this.lastQuaternion;
    }

    #computeGeometryEnergy(uniforms, index) {
        if (!uniforms) {
            return 0;
        }
        const pairIndex = index < this.geometryPairs.length
            ? index
            : this.geometryPairs.length - 1;
        const keys = this.geometryPairs[pairIndex] || [];
        if (!keys.length) {
            return 0;
        }
        let sum = 0;
        let count = 0;
        for (const key of keys) {
            if (!key) {
                continue;
            }
            const value = Number(uniforms[key]);
            if (Number.isFinite(value)) {
                sum += Math.abs(value);
                count += 1;
            }
        }
        if (!count) {
            return 0;
        }
        const average = sum / count;
        const normalized = average / this.rotationNormalization;
        return clamp(normalized, 0, 1);
    }

    #analyzeFrame(values, metadata = {}) {
        const voiceCount = Math.max(1, this.voicePool.length || this.voiceCount);
        if (!voiceCount) {
            return null;
        }
        const scratch = this.#ensureVoiceScratch(voiceCount);
        const sums = scratch.sums;
        const counts = scratch.counts;
        const averages = scratch.averages;
        const geometryEnergies = scratch.energies;
        sums.fill(0, 0, voiceCount);
        counts.fill(0, 0, voiceCount);
        averages.fill(0, 0, voiceCount);
        geometryEnergies.fill(0, 0, voiceCount);
        for (let index = 0; index < values.length; index += 1) {
            const voiceIndex = index % voiceCount;
            sums[voiceIndex] += values[index];
            counts[voiceIndex] += 1;
        }
        for (let index = 0; index < voiceCount; index += 1) {
            averages[index] = counts[index] ? sums[index] / counts[index] : 0;
        }
        const visualUniforms = metadata.visualUniforms || metadata.uniforms || null;
        const derivedUniforms = metadata.derivedUniforms || null;
        const transport = metadata.transport || this.transportState;
        const timelineProgress = Number.isFinite(metadata.progress)
            ? clamp(metadata.progress, 0, 1)
            : Number.isFinite(transport.progress)
                ? clamp(transport.progress, 0, 1)
                    : metadata.playbackFrame && Number.isFinite(metadata.playbackFrame.progress)
                        ? clamp(metadata.playbackFrame.progress, 0, 1)
                        : 0;

        const quaternionTelemetry = this.#resolveQuaternionTelemetry(visualUniforms, derivedUniforms);
        const quaternionBridge = quaternionTelemetry && Array.isArray(quaternionTelemetry.normalizedBridge)
            ? quaternionTelemetry.normalizedBridge
            : null;
        const hopfFiber = quaternionTelemetry && Array.isArray(quaternionTelemetry.hopfFiber)
            ? quaternionTelemetry.hopfFiber
            : null;
        const spinorCoupling = quaternionTelemetry
            ? deriveSpinorHarmonics(quaternionTelemetry)
            : null;
        const spinorRatios = spinorCoupling && Array.isArray(spinorCoupling.ratios) && spinorCoupling.ratios.length
            ? spinorCoupling.ratios
            : null;
        const spinorPanOrbit = spinorCoupling && Array.isArray(spinorCoupling.panOrbit) && spinorCoupling.panOrbit.length
            ? spinorCoupling.panOrbit
            : null;
        const spinorPhaseOrbit = spinorCoupling && Array.isArray(spinorCoupling.phaseOrbit) && spinorCoupling.phaseOrbit.length
            ? spinorCoupling.phaseOrbit
            : null;
        const leftAngle = quaternionTelemetry ? quaternionTelemetry.leftAngle : 0;
        const rightAngle = quaternionTelemetry ? quaternionTelemetry.rightAngle : 0;
        const quaternionDot = quaternionTelemetry ? quaternionTelemetry.dot : 0;

        for (let index = 0; index < voiceCount; index += 1) {
            const visualEnergy = this.#computeGeometryEnergy(visualUniforms, index);
            const derivedEnergy = this.#computeGeometryEnergy(derivedUniforms, index);
            geometryEnergies[index] = visualEnergy > 0 ? visualEnergy : derivedEnergy;
        }

        const voices = [];
        const modeDescriptor = this.outputMode === 'analysis'
            ? 'Silent geometry telemetry'
            : 'Dual-stream geometry resonance';
        const channelWeight = (1 - this.geometryBlend) * (1 - this.quaternionBlend);
        const geometryWeight = this.geometryBlend * (1 - this.quaternionBlend);
        const quaternionWeight = this.quaternionBlend;
        for (let index = 0; index < voiceCount; index += 1) {
            const channelEnergy = clamp(averages[index], 0, 1);
            const geometryEnergy = clamp(geometryEnergies[index] || 0, 0, 1);
            const quaternionComponent = quaternionBridge ? quaternionBridge[index % quaternionBridge.length] : 0;
            const quaternionEnergy = quaternionBridge
                ? clamp(0.5 + 0.5 * quaternionComponent, 0, 1)
                : geometryEnergy;
            const harmonicCoordinate = clamp(
                channelEnergy * channelWeight +
                geometryEnergy * geometryWeight +
                quaternionEnergy * quaternionWeight,
                0,
                1
            );
            const rawSpinorRatio = spinorRatios ? spinorRatios[index % spinorRatios.length] : 1;
            const spinorRatio = Math.pow(Math.max(0.03125, rawSpinorRatio), this.spinorFrequencyGain);
            const frequency = this.baseFrequency * Math.pow(2, harmonicCoordinate * this.octaveSpan) * spinorRatio;
            const baseGain = this.voiceGainBase + harmonicCoordinate * this.voiceGainRange;
            const transportGain = transport.playing ? 1 : 0.4;
            const warp = (geometryEnergy - channelEnergy) * this.harmonicWarp;
            const quaternionWarp = quaternionBridge
                ? (quaternionEnergy - channelEnergy) * this.harmonicWarp * this.quaternionBlend
                : 0;
            const combinedWarp = warp + quaternionWarp;
            const gateEnergy = clamp(
                channelEnergy * (1 - this.spectralFocus) + geometryEnergy * this.spectralFocus,
                0,
                1
            );
            const gateBinary = gateEnergy >= this.gateThreshold ? 1 : 0;
            const gateEnvelope = gateBinary ? 1 : clamp(gateEnergy * 0.5, 0.08, 0.35);
            const gainValue = Math.max(0.0001, baseGain * transportGain * gateEnvelope);
            const filterFocus = quaternionBridge
                ? Math.max(0, 1 + quaternionEnergy * 0.5 + Math.abs(quaternionComponent) * 0.35)
                : 1;
            const filterCenter = frequency * (1.25 + geometryEnergy * 0.85 + this.ultraHarmonicBoost * gateEnergy)
                * filterFocus
                * Math.pow(spinorRatio, 0.35 * this.spinorFrequencyGain);
            const qValue = 1 + harmonicCoordinate * 12 + gateEnergy * 6;
            const drift = (geometryEnergy - 0.5) * this.driftFactor + combinedWarp * 0.35 + quaternionComponent * this.quaternionDrift;
            const basePan = voiceCount > 1
                ? -1 + (2 * index) / (voiceCount - 1)
                : 0;
            const hopfComponent = hopfFiber ? hopfFiber[index % hopfFiber.length] : 0;
            const spinorPanContribution = spinorPanOrbit
                ? (spinorPanOrbit[index % spinorPanOrbit.length] || 0) * this.spinorPanGain
                : 0;
            const panDrift = (timelineProgress - 0.5) * 0.8 + drift + hopfComponent * 0.5 + spinorPanContribution;
            const panValue = clamp(basePan + panDrift, -1, 1);
            const carriers = this.carrierSettings.map((carrierSetting, carrierIndex) => {
                const multiplier = Math.max(0.0625, carrierSetting.multiple);
                const spread = Number.isFinite(carrierSetting.spread) ? carrierSetting.spread : 0.5;
                const offset = this.carrierSettings.length > 1
                    ? (carrierIndex / (this.carrierSettings.length - 1)) - 0.5
                    : 0;
                const modulatedMultiple = multiplier * (
                    1 + spread * (geometryEnergy - 0.5) + (combinedWarp + quaternionComponent * 0.25) * this.carrierSpread * offset
                );
                const carrierFrequency = frequency * Math.max(0.0625, modulatedMultiple);
                const energyMix = clamp(
                    (channelEnergy * (1 - this.carrierBlend)) + (geometryEnergy * this.carrierBlend) + combinedWarp * offset,
                    0,
                    1
                );
                const amplitude = gainValue * Math.max(this.carrierEnergyFloor, energyMix);
                return {
                    band: carrierSetting.label || `band-${carrierIndex + 1}`,
                    multiple: modulatedMultiple,
                    frequency: carrierFrequency,
                    energy: energyMix,
                    amplitude: gateBinary ? amplitude : amplitude * 0.1,
                    active: gateBinary ? 1 : 0
                };
            });
            const spinorPhaseContribution = spinorPhaseOrbit
                ? (spinorPhaseOrbit[index % spinorPhaseOrbit.length] || 0) * this.spinorPhaseGain
                : 0;
            const phase = (timelineProgress
                + (index / Math.max(1, voiceCount))
                + Math.max(-0.25, Math.min(0.25, drift))
                + spinorPhaseContribution) % 1;
            const amRate = this.amRateBase + gateEnergy * this.amRateRange + timelineProgress * this.amOrbitRate;
            const amDepth = Math.min(gainValue * 0.95, gainValue * (this.amDepthBase + gateEnergy * this.amDepthRange));
            const fmRate = this.fmRateBase + gateEnergy * this.fmRateRange + (geometryEnergy * this.fmRateGeometry) + Math.abs(quaternionComponent) * this.fmRateGeometry * 0.5;
            const fmDepthCents = this.fmDepthBase + gateEnergy * this.fmDepthRange;
            const fmDepth = frequency * (Math.pow(2, fmDepthCents / 1200) - 1);
            const pulseRate = this.pulseRateBase
                + gateEnergy * this.pulseRateRange
                + timelineProgress * this.amOrbitRate
                + hopfComponent * 9.5
                + spinorPhaseContribution * 24;
            const dutyCycle = clamp(
                0.25
                    + (geometryEnergy - 0.5) * 0.5
                    + hopfComponent * 0.2
                    + spinorPhaseContribution * 0.2
                    + (transport.playing ? 0.1 : 0),
                0.05,
                0.95
            );
            const sequenceSlot = Math.floor((phase * this.sequenceResolution)) % this.sequenceResolution;
            const sequenceCode = sequenceSlot.toString(16).toUpperCase();

            voices.push({
                index,
                channelEnergy,
                geometryEnergy,
                quaternionEnergy,
                quaternionComponent,
                harmonicCoordinate,
                frequency,
                gain: gainValue,
                filterFrequency: filterCenter,
                filterQ: qValue,
                pan: panValue,
                drift,
                carriers,
                modulation: {
                    gate: gateEnergy,
                    binary: gateBinary,
                    amRate,
                    amDepth,
                    fmRate,
                    fmDepth,
                    pulseRate,
                    dutyCycle,
                    phase,
                    sequence: sequenceCode
                },
                quaternion: {
                    hopf: hopfComponent,
                    bridge: quaternionComponent,
                    weight: quaternionWeight
                },
                spinor: spinorCoupling
                    ? {
                        rawRatio: rawSpinorRatio,
                        ratio: spinorRatio,
                        pan: spinorPanContribution,
                        phase: spinorPhaseContribution,
                        coherence: spinorCoupling.coherence,
                        braid: spinorCoupling.braidDensity
                    }
                    : null
            });
        }

        const voiceSummaries = voices.map((voice) => {
            const degrees = Math.round(voice.harmonicCoordinate * 360);
            const glyph = voice.modulation.binary ? '▣' : '◇';
            const spinGlyph = quaternionBridge ? (voice.quaternion.bridge >= 0 ? '↻' : '↺') : '';
            return `Φ${voice.index + 1}:${degrees}°${glyph}${voice.modulation.sequence}${spinGlyph}`;
        });
        const descriptor = (() => {
            switch (transport.mode) {
                case 'playback':
                    return `${modeDescriptor} playback lattice`;
                case 'auto-stream':
                    return `${modeDescriptor} auto-stream drift`;
                case 'manual':
                    return `${modeDescriptor} manual chord`;
                case 'muted':
                    return `${modeDescriptor} muted`;
                default:
                    return `${modeDescriptor} field`;
            }
        })();

        const progressLabel = Math.round(timelineProgress * 100);
        const binaryGateDensity = voices.length
            ? voices.reduce((sum, voice) => sum + (voice.modulation.binary ? 1 : 0), 0) / voices.length
            : 0;
        const weightedGate = voices.length
            ? voices.reduce((sum, voice) => sum + (voice.modulation.binary ? 1 : voice.modulation.gate), 0) / voices.length
            : 0;
        const activeWeight = voices.reduce((sum, voice) => sum + (voice.modulation.binary ? 1 : voice.modulation.gate), 0);
        const spectralCentroid = activeWeight
            ? voices.reduce((sum, voice) => sum + voice.frequency * (voice.modulation.binary ? 1 : voice.modulation.gate), 0) / activeWeight
            : 0;
        const averageFrequency = voices.length
            ? voices.reduce((sum, voice) => sum + voice.frequency, 0) / voices.length
            : 0;
        const averageFmRate = voices.length
            ? voices.reduce((sum, voice) => sum + voice.modulation.fmRate, 0) / voices.length
            : this.fmRateBase;
        const averageAmRate = voices.length
            ? voices.reduce((sum, voice) => sum + voice.modulation.amRate, 0) / voices.length
            : this.amRateBase;
        const sequence = voices.map((voice) => voice.modulation.sequence).join('');
        const leftDegrees = Math.round(leftAngle * (180 / Math.PI));
        const rightDegrees = Math.round(rightAngle * (180 / Math.PI));
        const spinorCoherence = spinorCoupling ? spinorCoupling.coherence : 0;
        const spinDescriptor = quaternionTelemetry
            ? ` · SpinΔ${leftDegrees}°∥${rightDegrees}° · ζ${Math.round(spinorCoherence * 100)}%`
            : '';
        const resonanceAtlas = quaternionTelemetry
            ? buildSpinorResonanceAtlas({
                rotationMatrix: quaternionTelemetry.rotationMatrix,
                bridgeVector: quaternionTelemetry.bridgeVector,
                normalizedBridge: quaternionTelemetry.normalizedBridge,
                hopfFiber,
                spinor: spinorCoupling,
                voices,
                timelineProgress
            })
            : null;
        const signalFabric = buildSpinorSignalFabric({
            voices,
            quaternion: quaternionTelemetry,
            spinor: spinorCoupling,
            resonance: resonanceAtlas,
            transport,
            timelineProgress
        });
        const signalSnapshot = signalFabric ? cloneSpinorSignalFabric(signalFabric) : null;
        const transductionGrid = buildSpinorTransductionGrid({
            quaternion: quaternionTelemetry,
            spinor: spinorCoupling,
            resonance: resonanceAtlas,
            signal: signalFabric,
            voices,
            timelineProgress
        });
        const transductionSnapshot = transductionGrid ? cloneSpinorTransductionGrid(transductionGrid) : null;
        const metricManifold = buildSpinorMetricManifold({
            quaternion: quaternionTelemetry,
            spinor: spinorCoupling,
            resonance: resonanceAtlas,
            signal: signalFabric,
            transduction: transductionGrid,
            voices,
            timelineProgress
        });
        const manifoldSnapshot = metricManifold ? cloneSpinorMetricManifold(metricManifold) : null;
        const topologyWeave = buildSpinorTopologyWeave({
            quaternion: quaternionTelemetry,
            resonance: resonanceAtlas,
            signal: signalFabric,
            manifold: metricManifold,
            transport,
            timelineProgress
        });
        const topologySnapshot = topologyWeave ? cloneSpinorTopologyWeave(topologyWeave) : null;
        const fluxContinuum = buildSpinorFluxContinuum({
            quaternion: quaternionTelemetry,
            resonance: resonanceAtlas,
            signal: signalFabric,
            manifold: metricManifold,
            topology: topologyWeave,
            transport,
            timelineProgress
        });
        const continuumSnapshot = fluxContinuum ? cloneSpinorFluxContinuum(fluxContinuum) : null;
        const continuumLattice = buildSpinorContinuumLattice({
            quaternion: quaternionTelemetry,
            signal: signalFabric,
            manifold: metricManifold,
            topology: topologyWeave,
            continuum: fluxContinuum,
            transport,
            timelineProgress
        });
        const latticeSnapshot = continuumLattice ? cloneSpinorContinuumLattice(continuumLattice) : null;
        const continuumConstellation = buildSpinorContinuumConstellation({
            quaternion: quaternionTelemetry,
            continuum: fluxContinuum,
            lattice: continuumLattice,
            signal: signalFabric,
            manifold: metricManifold,
            transport,
            timelineProgress
        });
        const constellationSnapshot = continuumConstellation
            ? cloneSpinorContinuumConstellation(continuumConstellation)
            : null;
        const summary = `${descriptor} · ${voiceSummaries.join(' · ')} · ${progressLabel}% orbit · ${Math.round(binaryGateDensity * 100)}% gates · ${Math.round(spectralCentroid)}Hz centroid${spinDescriptor}`;
        return {
            timestamp: safeNow(),
            outputMode: this.outputMode,
            audioActive: this.audioActive,
            audioSupported: this.audioSupported,
            voiceCount,
            voices,
            transport: { ...transport },
            timelineProgress,
            summary,
            transmission: {
                gateDensity: binaryGateDensity,
                gateContinuity: clamp(weightedGate, 0, 1),
                spectralCentroid,
                averageFrequency,
                averageFmRate,
                averageAmRate,
                sequence,
                carriers: voices.map((voice) => ({
                    index: voice.index,
                    sequence: voice.modulation.sequence,
                    carriers: voice.carriers.map((carrier) => ({ ...carrier }))
                })),
                spinor: spinorCoupling
                    ? {
                        coherence: spinorCoupling.coherence,
                        braidDensity: spinorCoupling.braidDensity,
                        ratios: spinorCoupling.ratios.slice(),
                        panOrbit: spinorCoupling.panOrbit.slice(),
                        phaseOrbit: spinorCoupling.phaseOrbit.slice(),
                        pitchLattice: spinorCoupling.pitchLattice.map((entry) => ({ ...entry }))
                    }
                    : null,
                resonance: resonanceAtlas
                    ? {
                        ...resonanceAtlas,
                        matrix: resonanceAtlas.matrix.map((row) => row.slice()),
                        axes: resonanceAtlas.axes.map((axis) => axis.slice()),
                        bridge: resonanceAtlas.bridge
                            ? {
                                vector: Array.isArray(resonanceAtlas.bridge.vector)
                                    ? resonanceAtlas.bridge.vector.slice()
                                    : [],
                                normalized: Array.isArray(resonanceAtlas.bridge.normalized)
                                    ? resonanceAtlas.bridge.normalized.slice()
                                    : [],
                                magnitude: resonanceAtlas.bridge.magnitude,
                                coherence: resonanceAtlas.bridge.coherence,
                                braidDensity: resonanceAtlas.bridge.braidDensity
                            }
                            : null,
                        hopf: Array.isArray(resonanceAtlas.hopf) ? resonanceAtlas.hopf.slice() : null,
                        voices: resonanceAtlas.voices.map((voice) => ({
                            ...voice,
                            source: Array.isArray(voice.source) ? voice.source.slice() : [],
                            rotated: Array.isArray(voice.rotated) ? voice.rotated.slice() : [],
                            normalized: Array.isArray(voice.normalized) ? voice.normalized.slice() : null,
                            carriers: Array.isArray(voice.carriers)
                                ? voice.carriers.map((carrier) => ({
                                    ...carrier,
                                    source: Array.isArray(carrier.source) ? carrier.source.slice() : [],
                                    rotated: Array.isArray(carrier.rotated) ? carrier.rotated.slice() : []
                                }))
                                : []
                        })),
                        aggregate: resonanceAtlas.aggregate ? { ...resonanceAtlas.aggregate } : null
                    }
                    : null,
                signal: signalSnapshot ? cloneSpinorSignalFabric(signalSnapshot) : null,
                transduction: transductionSnapshot ? cloneSpinorTransductionGrid(transductionSnapshot) : null,
                manifold: manifoldSnapshot ? cloneSpinorMetricManifold(manifoldSnapshot) : null,
                topology: topologySnapshot ? cloneSpinorTopologyWeave(topologySnapshot) : null,
                continuum: continuumSnapshot ? cloneSpinorFluxContinuum(continuumSnapshot) : null,
                lattice: latticeSnapshot ? cloneSpinorContinuumLattice(latticeSnapshot) : null,
                constellation: constellationSnapshot
                    ? cloneSpinorContinuumConstellation(constellationSnapshot)
                    : null
            },
            quaternion: quaternionTelemetry
                ? {
                    left: quaternionTelemetry.leftQuaternion.slice(),
                    right: quaternionTelemetry.rightQuaternion.slice(),
                    leftAngle,
                    rightAngle,
                    dot: quaternionDot,
                    bridgeMagnitude: quaternionTelemetry.bridgeMagnitude,
                    bridgeVector: quaternionTelemetry.bridgeVector.slice(),
                    normalizedBridge: quaternionTelemetry.normalizedBridge.slice(),
                    hopfFiber: hopfFiber ? hopfFiber.slice() : []
                }
                : null,
            spinor: spinorCoupling
                ? {
                    ...spinorCoupling,
                    ratios: spinorCoupling.ratios.slice(),
                    panOrbit: spinorCoupling.panOrbit.slice(),
                    phaseOrbit: spinorCoupling.phaseOrbit.slice(),
                    axis: {
                        left: spinorCoupling.axis.left ? spinorCoupling.axis.left.slice() : null,
                        right: spinorCoupling.axis.right ? spinorCoupling.axis.right.slice() : null,
                        dot: spinorCoupling.axis.dot,
                        cross: spinorCoupling.axis.cross ? spinorCoupling.axis.cross.slice() : null
                    },
                    fiber: spinorCoupling.fiber.slice(),
                    pitchLattice: spinorCoupling.pitchLattice.map((entry) => ({ ...entry }))
                }
                : null,
            resonance: resonanceAtlas
                ? {
                    ...resonanceAtlas,
                    matrix: resonanceAtlas.matrix.map((row) => row.slice()),
                    axes: resonanceAtlas.axes.map((axis) => axis.slice()),
                    bridge: resonanceAtlas.bridge
                        ? {
                            vector: Array.isArray(resonanceAtlas.bridge.vector)
                                ? resonanceAtlas.bridge.vector.slice()
                                : [],
                            normalized: Array.isArray(resonanceAtlas.bridge.normalized)
                                ? resonanceAtlas.bridge.normalized.slice()
                                : [],
                            magnitude: resonanceAtlas.bridge.magnitude,
                            coherence: resonanceAtlas.bridge.coherence,
                            braidDensity: resonanceAtlas.bridge.braidDensity
                        }
                        : null,
                    hopf: Array.isArray(resonanceAtlas.hopf) ? resonanceAtlas.hopf.slice() : null,
                    voices: resonanceAtlas.voices.map((voice) => ({
                        ...voice,
                        source: Array.isArray(voice.source) ? voice.source.slice() : [],
                        rotated: Array.isArray(voice.rotated) ? voice.rotated.slice() : [],
                        normalized: Array.isArray(voice.normalized) ? voice.normalized.slice() : null,
                        carriers: Array.isArray(voice.carriers)
                            ? voice.carriers.map((carrier) => ({
                                ...carrier,
                                source: Array.isArray(carrier.source) ? carrier.source.slice() : [],
                                rotated: Array.isArray(carrier.rotated) ? carrier.rotated.slice() : []
                            }))
                            : []
                    })),
                    aggregate: resonanceAtlas.aggregate ? { ...resonanceAtlas.aggregate } : null
                }
                : null,
            signal: signalSnapshot,
            transduction: transductionSnapshot,
            manifold: manifoldSnapshot,
            topology: topologySnapshot,
            continuum: continuumSnapshot,
            lattice: latticeSnapshot,
            constellation: constellationSnapshot
        };
    }

    #renderAudio(analysis, { immediate = false } = {}) {
        if (!this.audioContext || !this.masterGain || !this.voicePool.length) {
            return false;
        }
        const now = this.audioContext.currentTime;
        const timeConstant = immediate ? 0.01 : this.transitionTime;
        const voiceCount = Math.min(this.voicePool.length, analysis.voices.length);
        for (let index = 0; index < voiceCount; index += 1) {
            const voice = this.voicePool[index];
            const target = analysis.voices[index];
            const modulation = target.modulation || {};
            voice.oscillator.frequency.setTargetAtTime(target.frequency, now, timeConstant);
            voice.filter.frequency.setTargetAtTime(target.filterFrequency, now, timeConstant);
            voice.filter.Q.setTargetAtTime(target.filterQ, now, timeConstant * 0.75);
            voice.gain.gain.setTargetAtTime(target.gain, now, timeConstant);
            if (voice.gate) {
                const gateLevel = modulation.binary ? 1 : clamp(modulation.gate || 0, 0, 1);
                voice.gate.gain.setTargetAtTime(Math.max(0.0001, gateLevel), now, timeConstant * 0.8);
            }
            if (voice.amModulator) {
                const amRate = Math.max(0.1, modulation.amRate || this.amRateBase);
                voice.amModulator.frequency.setTargetAtTime(amRate, now, timeConstant);
                const duty = clamp(typeof modulation.dutyCycle === 'number' ? modulation.dutyCycle : 0.5, 0.05, 0.95);
                const desiredShape = duty > 0.66 ? 'square' : duty < 0.33 ? 'sawtooth' : 'triangle';
                if (voice.amShape !== desiredShape) {
                    try {
                        voice.amModulator.type = desiredShape;
                        voice.amShape = desiredShape;
                    } catch (error) {
                        console.warn('SonicGeometryEngine could not adjust AM waveform', error);
                    }
                }
            }
            if (voice.amDepth) {
                const amDepthTarget = Math.max(0, modulation.amDepth || 0);
                voice.amDepth.gain.setTargetAtTime(amDepthTarget, now, timeConstant);
            }
            if (voice.fmModulator) {
                const fmRate = Math.max(0.1, modulation.fmRate || this.fmRateBase);
                voice.fmModulator.frequency.setTargetAtTime(fmRate, now, timeConstant);
            }
            if (voice.fmDepth) {
                const fmDepthTarget = Math.max(0, modulation.fmDepth || 0);
                voice.fmDepth.gain.setTargetAtTime(fmDepthTarget, now, timeConstant * 0.8);
            }
            if (voice.pan) {
                voice.pan.pan.setTargetAtTime(target.pan, now, timeConstant * 1.2);
            }
        }
        return true;
    }
}
