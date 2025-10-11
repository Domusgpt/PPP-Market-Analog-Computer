import { DataMapper } from './DataMapper.js';
import { HypercubeRenderer } from './HypercubeRenderer.js';
import { defaultMapping } from './defaultMapping.js';
import { builtInMappingPresets } from './mappingPresets.js';
import { ChannelMonitor } from './ChannelMonitor.js';
import { DevelopmentTracker } from './DevelopmentTracker.js';
import { DataRecorder } from './DataRecorder.js';
import { DataPlayer } from './DataPlayer.js';
import { defaultDevelopmentLog } from './developmentLog.js';
import { SonicGeometryEngine } from './SonicGeometryEngine.js';
import { cloneSpinorResonanceAtlas } from './SpinorResonanceAtlas.js';
import { cloneSpinorSignalFabric } from './SpinorSignalFabric.js';
import { cloneSpinorTransductionGrid } from './SpinorTransductionGrid.js';
import { cloneSpinorMetricManifold } from './SpinorMetricManifold.js';
import { cloneSpinorTopologyWeave } from './SpinorTopologyWeave.js';
import { cloneSpinorFluxContinuum } from './SpinorFluxContinuum.js';
import { clampValue, cloneMappingDefinition, formatDataArray, parseDataInput } from './utils.js';
import { DATA_CHANNEL_COUNT } from './constants.js';

const JSON_INDENT = 2;
const TIMELINE_SLIDER_MAX = 1000;

const debounce = (fn, delay = 150) => {
    let timer = null;
    return (...args) => {
        if (timer) {
            clearTimeout(timer);
        }
        timer = setTimeout(() => {
            timer = null;
            fn(...args);
        }, Math.max(0, delay));
    };
};

const updateUniformPreview = (renderer, target) => {
    const state = renderer.getUniformState();
    const resolution = Array.isArray(state.u_resolution) ? state.u_resolution : [];
    const resolutionText = resolution.length >= 2
        ? `${Math.round(resolution[0])}×${Math.round(resolution[1])}`
        : 'dynamic';
    const dimensionValue = typeof state.u_dimension === 'number' ? state.u_dimension : 4;
    const morphValue = typeof state.u_morphFactor === 'number' ? state.u_morphFactor : 0;
    const patternValue = typeof state.u_patternIntensity === 'number' ? state.u_patternIntensity : 0;
    const gridValue = typeof state.u_gridDensity === 'number' ? state.u_gridDensity : 0;
    const shellValue = typeof state.u_shellWidth === 'number' ? state.u_shellWidth : 0;
    const tetraValue = typeof state.u_tetraThickness === 'number' ? state.u_tetraThickness : 0;
    const glitchValue = typeof state.u_glitchIntensity === 'number' ? state.u_glitchIntensity : 0;
    const colorShiftValue = typeof state.u_colorShift === 'number' ? state.u_colorShift : 0;
    const universeValue = typeof state.u_universeModifier === 'number' ? state.u_universeModifier : 1;

    const lines = [
        `Dimension=${dimensionValue.toFixed(2)}  Resolution=${resolutionText}`,
        `Rotation [XY, XZ, XW, YZ, YW, ZW]: ${[
            state.u_rotXY,
            state.u_rotXZ,
            state.u_rotXW,
            state.u_rotYZ,
            state.u_rotYW,
            state.u_rotZW
        ].map((value) => (typeof value === 'number' ? value : 0).toFixed(2)).join(', ')}`,
        `Morph=${morphValue.toFixed(2)}  Pattern=${patternValue.toFixed(2)}  Grid=${gridValue.toFixed(2)}`,
        `Shell=${shellValue.toFixed(2)}  Tetra=${tetraValue.toFixed(2)}  Glitch=${glitchValue.toFixed(2)}`,
        `Color Shift=${colorShiftValue.toFixed(2)}  Universe Mod=${universeValue.toFixed(2)}`
    ];
    const channelPreview = state.u_dataChannels ? state.u_dataChannels.slice(0, 6) : [];
    if (channelPreview.length) {
        lines.push(`Channels[0-5]: ${channelPreview.map((value) => value.toFixed(2)).join(', ')}`);
    }
    target.textContent = lines.join('\n');
};

const baseDataArray = (count = 16) => {
    const length = Math.min(DATA_CHANNEL_COUNT, Math.max(1, Math.floor(count)));
    return Array.from({ length }, (_, idx) => 0.5 + 0.45 * Math.sin(idx * 0.75));
};

const createAutoStreamGenerator = (applyDataArray, statusMessage, getChannelCount) => {
    let handle = null;
    const start = () => {
        if (handle) {
            return;
        }
        handle = setInterval(() => {
            const time = performance.now() * 0.0015;
            const length = Math.min(
                DATA_CHANNEL_COUNT,
                Math.max(1, Math.floor(typeof getChannelCount === 'function' ? getChannelCount() : 16))
            );
            const values = Array.from({ length }, (_, idx) => {
                const primary = 0.5 + 0.5 * Math.sin(time + idx * 0.45);
                const secondary = 0.5 + 0.5 * Math.cos(time * 0.75 + idx * 0.33);
                return clampValue(primary * 0.6 + secondary * 0.4, 0, 1);
            });
            applyDataArray(values, { source: 'auto-stream' });
            statusMessage.textContent = 'Auto streaming synthetic data…';
        }, 140);
    };

    const stop = () => {
        if (handle) {
            clearInterval(handle);
            handle = null;
        }
    };

    return { start, stop };
};

window.addEventListener('DOMContentLoaded', () => {
    if (document && document.documentElement) {
        document.documentElement.dataset.pppConsole = 'booting';
        delete document.documentElement.dataset.pppConsoleMessage;
    }

    const canvas = document.getElementById('visualizerCanvas');
    const dataInput = document.getElementById('dataInput');
    const applyButton = document.getElementById('applyButton');
    const randomizeButton = document.getElementById('randomizeButton');
    const autoStreamToggle = document.getElementById('autoStream');
    const statusMessage = document.getElementById('statusMessage');
    const uniformPreview = document.getElementById('uniformPreview');
    const channelCount = document.getElementById('channelCount');
    const smoothingSlider = document.getElementById('smoothingSlider');
    const smoothingValue = document.getElementById('smoothingValue');
    const mappingSelect = document.getElementById('mappingSelect');
    const mappingSource = document.getElementById('mappingSource');
    const mappingDescription = document.getElementById('mappingDescription');
    const mappingInput = document.getElementById('mappingInput');
    const mappingHelper = document.getElementById('mappingHelper');
    const applyMappingButton = document.getElementById('applyMappingButton');
    const clearMappingButton = document.getElementById('clearMappingButton');
    const copyUniformButton = document.getElementById('copyUniformButton');
    const downloadMappingButton = document.getElementById('downloadMappingButton');
    const startRecorderButton = document.getElementById('startRecorderButton');
    const stopRecorderButton = document.getElementById('stopRecorderButton');
    const downloadRecorderButton = document.getElementById('downloadRecorderButton');
    const clearRecorderButton = document.getElementById('clearRecorderButton');
    const recorderStatus = document.getElementById('recorderStatus');
    const recorderHelper = document.getElementById('recorderHelper');
    const loadPlaybackButton = document.getElementById('loadPlaybackButton');
    const playbackPlayButton = document.getElementById('playbackPlayButton');
    const playbackPauseButton = document.getElementById('playbackPauseButton');
    const playbackStopButton = document.getElementById('playbackStopButton');
    const playbackStepButton = document.getElementById('playbackStepButton');
    const playbackLoopToggle = document.getElementById('playbackLoopToggle');
    const playbackUniformToggle = document.getElementById('playbackUniformToggle');
    const playbackStatus = document.getElementById('playbackStatus');
    const playbackHelper = document.getElementById('playbackHelper');
    const playbackSpeedSlider = document.getElementById('playbackSpeed');
    const playbackSpeedValue = document.getElementById('playbackSpeedValue');
    const playbackTimeline = document.getElementById('playbackTimeline');
    const playbackTimelineValue = document.getElementById('playbackTimelineValue');
    const playbackTimelineTime = document.getElementById('playbackTimelineTime');
    const playbackFileInput = document.getElementById('playbackFile');
    const sonicGeometryToggle = document.getElementById('sonicGeometryToggle');
    const sonicGeometryHelper = document.getElementById('sonicGeometryHelper');
    const sonicGeometryModeSelect = document.getElementById('sonicGeometryMode');
    const sonicGeometryModeHelper = document.getElementById('sonicGeometryModeHelper');
    const channelMonitorCanvas = document.getElementById('channelMonitor');
    const monitorHelper = document.getElementById('monitorHelper');
    const developmentTrackContainer = document.getElementById('developmentTrack');
    const developmentSummary = document.getElementById('developmentSummary');
    const developmentNotes = document.getElementById('developmentNotes');
    const controlPanel = document.getElementById('controlPanel');
    const controlPanelToggle = document.getElementById('controlPanelToggle');
    const globalConfig = window.PPP_CONFIG || {};
    const sonicGeometryConfig = typeof globalConfig.sonicGeometry === 'object' && globalConfig.sonicGeometry !== null
        ? globalConfig.sonicGeometry
        : {};

    const setPanelState = (expanded) => {
        if (!document.body || !controlPanel || !controlPanelToggle) {
            return;
        }
        const state = expanded ? 'expanded' : 'collapsed';
        document.body.dataset.panelState = state;
        controlPanelToggle.setAttribute('aria-expanded', expanded ? 'true' : 'false');
        controlPanelToggle.setAttribute(
            'aria-label',
            expanded ? 'Hide Hypercube controls panel' : 'Show Hypercube controls panel'
        );
        controlPanel.setAttribute('aria-hidden', expanded ? 'false' : 'true');
        if (expanded) {
            requestAnimationFrame(() => {
                if (typeof controlPanel.focus === 'function') {
                    controlPanel.focus({ preventScroll: true });
                }
            });
        }
        controlPanelToggle.textContent = expanded ? 'Hide Controls' : 'Show Controls';
    };

    let userPanelPreference = false;
    const evaluateDefaultPanelState = () => {
        if (!controlPanel || !controlPanelToggle) {
            return;
        }
        if (userPanelPreference) {
            return;
        }
        const shouldExpand = window.matchMedia('(min-width: 1600px)').matches;
        setPanelState(shouldExpand);
    };

    if (controlPanel && controlPanelToggle) {
        setPanelState(false);
        controlPanelToggle.addEventListener('click', () => {
            userPanelPreference = true;
            const expanded = controlPanelToggle.getAttribute('aria-expanded') === 'true';
            setPanelState(!expanded);
        });
        const handleResize = debounce(() => {
            evaluateDefaultPanelState();
        }, 200);
        window.addEventListener('resize', handleResize);
        evaluateDefaultPanelState();
    }

    if (channelCount) {
        channelCount.textContent = `0 / ${DATA_CHANNEL_COUNT} channels`;
    }
    if (monitorHelper) {
        monitorHelper.textContent = channelMonitorCanvas ? 'Initializing monitor…' : 'Monitor unavailable';
    }

    let renderer;
    let dataPlayer = null;
    let autoStream = null;
    let sonicGeometryEngine = null;
    const sonicAnalysisListeners = new Set();
    const sonicSignalListeners = new Set();
    const sonicTransductionListeners = new Set();
    const sonicManifoldListeners = new Set();
    const sonicTopologyListeners = new Set();
    const sonicContinuumListeners = new Set();
    const sonicLatticeListeners = new Set();
    try {
        renderer = new HypercubeRenderer(canvas);
        statusMessage.textContent = 'Renderer ready. Stream or paste data to drive the visualization.';
        if (document && document.documentElement) {
            document.documentElement.dataset.pppConsole = 'ready';
            delete document.documentElement.dataset.pppConsoleMessage;
        }
        if (window.parent && window.parent !== window) {
            window.requestAnimationFrame(() => {
                window.parent.postMessage({ type: 'ppp-console-ready' }, '*');
            });
        }
    } catch (error) {
        console.error(error);
        statusMessage.textContent = `Renderer initialization failed: ${error.message}`;
        uniformPreview.textContent = 'WebGL unavailable. Try a compatible browser to view the visualization.';
        if (document && document.documentElement) {
            document.documentElement.dataset.pppConsole = 'error';
            document.documentElement.dataset.pppConsoleMessage = error.message || 'Renderer initialization failed.';
        }
        if (window.parent && window.parent !== window) {
            window.parent.postMessage({ type: 'ppp-console-error', message: error.message }, '*');
        }
        return;
    }

    const initialSmoothing = typeof globalConfig.smoothing === 'number'
        ? clampValue(globalConfig.smoothing, 0, 1)
        : parseFloat(smoothingSlider.value);
    smoothingSlider.value = initialSmoothing;
    smoothingValue.textContent = initialSmoothing.toFixed(2);

    const dataMapper = new DataMapper({ mapping: defaultMapping, smoothing: initialSmoothing });

    const trackerBaseEntries = Array.isArray(globalConfig.developmentLog)
        ? globalConfig.developmentLog
        : defaultDevelopmentLog;
    const developmentTracker = new DevelopmentTracker({ entries: trackerBaseEntries });
    if (Array.isArray(globalConfig.additionalDevelopmentEntries)) {
        developmentTracker.addEntries(globalConfig.additionalDevelopmentEntries);
    }
    const currentSessionEntry = globalConfig.currentDevelopmentEntry || {
        id: 'session-12',
        sequence: 12,
        title: 'Session 12 – Sonic geometry resonance',
        summary: 'Transduced polytopal playback into a Web Audio harmonic field with UI control and PPP hooks.',
        highlights: [
            'Introduced a SonicGeometryEngine that maps channel clusters and rotation uniforms into a four-voice resonant lattice',
            'Wove playback, auto-stream, and manual inputs into adaptive harmonic transport with PPP API exposure',
            'Extended the control panel with a Sonic Geometry toggle and live resonance summaries across docs and helper copy'
        ],
        analysis: 'Unified the visual and auditory surfaces so analysts can audit multidimensional motion through emergent harmonic cues.'
    };
    const appliedSessionEntry = developmentTracker.addEntry(currentSessionEntry);

    const monitorConfig = typeof globalConfig.monitor === 'object' && globalConfig.monitor !== null
        ? globalConfig.monitor
        : {};
    const monitorEnabled = Boolean(channelMonitorCanvas) && monitorConfig.enabled !== false;
    const channelMonitor = monitorEnabled ? new ChannelMonitor(channelMonitorCanvas, monitorConfig) : null;
    if (channelMonitor && monitorHelper) {
        monitorHelper.textContent = 'Monitor ready';
    } else if (monitorHelper && !channelMonitor) {
        monitorHelper.textContent = channelMonitorCanvas ? 'Monitor disabled' : 'Monitor unavailable';
    }
    if (channelMonitor) {
        window.addEventListener('resize', () => channelMonitor.resize());
    }

    const recorderConfig = typeof globalConfig.recorder === 'object' && globalConfig.recorder !== null
        ? globalConfig.recorder
        : {};
    const dataRecorder = new DataRecorder(recorderConfig);
    const recorderAutoStart = recorderConfig.autoStart === true;
    const recorderClearOnStart = recorderConfig.clearOnStart !== false;
    const recorderFilename = typeof recorderConfig.filename === 'string' && recorderConfig.filename.trim().length
        ? recorderConfig.filename.trim()
        : 'ppp-recorder-export.json';
    const recorderIncludeUniforms = recorderConfig.includeUniforms !== undefined ? recorderConfig.includeUniforms !== false : true;
    dataRecorder.setIncludeUniforms(recorderIncludeUniforms);
    if (Number.isFinite(recorderConfig.maxEntries)) {
        dataRecorder.setMaxEntries(recorderConfig.maxEntries);
    }
    let shouldAutoStartRecorder = recorderAutoStart;
    let previousRecorderSummary = null;

    const configuredSonicMode = typeof sonicGeometryConfig.defaultMode === 'string'
        ? sonicGeometryConfig.defaultMode.toLowerCase()
        : null;
    const initialSonicMode = configuredSonicMode === 'analysis' ? 'analysis' : 'hybrid';
    const defaultSonicHelperText = 'Enable to translate polytopal motion into double-quaternion harmonic descriptors with optional resonance audio while the spinor coupler, resonance atlas, signal fabric, transduction grid, metric manifold, topology weave, flux continuum, and continuum lattice map the 4D rotation core into harmonic lattices. Use the mode selector to combine sound with telemetry or stream silent carrier matrices, bit lattices, manifold metrics, topology braiding analytics, flux-continuum alignment data, and lattice synergy metrics.';
    sonicGeometryEngine = new SonicGeometryEngine({ outputMode: initialSonicMode });
    const sonicAudioSupported = sonicGeometryEngine.hasAudioSupport();
    if (sonicGeometryHelper) {
        if (sonicAudioSupported) {
            setSonicHelperText(defaultSonicHelperText);
        } else {
            setSonicHelperText('Audio APIs unavailable. Sonic geometry will operate in silent analysis mode until a compatible context is available.');
        }
    }
    if (sonicGeometryModeSelect) {
        if (!sonicAudioSupported) {
            const hybridOption = sonicGeometryModeSelect.querySelector('option[value="hybrid"]');
            if (hybridOption) {
                hybridOption.disabled = true;
            }
        }
        const engineMode = sonicGeometryEngine.getOutputMode();
        if (engineMode && sonicGeometryModeSelect.value !== engineMode) {
            sonicGeometryModeSelect.value = engineMode;
        }
        setSonicModeHelperText(sonicGeometryModeSelect.value, { audioSupported: sonicAudioSupported });
    }

    dataPlayer = new DataPlayer({
        onFrame: (frame) => {
            handlePlaybackFrame(frame);
        },
        onStatusChange: (status) => {
            updatePlaybackStatus(status);
            if (status.state === 'finished') {
                setPlaybackHelperText(`Finished playback of ${status.frameCount} frame${status.frameCount === 1 ? '' : 's'} (${formatPlaybackTime(status.duration)}). Press Play to replay or Step to inspect.`);
            }
        }
    });

    let lastDataValues = [];
    let mappingEditorDirty = false;
    let timelineScrubbing = false;
    let playbackStatusSnapshot = null;

    // Recorder status will be initialized after helper definitions.

    const stringifyJson = (value) => {
        try {
            return JSON.stringify(value, null, JSON_INDENT);
        } catch (error) {
            console.error('Unable to stringify value for JSON editor.', error);
            return '';
        }
    };

    const setMappingHelperText = (text) => {
        if (mappingHelper) {
            mappingHelper.textContent = text;
        }
    };

    const setRecorderHelperText = (text) => {
        if (recorderHelper) {
            recorderHelper.textContent = text;
        }
    };

    const setPlaybackHelperText = (text) => {
        if (playbackHelper) {
            playbackHelper.textContent = text;
        }
    };

    function setSonicHelperText(text) {
        if (sonicGeometryHelper) {
            sonicGeometryHelper.textContent = text;
        }
    }

    function setSonicModeHelperText(mode, { audioSupported = true } = {}) {
        if (!sonicGeometryModeHelper) {
            return;
        }
        if (!audioSupported && mode !== 'analysis') {
            sonicGeometryModeHelper.textContent = 'Audio APIs unavailable; operating in silent analysis mode while streaming quaternion bridges, spinor lattices, resonance atlases, transduction grids, metric manifolds, topology weaves, flux continua, continuum lattices, carrier matrices, and gate metrics for multimodal pipelines.';
            return;
        }
        if (mode === 'analysis') {
            sonicGeometryModeHelper.textContent = 'Silent analysis streams quaternion bridges, spinor lattices, resonance atlases, spinor signal fabric payloads, transduction grids, metric manifolds, topology weaves, flux continua, continuum lattices, carrier matrices, gate density metrics, and harmonic descriptors without engaging the AudioContext—ideal for multimodal transformer training or robotics review.';
            return;
        }
        sonicGeometryModeHelper.textContent = 'Dual-stream mode couples resonance audio with the quaternion bridge, spinor harmonic lattice, resonance atlas, signal fabric, transduction grid, metric manifold, topology weave, flux continuum, continuum lattice, carrier matrix, and gate sequencing telemetry so multimodal systems ingest synchronized sound and data.';
    }

    const cloneSonicSignal = (signal) => (signal ? cloneSpinorSignalFabric(signal) : null);
    const cloneSonicTransduction = (transduction) => (transduction ? cloneSpinorTransductionGrid(transduction) : null);
    const cloneSonicManifold = (manifold) => (manifold ? cloneSpinorMetricManifold(manifold) : null);
    const cloneSonicTopology = (topology) => (topology ? cloneSpinorTopologyWeave(topology) : null);
    const cloneSonicContinuum = (continuum) => (continuum ? cloneSpinorFluxContinuum(continuum) : null);
    const cloneSonicLattice = (lattice) => (lattice ? cloneSpinorContinuumLattice(lattice) : null);

    const cloneSonicAnalysis = (analysis) => {
        if (!analysis || typeof analysis !== 'object') {
            return {
                voices: [],
                transmission: null,
                quaternion: null,
                spinor: null,
                resonance: null,
                signal: cloneSonicSignal(null),
                transduction: cloneSonicTransduction(null),
                manifold: cloneSonicManifold(null),
                topology: cloneSonicTopology(null),
                continuum: cloneSonicContinuum(null)
            };
        }

        const transmissionSource =
            analysis && typeof analysis.transmission === 'object'
                ? analysis.transmission
                : null;
        const quaternionSource =
            analysis && typeof analysis.quaternion === 'object'
                ? analysis.quaternion
                : null;
        const spinorSource =
            analysis && typeof analysis.spinor === 'object'
                ? analysis.spinor
                : null;
        const resonanceSource =
            analysis && typeof analysis.resonance === 'object'
                ? analysis.resonance
                : null;

        return {
            ...analysis,
            voices: Array.isArray(analysis.voices)
                ? analysis.voices.map((voice) => ({
                    ...voice,
                    carriers: Array.isArray(voice.carriers)
                        ? voice.carriers.map((carrier) => ({ ...carrier }))
                        : [],
                    modulation: voice.modulation ? { ...voice.modulation } : null,
                    spinor: voice.spinor ? { ...voice.spinor } : null
                }))
                : [],
            transmission: transmissionSource
                ? {
                    ...transmissionSource,
                    carriers: Array.isArray(transmissionSource.carriers)
                        ? transmissionSource.carriers.map((carrier) => ({
                            ...carrier,
                            carriers: Array.isArray(carrier.carriers)
                                ? carrier.carriers.map((band) => ({ ...band }))
                                : []
                        }))
                        : [],
                    spinor: transmissionSource.spinor
                        ? {
                            ...transmissionSource.spinor,
                            ratios: Array.isArray(transmissionSource.spinor.ratios)
                                ? transmissionSource.spinor.ratios.slice()
                                : [],
                            panOrbit: Array.isArray(transmissionSource.spinor.panOrbit)
                                ? transmissionSource.spinor.panOrbit.slice()
                                : [],
                            phaseOrbit: Array.isArray(transmissionSource.spinor.phaseOrbit)
                                ? transmissionSource.spinor.phaseOrbit.slice()
                                : [],
                            pitchLattice: Array.isArray(transmissionSource.spinor.pitchLattice)
                                ? transmissionSource.spinor.pitchLattice.map((entry) => ({ ...entry }))
                                : []
                        }
                        : null,
                    resonance: transmissionSource.resonance
                        ? cloneSpinorResonanceAtlas(transmissionSource.resonance)
                        : null,
                    signal: cloneSonicSignal(transmissionSource.signal),
                    transduction: cloneSonicTransduction(transmissionSource.transduction),
                    manifold: cloneSonicManifold(transmissionSource.manifold),
                    topology: cloneSonicTopology(transmissionSource.topology),
                    continuum: cloneSonicContinuum(transmissionSource.continuum)
                }
                : null,
            quaternion: quaternionSource
                ? {
                    ...quaternionSource,
                    left: Array.isArray(quaternionSource.left)
                        ? quaternionSource.left.slice()
                        : [],
                    right: Array.isArray(quaternionSource.right)
                        ? quaternionSource.right.slice()
                        : [],
                    bridgeVector: Array.isArray(quaternionSource.bridgeVector)
                        ? quaternionSource.bridgeVector.slice()
                        : [],
                    normalizedBridge: Array.isArray(quaternionSource.normalizedBridge)
                        ? quaternionSource.normalizedBridge.slice()
                        : [],
                    hopfFiber: Array.isArray(quaternionSource.hopfFiber)
                        ? quaternionSource.hopfFiber.slice()
                        : []
                }
                : null,
            spinor: spinorSource
                ? {
                    ...spinorSource,
                    ratios: Array.isArray(spinorSource.ratios)
                        ? spinorSource.ratios.slice()
                        : [],
                    panOrbit: Array.isArray(spinorSource.panOrbit)
                        ? spinorSource.panOrbit.slice()
                        : [],
                    phaseOrbit: Array.isArray(spinorSource.phaseOrbit)
                        ? spinorSource.phaseOrbit.slice()
                        : [],
                    axis: spinorSource.axis
                        ? {
                            ...spinorSource.axis,
                            left: Array.isArray(spinorSource.axis.left)
                                ? spinorSource.axis.left.slice()
                                : null,
                            right: Array.isArray(spinorSource.axis.right)
                                ? spinorSource.axis.right.slice()
                                : null,
                            cross: Array.isArray(spinorSource.axis.cross)
                                ? spinorSource.axis.cross.slice()
                                : null
                        }
                        : null,
                    fiber: Array.isArray(spinorSource.fiber)
                        ? spinorSource.fiber.slice()
                        : [],
                    pitchLattice: Array.isArray(spinorSource.pitchLattice)
                        ? spinorSource.pitchLattice.map((entry) => ({ ...entry }))
                        : []
                }
                : null,
            resonance: resonanceSource
                ? cloneSpinorResonanceAtlas(resonanceSource)
                : null,
            signal: cloneSonicSignal(analysis.signal),
            transduction: cloneSonicTransduction(analysis.transduction),
            manifold: cloneSonicManifold(analysis.manifold),
            topology: cloneSonicTopology(analysis.topology),
            continuum: cloneSonicContinuum(analysis.continuum)
        };
    };
    const notifySonicAnalysis = (analysis) => {
        if (!analysis) {
            return;
        }
        if (typeof globalConfig.onSonicAnalysis === 'function') {
            try {
                globalConfig.onSonicAnalysis(cloneSonicAnalysis(analysis));
            } catch (error) {
                console.error('PPP_CONFIG.onSonicAnalysis error', error);
            }
        }
        sonicAnalysisListeners.forEach((listener) => {
            if (typeof listener !== 'function') {
                return;
            }
            try {
                listener(cloneSonicAnalysis(analysis));
            } catch (error) {
                console.error('PPP sonic analysis listener error', error);
            }
        });
    };

    const notifySonicSignal = (signal) => {
        if (!signal) {
            return;
        }
        if (typeof globalConfig.onSonicSignal === 'function') {
            try {
                globalConfig.onSonicSignal(cloneSonicSignal(signal));
            } catch (error) {
                console.error('PPP_CONFIG.onSonicSignal error', error);
            }
        }
        sonicSignalListeners.forEach((listener) => {
            if (typeof listener !== 'function') {
                return;
            }
            try {
                listener(cloneSonicSignal(signal));
            } catch (error) {
                console.error('PPP sonic signal listener error', error);
            }
        });
    };

    const notifySonicTransduction = (transduction) => {
        if (!transduction) {
            return;
        }
        if (typeof globalConfig.onSonicTransduction === 'function') {
            try {
                globalConfig.onSonicTransduction(cloneSonicTransduction(transduction));
            } catch (error) {
                console.error('PPP_CONFIG.onSonicTransduction error', error);
            }
        }
        sonicTransductionListeners.forEach((listener) => {
            if (typeof listener !== 'function') {
                return;
            }
            try {
                listener(cloneSonicTransduction(transduction));
            } catch (error) {
                console.error('PPP sonic transduction listener error', error);
            }
        });
    };

    const notifySonicManifold = (manifold) => {
        if (!manifold) {
            return;
        }
        if (typeof globalConfig.onSonicManifold === 'function') {
            try {
                globalConfig.onSonicManifold(cloneSonicManifold(manifold));
            } catch (error) {
                console.error('PPP_CONFIG.onSonicManifold error', error);
            }
        }
        sonicManifoldListeners.forEach((listener) => {
            if (typeof listener !== 'function') {
                return;
            }
            try {
                listener(cloneSonicManifold(manifold));
            } catch (error) {
                console.error('PPP sonic manifold listener error', error);
            }
        });
    };

    const notifySonicTopology = (topology) => {
        if (!topology) {
            return;
        }
        if (typeof globalConfig.onSonicTopology === 'function') {
            try {
                globalConfig.onSonicTopology(cloneSonicTopology(topology));
            } catch (error) {
                console.error('PPP_CONFIG.onSonicTopology error', error);
            }
        }
        sonicTopologyListeners.forEach((listener) => {
            if (typeof listener !== 'function') {
                return;
            }
            try {
                listener(cloneSonicTopology(topology));
            } catch (error) {
                console.error('PPP sonic topology listener error', error);
            }
        });
    };

    const notifySonicContinuum = (continuum) => {
        if (!continuum) {
            return;
        }
        if (typeof globalConfig.onSonicContinuum === 'function') {
            try {
                globalConfig.onSonicContinuum(cloneSonicContinuum(continuum));
            } catch (error) {
                console.error('PPP_CONFIG.onSonicContinuum error', error);
            }
        }
        sonicContinuumListeners.forEach((listener) => {
            if (typeof listener !== 'function') {
                return;
            }
            try {
                listener(cloneSonicContinuum(continuum));
            } catch (error) {
                console.error('PPP sonic continuum listener error', error);
            }
        });
    };

    const notifySonicLattice = (lattice) => {
        if (!lattice) {
            return;
        }
        if (typeof globalConfig.onSonicLattice === 'function') {
            try {
                globalConfig.onSonicLattice(cloneSonicLattice(lattice));
            } catch (error) {
                console.error('PPP_CONFIG.onSonicLattice error', error);
            }
        }
        sonicLatticeListeners.forEach((listener) => {
            if (typeof listener !== 'function') {
                return;
            }
            try {
                listener(cloneSonicLattice(lattice));
            } catch (error) {
                console.error('PPP sonic lattice listener error', error);
            }
        });
    };

    const applySonicOutputMode = async (mode, { updateHelper = true } = {}) => {
        if (!sonicGeometryEngine) {
            return 'analysis';
        }
        const resolved = await sonicGeometryEngine.setOutputMode(mode);
        if (sonicGeometryModeSelect && sonicGeometryModeSelect.value !== resolved) {
            sonicGeometryModeSelect.value = resolved;
        }
        setSonicModeHelperText(resolved, { audioSupported: sonicGeometryEngine.hasAudioSupport() });
        if (updateHelper && sonicGeometryToggle && sonicGeometryToggle.checked) {
            const summary = sonicGeometryEngine.getLastSummary();
            if (summary) {
                setSonicHelperText(summary);
            } else if (resolved === 'analysis') {
                setSonicHelperText('Sonic geometry analysis active. Audio muted by mode selection.');
            } else {
                setSonicHelperText(defaultSonicHelperText);
            }
        }
        return resolved;
    };

    const formatPlaybackTime = (ms) => {
        if (!Number.isFinite(ms) || ms < 0) {
            return '0.00s';
        }
        return `${(ms / 1000).toFixed(2)}s`;
    };

    const setTimelineDisplay = (progress = 0, elapsed = 0, duration = 0, { force = false } = {}) => {
        const normalized = clampValue(Number.isFinite(progress) ? progress : 0, 0, 1);
        if (playbackTimeline && (!timelineScrubbing || force)) {
            playbackTimeline.value = String(Math.round(normalized * TIMELINE_SLIDER_MAX));
        }
        if (playbackTimelineValue) {
            playbackTimelineValue.textContent = `${Math.round(normalized * 100)}%`;
        }
        const safeDuration = Number.isFinite(duration) && duration > 0 ? duration : 0;
        let safeElapsed = Number.isFinite(elapsed) ? Math.max(0, elapsed) : 0;
        if (safeDuration > 0) {
            safeElapsed = Math.min(safeElapsed, safeDuration);
        }
        if (playbackTimelineTime) {
            playbackTimelineTime.textContent = `${formatPlaybackTime(safeElapsed)} / ${formatPlaybackTime(safeDuration)}`;
        }
    };

    const updatePlaybackStatus = (status) => {
        if (!playbackStatus) {
            return;
        }
        const snapshot = status || (dataPlayer ? dataPlayer.getStatus() : null);
        const hasFrames = snapshot && snapshot.frameCount > 0;
        playbackStatusSnapshot = hasFrames ? { ...snapshot } : null;
        playbackStatus.classList.remove('idle', 'recording', 'paused', 'playing', 'finished');
        if (!hasFrames) {
            playbackStatus.textContent = 'No recording loaded';
            playbackStatus.classList.add('idle');
            if (!status) {
                setPlaybackHelperText('Load a recorder JSON export to replay captured channel streams. Use Space to play or pause, ←/→ to step, and Shift+←/→ to jump 10 frames.');
            }
            if (sonicGeometryEngine) {
                const autoStreaming = Boolean(autoStreamToggle && autoStreamToggle.checked);
                sonicGeometryEngine.setTransportState({
                    playing: autoStreaming,
                    mode: autoStreaming ? 'auto-stream' : 'idle',
                    progress: 0,
                    frameIndex: -1,
                    frameCount: 0
                });
            }
            if (playbackLoopToggle) {
                playbackLoopToggle.checked = false;
            }
            if (playbackSpeedValue) {
                playbackSpeedValue.textContent = `${parseFloat(playbackSpeedSlider ? playbackSpeedSlider.value : '1').toFixed(2)}×`;
            }
            setTimelineDisplay(0, 0, 0, { force: true });
            return;
        }

        let label = '';
        let stateClass = 'paused';
        switch (snapshot.state) {
            case 'playing':
                label = `Playing ${Math.max(1, (snapshot.currentIndex || 0) + 1)}/${snapshot.frameCount}`;
                stateClass = 'playing';
                break;
            case 'finished':
                label = `Finished ${snapshot.frameCount} frames`;
                stateClass = 'finished';
                break;
            case 'loaded':
                label = `Loaded ${snapshot.frameCount} frames`;
                stateClass = 'idle';
                break;
            default:
                label = `Paused ${Math.max(1, (snapshot.currentIndex || 0) + 1)}/${snapshot.frameCount}`;
                stateClass = 'paused';
                break;
        }

        playbackStatus.textContent = label;
        playbackStatus.classList.add(stateClass);

        if (playbackLoopToggle && typeof snapshot.loop === 'boolean' && playbackLoopToggle.checked !== snapshot.loop) {
            playbackLoopToggle.checked = snapshot.loop;
        }
        if (playbackSpeedValue && Number.isFinite(snapshot.speed)) {
            playbackSpeedValue.textContent = `${snapshot.speed.toFixed(2)}×`;
        }
        setTimelineDisplay(snapshot.progress, snapshot.currentElapsed, snapshot.duration);

        if (sonicGeometryEngine) {
            sonicGeometryEngine.setTransportState({
                playing: snapshot.playing,
                loop: snapshot.loop,
                progress: snapshot.progress,
                mode: 'playback',
                frameIndex: snapshot.currentIndex,
                frameCount: snapshot.frameCount
            });
        }
    };

    updatePlaybackStatus();

    const stopAutoStreamIfActive = () => {
        const wasStreaming = Boolean(autoStreamToggle && autoStreamToggle.checked);
        if (autoStreamToggle) {
            autoStreamToggle.checked = false;
        }
        if (autoStream && typeof autoStream.stop === 'function') {
            autoStream.stop();
        }
        if (wasStreaming && sonicGeometryEngine) {
            const fallbackMode = playbackStatusSnapshot && playbackStatusSnapshot.frameCount > 0
                ? 'playback'
                : 'manual';
            sonicGeometryEngine.setTransportState({
                playing: false,
                mode: fallbackMode,
                progress: playbackStatusSnapshot ? playbackStatusSnapshot.progress : 0,
                frameIndex: playbackStatusSnapshot ? playbackStatusSnapshot.currentIndex : -1,
                frameCount: playbackStatusSnapshot ? playbackStatusSnapshot.frameCount : 0
            });
            if (sonicGeometryToggle && sonicGeometryToggle.checked) {
                setSonicHelperText('Auto stream disengaged. Awaiting next frame.');
            }
        }
    };

    const handlePlaybackFrame = (frame) => {
        if (!frame || !Array.isArray(frame.data)) {
            return;
        }
        stopAutoStreamIfActive();
        const uniformOverride = playbackUniformToggle && playbackUniformToggle.checked && frame.uniforms
            ? frame.uniforms
            : null;
        applyDataArray(frame.data, {
            updateTextarea: true,
            source: 'playback',
            uniformOverride,
            playbackFrame: {
                index: frame.index,
                frameCount: frame.frameCount,
                progress: frame.progress,
                elapsed: frame.elapsed,
                duration: frame.duration,
                sourceLabel: frame.sourceLabel || ''
            }
        });
        const label = `${frame.index + 1}/${frame.frameCount}`;
        const status = dataPlayer ? dataPlayer.getStatus() : null;
        const speed = status && Number.isFinite(status.speed) ? status.speed : 1;
        statusMessage.textContent = `Playback frame ${label} · ${speed.toFixed(2)}×`;
        const progressPercent = Number.isFinite(frame.progress)
            ? Math.round(frame.progress * 100)
            : Math.round((frame.index / Math.max(1, frame.frameCount - 1)) * 100);
        setPlaybackHelperText(`Frame ${label} · ${formatPlaybackTime(frame.elapsed)} / ${formatPlaybackTime(frame.duration)} (${progressPercent}% of run)`);
        setTimelineDisplay(frame.progress, frame.elapsed, frame.duration, { force: frame.reason === 'seek' || frame.reason === 'step' });
    };

    const shouldIgnorePlaybackHotkey = (event) => {
        const active = document.activeElement;
        if (!active || active === document.body) {
            return false;
        }
        if (active.isContentEditable) {
            return true;
        }
        const tagName = active.tagName ? active.tagName.toLowerCase() : '';
        if (tagName === 'input' || tagName === 'textarea' || tagName === 'select') {
            return true;
        }
        const role = active.getAttribute ? active.getAttribute('role') : null;
        if (role === 'textbox' || role === 'combobox' || role === 'slider') {
            return true;
        }
        if (tagName === 'button' || role === 'button') {
            return event.key === ' ' || event.key === 'Spacebar';
        }
        return false;
    };

    const togglePlaybackState = () => {
        if (!dataPlayer || !dataPlayer.hasFrames()) {
            setPlaybackHelperText('Load a recording before toggling playback.');
            return null;
        }
        const status = playbackStatusSnapshot || dataPlayer.getStatus();
        if (status && status.playing) {
            pausePlaybackControl();
            return 'paused';
        }
        const shouldRestart = status
            ? status.state === 'finished' || status.currentIndex >= status.frameCount - 1
            : false;
        playLoadedRecording({ restart: shouldRestart });
        return 'playing';
    };

    const jumpPlaybackByFrames = (delta) => {
        if (!dataPlayer || !dataPlayer.hasFrames()) {
            setPlaybackHelperText('Load a recording before stepping through frames.');
            return null;
        }
        if (!Number.isFinite(delta) || delta === 0) {
            return null;
        }
        const status = dataPlayer.getStatus();
        if (!status || status.frameCount === 0) {
            return null;
        }
        const previousIndex = status.currentIndex;
        let targetIndex;
        if (previousIndex < 0) {
            targetIndex = delta > 0 ? 0 : status.frameCount - 1;
        } else {
            targetIndex = previousIndex + delta;
        }
        if (targetIndex < 0) {
            targetIndex = 0;
        }
        if (targetIndex >= status.frameCount) {
            targetIndex = status.frameCount - 1;
        }
        if (previousIndex === targetIndex && previousIndex >= 0) {
            setPlaybackHelperText(delta < 0 ? 'Already at the first frame.' : 'Already at the final frame.');
            return null;
        }
        const shouldResume = status.playing;
        const frame = dataPlayer.seek(targetIndex);
        if (shouldResume) {
            dataPlayer.play();
        }
        if (frame) {
            setPlaybackHelperText(`Moved to frame ${Math.max(1, frame.index + 1)} of ${frame.frameCount}.`);
        }
        return frame;
    };

    const seekPlaybackBoundary = (position) => {
        if (!dataPlayer || !dataPlayer.hasFrames()) {
            setPlaybackHelperText('Load a recording before navigating playback.');
            return null;
        }
        const status = dataPlayer.getStatus();
        if (!status || status.frameCount === 0) {
            return null;
        }
        const targetIndex = position === 'end' ? status.frameCount - 1 : 0;
        if (status.currentIndex === targetIndex) {
            setPlaybackHelperText(position === 'end' ? 'Already at the final frame.' : 'Already at the first frame.');
            return null;
        }
        const shouldResume = status.playing;
        const frame = dataPlayer.seek(targetIndex);
        if (shouldResume) {
            dataPlayer.play();
        }
        if (frame) {
            const descriptor = position === 'end' ? 'final' : 'first';
            setPlaybackHelperText(`Jumped to the ${descriptor} frame (${Math.max(1, frame.index + 1)} of ${frame.frameCount}).`);
        }
        return frame;
    };

    const handlePlaybackHotkeys = (event) => {
        if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) {
            return;
        }
        if (shouldIgnorePlaybackHotkey(event)) {
            return;
        }
        const hasFrames = dataPlayer && dataPlayer.hasFrames && dataPlayer.hasFrames();
        switch (event.key) {
            case ' ':
            case 'Spacebar':
                if (!hasFrames) {
                    return;
                }
                event.preventDefault();
                togglePlaybackState();
                break;
            case 'ArrowRight':
                if (!hasFrames) {
                    return;
                }
                event.preventDefault();
                if (event.shiftKey) {
                    jumpPlaybackByFrames(10);
                } else {
                    jumpPlaybackByFrames(1);
                }
                break;
            case 'ArrowLeft':
                if (!hasFrames) {
                    return;
                }
                event.preventDefault();
                if (event.shiftKey) {
                    jumpPlaybackByFrames(-10);
                } else {
                    jumpPlaybackByFrames(-1);
                }
                break;
            case 'Home':
                if (!hasFrames) {
                    return;
                }
                event.preventDefault();
                seekPlaybackBoundary('start');
                break;
            case 'End':
                if (!hasFrames) {
                    return;
                }
                event.preventDefault();
                seekPlaybackBoundary('end');
                break;
            default:
                break;
        }
    };

    const updateRecorderStatus = ({ latestRecord = null } = {}) => {
        const stats = dataRecorder.getStats();
        if (recorderStatus) {
            recorderStatus.classList.remove('idle', 'recording', 'paused');
            let label = 'Idle';
            let stateClass = 'idle';
            if (stats.recording) {
                label = `Recording ${stats.frameCount}/${stats.maxEntries}`;
                stateClass = 'recording';
            } else if (stats.frameCount > 0) {
                label = `Paused ${stats.frameCount}/${stats.maxEntries}`;
                stateClass = 'paused';
            }
            recorderStatus.textContent = label;
            recorderStatus.classList.add(stateClass);
        }
        if (latestRecord && recorderHelper) {
            const channelCountLabel = latestRecord.data ? latestRecord.data.length : lastDataValues.length;
            setRecorderHelperText(`Captured frame ${latestRecord.index} (${channelCountLabel} channels, elapsed ${latestRecord.elapsed.toFixed(0)}ms).`);
        } else if (recorderHelper && !stats.recording && stats.frameCount === 0) {
            setRecorderHelperText('Capture streamed data arrays with timestamps and optional uniform snapshots. Configure limits via PPP_CONFIG.recorder.');
        }
        const shouldNotify = latestRecord
            || !previousRecorderSummary
            || previousRecorderSummary.recording !== stats.recording
            || previousRecorderSummary.frameCount !== stats.frameCount
            || previousRecorderSummary.maxEntries !== stats.maxEntries;
        if (shouldNotify && typeof globalConfig.onRecordingUpdate === 'function') {
            globalConfig.onRecordingUpdate({ ...stats }, latestRecord ? { ...latestRecord } : null);
        }
        if (shouldNotify) {
            previousRecorderSummary = {
                recording: stats.recording,
                frameCount: stats.frameCount,
                maxEntries: stats.maxEntries
            };
        }
    };

    updateRecorderStatus();

    const populateMappingEditor = (mappingDefinition, { force = false } = {}) => {
        if (!mappingInput) {
            return;
        }
        if (!mappingEditorDirty || force) {
            mappingInput.value = stringifyJson(mappingDefinition);
            mappingEditorDirty = false;
            setMappingHelperText('Synced with active preset');
        } else {
            setMappingHelperText('Edited (JSON differs from preset)');
        }
    };

    const notifyDevelopmentUpdate = (latestEntry) => {
        if (typeof globalConfig.onDevelopmentUpdate !== 'function') {
            return;
        }
        try {
            globalConfig.onDevelopmentUpdate(
                developmentTracker.getEntries(),
                latestEntry || developmentTracker.getLatestEntry()
            );
        } catch (error) {
            console.error('PPP_CONFIG.onDevelopmentUpdate error', error);
        }
    };

    const renderDevelopmentTrack = (latestEntry) => {
        if (!developmentTrackContainer) {
            notifyDevelopmentUpdate(latestEntry);
            return;
        }
        const entries = developmentTracker.getEntries();
        developmentTrackContainer.innerHTML = '';
        if (!entries.length) {
            const empty = document.createElement('article');
            empty.className = 'development-entry';
            const message = document.createElement('p');
            message.textContent = 'No development sessions recorded yet.';
            empty.appendChild(message);
            developmentTrackContainer.appendChild(empty);
        } else {
            const fragment = document.createDocumentFragment();
            entries.forEach((entry) => {
                const article = document.createElement('article');
                article.className = 'development-entry';

                const heading = document.createElement('h3');
                heading.textContent = entry.title;
                article.appendChild(heading);

                if (entry.summary) {
                    const summaryParagraph = document.createElement('p');
                    summaryParagraph.textContent = entry.summary;
                    article.appendChild(summaryParagraph);
                }

                if (entry.highlights.length) {
                    const list = document.createElement('ul');
                    list.className = 'development-highlights';
                    entry.highlights.forEach((highlight) => {
                        const item = document.createElement('li');
                        item.textContent = highlight;
                        list.appendChild(item);
                    });
                    article.appendChild(list);
                }

                if (entry.analysis) {
                    const analysisParagraph = document.createElement('p');
                    analysisParagraph.textContent = entry.analysis;
                    article.appendChild(analysisParagraph);
                }

                fragment.appendChild(article);
            });
            developmentTrackContainer.appendChild(fragment);
        }

        const summary = developmentTracker.getSummary();
        if (developmentSummary) {
            developmentSummary.textContent = summary.sessionCount
                ? `Session ${summary.latestSequence} of ${summary.sessionCount}`
                : 'No sessions logged';
        }
        if (developmentNotes) {
            developmentNotes.textContent = summary.latestAnalysis || 'Add an entry to populate the analysis log.';
        }

        notifyDevelopmentUpdate(latestEntry);
    };

    renderDevelopmentTrack(appliedSessionEntry);

    const loadPlaybackFromPayload = (payload, { label } = {}) => {
        if (!dataPlayer) {
            return 0;
        }
        let frameCount = 0;
        try {
            frameCount = dataPlayer.loadFromRecorderExport(payload, { label });
        } catch (error) {
            console.error('Playback load failed.', error);
            frameCount = 0;
        }
        if (playbackSpeedSlider) {
            playbackSpeedSlider.value = '1';
        }
        dataPlayer.setSpeed(playbackSpeedSlider ? parseFloat(playbackSpeedSlider.value) : 1);
        updatePlaybackStatus();
        if (frameCount > 0) {
            stopAutoStreamIfActive();
            const sourceLabel = label && label.trim().length ? ` from ${label}` : '';
            setPlaybackHelperText(`Loaded ${frameCount} frame${frameCount === 1 ? '' : 's'}${sourceLabel}. Press Play, tap Space, or scrub the timeline to inspect the recording.`);
            statusMessage.textContent = `Recording loaded (${frameCount} frames).`;
        } else {
            setPlaybackHelperText('Recording load failed. Verify the JSON export format.');
            statusMessage.textContent = 'Playback load failed.';
        }
        return frameCount;
    };

    const playLoadedRecording = ({ restart = false } = {}) => {
        if (!dataPlayer || !dataPlayer.hasFrames()) {
            setPlaybackHelperText('Load a recording before pressing Play.');
            return false;
        }
        stopAutoStreamIfActive();
        const started = dataPlayer.play({ restart });
        if (started) {
            setPlaybackHelperText('Playing recording…');
        }
        return started;
    };

    const pausePlaybackControl = () => {
        if (!dataPlayer) {
            return;
        }
        const status = dataPlayer.pause();
        if (status.frameCount) {
            setPlaybackHelperText(`Playback paused at frame ${Math.max(1, (status.currentIndex || 0) + 1)} of ${status.frameCount}.`);
        }
    };

    const stopPlaybackControl = () => {
        if (!dataPlayer) {
            return;
        }
        const hadFrames = dataPlayer.hasFrames();
        dataPlayer.stop({ reset: true });
        updatePlaybackStatus();
        if (hadFrames) {
            setPlaybackHelperText('Playback stopped. Press Play to restart or Step to inspect frames.');
        }
    };

    const stepPlaybackControl = (direction = 1) => {
        if (!dataPlayer || !dataPlayer.hasFrames()) {
            setPlaybackHelperText('Load a recording before stepping through frames.');
            return null;
        }
        const frame = dataPlayer.step(direction);
        if (!frame) {
            setPlaybackHelperText(direction < 0 ? 'Already at the first frame.' : 'Already at the final frame.');
        }
        return frame;
    };

    function applyDataArray(values, options = {}) {
        const {
            updateTextarea = true,
            suppressCallbacks = false,
            uniformOverride = null,
            source = 'manual',
            playbackFrame = null
        } = options;
        const normalizedValues = Array.isArray(values)
            ? values.slice()
            : values && typeof values.length === 'number'
                ? Array.from(values)
                : [];

        dataMapper.updateData(normalizedValues);
        const snapshot = dataMapper.getUniformSnapshot();
        const hasOverride = uniformOverride && typeof uniformOverride === 'object';
        const appliedUniforms = hasOverride ? uniformOverride : snapshot;
        renderer.setUniformState(appliedUniforms);
        updateUniformPreview(renderer, uniformPreview);
        let monitorSummary = null;
        if (updateTextarea) {
            dataInput.value = formatDataArray(normalizedValues);
        }
        if (channelCount) {
            const displayLength = Math.min(normalizedValues.length, DATA_CHANNEL_COUNT);
            channelCount.textContent = `${displayLength} / ${DATA_CHANNEL_COUNT} channels`;
        }
        if (updateTextarea && normalizedValues.length > DATA_CHANNEL_COUNT) {
            statusMessage.textContent = `Ingested ${normalizedValues.length} values (first ${DATA_CHANNEL_COUNT} mapped to u_dataChannels).`;
        }
        if (channelMonitor) {
            monitorSummary = channelMonitor.update(normalizedValues);
            if (monitorHelper && monitorSummary) {
                monitorHelper.textContent = `Avg ${monitorSummary.average.toFixed(2)} · Max C${monitorSummary.maxIndex}=${monitorSummary.maxValue.toFixed(2)}`;
            }
        }
        lastDataValues = normalizedValues;

        const latestRecord = dataRecorder.capture(normalizedValues, snapshot);
        updateRecorderStatus({ latestRecord });

        if (!suppressCallbacks && typeof globalConfig.onDataApplied === 'function') {
            globalConfig.onDataApplied(normalizedValues.slice(), snapshot);
        }
        if (!suppressCallbacks && typeof globalConfig.onMonitorUpdate === 'function' && monitorSummary) {
            globalConfig.onMonitorUpdate({ ...monitorSummary });
        }

        if (sonicGeometryEngine) {
            const normalizedSource = typeof source === 'string' && source.trim().length
                ? source
                : 'manual';
            const baseTransport = (() => {
                if (normalizedSource === 'playback' && playbackStatusSnapshot) {
                    return {
                        playing: playbackStatusSnapshot.playing,
                        loop: playbackStatusSnapshot.loop,
                        progress: playbackStatusSnapshot.progress,
                        frameIndex: playbackStatusSnapshot.currentIndex,
                        frameCount: playbackStatusSnapshot.frameCount,
                        mode: 'playback'
                    };
                }
                if (normalizedSource === 'auto-stream') {
                    const autoStreaming = Boolean(autoStreamToggle && autoStreamToggle.checked);
                    return {
                        playing: autoStreaming,
                        loop: false,
                        progress: playbackStatusSnapshot ? playbackStatusSnapshot.progress : 0,
                        frameIndex: playbackStatusSnapshot ? playbackStatusSnapshot.currentIndex : -1,
                        frameCount: playbackStatusSnapshot ? playbackStatusSnapshot.frameCount : 0,
                        mode: 'auto-stream'
                    };
                }
                return {
                    playing: normalizedSource !== 'system' && normalizedSource !== 'reapply',
                    loop: false,
                    progress: playbackStatusSnapshot ? playbackStatusSnapshot.progress : 0,
                    frameIndex: playbackStatusSnapshot ? playbackStatusSnapshot.currentIndex : -1,
                    frameCount: playbackStatusSnapshot ? playbackStatusSnapshot.frameCount : 0,
                    mode: normalizedSource
                };
            })();

            const metadata = {
                source: normalizedSource,
                transport: baseTransport,
                visualUniforms: appliedUniforms,
                derivedUniforms: snapshot,
                playbackFrame,
                timestamp: typeof performance !== 'undefined' && typeof performance.now === 'function'
                    ? performance.now()
                    : Date.now()
            };
            const analysis = sonicGeometryEngine.updateFromData(normalizedValues, metadata);
            if (analysis) {
                notifySonicAnalysis(analysis);
                if (analysis.signal) {
                    notifySonicSignal(analysis.signal);
                }
                if (analysis.transduction) {
                    notifySonicTransduction(analysis.transduction);
                }
            if (analysis.manifold) {
                notifySonicManifold(analysis.manifold);
            }
            if (analysis.topology) {
                notifySonicTopology(analysis.topology);
            }
            if (analysis.continuum) {
                notifySonicContinuum(analysis.continuum);
            }
            if (analysis.lattice) {
                notifySonicLattice(analysis.lattice);
            }
            }
            if (analysis && sonicGeometryToggle && sonicGeometryToggle.checked) {
                const summary = analysis.summary || sonicGeometryEngine.getLastSummary();
                if (summary) {
                    setSonicHelperText(summary);
                }
            }
        }

        return snapshot;
    }

    const downloadRecording = (filename = recorderFilename) => {
        const payload = dataRecorder.toJSON();
        const frameCount =
            payload && payload.meta && typeof payload.meta.frameCount === 'number'
                ? payload.meta.frameCount
                : 0;
        if (!frameCount) {
            setRecorderHelperText('Recorder export skipped: no frames captured yet.');
            return false;
        }
        try {
            const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            setTimeout(() => {
                URL.revokeObjectURL(link.href);
                document.body.removeChild(link);
            }, 0);
            setRecorderHelperText(`Downloaded ${frameCount} frame${frameCount === 1 ? '' : 's'} to ${filename}.`);
            return true;
        } catch (error) {
            console.error('Failed to download recorder payload.', error);
            setRecorderHelperText('Recorder export failed. Check console for details.');
            return false;
        }
    };

    const startRecording = ({ clear = recorderClearOnStart } = {}) => {
        dataRecorder.start({ clear });
        updateRecorderStatus();
        const stats = dataRecorder.getStats();
        setRecorderHelperText(`Recorder armed. Capturing up to ${stats.maxEntries} frames${recorderIncludeUniforms ? ' with uniforms' : ''}.`);
        return stats;
    };

    const stopRecording = () => {
        const records = dataRecorder.stop();
        updateRecorderStatus();
        setRecorderHelperText(`Recording paused at ${records.length} frame${records.length === 1 ? '' : 's'}. Download to export.`);
        return records;
    };

    const clearRecording = () => {
        dataRecorder.clear();
        previousRecorderSummary = null;
        updateRecorderStatus();
        setRecorderHelperText('Recorder cleared. Start recording to capture the next stream.');
        return dataRecorder.getStats();
    };

    if (shouldAutoStartRecorder) {
        startRecording();
        shouldAutoStartRecorder = false;
    }

    const presetRegistry = new Map();
    let currentPresetId = null;

    if (mappingSelect) {
        mappingSelect.innerHTML = '';
    }

    const registerPreset = (preset, defaultSource = 'custom') => {
        if (!preset || typeof preset !== 'object' || !preset.mapping) {
            return;
        }
        const idBase = preset.id ?? `preset-${presetRegistry.size}`;
        const id = String(idBase);
        const label = preset.label || id;
        const description = typeof preset.description === 'string' ? preset.description : '';
        const source = preset.source || defaultSource;
        const sourceLabel = preset.sourceLabel || (source === 'built-in' ? 'Built-in preset' : 'Custom preset');
        const mappingClone = cloneMappingDefinition(preset.mapping);
        const entry = {
            id,
            label,
            description,
            mapping: mappingClone,
            source,
            sourceLabel
        };
        presetRegistry.set(id, entry);

        if (mappingSelect) {
            let option = Array.from(mappingSelect.options).find((opt) => opt.value === id);
            if (!option) {
                option = document.createElement('option');
                option.value = id;
                mappingSelect.appendChild(option);
            }
            option.textContent = label;
            option.dataset.source = source;
        }

        return id;
    };

    const setPresetSelection = (presetId, { reapply = true, updateStatus = true } = {}) => {
        if (!presetRegistry.size) {
            return false;
        }

        let entry = presetRegistry.get(presetId);
        if (!entry) {
            entry = presetRegistry.get('default') || Array.from(presetRegistry.values())[0];
        }
        if (!entry) {
            return false;
        }

        if (mappingSelect && mappingSelect.value !== entry.id) {
            mappingSelect.value = entry.id;
        }

        dataMapper.setMapping(cloneMappingDefinition(entry.mapping));
        currentPresetId = entry.id;

        populateMappingEditor(entry.mapping, { force: true });

        if (mappingSource) {
            mappingSource.textContent = entry.sourceLabel || (entry.source === 'built-in' ? 'Built-in preset' : 'Custom preset');
        }
        if (mappingDescription) {
            mappingDescription.textContent = entry.description || 'No description provided.';
        }

        if (reapply && lastDataValues.length) {
            applyDataArray(lastDataValues, { updateTextarea: false, source: 'reapply' });
        }

        if (updateStatus) {
            statusMessage.textContent = `Mapping preset: ${entry.label}`;
        }

        if (updateStatus && typeof globalConfig.onPresetChange === 'function') {
            const entryClone = {
                id: entry.id,
                label: entry.label,
                description: entry.description,
                source: entry.source,
                sourceLabel: entry.sourceLabel,
                mapping: cloneMappingDefinition(entry.mapping)
            };
            globalConfig.onPresetChange(entry.id, entryClone);
        }

        return true;
    };

    builtInMappingPresets.forEach((preset) => registerPreset(preset, preset.source || 'built-in'));

    const userPresets = globalConfig.mappingPresets;
    if (Array.isArray(userPresets)) {
        userPresets.forEach((preset, index) => {
            if (preset && typeof preset === 'object') {
                registerPreset({
                    id: preset.id ?? `custom-${index}`,
                    label: preset.label,
                    description: preset.description,
                    mapping: preset.mapping,
                    source: preset.source || 'custom',
                    sourceLabel: preset.sourceLabel
                }, 'custom');
            }
        });
    } else if (userPresets && typeof userPresets === 'object') {
        Object.entries(userPresets).forEach(([id, preset]) => {
            if (preset && typeof preset === 'object') {
                registerPreset({
                    id,
                    label: preset.label,
                    description: preset.description,
                    mapping: preset.mapping,
                    source: preset.source || 'custom',
                    sourceLabel: preset.sourceLabel
                }, 'custom');
            }
        });
    }

    let configMappingId = null;
    if (globalConfig.mapping) {
        configMappingId = globalConfig.mappingId || 'config-mapping';
        registerPreset({
            id: configMappingId,
            label: globalConfig.mappingLabel || 'Config Mapping',
            description: globalConfig.mappingDescription || 'Mapping supplied via window.PPP_CONFIG.mapping.',
            mapping: globalConfig.mapping,
            source: 'config',
            sourceLabel: 'Config override'
        }, 'config');
    }

    let initialPresetId = 'default';
    if (globalConfig.initialPreset && presetRegistry.has(globalConfig.initialPreset)) {
        initialPresetId = globalConfig.initialPreset;
    } else if (configMappingId && presetRegistry.has(configMappingId)) {
        initialPresetId = configMappingId;
    }

    setPresetSelection(initialPresetId, { reapply: false, updateStatus: false });
    const initialEntry = presetRegistry.get(initialPresetId);
    if (initialEntry) {
        populateMappingEditor(initialEntry.mapping, { force: true });
    }

    if (mappingSelect) {
        mappingSelect.addEventListener('change', (event) => {
            setPresetSelection(event.target.value, { reapply: true, updateStatus: true });
        });
    }

    if (mappingInput) {
        mappingInput.addEventListener('input', () => {
            mappingEditorDirty = true;
            setMappingHelperText('Edited (not yet imported)');
        });
    }

    if (clearMappingButton) {
        clearMappingButton.addEventListener('click', () => {
            if (mappingInput) {
                mappingInput.value = '';
            }
            mappingEditorDirty = false;
            setMappingHelperText('Editor cleared');
            statusMessage.textContent = 'Mapping editor cleared. Paste JSON to import a new mapping.';
        });
    }

    if (applyMappingButton) {
        applyMappingButton.addEventListener('click', () => {
            if (!mappingInput) {
                statusMessage.textContent = 'Mapping editor unavailable in this layout.';
                return;
            }
            const raw = mappingInput.value.trim();
            if (!raw) {
                statusMessage.textContent = 'Paste a mapping object or preset JSON to import.';
                return;
            }
            let parsed;
            try {
                parsed = JSON.parse(raw);
            } catch (error) {
                statusMessage.textContent = `Invalid mapping JSON: ${error.message}`;
                return;
            }

            let mappingDefinition = null;
            let presetMeta = {};
            if (parsed && typeof parsed === 'object') {
                if (parsed.mapping && typeof parsed.mapping === 'object') {
                    mappingDefinition = parsed.mapping;
                    presetMeta = parsed;
                } else {
                    mappingDefinition = parsed;
                }
            }

            if (!mappingDefinition || typeof mappingDefinition !== 'object') {
                statusMessage.textContent = 'Provided JSON does not describe a mapping object.';
                return;
            }

            const presetId = registerPreset({
                id: presetMeta.id,
                label: presetMeta.label || 'Imported mapping',
                description: presetMeta.description || 'Mapping imported from JSON editor.',
                mapping: mappingDefinition,
                source: 'imported',
                sourceLabel: presetMeta.sourceLabel || 'Imported JSON'
            }, 'imported');

            if (!presetId) {
                statusMessage.textContent = 'Unable to register imported mapping. Verify the JSON structure.';
                return;
            }

            setPresetSelection(presetId, { reapply: true, updateStatus: true });
            const entry = presetRegistry.get(presetId);
            if (entry) {
                populateMappingEditor(entry.mapping, { force: true });
            }
            statusMessage.textContent = `Imported mapping preset “${presetMeta.label || presetId}”.`;
        });
    }

    const exportCurrentMappingDefinition = () => dataMapper.getMappingDefinition();

    if (downloadMappingButton) {
        downloadMappingButton.addEventListener('click', () => {
            const mappingDefinition = exportCurrentMappingDefinition();
            const text = stringifyJson(mappingDefinition);
            if (!text) {
                statusMessage.textContent = 'Unable to serialize mapping for download.';
                return;
            }
            const blob = new Blob([text], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const presetId = currentPresetId || 'custom';
            const anchor = document.createElement('a');
            anchor.href = url;
            anchor.download = `ppp-mapping-${presetId}.json`;
            document.body.appendChild(anchor);
            anchor.click();
            document.body.removeChild(anchor);
            URL.revokeObjectURL(url);
            statusMessage.textContent = 'Downloaded mapping JSON for the active configuration.';
        });
    }

    const copyUniformSnapshot = async () => {
        const snapshot = renderer.getUniformState();
        const text = stringifyJson(snapshot);
        if (!text) {
            throw new Error('Unable to serialize uniform snapshot.');
        }
        if (typeof navigator !== 'undefined' && navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(text);
            return true;
        }
        const tempArea = document.createElement('textarea');
        tempArea.value = text;
        tempArea.style.position = 'fixed';
        tempArea.style.opacity = '0';
        document.body.appendChild(tempArea);
        tempArea.select();
        const succeeded = document.execCommand('copy');
        document.body.removeChild(tempArea);
        if (!succeeded) {
            throw new Error('Document copy command was rejected.');
        }
        return true;
    };

    if (copyUniformButton) {
        copyUniformButton.addEventListener('click', async () => {
            try {
                await copyUniformSnapshot();
                statusMessage.textContent = 'Uniform snapshot copied to clipboard.';
            } catch (error) {
                console.error('Unable to copy uniforms.', error);
                statusMessage.textContent = `Copy failed: ${error.message}`;
            }
        });
    }

    const requestedChannelCountRaw = typeof globalConfig.channelCount === 'number' ? globalConfig.channelCount : 16;
    const requestedChannelCount = clampValue(requestedChannelCountRaw, 1, DATA_CHANNEL_COUNT);
    const baseData = baseDataArray(requestedChannelCount);
    const initialDataSource = globalConfig.initialData;
    const initialData = Array.isArray(initialDataSource)
        ? initialDataSource.slice()
        : initialDataSource && typeof initialDataSource.length === 'number'
            ? Array.from(initialDataSource)
            : baseData;

    dataInput.value = formatDataArray(initialData);
    applyDataArray(initialData, { updateTextarea: false, suppressCallbacks: true, source: 'system' });
    updateUniformPreview(renderer, uniformPreview);

    if (globalConfig.palette && typeof globalConfig.palette === 'object') {
        const paletteUpdates = {};
        if (globalConfig.palette.primary) {
            paletteUpdates.u_primaryColor = globalConfig.palette.primary;
        }
        if (globalConfig.palette.secondary) {
            paletteUpdates.u_secondaryColor = globalConfig.palette.secondary;
        }
        if (globalConfig.palette.background) {
            paletteUpdates.u_backgroundColor = globalConfig.palette.background;
        }
        if (Object.keys(paletteUpdates).length) {
            renderer.setUniformState(paletteUpdates);
        }
    }

    if (globalConfig.uniformOverrides && typeof globalConfig.uniformOverrides === 'object') {
        renderer.setUniformState(globalConfig.uniformOverrides);
    }

    updateUniformPreview(renderer, uniformPreview);

    smoothingSlider.addEventListener('input', (event) => {
        const value = clampValue(parseFloat(event.target.value), 0, 1);
        dataMapper.setSmoothing(value);
        smoothingValue.textContent = value.toFixed(2);
        if (lastDataValues.length) {
            applyDataArray(lastDataValues, { updateTextarea: false, suppressCallbacks: true, source: 'system' });
        }
    });

    autoStream = createAutoStreamGenerator(applyDataArray, statusMessage, () => lastDataValues.length || requestedChannelCount);

    applyButton.addEventListener('click', () => {
        const values = parseDataInput(dataInput.value);
        if (!values.length) {
            statusMessage.textContent = 'No numeric values detected. Provide a comma or space separated list of numbers.';
            return;
        }
        autoStreamToggle.checked = false;
        autoStream.stop();
        applyDataArray(values, { source: 'manual' });
        statusMessage.textContent = `Applied ${values.length} values.`;
    });

    randomizeButton.addEventListener('click', () => {
        const length = Math.min(DATA_CHANNEL_COUNT, Math.max(1, lastDataValues.length || requestedChannelCount));
        const values = Array.from({ length }, () => Math.random());
        autoStreamToggle.checked = false;
        autoStream.stop();
        applyDataArray(values, { source: 'manual' });
        statusMessage.textContent = 'Randomized data injected.';
    });

    autoStreamToggle.addEventListener('change', (event) => {
        if (event.target.checked) {
            autoStream.start();
            statusMessage.textContent = 'Auto streaming synthetic data…';
            if (sonicGeometryEngine && sonicGeometryToggle && sonicGeometryToggle.checked) {
                sonicGeometryEngine.setTransportState({ playing: true, mode: 'auto-stream' });
                setSonicHelperText('Synthetic resonance drift engaged.');
            }
        } else {
            autoStream.stop();
            statusMessage.textContent = 'Auto stream paused. Apply or randomize data to continue.';
            if (sonicGeometryEngine) {
                sonicGeometryEngine.setTransportState({ playing: false, mode: 'idle' });
                if (sonicGeometryToggle && sonicGeometryToggle.checked) {
                    setSonicHelperText('Auto stream paused. Sonic lattice idling.');
                }
            }
        }
    });

    if (startRecorderButton) {
        startRecorderButton.addEventListener('click', (event) => {
            startRecording({ clear: event.shiftKey ? false : recorderClearOnStart });
        });
    }
    if (stopRecorderButton) {
        stopRecorderButton.addEventListener('click', () => {
            stopRecording();
        });
    }
    if (downloadRecorderButton) {
        downloadRecorderButton.addEventListener('click', () => {
            if (dataRecorder.isRecording()) {
                stopRecording();
            }
            downloadRecording(recorderFilename);
        });
    }
    if (clearRecorderButton) {
        clearRecorderButton.addEventListener('click', () => {
            clearRecording();
        });
    }

    if (sonicGeometryToggle) {
        sonicGeometryToggle.addEventListener('change', async (event) => {
            if (!sonicGeometryEngine || !sonicGeometryEngine.isSupported()) {
                event.target.checked = false;
                return;
            }
            if (event.target.checked) {
                const enabled = await sonicGeometryEngine.enable();
                if (!enabled) {
                    sonicGeometryToggle.checked = false;
                    setSonicHelperText('Browser blocked the audio context. Interact with the page and toggle again.');
                    return;
                }
                const desiredMode = sonicGeometryModeSelect
                    ? sonicGeometryModeSelect.value
                    : sonicGeometryEngine.getOutputMode();
                const resolvedMode = await applySonicOutputMode(desiredMode, { updateHelper: false });
                const currentMode = (() => {
                    if (playbackStatusSnapshot && playbackStatusSnapshot.frameCount > 0) {
                        return 'playback';
                    }
                    if (autoStreamToggle && autoStreamToggle.checked) {
                        return 'auto-stream';
                    }
                    return 'manual';
                })();
                const transport = (() => {
                    if (currentMode === 'playback' && playbackStatusSnapshot) {
                        return {
                            playing: playbackStatusSnapshot.playing,
                            loop: playbackStatusSnapshot.loop,
                            progress: playbackStatusSnapshot.progress,
                            frameIndex: playbackStatusSnapshot.currentIndex,
                            frameCount: playbackStatusSnapshot.frameCount,
                            mode: 'playback'
                        };
                    }
                    if (currentMode === 'auto-stream') {
                        return {
                            playing: true,
                            loop: false,
                            progress: playbackStatusSnapshot ? playbackStatusSnapshot.progress : 0,
                            frameIndex: playbackStatusSnapshot ? playbackStatusSnapshot.currentIndex : -1,
                            frameCount: playbackStatusSnapshot ? playbackStatusSnapshot.frameCount : 0,
                            mode: 'auto-stream'
                        };
                    }
                    return {
                        playing: true,
                        loop: false,
                        progress: playbackStatusSnapshot ? playbackStatusSnapshot.progress : 0,
                        frameIndex: playbackStatusSnapshot ? playbackStatusSnapshot.currentIndex : -1,
                        frameCount: playbackStatusSnapshot ? playbackStatusSnapshot.frameCount : 0,
                        mode: 'manual'
                    };
                })();
                sonicGeometryEngine.setTransportState(transport);
                let bootAnalysis = null;
                if (lastDataValues.length) {
                    const visualUniforms = renderer.getUniformState();
                    const derivedUniforms = dataMapper.getUniformSnapshot();
                    bootAnalysis = sonicGeometryEngine.updateFromData(lastDataValues, {
                        source: currentMode,
                        transport,
                        visualUniforms,
                        derivedUniforms
                    });
                }
                if (bootAnalysis) {
                    notifySonicAnalysis(bootAnalysis);
                    if (bootAnalysis.signal) {
                        notifySonicSignal(bootAnalysis.signal);
                    }
                    if (bootAnalysis.transduction) {
                        notifySonicTransduction(bootAnalysis.transduction);
                    }
                    if (bootAnalysis.manifold) {
                        notifySonicManifold(bootAnalysis.manifold);
                    }
                    if (bootAnalysis.topology) {
                        notifySonicTopology(bootAnalysis.topology);
                    }
                    if (bootAnalysis.continuum) {
                        notifySonicContinuum(bootAnalysis.continuum);
                    }
                    if (bootAnalysis.lattice) {
                        notifySonicLattice(bootAnalysis.lattice);
                    }
                }
                const summary = (bootAnalysis && bootAnalysis.summary) || sonicGeometryEngine.getLastSummary();
                if (summary) {
                    setSonicHelperText(summary);
                } else {
                    setSonicHelperText(resolvedMode === 'analysis'
                        ? 'Sonic geometry analysis active. Move data to stream harmonic descriptors.'
                        : 'Sonic geometry active. Move data to weave resonance.');
                }
            } else {
                sonicGeometryEngine.disable();
                setSonicHelperText('Sonic geometry analysis muted. Toggle to re-engage.');
            }
        });
    }

    if (sonicGeometryModeSelect) {
        sonicGeometryModeSelect.addEventListener('change', async (event) => {
            const requestedMode = event.target.value === 'analysis' ? 'analysis' : 'hybrid';
            const resolved = await applySonicOutputMode(requestedMode);
            if (resolved !== requestedMode) {
                event.target.value = resolved;
            }
        });
    }

    if (loadPlaybackButton) {
        loadPlaybackButton.addEventListener('click', () => {
            if (playbackFileInput) {
                playbackFileInput.click();
            } else {
                setPlaybackHelperText('Playback file input unavailable in this layout.');
            }
        });
    }

    if (playbackFileInput) {
        playbackFileInput.addEventListener('change', (event) => {
            const file = event.target.files && event.target.files[0];
            if (!file) {
                return;
            }
            const reader = new FileReader();
            reader.onload = () => {
                try {
                    const parsed = JSON.parse(reader.result);
                    loadPlaybackFromPayload(parsed, { label: file.name });
                } catch (error) {
                    console.error('Failed to parse playback JSON.', error);
                    setPlaybackHelperText(`Playback load failed: ${error.message}`);
                    statusMessage.textContent = 'Playback load failed.';
                }
            };
            reader.onerror = () => {
                setPlaybackHelperText('Unable to read the selected file.');
                statusMessage.textContent = 'Playback load failed.';
            };
            reader.readAsText(file);
            playbackFileInput.value = '';
        });
    }

    if (playbackPlayButton) {
        playbackPlayButton.addEventListener('click', () => {
            playLoadedRecording();
        });
    }

    if (playbackPauseButton) {
        playbackPauseButton.addEventListener('click', () => {
            pausePlaybackControl();
        });
    }

    if (playbackStopButton) {
        playbackStopButton.addEventListener('click', () => {
            stopPlaybackControl();
        });
    }

    if (playbackStepButton) {
        playbackStepButton.addEventListener('click', (event) => {
            stepPlaybackControl(event.shiftKey ? -1 : 1);
        });
    }

    window.addEventListener('keydown', handlePlaybackHotkeys);

    if (playbackLoopToggle) {
        playbackLoopToggle.addEventListener('change', (event) => {
            if (dataPlayer) {
                dataPlayer.setLoop(event.target.checked);
            }
            setPlaybackHelperText(event.target.checked ? 'Loop enabled for playback.' : 'Loop disabled for playback.');
        });
    }

    if (playbackSpeedSlider) {
        const updateSpeedLabel = (value) => {
            if (playbackSpeedValue) {
                playbackSpeedValue.textContent = `${value.toFixed(2)}×`;
            }
        };
        const initialSpeed = parseFloat(playbackSpeedSlider.value) || 1;
        updateSpeedLabel(initialSpeed);
        playbackSpeedSlider.addEventListener('input', (event) => {
            const value = clampValue(parseFloat(event.target.value), 0.25, 2);
            updateSpeedLabel(value);
            if (dataPlayer) {
                dataPlayer.setSpeed(value);
            }
        });
    }

    const commitTimelineSeek = (progress) => {
        timelineScrubbing = false;
        if (!dataPlayer || typeof dataPlayer.hasFrames !== 'function' || !dataPlayer.hasFrames()) {
            setTimelineDisplay(progress, 0, 0, { force: true });
            return;
        }
        const statusBeforeSeek = dataPlayer.getStatus();
        if (!statusBeforeSeek || statusBeforeSeek.frameCount === 0) {
            setTimelineDisplay(
                progress,
                statusBeforeSeek ? statusBeforeSeek.currentElapsed : 0,
                statusBeforeSeek ? statusBeforeSeek.duration : 0,
                { force: true }
            );
            return;
        }
        const shouldResume = statusBeforeSeek.playing;
        const frame = dataPlayer.seekToProgress(progress);
        if (shouldResume) {
            dataPlayer.play();
        }
        if (!frame) {
            setTimelineDisplay(progress, statusBeforeSeek.duration * progress, statusBeforeSeek.duration, { force: true });
        }
    };

    if (playbackTimeline) {
        playbackTimeline.value = '0';
        playbackTimeline.addEventListener('pointerdown', () => {
            timelineScrubbing = true;
        });
        playbackTimeline.addEventListener('input', (event) => {
            const sliderValue = parseFloat(event.target.value) || 0;
            const progress = clampValue(sliderValue / TIMELINE_SLIDER_MAX, 0, 1);
            timelineScrubbing = true;
            const status = dataPlayer ? dataPlayer.getStatus() : null;
            const duration = status ? status.duration : 0;
            const elapsed = duration > 0 ? progress * duration : 0;
            setTimelineDisplay(progress, elapsed, duration);
        });
        playbackTimeline.addEventListener('change', (event) => {
            const sliderValue = parseFloat(event.target.value) || 0;
            const progress = clampValue(sliderValue / TIMELINE_SLIDER_MAX, 0, 1);
            commitTimelineSeek(progress);
        });
        playbackTimeline.addEventListener('pointerup', () => {
            if (!timelineScrubbing) {
                return;
            }
            const sliderValue = parseFloat(playbackTimeline.value) || 0;
            const progress = clampValue(sliderValue / TIMELINE_SLIDER_MAX, 0, 1);
            commitTimelineSeek(progress);
        });
        playbackTimeline.addEventListener('pointerleave', (event) => {
            if (!timelineScrubbing || event.buttons !== 0) {
                return;
            }
            const sliderValue = parseFloat(playbackTimeline.value) || 0;
            const progress = clampValue(sliderValue / TIMELINE_SLIDER_MAX, 0, 1);
            commitTimelineSeek(progress);
        });
        playbackTimeline.addEventListener('blur', () => {
            timelineScrubbing = false;
        });
    }

    if (playbackUniformToggle) {
        playbackUniformToggle.addEventListener('change', (event) => {
            setPlaybackHelperText(event.target.checked
                ? 'Recorded uniform snapshots will be applied during playback.'
                : 'Playback will rely on the active mapping (recorded uniforms disabled).');
        });
    }

    const api = {
        renderer,
        dataMapper,
        developmentTracker,
        dataRecorder,
        dataPlayer,
        sonicGeometry: {
            engine: sonicGeometryEngine,
            isSupported: () => (sonicGeometryEngine ? sonicGeometryEngine.isSupported() : false),
            hasAudioSupport: () => (sonicGeometryEngine ? sonicGeometryEngine.hasAudioSupport() : false),
            getOutputMode: () => (sonicGeometryEngine ? sonicGeometryEngine.getOutputMode() : 'analysis'),
            getState: () => (sonicGeometryEngine ? sonicGeometryEngine.getState() : null),
            enable: async (mode) => {
                if (!sonicGeometryEngine || !sonicGeometryEngine.isSupported()) {
                    return false;
                }
                const activated = await sonicGeometryEngine.enable();
                if (!activated) {
                    return false;
                }
                const requestedMode = typeof mode === 'string' ? mode : (sonicGeometryModeSelect ? sonicGeometryModeSelect.value : null);
                const resolvedMode = await applySonicOutputMode(requestedMode || 'hybrid');
                if (sonicGeometryToggle && !sonicGeometryToggle.checked) {
                    sonicGeometryToggle.checked = true;
                }
                const summary = sonicGeometryEngine.getLastSummary();
                if (summary) {
                    setSonicHelperText(summary);
                } else {
                    setSonicHelperText(resolvedMode === 'analysis'
                        ? 'Sonic geometry analysis active. Move data to stream harmonic descriptors.'
                        : defaultSonicHelperText);
                }
                const lastAnalysis = sonicGeometryEngine.getLastAnalysis();
                if (lastAnalysis) {
                    notifySonicAnalysis(lastAnalysis);
                    if (lastAnalysis.signal) {
                        notifySonicSignal(lastAnalysis.signal);
                    }
                    if (lastAnalysis.transduction) {
                        notifySonicTransduction(lastAnalysis.transduction);
                    }
                    if (lastAnalysis.manifold) {
                        notifySonicManifold(lastAnalysis.manifold);
                    }
                    if (lastAnalysis.topology) {
                        notifySonicTopology(lastAnalysis.topology);
                    }
                    if (lastAnalysis.continuum) {
                        notifySonicContinuum(lastAnalysis.continuum);
                    }
                    if (lastAnalysis.lattice) {
                        notifySonicLattice(lastAnalysis.lattice);
                    }
                }
                return true;
            },
            disable: (options) => {
                if (!sonicGeometryEngine) {
                    return false;
                }
                sonicGeometryEngine.disable(options || {});
                if (sonicGeometryToggle) {
                    sonicGeometryToggle.checked = false;
                }
                setSonicHelperText('Sonic geometry analysis muted. Toggle to re-engage.');
                return true;
            },
            setOutputMode: async (mode) => applySonicOutputMode(mode),
            update: (values, metadata) => {
                if (!sonicGeometryEngine) {
                    return null;
                }
                const analysis = sonicGeometryEngine.updateFromData(values, metadata);
                if (analysis) {
                    notifySonicAnalysis(analysis);
                    if (analysis.signal) {
                        notifySonicSignal(analysis.signal);
                    }
                    if (analysis.transduction) {
                        notifySonicTransduction(analysis.transduction);
                    }
                    if (analysis.manifold) {
                        notifySonicManifold(analysis.manifold);
                    }
                    if (analysis.topology) {
                        notifySonicTopology(analysis.topology);
                    }
                    if (analysis.continuum) {
                        notifySonicContinuum(analysis.continuum);
                    }
                    if (analysis.lattice) {
                        notifySonicLattice(analysis.lattice);
                    }
                    if (sonicGeometryToggle && sonicGeometryToggle.checked) {
                        const summary = analysis.summary || sonicGeometryEngine.getLastSummary();
                        if (summary) {
                            setSonicHelperText(summary);
                        }
                    }
                }
                return analysis;
            },
            getAnalysis: () => (sonicGeometryEngine ? sonicGeometryEngine.getLastAnalysis() : null),
            getSignal: () => (sonicGeometryEngine ? sonicGeometryEngine.getLastSignal() : null),
            getTransduction: () => (sonicGeometryEngine ? sonicGeometryEngine.getLastTransduction() : null),
            getManifold: () => (sonicGeometryEngine ? sonicGeometryEngine.getLastManifold() : null),
            getTopology: () => (sonicGeometryEngine ? sonicGeometryEngine.getLastTopology() : null),
            getContinuum: () => (sonicGeometryEngine ? sonicGeometryEngine.getLastContinuum() : null),
            getLattice: () => (sonicGeometryEngine ? sonicGeometryEngine.getLastLattice() : null),
            getResonance: () => {
                const latest = sonicGeometryEngine ? sonicGeometryEngine.getLastAnalysis() : null;
                return latest && latest.resonance ? cloneSpinorResonanceAtlas(latest.resonance) : null;
            },
            onAnalysis: (listener) => {
                if (typeof listener !== 'function') {
                    return () => {};
                }
                sonicAnalysisListeners.add(listener);
                const latest = sonicGeometryEngine ? sonicGeometryEngine.getLastAnalysis() : null;
                if (latest) {
                    try {
                        listener(cloneSonicAnalysis(latest));
                    } catch (error) {
                        console.error('PPP sonic analysis listener error', error);
                    }
                }
                return () => {
                    sonicAnalysisListeners.delete(listener);
                };
            },
            onSignal: (listener) => {
                if (typeof listener !== 'function') {
                    return () => {};
                }
                sonicSignalListeners.add(listener);
                const latestSignal = sonicGeometryEngine ? sonicGeometryEngine.getLastSignal() : null;
                if (latestSignal) {
                    try {
                        listener(cloneSonicSignal(latestSignal));
                    } catch (error) {
                        console.error('PPP sonic signal listener error', error);
                    }
                }
                return () => {
                    sonicSignalListeners.delete(listener);
                };
            },
            onTransduction: (listener) => {
                if (typeof listener !== 'function') {
                    return () => {};
                }
                sonicTransductionListeners.add(listener);
                const latestTransduction = sonicGeometryEngine ? sonicGeometryEngine.getLastTransduction() : null;
                if (latestTransduction) {
                    try {
                        listener(cloneSonicTransduction(latestTransduction));
                    } catch (error) {
                        console.error('PPP sonic transduction listener error', error);
                    }
                }
                return () => {
                    sonicTransductionListeners.delete(listener);
                };
            },
            onManifold: (listener) => {
                if (typeof listener !== 'function') {
                    return () => {};
                }
                sonicManifoldListeners.add(listener);
                const latestManifold = sonicGeometryEngine ? sonicGeometryEngine.getLastManifold() : null;
                if (latestManifold) {
                    try {
                        listener(cloneSonicManifold(latestManifold));
                    } catch (error) {
                        console.error('PPP sonic manifold listener error', error);
                    }
                }
                return () => {
                    sonicManifoldListeners.delete(listener);
                };
            },
            onTopology: (listener) => {
                if (typeof listener !== 'function') {
                    return () => {};
                }
                sonicTopologyListeners.add(listener);
                const latestTopology = sonicGeometryEngine ? sonicGeometryEngine.getLastTopology() : null;
                if (latestTopology) {
                    try {
                        listener(cloneSonicTopology(latestTopology));
                    } catch (error) {
                        console.error('PPP sonic topology listener error', error);
                    }
                }
                return () => {
                    sonicTopologyListeners.delete(listener);
                };
            },
            onContinuum: (listener) => {
                if (typeof listener !== 'function') {
                    return () => {};
                }
                sonicContinuumListeners.add(listener);
                const latestContinuum = sonicGeometryEngine ? sonicGeometryEngine.getLastContinuum() : null;
                if (latestContinuum) {
                    try {
                        listener(cloneSonicContinuum(latestContinuum));
                    } catch (error) {
                        console.error('PPP sonic continuum listener error', error);
                    }
                }
                return () => {
                    sonicContinuumListeners.delete(listener);
                };
            },
            onLattice: (listener) => {
                if (typeof listener !== 'function') {
                    return () => {};
                }
                sonicLatticeListeners.add(listener);
                const latestLattice = sonicGeometryEngine ? sonicGeometryEngine.getLastLattice() : null;
                if (latestLattice) {
                    try {
                        listener(cloneSonicLattice(latestLattice));
                    } catch (error) {
                        console.error('PPP sonic lattice listener error', error);
                    }
                }
                return () => {
                    sonicLatticeListeners.delete(listener);
                };
            }
        },
        applyDataArray: (values, options) => applyDataArray(values, options),
        startRecording: (options) => startRecording(options || {}),
        stopRecording: () => stopRecording(),
        clearRecording: () => clearRecording(),
        downloadRecording: (filename) => downloadRecording(typeof filename === 'string' ? filename : recorderFilename),
        getRecording: () => dataRecorder.getRecords(),
        getRecordingStats: () => dataRecorder.getStats(),
        loadPlayback: (payload, options) => loadPlaybackFromPayload(payload, options || {}),
        playPlayback: (options) => playLoadedRecording(options || {}),
        pausePlayback: () => {
            pausePlaybackControl();
            return dataPlayer ? dataPlayer.getStatus() : null;
        },
        stopPlayback: () => {
            stopPlaybackControl();
            return dataPlayer ? dataPlayer.getStatus() : null;
        },
        stepPlayback: (direction) => stepPlaybackControl(direction),
        togglePlayback: () => togglePlaybackState(),
        jumpPlaybackByFrames: (delta) => jumpPlaybackByFrames(delta),
        seekPlaybackStart: () => seekPlaybackBoundary('start'),
        seekPlaybackEnd: () => seekPlaybackBoundary('end'),
        setPlaybackLoop: (value) => (dataPlayer ? dataPlayer.setLoop(value) : false),
        setPlaybackSpeed: (value) => (dataPlayer ? dataPlayer.setSpeed(value) : 1),
        getPlaybackStatus: () => (dataPlayer ? dataPlayer.getStatus() : null),
        setPlaybackUniformMode: (enable) => {
            if (playbackUniformToggle) {
                playbackUniformToggle.checked = enable !== false;
                setPlaybackHelperText(playbackUniformToggle.checked
                    ? 'Recorded uniform snapshots will be applied during playback.'
                    : 'Playback will rely on the active mapping (recorded uniforms disabled).');
            }
            return playbackUniformToggle ? playbackUniformToggle.checked : enable !== false;
        },
        recordDevelopmentEntry: (entry) => {
            const result = developmentTracker.addEntry(entry);
            renderDevelopmentTrack(result);
            return result;
        },
        getDevelopmentTrack: () => developmentTracker.getEntries(),
        setDevelopmentTrack: (entries, { render = true } = {}) => {
            developmentTracker.setEntries(entries);
            if (render) {
                renderDevelopmentTrack(developmentTracker.getLatestEntry());
            }
        },
        appendDevelopmentEntries: (entries, { render = true } = {}) => {
            developmentTracker.addEntries(entries);
            if (render) {
                renderDevelopmentTrack(developmentTracker.getLatestEntry());
            }
        },
        setMapping: (nextMapping, { reapply = true } = {}) => {
            dataMapper.setMapping(nextMapping);
            currentPresetId = null;
            if (reapply && lastDataValues.length) {
                applyDataArray(lastDataValues, { updateTextarea: false, source: 'reapply' });
            }
            if (mappingSource) {
                mappingSource.textContent = 'Manual override';
            }
            if (mappingDescription) {
                mappingDescription.textContent = 'Mapping supplied programmatically via API.';
            }
            populateMappingEditor(dataMapper.getMappingDefinition(), { force: true });
        },
        channelMonitor,
        setMonitorHighlight: (indices) => {
            if (channelMonitor) {
                channelMonitor.setHighlightIndices(indices);
            }
        },
        setMonitorRange: (range) => {
            if (channelMonitor) {
                channelMonitor.setValueRange(range);
                channelMonitor.render();
            }
        },
        setMonitorSmoothing: (value) => {
            if (channelMonitor) {
                channelMonitor.setSmoothing(value);
            }
        },
        getMonitorSummary: () => (channelMonitor ? channelMonitor.getSummary() : null),
        resizeMonitor: () => {
            if (channelMonitor) {
                channelMonitor.resize();
            }
        },
        setPreset: (presetId, options = {}) => setPresetSelection(presetId, {
            reapply: options.reapply !== false,
            updateStatus: options.updateStatus !== false
        }),
        registerPreset: (preset, options = {}) => {
            const normalizedPreset = {
                ...preset,
                source: (preset && preset.source) || options.source,
                sourceLabel: (preset && preset.sourceLabel) || options.sourceLabel
            };
            const id = registerPreset(normalizedPreset, options.source || 'custom');
            if (!id) {
                return null;
            }
            if (options.activate !== false) {
                setPresetSelection(id, {
                    reapply: options.reapply !== false,
                    updateStatus: options.updateStatus !== false
                });
            }
            return id;
        },
        listPresets: () => Array.from(presetRegistry.values()).map((entry) => ({
            id: entry.id,
            label: entry.label,
            description: entry.description,
            source: entry.source,
            sourceLabel: entry.sourceLabel
        })),
        getPresetDefinition: (presetId) => {
            const entry = presetRegistry.get(presetId ?? currentPresetId);
            if (!entry) {
                return null;
            }
            return {
                id: entry.id,
                label: entry.label,
                description: entry.description,
                source: entry.source,
                sourceLabel: entry.sourceLabel,
                mapping: cloneMappingDefinition(entry.mapping)
            };
        },
        getCurrentPresetId: () => currentPresetId,
        exportMappingDefinition: () => exportCurrentMappingDefinition(),
        exportUniformSnapshot: () => renderer.getUniformState(),
        getLastDataArray: () => lastDataValues.slice(),
        startAutoStream: () => {
            autoStreamToggle.checked = true;
            autoStream.start();
            statusMessage.textContent = 'Auto streaming synthetic data…';
        },
        stopAutoStream: () => {
            autoStreamToggle.checked = false;
            autoStream.stop();
            statusMessage.textContent = 'Auto stream paused via API.';
        }
    };

    if (globalConfig.exposeApi !== false) {
        window.PPP = api;
    }

    if (typeof globalConfig.onReady === 'function') {
        globalConfig.onReady(api);
    }

    if (globalConfig.autoStream === false) {
        autoStreamToggle.checked = false;
        autoStream.stop();
        statusMessage.textContent = 'Auto stream disabled by configuration. Apply or randomize data to visualize.';
        if (sonicGeometryEngine) {
            sonicGeometryEngine.setTransportState({
                playing: false,
                mode: 'idle',
                progress: playbackStatusSnapshot ? playbackStatusSnapshot.progress : 0,
                frameIndex: playbackStatusSnapshot ? playbackStatusSnapshot.currentIndex : -1,
                frameCount: playbackStatusSnapshot ? playbackStatusSnapshot.frameCount : 0
            });
            if (sonicGeometryToggle && sonicGeometryToggle.checked) {
                setSonicHelperText('Auto stream disabled. Sonic lattice awaiting manual input.');
            }
        }
    } else {
        autoStreamToggle.checked = true;
        autoStream.start();
        statusMessage.textContent = 'Auto streaming synthetic data…';
        if (sonicGeometryEngine && sonicGeometryToggle && sonicGeometryToggle.checked) {
            sonicGeometryEngine.setTransportState({
                playing: true,
                mode: 'auto-stream',
                progress: playbackStatusSnapshot ? playbackStatusSnapshot.progress : 0,
                frameIndex: playbackStatusSnapshot ? playbackStatusSnapshot.currentIndex : -1,
                frameCount: playbackStatusSnapshot ? playbackStatusSnapshot.frameCount : 0
            });
            const summary = sonicGeometryEngine.getLastSummary();
            setSonicHelperText(summary || 'Synthetic resonance drift engaged.');
        }
    }
});
