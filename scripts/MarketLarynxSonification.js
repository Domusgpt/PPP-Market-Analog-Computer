/**
 * Market Larynx Sonification Module
 *
 * Implements the "Harmonic Alpha" paper's music theory to financial markets mapping.
 * This module connects to the WASM Market Larynx engine and produces audio based on
 * market tension levels.
 *
 * Musical Interval Mapping:
 * - Low Tension (< 0.3): Consonance - Perfect 5ths (ratio 3:2) = Bull Market
 * - Medium Tension (0.3-0.7): Partial dissonance = Uncertain Market
 * - High Tension (> 0.7): Dissonance - Tritones (ratio sqrt(2):1) = Bear/Crash Risk
 * - Gamma Event: Resolution - I-IV-V chord progression = Market regime change
 *
 * Integration with SonicGeometryEngine:
 * This module can run alongside or as a layer on top of SonicGeometryEngine,
 * modulating its parameters based on market state.
 */

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

/**
 * Musical intervals with their frequency ratios and consonance values
 */
const MUSICAL_INTERVALS = {
    UNISON: { ratio: 1.0, consonance: 1.0, name: 'Unison' },
    MINOR_SECOND: { ratio: 16/15, consonance: 0.2, name: 'Minor 2nd' },
    MAJOR_SECOND: { ratio: 9/8, consonance: 0.4, name: 'Major 2nd' },
    MINOR_THIRD: { ratio: 6/5, consonance: 0.7, name: 'Minor 3rd' },
    MAJOR_THIRD: { ratio: 5/4, consonance: 0.8, name: 'Major 3rd' },
    PERFECT_FOURTH: { ratio: 4/3, consonance: 0.9, name: 'Perfect 4th' },
    TRITONE: { ratio: Math.sqrt(2), consonance: 0.0, name: 'Tritone' },
    PERFECT_FIFTH: { ratio: 3/2, consonance: 0.95, name: 'Perfect 5th' },
    MINOR_SIXTH: { ratio: 8/5, consonance: 0.65, name: 'Minor 6th' },
    MAJOR_SIXTH: { ratio: 5/3, consonance: 0.75, name: 'Major 6th' },
    MINOR_SEVENTH: { ratio: 9/5, consonance: 0.5, name: 'Minor 7th' },
    MAJOR_SEVENTH: { ratio: 15/8, consonance: 0.45, name: 'Major 7th' },
    OCTAVE: { ratio: 2.0, consonance: 1.0, name: 'Octave' }
};

/**
 * Get interval based on tension level
 */
function getIntervalFromTension(tension) {
    const t = clamp(tension, 0, 1);

    if (t < 0.1) return MUSICAL_INTERVALS.UNISON;
    if (t < 0.2) return MUSICAL_INTERVALS.PERFECT_FIFTH;
    if (t < 0.3) return MUSICAL_INTERVALS.PERFECT_FOURTH;
    if (t < 0.4) return MUSICAL_INTERVALS.MAJOR_THIRD;
    if (t < 0.5) return MUSICAL_INTERVALS.MINOR_THIRD;
    if (t < 0.6) return MUSICAL_INTERVALS.MAJOR_SECOND;
    if (t < 0.7) return MUSICAL_INTERVALS.MINOR_SECOND;
    if (t < 0.85) return MUSICAL_INTERVALS.MINOR_SEVENTH;
    return MUSICAL_INTERVALS.TRITONE;
}

/**
 * Market regime to chord mapping
 */
const REGIME_CHORDS = {
    Bull: {
        intervals: [1, 5/4, 3/2], // Major triad
        name: 'Major (I)',
        color: '#00ff88'
    },
    MildBull: {
        intervals: [1, 5/4, 3/2, 9/8], // Major add9
        name: 'Major Add9',
        color: '#88ff00'
    },
    Neutral: {
        intervals: [1, 9/8, 3/2], // Sus2
        name: 'Suspended 2nd',
        color: '#ffff00'
    },
    MildBear: {
        intervals: [1, 6/5, 3/2], // Minor triad
        name: 'Minor (i)',
        color: '#ff8800'
    },
    Bear: {
        intervals: [1, 6/5, 3/2, 9/5], // Minor 7
        name: 'Minor 7th',
        color: '#ff4400'
    },
    CrashRisk: {
        intervals: [1, 6/5, Math.sqrt(2), 9/5], // Diminished with tritone
        name: 'Diminished 7th',
        color: '#ff0000'
    },
    GammaEvent: {
        // Resolution: V7 -> I (Dominant to Tonic)
        intervals: [1, 5/4, 3/2, 2], // Major with octave
        name: 'Resolution (V7→I)',
        color: '#ff00ff'
    }
};

/**
 * Market Larynx Sonification Engine
 */
export class MarketLarynxSonification {
    constructor(options = {}) {
        this.baseFrequency = options.baseFrequency || 110; // A2
        this.masterVolume = options.masterVolume || 0.3;
        this.transitionTime = options.transitionTime || 0.15;
        this.voiceCount = options.voiceCount || 4;

        this.audioContext = null;
        this.masterGain = null;
        this.voices = [];
        this.isActive = false;

        // State tracking
        this.currentTension = 0;
        this.currentRegime = 'Neutral';
        this.isGammaActive = false;
        this.lastUpdate = 0;

        // Analysis data
        this.tensionHistory = [];
        this.historyMaxLength = 100;

        // Event callbacks
        this.onRegimeChange = options.onRegimeChange || null;
        this.onGammaEvent = options.onGammaEvent || null;
        this.onTensionUpdate = options.onTensionUpdate || null;
    }

    /**
     * Initialize the audio system
     */
    async initialize() {
        if (this.audioContext) return true;

        try {
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            if (!AudioContextClass) {
                console.warn('Web Audio API not supported');
                return false;
            }

            this.audioContext = new AudioContextClass();

            // Create master gain
            this.masterGain = this.audioContext.createGain();
            this.masterGain.gain.value = 0;
            this.masterGain.connect(this.audioContext.destination);

            // Create voice pool
            this._createVoices();

            return true;
        } catch (error) {
            console.error('Failed to initialize audio:', error);
            return false;
        }
    }

    /**
     * Create the voice pool with oscillators
     */
    _createVoices() {
        for (let i = 0; i < this.voiceCount; i++) {
            const osc = this.audioContext.createOscillator();
            const gain = this.audioContext.createGain();
            const filter = this.audioContext.createBiquadFilter();
            const panner = this.audioContext.createStereoPanner();

            // FM modulator for tension effects
            const fmOsc = this.audioContext.createOscillator();
            const fmGain = this.audioContext.createGain();

            filter.type = 'lowpass';
            filter.frequency.value = 2000;
            filter.Q.value = 1;

            gain.gain.value = 0;
            fmGain.gain.value = 0;

            // FM routing
            fmOsc.connect(fmGain);
            fmGain.connect(osc.frequency);

            // Main signal path
            osc.connect(filter);
            filter.connect(gain);
            gain.connect(panner);
            panner.connect(this.masterGain);

            // Start oscillators
            osc.type = 'sine';
            osc.frequency.value = this.baseFrequency;
            fmOsc.type = 'sine';
            fmOsc.frequency.value = 5;

            osc.start();
            fmOsc.start();

            // Spread voices across stereo field
            const panPosition = this.voiceCount > 1
                ? -0.8 + (1.6 * i / (this.voiceCount - 1))
                : 0;
            panner.pan.value = panPosition;

            this.voices.push({
                oscillator: osc,
                gain,
                filter,
                panner,
                fmOsc,
                fmGain,
                index: i
            });
        }
    }

    /**
     * Start audio output
     */
    async enable() {
        if (!this.audioContext) {
            const initialized = await this.initialize();
            if (!initialized) return false;
        }

        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }

        this.isActive = true;
        const now = this.audioContext.currentTime;
        this.masterGain.gain.setTargetAtTime(this.masterVolume, now, this.transitionTime);

        return true;
    }

    /**
     * Stop audio output
     */
    disable() {
        if (!this.audioContext) return;

        this.isActive = false;
        const now = this.audioContext.currentTime;
        this.masterGain.gain.setTargetAtTime(0, now, this.transitionTime);
    }

    /**
     * Update from WASM Market Larynx state
     * @param {Object} wasmEngine - WebEngine WASM instance
     */
    updateFromWasm(wasmEngine) {
        if (!wasmEngine) return;

        try {
            const tension = wasmEngine.get_market_tension?.() || 0;
            const regime = wasmEngine.get_market_regime?.() || 'Neutral';
            const gammaActive = wasmEngine.is_market_gamma_active?.() || false;
            const frequencies = wasmEngine.get_market_sonification_frequencies?.() || [this.baseFrequency, this.baseFrequency * 1.5, this.baseFrequency * 2];

            this.update(tension, regime, gammaActive, frequencies);
        } catch (error) {
            console.warn('Error reading WASM state:', error);
        }
    }

    /**
     * Update the sonification state
     * @param {number} tension - Market tension (0-1)
     * @param {string} regime - Market regime
     * @param {boolean} gammaActive - Whether gamma event is active
     * @param {number[]} frequencies - Suggested frequencies from WASM
     */
    update(tension, regime, gammaActive, frequencies = null) {
        const now = this.audioContext?.currentTime || 0;
        const prevRegime = this.currentRegime;
        const wasGammaActive = this.isGammaActive;

        this.currentTension = tension;
        this.currentRegime = regime;
        this.isGammaActive = gammaActive;
        this.lastUpdate = Date.now();

        // Record history
        this.tensionHistory.push({
            tension,
            regime,
            timestamp: this.lastUpdate
        });
        if (this.tensionHistory.length > this.historyMaxLength) {
            this.tensionHistory.shift();
        }

        // Fire callbacks
        if (regime !== prevRegime && this.onRegimeChange) {
            this.onRegimeChange(regime, prevRegime);
        }
        if (gammaActive && !wasGammaActive && this.onGammaEvent) {
            this.onGammaEvent(tension, regime);
        }
        if (this.onTensionUpdate) {
            this.onTensionUpdate(tension, regime, gammaActive);
        }

        // Update audio if active
        if (this.isActive && this.audioContext) {
            this._updateAudio(tension, regime, gammaActive, frequencies, now);
        }
    }

    /**
     * Update audio parameters based on market state
     */
    _updateAudio(tension, regime, gammaActive, frequencies, now) {
        const interval = getIntervalFromTension(tension);
        const chord = REGIME_CHORDS[regime] || REGIME_CHORDS.Neutral;

        // Base frequency modulated by tension
        const baseFreq = frequencies?.[0] || this.baseFrequency;

        // Update each voice
        for (let i = 0; i < this.voices.length && i < chord.intervals.length; i++) {
            const voice = this.voices[i];
            const targetFreq = baseFreq * chord.intervals[i];

            // Smooth frequency transition
            voice.oscillator.frequency.setTargetAtTime(targetFreq, now, this.transitionTime);

            // Gain based on consonance and voice position
            const voiceGain = gammaActive
                ? (1 - Math.abs(i - this.voiceCount / 2) / this.voiceCount) * 0.8
                : (0.3 + interval.consonance * 0.4) / Math.sqrt(i + 1);
            voice.gain.gain.setTargetAtTime(voiceGain, now, this.transitionTime);

            // FM modulation increases with tension
            const fmAmount = tension * 50 * (1 + i * 0.5);
            voice.fmGain.gain.setTargetAtTime(fmAmount, now, this.transitionTime);
            voice.fmOsc.frequency.setTargetAtTime(3 + tension * 20, now, this.transitionTime);

            // Filter cutoff decreases with tension (darker sound)
            const filterFreq = 3000 - tension * 2000 + (gammaActive ? 1000 : 0);
            voice.filter.frequency.setTargetAtTime(filterFreq, now, this.transitionTime);

            // Waveform changes with regime
            const waveform = this._getWaveformForRegime(regime, tension);
            if (voice.oscillator.type !== waveform) {
                voice.oscillator.type = waveform;
            }
        }

        // Mute unused voices
        for (let i = chord.intervals.length; i < this.voices.length; i++) {
            this.voices[i].gain.gain.setTargetAtTime(0, now, this.transitionTime);
        }

        // Gamma event: play resolution sequence
        if (gammaActive) {
            this._playGammaResolution(now);
        }
    }

    /**
     * Get appropriate waveform for the market regime
     */
    _getWaveformForRegime(regime, tension) {
        switch (regime) {
            case 'Bull':
            case 'MildBull':
                return 'sine';
            case 'Neutral':
                return 'triangle';
            case 'MildBear':
                return tension > 0.5 ? 'sawtooth' : 'triangle';
            case 'Bear':
                return 'sawtooth';
            case 'CrashRisk':
            case 'GammaEvent':
                return 'square';
            default:
                return 'sine';
        }
    }

    /**
     * Play the gamma (crash/resolution) audio event
     */
    _playGammaResolution(now) {
        // Create a short "resolution" sound effect
        // V7 -> I cadence over 1 second

        const osc = this.audioContext.createOscillator();
        const gain = this.audioContext.createGain();
        const filter = this.audioContext.createBiquadFilter();

        filter.type = 'lowpass';
        filter.frequency.value = 1500;

        osc.connect(filter);
        filter.connect(gain);
        gain.connect(this.masterGain);

        // Dominant 7th chord arpeggio -> Resolution
        const baseFreq = this.baseFrequency * 1.5; // Start on dominant
        const sequence = [
            { freq: baseFreq, time: 0 },           // V root
            { freq: baseFreq * 5/4, time: 0.1 },   // V major 3rd
            { freq: baseFreq * 3/2, time: 0.2 },   // V 5th
            { freq: baseFreq * 9/5, time: 0.3 },   // V 7th
            { freq: this.baseFrequency, time: 0.5 }, // Resolve to I
        ];

        // Program the sequence
        osc.type = 'triangle';
        for (const step of sequence) {
            osc.frequency.setValueAtTime(step.freq, now + step.time);
        }

        // Envelope
        gain.gain.setValueAtTime(0, now);
        gain.gain.linearRampToValueAtTime(0.4, now + 0.05);
        gain.gain.setValueAtTime(0.4, now + 0.4);
        gain.gain.linearRampToValueAtTime(0.6, now + 0.5);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 1.5);

        // Start and stop
        osc.start(now);
        osc.stop(now + 1.6);
    }

    /**
     * Get the current state for visualization/debugging
     */
    getState() {
        const interval = getIntervalFromTension(this.currentTension);
        const chord = REGIME_CHORDS[this.currentRegime] || REGIME_CHORDS.Neutral;

        return {
            tension: this.currentTension,
            regime: this.currentRegime,
            gammaActive: this.isGammaActive,
            isActive: this.isActive,
            interval: {
                name: interval.name,
                ratio: interval.ratio,
                consonance: interval.consonance
            },
            chord: {
                name: chord.name,
                intervals: chord.intervals,
                color: chord.color
            },
            frequencies: this.voices.map(v => v.oscillator.frequency.value),
            history: this.tensionHistory.slice(-20)
        };
    }

    /**
     * Dispose of audio resources
     */
    dispose() {
        this.disable();

        if (this.audioContext) {
            for (const voice of this.voices) {
                voice.oscillator.stop();
                voice.fmOsc.stop();
            }
            this.audioContext.close();
            this.audioContext = null;
        }

        this.voices = [];
    }
}

/**
 * Integration helper to connect with SonicGeometryEngine
 */
export class MarketLarynxIntegration {
    constructor(sonicEngine, wasmEngine) {
        this.sonicEngine = sonicEngine;
        this.wasmEngine = wasmEngine;
        this.larynx = new MarketLarynxSonification();
        this.blendRatio = 0.5; // How much market audio vs geometry audio
    }

    async initialize() {
        await this.larynx.initialize();
    }

    async enable() {
        await this.larynx.enable();
    }

    disable() {
        this.larynx.disable();
    }

    /**
     * Update both engines
     */
    update() {
        if (this.wasmEngine) {
            this.larynx.updateFromWasm(this.wasmEngine);

            // Optionally modulate SonicGeometryEngine based on market state
            if (this.sonicEngine && this.larynx.currentTension > 0.5) {
                // High market tension: modify geometry audio parameters
                // This creates a unified audio experience
            }
        }
    }

    getState() {
        return {
            larynx: this.larynx.getState(),
            blend: this.blendRatio
        };
    }

    dispose() {
        this.larynx.dispose();
    }
}

export default MarketLarynxSonification;
