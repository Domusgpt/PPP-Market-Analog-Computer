/**
 * GeminiAudioOracle.ts
 *
 * Multimodal audio analysis using Gemini 3 Pro for CPE calibration.
 * Provides ground-truth validation of geometric-musical hypotheses
 * through direct acoustic perception.
 *
 * @version 1.0.0
 * @license MIT
 *
 * AUDIO SPECIFICATIONS:
 * - Format: WAV (PCM), MP3, FLAC, OGG supported
 * - Sample Rate: 48kHz recommended (44.1kHz minimum)
 * - Bit Depth: 24-bit recommended (16-bit minimum)
 * - Channels: Stereo or Mono
 * - Max Duration: 9.5 hours per prompt
 * - Max File Size: 20MB inline, unlimited via Files API
 */

import { MusicGeometryDomain, Vector4D, ChordGeometry } from './MusicGeometryDomain.js';
import * as fs from 'fs';
import * as path from 'path';

// =============================================================================
// TYPES
// =============================================================================

export interface GeminiAudioConfig {
    apiKey: string;
    model: 'gemini-3-pro' | 'gemini-3-flash' | 'gemini-2.5-flash-native-audio';
    temperature: number;
    maxOutputTokens: number;
    safetySettings?: SafetySetting[];
}

export interface SafetySetting {
    category: string;
    threshold: string;
}

export interface TensionAnalysis {
    tensionRating: number;          // 1-10 scale
    stability: 'stable' | 'unstable' | 'neutral';
    resolutionNeeded: boolean;
    dominantQuality: 'consonant' | 'dissonant' | 'mixed';
    confidence: number;
    rawResponse?: string;
}

export interface EmotionAnalysis {
    primaryEmotion: string;
    confidence: number;
    secondaryEmotion: string | null;
    valence: number;                // -1 (negative) to +1 (positive)
    arousal: number;                // 0 (calm) to 1 (excited)
    rawResponse?: string;
}

export interface IntervalAnalysis {
    consonance: number;             // 0-1 scale
    roughness: number;              // 0-1 scale (psychoacoustic)
    beating: boolean;               // Audible beating present
    intervalQuality: string;        // e.g., "perfect fifth", "minor second"
    cents: number;                  // Deviation from equal temperament
    rawResponse?: string;
}

export interface ProgressionAnalysis {
    musicalQuality: number;         // 1-10 scale
    voiceLeadingSmooth: boolean;
    cadenceType: string;
    harmonicLogic: number;          // 1-10 scale
    preferredProgression?: 'A' | 'B' | 'equal';
    rawResponse?: string;
}

export interface CommaAnalysis {
    pitchMatch: boolean;            // Do the two pitches sound identical?
    centsDifference: number;        // Perceived difference in cents
    tuningSystem: string;           // Detected tuning system
    commaAudible: boolean;          // Is the Pythagorean comma audible?
    rawResponse?: string;
}

export interface CalibrationResult {
    hypothesis: string;
    supported: boolean;
    correlation: number;
    pValue: number;
    effectSize: number;
    details: Record<string, any>;
}

// =============================================================================
// DEFAULT CONFIGURATION
// =============================================================================

const DEFAULT_CONFIG: Partial<GeminiAudioConfig> = {
    model: 'gemini-3-pro',
    temperature: 0.1,               // Low for consistency
    maxOutputTokens: 512,
    safetySettings: [
        { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_NONE' },
        { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_NONE' },
        { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'BLOCK_NONE' },
        { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_NONE' }
    ]
};

// =============================================================================
// PROMPTS
// =============================================================================

const PROMPTS = {
    tension: `You are an expert music theorist and psychoacoustician analyzing musical tension.

Listen carefully to this audio clip of a musical chord or passage.

Rate the MUSICAL TENSION on a scale of 1-10:
1 = Completely stable, resolved, at rest (e.g., major triad in root position)
5 = Moderate tension, some instability
10 = Maximum tension, extremely unstable, demands immediate resolution

Consider these psychoacoustic factors:
- Interval ratios (simple ratios = consonant, complex = dissonant)
- Roughness (beating between close frequencies)
- Harmonic series alignment
- Tonal stability (relation to implied tonic)

Respond with ONLY valid JSON:
{"tension_rating": <1-10>, "stability": "<stable|unstable|neutral>", "resolution_needed": <true|false>, "dominant_quality": "<consonant|dissonant|mixed>", "confidence": <0.0-1.0>}`,

    emotion: `You are an expert in music cognition and emotion.

Listen to this musical audio and identify its emotional quality.

PRIMARY emotions: joy, sadness, tension, peace, triumph, mystery, anger, nostalgia, neutral

Consider:
- Mode (major/minor associations)
- Register (high = bright, low = dark)
- Harmonic density
- Spectral characteristics

Respond with ONLY valid JSON:
{"primary_emotion": "<emotion>", "confidence": <0.0-1.0>, "secondary_emotion": "<emotion|null>", "valence": <-1.0 to 1.0>, "arousal": <0.0-1.0>}`,

    interval: `You are a psychoacoustician analyzing musical intervals.

Listen to these two pitches played together (or in sequence).

Analyze:
1. CONSONANCE: How pleasant/stable does the interval sound? (0=harsh, 1=pure)
2. ROUGHNESS: Is there audible beating or roughness? (0=none, 1=severe)
3. INTERVAL QUALITY: What interval is this? (e.g., "perfect fifth", "minor third")
4. TUNING: Any deviation from equal temperament?

Respond with ONLY valid JSON:
{"consonance": <0.0-1.0>, "roughness": <0.0-1.0>, "beating": <true|false>, "interval_quality": "<string>", "cents_deviation": <number>}`,

    progression_compare: `You are a master composer evaluating chord progressions.

You will hear TWO chord progressions. Both share the same starting and ending chords but take different harmonic paths.

PROGRESSION A plays first, then PROGRESSION B.

Evaluate which progression is MORE MUSICAL based on:
- Voice leading smoothness (minimal movement between voices)
- Harmonic logic (functional relationships)
- Aesthetic quality (does it "sound right"?)

Respond with ONLY valid JSON:
{"preferred": "<A|B|equal>", "confidence": <0.0-1.0>, "voice_leading_smooth_a": <true|false>, "voice_leading_smooth_b": <true|false>, "harmonic_logic_a": <1-10>, "harmonic_logic_b": <1-10>, "reason": "<brief explanation>"}`,

    comma_detection: `You are a tuning specialist with perfect pitch.

You will hear two pitches that should theoretically be the same note (both are "C" or equivalent).

One is the starting pitch. The other is reached after 12 perfect fifths (which theoretically returns to the same pitch class).

Listen carefully:
1. Do these pitches sound IDENTICAL, or is there a slight difference?
2. If different, which is sharper/higher?
3. Estimate the difference in cents (100 cents = 1 semitone)

Respond with ONLY valid JSON:
{"pitches_identical": <true|false>, "cents_difference": <number>, "higher_pitch": "<first|second|same>", "comma_audible": <true|false>, "tuning_assessment": "<string>"}`,

    dimensional_analysis: `You are analyzing the GEOMETRIC properties of sound.

Listen to this audio and describe its characteristics in terms of:
1. BRIGHTNESS (spectral centroid): -1 (dark) to +1 (bright)
2. DENSITY (harmonic complexity): 0 (sparse) to 1 (dense)
3. MOVEMENT (temporal evolution): 0 (static) to 1 (dynamic)
4. DEPTH (spatial/reverberant): 0 (dry/close) to 1 (wet/distant)

These four dimensions form a 4D coordinate for this sound.

Respond with ONLY valid JSON:
{"brightness": <-1.0 to 1.0>, "density": <0.0-1.0>, "movement": <0.0-1.0>, "depth": <0.0-1.0>, "dominant_dimension": "<brightness|density|movement|depth>"}`
};

// =============================================================================
// MAIN CLASS
// =============================================================================

export class GeminiAudioOracle {
    private config: GeminiAudioConfig;
    private musicDomain: MusicGeometryDomain;
    private callCount: number = 0;
    private lastCallTime: number = 0;

    constructor(apiKey: string, config: Partial<GeminiAudioConfig> = {}) {
        this.config = {
            apiKey,
            ...DEFAULT_CONFIG,
            ...config
        } as GeminiAudioConfig;

        this.musicDomain = new MusicGeometryDomain();
    }

    // =========================================================================
    // CORE API METHODS
    // =========================================================================

    /**
     * Analyze musical tension from audio
     */
    async analyzeTension(audioPath: string): Promise<TensionAnalysis> {
        const response = await this.callGeminiWithAudio(audioPath, PROMPTS.tension);
        return this.parseTensionResponse(response);
    }

    /**
     * Analyze emotional content from audio
     */
    async analyzeEmotion(audioPath: string): Promise<EmotionAnalysis> {
        const response = await this.callGeminiWithAudio(audioPath, PROMPTS.emotion);
        return this.parseEmotionResponse(response);
    }

    /**
     * Analyze interval characteristics
     */
    async analyzeInterval(audioPath: string): Promise<IntervalAnalysis> {
        const response = await this.callGeminiWithAudio(audioPath, PROMPTS.interval);
        return this.parseIntervalResponse(response);
    }

    /**
     * Compare two progressions (A vs B)
     */
    async compareProgressions(audioPathA: string, audioPathB: string): Promise<ProgressionAnalysis> {
        // Concatenate or send both files
        const response = await this.callGeminiWithMultipleAudio(
            [audioPathA, audioPathB],
            PROMPTS.progression_compare
        );
        return this.parseProgressionResponse(response);
    }

    /**
     * Detect Pythagorean comma
     */
    async detectComma(audioPath: string): Promise<CommaAnalysis> {
        const response = await this.callGeminiWithAudio(audioPath, PROMPTS.comma_detection);
        return this.parseCommaResponse(response);
    }

    /**
     * Get 4D perceptual coordinates for audio
     */
    async get4DCoordinates(audioPath: string): Promise<Vector4D> {
        const response = await this.callGeminiWithAudio(audioPath, PROMPTS.dimensional_analysis);
        const parsed = this.parseJSON(response);

        return [
            parsed.brightness ?? 0,
            parsed.density ?? 0,
            parsed.movement ?? 0,
            parsed.depth ?? 0
        ];
    }

    // =========================================================================
    // HYPOTHESIS VALIDATION
    // =========================================================================

    /**
     * Validate Hypothesis 1: Tension Transfer
     * Tests correlation between geometric tension and perceived tension
     */
    async validateH1_TensionTransfer(
        stimuli: Array<{ audioPath: string; chord: string[] }>
    ): Promise<CalibrationResult> {
        const geometricTensions: number[] = [];
        const perceivedTensions: number[] = [];

        console.log(`\nValidating H1: Tension Transfer (n=${stimuli.length})`);

        for (let i = 0; i < stimuli.length; i++) {
            const { audioPath, chord } = stimuli[i];

            // Geometric tension from MusicGeometryDomain
            const geom = this.musicDomain.chordToPolytope(chord);
            geometricTensions.push(geom.tension);

            // Perceived tension from Gemini
            const analysis = await this.analyzeTension(audioPath);
            perceivedTensions.push(analysis.tensionRating / 10); // Normalize to 0-1

            console.log(`  [${i + 1}/${stimuli.length}] ${chord.join('-')}: ` +
                `geom=${geom.tension.toFixed(3)}, perceived=${(analysis.tensionRating / 10).toFixed(3)}`);
        }

        // Calculate Pearson correlation
        const r = this.pearsonCorrelation(geometricTensions, perceivedTensions);
        const n = stimuli.length;
        const t = r * Math.sqrt((n - 2) / (1 - r * r));
        const pValue = this.tDistributionPValue(t, n - 2);

        return {
            hypothesis: 'H1: Tension Transfer',
            supported: r >= 0.5 && pValue < 0.05,
            correlation: r,
            pValue,
            effectSize: r,  // For correlation, r is the effect size
            details: {
                n,
                geometricMean: this.mean(geometricTensions),
                perceivedMean: this.mean(perceivedTensions),
                geometricStd: this.std(geometricTensions),
                perceivedStd: this.std(perceivedTensions)
            }
        };
    }

    /**
     * Validate Hypothesis 2: Geodesic Voice Leading
     * Tests if expert voice leading follows geometric geodesics
     */
    async validateH2_GeodesicVoiceLeading(
        expertProgressions: Array<{ audioPath: string; chords: string[][] }>,
        randomProgressions: Array<{ audioPath: string; chords: string[][] }>
    ): Promise<CalibrationResult> {
        const expertPathLengths: number[] = [];
        const randomPathLengths: number[] = [];
        const expertQualityRatings: number[] = [];
        const randomQualityRatings: number[] = [];

        console.log(`\nValidating H2: Geodesic Voice Leading`);
        console.log(`  Expert: n=${expertProgressions.length}, Random: n=${randomProgressions.length}`);

        // Analyze expert progressions
        for (const prog of expertProgressions) {
            const path = this.musicDomain.progressionToPath(prog.chords);
            expertPathLengths.push(path.length);

            // Optional: get quality rating if audio available
            if (prog.audioPath) {
                const analysis = await this.analyzeTension(prog.audioPath);
                expertQualityRatings.push(10 - analysis.tensionRating); // Invert: low tension = high quality
            }
        }

        // Analyze random progressions
        for (const prog of randomProgressions) {
            const path = this.musicDomain.progressionToPath(prog.chords);
            randomPathLengths.push(path.length);
        }

        // Independent samples t-test
        const expertMean = this.mean(expertPathLengths);
        const randomMean = this.mean(randomPathLengths);
        const expertStd = this.std(expertPathLengths);
        const randomStd = this.std(randomPathLengths);

        const pooledStd = Math.sqrt(
            ((expertPathLengths.length - 1) * expertStd * expertStd +
                (randomPathLengths.length - 1) * randomStd * randomStd) /
            (expertPathLengths.length + randomPathLengths.length - 2)
        );

        const t = (expertMean - randomMean) /
            (pooledStd * Math.sqrt(1 / expertPathLengths.length + 1 / randomPathLengths.length));

        const df = expertPathLengths.length + randomPathLengths.length - 2;
        const pValue = this.tDistributionPValue(Math.abs(t), df);

        // Cohen's d
        const d = (expertMean - randomMean) / pooledStd;

        return {
            hypothesis: 'H2: Geodesic Voice Leading',
            supported: expertMean < randomMean && pValue < 0.05 && Math.abs(d) >= 0.5,
            correlation: 0,  // Not applicable for t-test
            pValue,
            effectSize: d,
            details: {
                expertMean,
                randomMean,
                expertStd,
                randomStd,
                tStatistic: t,
                degreesOfFreedom: df,
                interpretation: expertMean < randomMean ?
                    'Expert paths are shorter (more efficient)' :
                    'Expert paths are NOT shorter than random'
            }
        };
    }

    /**
     * Validate Hypothesis 3: Pythagorean Comma
     * Tests if the comma is geometrically and acoustically detectable
     */
    async validateH3_PythagoreanComma(
        commaAudioPath: string
    ): Promise<CalibrationResult> {
        console.log(`\nValidating H3: Pythagorean Comma`);

        // Geometric calculation: trace 12 fifths
        const startCoord = this.musicDomain.noteToCoordinate('C4');
        let currentCoord = [...startCoord] as Vector4D;

        // In Pythagorean tuning, each fifth is exactly 3:2 ratio
        // After 12 fifths, we should NOT return to exactly the same point
        const fifthRotation = 7; // 7 semitones
        let cumulativeSemitones = 0;

        for (let i = 0; i < 12; i++) {
            cumulativeSemitones += fifthRotation;
            // Map the cumulative pitch class to coordinate
            const pitchClass = cumulativeSemitones % 12;
            const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
            currentCoord = this.musicDomain.noteToCoordinate(noteNames[pitchClass] + '4');
        }

        // Calculate geometric displacement
        const displacement = Math.sqrt(
            Math.pow(currentCoord[0] - startCoord[0], 2) +
            Math.pow(currentCoord[1] - startCoord[1], 2) +
            Math.pow(currentCoord[2] - startCoord[2], 2) +
            Math.pow(currentCoord[3] - startCoord[3], 2)
        );

        // Acoustic analysis via Gemini
        const commaAnalysis = await this.detectComma(commaAudioPath);

        // Expected Pythagorean comma: 23.46 cents
        const expectedCents = 23.46;
        const measuredCents = commaAnalysis.centsDifference;
        const centsError = Math.abs(measuredCents - expectedCents);

        return {
            hypothesis: 'H3: Pythagorean Comma Manifestation',
            supported: commaAnalysis.commaAudible && centsError < 10,
            correlation: 0,  // Not applicable
            pValue: 0,       // Not applicable (direct measurement)
            effectSize: displacement,
            details: {
                geometricDisplacement: displacement,
                expectedCents,
                measuredCents,
                centsError,
                commaAudible: commaAnalysis.commaAudible,
                pitchesIdentical: commaAnalysis.pitchMatch,
                startCoord,
                endCoord: currentCoord,
                tuningAssessment: commaAnalysis.tuningAssessment
            }
        };
    }

    // =========================================================================
    // GEMINI API CALLS
    // =========================================================================

    /**
     * Call Gemini API with audio file
     */
    private async callGeminiWithAudio(audioPath: string, prompt: string): Promise<string> {
        const { execSync } = await import('child_process');

        // Read audio file and encode as base64
        const audioBuffer = fs.readFileSync(audioPath);
        const base64Audio = audioBuffer.toString('base64');

        // Determine MIME type
        const ext = path.extname(audioPath).toLowerCase();
        const mimeTypes: Record<string, string> = {
            '.wav': 'audio/wav',
            '.mp3': 'audio/mp3',
            '.flac': 'audio/flac',
            '.ogg': 'audio/ogg',
            '.m4a': 'audio/m4a',
            '.aac': 'audio/aac'
        };
        const mimeType = mimeTypes[ext] || 'audio/wav';

        // Build request payload
        const payload = {
            contents: [{
                parts: [
                    {
                        inline_data: {
                            mime_type: mimeType,
                            data: base64Audio
                        }
                    },
                    { text: prompt }
                ]
            }],
            generationConfig: {
                temperature: this.config.temperature,
                maxOutputTokens: this.config.maxOutputTokens
            },
            safetySettings: this.config.safetySettings
        };

        // Write payload to temp file (to avoid command line length limits)
        const tempFile = `/tmp/gemini_request_${Date.now()}.json`;
        fs.writeFileSync(tempFile, JSON.stringify(payload));

        try {
            const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${this.config.model}:generateContent?key=${this.config.apiKey}`;

            const result = execSync(
                `curl -s -X POST "${endpoint}" -H "Content-Type: application/json" -d @${tempFile}`,
                { encoding: 'utf-8', timeout: 120000, maxBuffer: 50 * 1024 * 1024 }
            );

            // Parse response
            const response = JSON.parse(result);

            if (response.error) {
                throw new Error(`Gemini API Error: ${response.error.message}`);
            }

            const text = response.candidates?.[0]?.content?.parts?.[0]?.text || '';
            this.callCount++;

            return text;
        } finally {
            // Clean up temp file
            if (fs.existsSync(tempFile)) {
                fs.unlinkSync(tempFile);
            }
        }
    }

    /**
     * Call Gemini with multiple audio files
     */
    private async callGeminiWithMultipleAudio(audioPaths: string[], prompt: string): Promise<string> {
        const { execSync } = await import('child_process');

        const parts: any[] = [];

        // Add each audio file
        for (const audioPath of audioPaths) {
            const audioBuffer = fs.readFileSync(audioPath);
            const base64Audio = audioBuffer.toString('base64');
            const ext = path.extname(audioPath).toLowerCase();
            const mimeType = ext === '.mp3' ? 'audio/mp3' : 'audio/wav';

            parts.push({
                inline_data: {
                    mime_type: mimeType,
                    data: base64Audio
                }
            });
        }

        // Add prompt
        parts.push({ text: prompt });

        const payload = {
            contents: [{ parts }],
            generationConfig: {
                temperature: this.config.temperature,
                maxOutputTokens: this.config.maxOutputTokens
            }
        };

        const tempFile = `/tmp/gemini_multi_${Date.now()}.json`;
        fs.writeFileSync(tempFile, JSON.stringify(payload));

        try {
            const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${this.config.model}:generateContent?key=${this.config.apiKey}`;

            const result = execSync(
                `curl -s -X POST "${endpoint}" -H "Content-Type: application/json" -d @${tempFile}`,
                { encoding: 'utf-8', timeout: 180000, maxBuffer: 100 * 1024 * 1024 }
            );

            const response = JSON.parse(result);

            if (response.error) {
                throw new Error(`Gemini API Error: ${response.error.message}`);
            }

            return response.candidates?.[0]?.content?.parts?.[0]?.text || '';
        } finally {
            if (fs.existsSync(tempFile)) {
                fs.unlinkSync(tempFile);
            }
        }
    }

    // =========================================================================
    // RESPONSE PARSERS
    // =========================================================================

    private parseJSON(text: string): any {
        // Extract JSON from response (may have markdown code blocks)
        const jsonMatch = text.match(/\{[\s\S]*\}/);
        if (!jsonMatch) {
            console.warn('No JSON found in response:', text);
            return {};
        }
        try {
            return JSON.parse(jsonMatch[0]);
        } catch (e) {
            console.warn('Failed to parse JSON:', jsonMatch[0]);
            return {};
        }
    }

    private parseTensionResponse(text: string): TensionAnalysis {
        const json = this.parseJSON(text);
        return {
            tensionRating: json.tension_rating ?? 5,
            stability: json.stability ?? 'neutral',
            resolutionNeeded: json.resolution_needed ?? false,
            dominantQuality: json.dominant_quality ?? 'mixed',
            confidence: json.confidence ?? 0.5,
            rawResponse: text
        };
    }

    private parseEmotionResponse(text: string): EmotionAnalysis {
        const json = this.parseJSON(text);
        return {
            primaryEmotion: json.primary_emotion ?? 'neutral',
            confidence: json.confidence ?? 0.5,
            secondaryEmotion: json.secondary_emotion ?? null,
            valence: json.valence ?? 0,
            arousal: json.arousal ?? 0.5,
            rawResponse: text
        };
    }

    private parseIntervalResponse(text: string): IntervalAnalysis {
        const json = this.parseJSON(text);
        return {
            consonance: json.consonance ?? 0.5,
            roughness: json.roughness ?? 0.5,
            beating: json.beating ?? false,
            intervalQuality: json.interval_quality ?? 'unknown',
            cents: json.cents_deviation ?? 0,
            rawResponse: text
        };
    }

    private parseProgressionResponse(text: string): ProgressionAnalysis {
        const json = this.parseJSON(text);
        return {
            musicalQuality: (json.harmonic_logic_a + json.harmonic_logic_b) / 2 || 5,
            voiceLeadingSmooth: json.voice_leading_smooth_a || json.voice_leading_smooth_b || false,
            cadenceType: 'unknown',
            harmonicLogic: json.harmonic_logic_a ?? 5,
            preferredProgression: json.preferred ?? 'equal',
            rawResponse: text
        };
    }

    private parseCommaResponse(text: string): CommaAnalysis {
        const json = this.parseJSON(text);
        return {
            pitchMatch: json.pitches_identical ?? true,
            centsDifference: json.cents_difference ?? 0,
            tuningSystem: json.tuning_assessment ?? 'unknown',
            commaAudible: json.comma_audible ?? false,
            rawResponse: text
        };
    }

    // =========================================================================
    // STATISTICAL UTILITIES
    // =========================================================================

    private mean(arr: number[]): number {
        return arr.reduce((a, b) => a + b, 0) / arr.length;
    }

    private std(arr: number[]): number {
        const m = this.mean(arr);
        return Math.sqrt(arr.reduce((acc, val) => acc + Math.pow(val - m, 2), 0) / arr.length);
    }

    private pearsonCorrelation(x: number[], y: number[]): number {
        const n = x.length;
        const meanX = this.mean(x);
        const meanY = this.mean(y);

        let numerator = 0;
        let denomX = 0;
        let denomY = 0;

        for (let i = 0; i < n; i++) {
            const dx = x[i] - meanX;
            const dy = y[i] - meanY;
            numerator += dx * dy;
            denomX += dx * dx;
            denomY += dy * dy;
        }

        const denominator = Math.sqrt(denomX * denomY);
        return denominator === 0 ? 0 : numerator / denominator;
    }

    private tDistributionPValue(t: number, df: number): number {
        // Approximation using normal distribution for large df
        if (df > 30) {
            // Two-tailed p-value from standard normal
            const z = Math.abs(t);
            return 2 * (1 - this.normalCDF(z));
        }
        // For small df, use approximation
        const x = df / (df + t * t);
        return this.incompleteBeta(df / 2, 0.5, x);
    }

    private normalCDF(x: number): number {
        const a1 = 0.254829592;
        const a2 = -0.284496736;
        const a3 = 1.421413741;
        const a4 = -1.453152027;
        const a5 = 1.061405429;
        const p = 0.3275911;

        const sign = x < 0 ? -1 : 1;
        x = Math.abs(x) / Math.sqrt(2);

        const t = 1.0 / (1.0 + p * x);
        const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

        return 0.5 * (1.0 + sign * y);
    }

    private incompleteBeta(a: number, b: number, x: number): number {
        // Simplified approximation
        if (x === 0) return 0;
        if (x === 1) return 1;
        return x; // Placeholder - full implementation would use continued fraction
    }

    // =========================================================================
    // GETTERS
    // =========================================================================

    getCallCount(): number {
        return this.callCount;
    }

    getMusicDomain(): MusicGeometryDomain {
        return this.musicDomain;
    }
}

// =============================================================================
// FACTORY FUNCTIONS
// =============================================================================

/**
 * Create oracle with Gemini 3 Pro (recommended for research)
 */
export function createGemini3ProOracle(apiKey: string): GeminiAudioOracle {
    return new GeminiAudioOracle(apiKey, { model: 'gemini-3-pro' });
}

/**
 * Create oracle with Gemini 3 Flash (faster, cheaper)
 */
export function createGemini3FlashOracle(apiKey: string): GeminiAudioOracle {
    return new GeminiAudioOracle(apiKey, { model: 'gemini-3-flash' });
}

export default GeminiAudioOracle;
