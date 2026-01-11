/**
 * Gemini Audio Analysis Service
 *
 * Uses Gemini's multimodal capabilities to analyze audio and verify
 * our geometric model predictions with real audio perception.
 *
 * CALIBRATION STRATEGY:
 * 1. Geometric Model predicts relationships (e.g., "G7→C resolves")
 * 2. Generate actual audio for the musical concepts
 * 3. Gemini analyzes the audio (oracle/ground truth)
 * 4. Compare predictions to perception
 *
 * Models:
 * - gemini-2.0-flash-exp: Latest multimodal with audio understanding
 * - gemini-1.5-pro: Stable multimodal
 */

// ============================================================================
// Types
// ============================================================================

export interface GeminiAudioConfig {
  apiKey: string;
  model?: string;
}

export interface AudioAnalysisResult {
  /** Raw response from Gemini */
  rawResponse: string;
  /** Parsed musical analysis */
  analysis: {
    /** Detected key/tonality */
    detectedKey?: string;
    /** Detected mode (major/minor) */
    detectedMode?: 'major' | 'minor' | 'unknown';
    /** Detected chord or notes */
    detectedChord?: string;
    /** Perceived tension level (0-1) */
    tensionLevel?: number;
    /** Perceived brightness/darkness */
    brightness?: 'bright' | 'neutral' | 'dark';
    /** Emotional quality */
    emotion?: string;
    /** Consonance rating (0-1, 1=very consonant) */
    consonance?: number;
  };
  /** Model used */
  model: string;
  /** Timestamp */
  timestamp: string;
}

export interface AudioComparisonResult {
  /** First audio analysis */
  audioA: AudioAnalysisResult;
  /** Second audio analysis */
  audioB: AudioAnalysisResult;
  /** Comparison analysis */
  comparison: {
    /** Are they in the same key? */
    sameKey: boolean;
    /** Perceived similarity (0-1) */
    similarity: number;
    /** Which feels more tense? */
    moreTense: 'A' | 'B' | 'equal';
    /** Does A resolve to B? */
    resolvesAtoB: boolean;
    /** Relationship description */
    relationship: string;
    /** Circle of fifths distance (if detectable) */
    fifthsDistance?: number;
  };
  /** Raw comparison response */
  rawResponse: string;
}

export interface AudioPart {
  inlineData: {
    mimeType: string;
    data: string; // base64
  };
}

export interface TextPart {
  text: string;
}

type ContentPart = AudioPart | TextPart;

// ============================================================================
// Constants
// ============================================================================

const GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models';
const DEFAULT_AUDIO_MODEL = 'gemini-2.0-flash-exp';

// ============================================================================
// Audio Utilities
// ============================================================================

/**
 * Convert Float32Array audio samples to WAV format base64
 */
export function samplesToWavBase64(
  samples: Float32Array,
  sampleRate: number = 44100
): string {
  // WAV header + PCM data
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
  const blockAlign = numChannels * (bitsPerSample / 8);
  const dataSize = samples.length * (bitsPerSample / 8);
  const fileSize = 44 + dataSize;

  const buffer = new ArrayBuffer(fileSize);
  const view = new DataView(buffer);

  // RIFF header
  writeString(view, 0, 'RIFF');
  view.setUint32(4, fileSize - 8, true);
  writeString(view, 8, 'WAVE');

  // fmt chunk
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // chunk size
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);

  // data chunk
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  // PCM samples (convert float to 16-bit int)
  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    const intSample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
    view.setInt16(offset, intSample, true);
    offset += 2;
  }

  // Convert to base64
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function writeString(view: DataView, offset: number, str: string): void {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

// ============================================================================
// Gemini Audio Service
// ============================================================================

export class GeminiAudioService {
  private apiKey: string;
  private model: string;

  constructor(config: GeminiAudioConfig) {
    this.apiKey = config.apiKey;
    this.model = config.model || DEFAULT_AUDIO_MODEL;
  }

  /**
   * Analyze a single audio sample
   */
  async analyzeAudio(
    audioBase64: string,
    mimeType: string = 'audio/wav',
    prompt?: string
  ): Promise<AudioAnalysisResult> {
    const defaultPrompt = `Analyze this audio sample musically. Respond in JSON format:
{
  "detectedKey": "C" | "C#" | "D" | etc. or null if unclear,
  "detectedMode": "major" | "minor" | "unknown",
  "detectedChord": "description of chord/notes heard",
  "tensionLevel": 0.0-1.0 (0=very relaxed, 1=very tense),
  "brightness": "bright" | "neutral" | "dark",
  "emotion": "description of emotional quality",
  "consonance": 0.0-1.0 (1=very consonant/pleasant)
}
Only output JSON, no other text.`;

    const contents: ContentPart[] = [
      {
        inlineData: {
          mimeType,
          data: audioBase64,
        },
      },
      {
        text: prompt || defaultPrompt,
      },
    ];

    const response = await this.callGemini(contents);
    const analysis = this.parseAnalysisResponse(response);

    return {
      rawResponse: response,
      analysis,
      model: this.model,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Compare two audio samples
   */
  async compareAudio(
    audioABase64: string,
    audioBBase64: string,
    mimeType: string = 'audio/wav'
  ): Promise<AudioComparisonResult> {
    // First, analyze each audio individually
    const [audioA, audioB] = await Promise.all([
      this.analyzeAudio(audioABase64, mimeType),
      this.analyzeAudio(audioBBase64, mimeType),
    ]);

    // Then compare them together
    const comparisonPrompt = `You are hearing two audio samples. Compare them musically.

Respond in JSON format:
{
  "sameKey": true/false,
  "similarity": 0.0-1.0 (how similar they sound),
  "moreTense": "A" | "B" | "equal",
  "resolvesAtoB": true/false (does the first naturally lead to the second?),
  "relationship": "description of musical relationship",
  "fifthsDistance": number or null (circle of fifths distance if detectable)
}
Only output JSON.`;

    const contents: ContentPart[] = [
      { text: 'Audio A:' },
      { inlineData: { mimeType, data: audioABase64 } },
      { text: 'Audio B:' },
      { inlineData: { mimeType, data: audioBBase64 } },
      { text: comparisonPrompt },
    ];

    const comparisonResponse = await this.callGemini(contents);
    const comparison = this.parseComparisonResponse(comparisonResponse);

    return {
      audioA,
      audioB,
      comparison,
      rawResponse: comparisonResponse,
    };
  }

  /**
   * Ask a specific question about audio
   */
  async queryAudio(
    audioBase64: string,
    question: string,
    mimeType: string = 'audio/wav'
  ): Promise<string> {
    const contents: ContentPart[] = [
      { inlineData: { mimeType, data: audioBase64 } },
      { text: question },
    ];

    return this.callGemini(contents);
  }

  /**
   * Rate musical tension/resolution between two audio samples
   */
  async rateTensionResolution(
    audioFromBase64: string,
    audioToBase64: string,
    mimeType: string = 'audio/wav'
  ): Promise<{
    tensionChange: number; // negative = resolution, positive = building tension
    isResolution: boolean;
    strength: number; // 0-1
    explanation: string;
  }> {
    const prompt = `Listen to these two audio samples in sequence.

Audio 1 is the "from" state, Audio 2 is the "to" state.

Evaluate the tension/resolution:
- Does Audio 1 naturally want to move to Audio 2?
- Is this a resolution (tension decreasing) or tension building?
- How strong is this effect?

Respond in JSON:
{
  "tensionChange": -1.0 to 1.0 (negative=resolution, positive=building),
  "isResolution": true/false,
  "strength": 0.0-1.0,
  "explanation": "why this works or doesn't work"
}
Only JSON.`;

    const contents: ContentPart[] = [
      { text: 'Audio 1 (FROM):' },
      { inlineData: { mimeType, data: audioFromBase64 } },
      { text: 'Audio 2 (TO):' },
      { inlineData: { mimeType, data: audioToBase64 } },
      { text: prompt },
    ];

    const response = await this.callGemini(contents);

    try {
      const parsed = this.extractJson(response);
      return {
        tensionChange: parsed.tensionChange ?? 0,
        isResolution: parsed.isResolution ?? false,
        strength: parsed.strength ?? 0.5,
        explanation: parsed.explanation ?? response,
      };
    } catch {
      return {
        tensionChange: 0,
        isResolution: false,
        strength: 0.5,
        explanation: response,
      };
    }
  }

  // ==========================================================================
  // Private Methods
  // ==========================================================================

  private async callGemini(contents: ContentPart[]): Promise<string> {
    const url = `${GEMINI_API_URL}/${this.model}:generateContent?key=${this.apiKey}`;

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents: [{ parts: contents }],
        generationConfig: {
          temperature: 0.3, // Lower for more consistent analysis
          maxOutputTokens: 1024,
        },
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Gemini audio API error: ${response.status} - ${error}`);
    }

    const data = await response.json();

    if (!data.candidates?.[0]?.content?.parts?.[0]?.text) {
      throw new Error('Invalid Gemini audio response');
    }

    return data.candidates[0].content.parts[0].text;
  }

  private extractJson(text: string): Record<string, unknown> {
    // Try to extract JSON from response
    let jsonStr = text.trim();

    // Handle markdown code blocks
    const jsonMatch = jsonStr.match(/```(?:json)?\s*([\s\S]*?)```/);
    if (jsonMatch) {
      jsonStr = jsonMatch[1].trim();
    }

    return JSON.parse(jsonStr);
  }

  private parseAnalysisResponse(response: string): AudioAnalysisResult['analysis'] {
    try {
      const parsed = this.extractJson(response);
      return {
        detectedKey: parsed.detectedKey as string | undefined,
        detectedMode: parsed.detectedMode as 'major' | 'minor' | 'unknown' | undefined,
        detectedChord: parsed.detectedChord as string | undefined,
        tensionLevel: parsed.tensionLevel as number | undefined,
        brightness: parsed.brightness as 'bright' | 'neutral' | 'dark' | undefined,
        emotion: parsed.emotion as string | undefined,
        consonance: parsed.consonance as number | undefined,
      };
    } catch {
      return {};
    }
  }

  private parseComparisonResponse(response: string): AudioComparisonResult['comparison'] {
    try {
      const parsed = this.extractJson(response);
      return {
        sameKey: parsed.sameKey as boolean ?? false,
        similarity: parsed.similarity as number ?? 0.5,
        moreTense: (parsed.moreTense as 'A' | 'B' | 'equal') ?? 'equal',
        resolvesAtoB: parsed.resolvesAtoB as boolean ?? false,
        relationship: parsed.relationship as string ?? 'unknown',
        fifthsDistance: parsed.fifthsDistance as number | undefined,
      };
    } catch {
      return {
        sameKey: false,
        similarity: 0.5,
        moreTense: 'equal',
        resolvesAtoB: false,
        relationship: response,
      };
    }
  }
}

// ============================================================================
// Singleton
// ============================================================================

let globalGeminiAudioService: GeminiAudioService | null = null;

export function initializeGeminiAudio(apiKey: string, model?: string): GeminiAudioService {
  globalGeminiAudioService = new GeminiAudioService({ apiKey, model });
  return globalGeminiAudioService;
}

export function getGeminiAudioService(): GeminiAudioService {
  if (!globalGeminiAudioService) {
    throw new Error('Gemini audio service not initialized. Call initializeGeminiAudio(apiKey) first.');
  }
  return globalGeminiAudioService;
}

export function resetGeminiAudioService(): void {
  globalGeminiAudioService = null;
}
