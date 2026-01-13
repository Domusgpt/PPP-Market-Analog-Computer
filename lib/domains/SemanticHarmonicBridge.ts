/**
 * SemanticHarmonicBridge.ts
 *
 * Integrates HDCEncoder semantic embeddings with MusicGeometryDomain
 * to create a unified neural-geometric-harmonic space.
 *
 * This allows natural language descriptions like "melancholic minor key"
 * to influence the 4D polytope geometry alongside musical structure.
 *
 * @version 1.0.0
 */

import { MusicGeometryDomain, Vector4D, ChordGeometry, Path4D } from './MusicGeometryDomain.js';

// =============================================================================
// TYPES
// =============================================================================

export interface SemanticHarmonicConfig {
    // Weight for semantic vs structural components (0 = pure music, 1 = pure semantic)
    semanticWeight: number;

    // Voyage API key for embeddings
    voyageApiKey?: string;

    // Rate limiting (requests per minute)
    rateLimit: number;

    // Cache embeddings to reduce API calls
    cacheEnabled: boolean;

    // Musical emotion archetypes
    emotionArchetypes: EmotionArchetype[];
}

export interface EmotionArchetype {
    name: string;
    keywords: string[];
    musicalFeatures: {
        preferredMode: 'major' | 'minor' | 'any';
        tensionRange: [number, number];  // 0-1
        tempoAffinity: 'slow' | 'medium' | 'fast' | 'any';
    };
    baseVector?: Vector4D;  // Computed from embedding
}

export interface SemanticMusicalState {
    // Semantic component
    semanticVector: Vector4D;
    semanticDescription: string;
    dominantEmotion: string;
    emotionConfidence: number;

    // Musical component
    musicalVector: Vector4D;
    currentChord?: ChordGeometry;
    tension: number;

    // Combined
    combinedVector: Vector4D;
    harmonicAlignment: number;  // How well semantic matches musical structure
}

export interface HarmonicSuggestion {
    chord: string[];
    reason: string;
    alignmentScore: number;
    tensionChange: number;
}

// =============================================================================
// DEFAULT EMOTION ARCHETYPES
// =============================================================================

export const DEFAULT_EMOTION_ARCHETYPES: EmotionArchetype[] = [
    {
        name: 'joy',
        keywords: ['happy', 'joyful', 'bright', 'cheerful', 'uplifting', 'elated'],
        musicalFeatures: {
            preferredMode: 'major',
            tensionRange: [0.1, 0.3],
            tempoAffinity: 'fast'
        }
    },
    {
        name: 'sadness',
        keywords: ['sad', 'melancholic', 'sorrowful', 'mournful', 'grief', 'despair'],
        musicalFeatures: {
            preferredMode: 'minor',
            tensionRange: [0.2, 0.5],
            tempoAffinity: 'slow'
        }
    },
    {
        name: 'tension',
        keywords: ['tense', 'anxious', 'suspenseful', 'nervous', 'uneasy', 'dramatic'],
        musicalFeatures: {
            preferredMode: 'minor',
            tensionRange: [0.5, 0.9],
            tempoAffinity: 'medium'
        }
    },
    {
        name: 'peace',
        keywords: ['peaceful', 'calm', 'serene', 'tranquil', 'relaxed', 'gentle'],
        musicalFeatures: {
            preferredMode: 'major',
            tensionRange: [0.0, 0.2],
            tempoAffinity: 'slow'
        }
    },
    {
        name: 'triumph',
        keywords: ['triumphant', 'victorious', 'powerful', 'epic', 'heroic', 'majestic'],
        musicalFeatures: {
            preferredMode: 'major',
            tensionRange: [0.3, 0.6],
            tempoAffinity: 'medium'
        }
    },
    {
        name: 'mystery',
        keywords: ['mysterious', 'enigmatic', 'eerie', 'haunting', 'otherworldly'],
        musicalFeatures: {
            preferredMode: 'minor',
            tensionRange: [0.3, 0.7],
            tempoAffinity: 'slow'
        }
    },
    {
        name: 'anger',
        keywords: ['angry', 'furious', 'aggressive', 'intense', 'fierce', 'rage'],
        musicalFeatures: {
            preferredMode: 'minor',
            tensionRange: [0.6, 1.0],
            tempoAffinity: 'fast'
        }
    },
    {
        name: 'nostalgia',
        keywords: ['nostalgic', 'wistful', 'bittersweet', 'longing', 'reminiscent'],
        musicalFeatures: {
            preferredMode: 'any',
            tensionRange: [0.2, 0.4],
            tempoAffinity: 'medium'
        }
    }
];

// =============================================================================
// DEFAULT CONFIG
// =============================================================================

const DEFAULT_CONFIG: SemanticHarmonicConfig = {
    semanticWeight: 0.3,
    rateLimit: 300,  // After payment method added
    cacheEnabled: true,
    emotionArchetypes: DEFAULT_EMOTION_ARCHETYPES
};

// =============================================================================
// MAIN CLASS
// =============================================================================

export class SemanticHarmonicBridge {
    private config: SemanticHarmonicConfig;
    private musicDomain: MusicGeometryDomain;
    private embeddingCache: Map<string, number[]>;
    private archetypeVectors: Map<string, Vector4D>;
    private lastApiCall: number = 0;
    private initialized: boolean = false;

    constructor(
        musicDomain?: MusicGeometryDomain,
        config: Partial<SemanticHarmonicConfig> = {}
    ) {
        this.config = { ...DEFAULT_CONFIG, ...config };
        this.musicDomain = musicDomain || new MusicGeometryDomain();
        this.embeddingCache = new Map();
        this.archetypeVectors = new Map();
    }

    // =========================================================================
    // INITIALIZATION
    // =========================================================================

    /**
     * Initialize archetype vectors by embedding their keywords
     * Call this once at startup
     */
    async initialize(): Promise<void> {
        if (!this.config.voyageApiKey) {
            console.warn('No Voyage API key - using fallback archetype vectors');
            this.initializeFallbackVectors();
            this.initialized = true;
            return;
        }

        console.log('Initializing emotion archetype embeddings...');

        for (const archetype of this.config.emotionArchetypes) {
            // Combine keywords into a semantic phrase
            const phrase = `${archetype.name}: ${archetype.keywords.join(', ')}`;

            try {
                const embedding = await this.getEmbedding(phrase);
                const vector4D = this.projectTo4D(embedding);
                archetype.baseVector = vector4D;
                this.archetypeVectors.set(archetype.name, vector4D);
                console.log(`  ✓ ${archetype.name}`);
            } catch (e) {
                console.warn(`  ✗ ${archetype.name}: ${e}`);
                // Use fallback
                this.archetypeVectors.set(archetype.name, this.getFallbackVector(archetype.name));
            }
        }

        this.initialized = true;
        console.log('Archetype initialization complete.');
    }

    /**
     * Fallback vectors when API is unavailable
     * Based on musical characteristics
     */
    private initializeFallbackVectors(): void {
        // These are hand-tuned vectors based on musical intuition
        // [brightness, tension, energy, temporal]
        const fallbacks: Record<string, Vector4D> = {
            'joy':      [0.8, -0.3, 0.7, 0.2],
            'sadness':  [-0.6, 0.3, -0.4, -0.3],
            'tension':  [0.1, 0.9, 0.5, 0.1],
            'peace':    [0.4, -0.8, -0.5, -0.2],
            'triumph':  [0.7, 0.4, 0.8, 0.3],
            'mystery':  [-0.3, 0.5, -0.2, -0.5],
            'anger':    [-0.2, 0.8, 0.9, 0.4],
            'nostalgia':[-0.1, 0.2, -0.3, -0.6]
        };

        for (const archetype of this.config.emotionArchetypes) {
            const vector = fallbacks[archetype.name] || [0, 0, 0, 0];
            archetype.baseVector = this.normalize(vector);
            this.archetypeVectors.set(archetype.name, archetype.baseVector);
        }
    }

    private getFallbackVector(name: string): Vector4D {
        const fallbacks: Record<string, Vector4D> = {
            'joy':      [0.8, -0.3, 0.7, 0.2],
            'sadness':  [-0.6, 0.3, -0.4, -0.3],
            'tension':  [0.1, 0.9, 0.5, 0.1],
            'peace':    [0.4, -0.8, -0.5, -0.2],
            'triumph':  [0.7, 0.4, 0.8, 0.3],
            'mystery':  [-0.3, 0.5, -0.2, -0.5],
            'anger':    [-0.2, 0.8, 0.9, 0.4],
            'nostalgia':[-0.1, 0.2, -0.3, -0.6]
        };
        return this.normalize(fallbacks[name] || [0, 0, 0, 0]);
    }

    // =========================================================================
    // EMBEDDING API
    // =========================================================================

    /**
     * Get embedding from Voyage API (with caching and rate limiting)
     */
    private async getEmbedding(text: string): Promise<number[]> {
        // Check cache
        if (this.config.cacheEnabled && this.embeddingCache.has(text)) {
            return this.embeddingCache.get(text)!;
        }

        // Rate limiting
        const now = Date.now();
        const minDelay = 60000 / this.config.rateLimit;  // ms between calls
        const timeSinceLast = now - this.lastApiCall;

        if (timeSinceLast < minDelay) {
            await new Promise(r => setTimeout(r, minDelay - timeSinceLast));
        }
        this.lastApiCall = Date.now();

        // Make API call using curl (Node fetch has issues in some envs)
        const { execSync } = await import('child_process');

        const payload = JSON.stringify({
            model: 'voyage-3',
            input: text,
            input_type: 'document'
        });

        const cmd = `curl -s --connect-timeout 10 --max-time 30 "https://api.voyageai.com/v1/embeddings" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer ${this.config.voyageApiKey}" \
            -d '${payload.replace(/'/g, "'\"'\"'")}'`;

        const result = execSync(cmd, { encoding: 'utf-8', timeout: 35000 });
        const data = JSON.parse(result);

        if (!data.data?.[0]?.embedding) {
            throw new Error('Invalid API response');
        }

        const embedding = data.data[0].embedding;

        // Cache result
        if (this.config.cacheEnabled) {
            this.embeddingCache.set(text, embedding);
        }

        return embedding;
    }

    /**
     * Project high-dimensional embedding to 4D using random projection
     */
    private projectTo4D(embedding: number[]): Vector4D {
        // Johnson-Lindenstrauss projection (deterministic seed for reproducibility)
        const inputDim = embedding.length;
        const outputDim = 4;
        const seed = 42;

        let state = seed;
        const random = () => {
            state = (state * 1103515245 + 12345) & 0x7fffffff;
            return state / 0x7fffffff;
        };

        const scale = Math.sqrt(2 / outputDim);
        const result: Vector4D = [0, 0, 0, 0];

        for (let i = 0; i < outputDim; i++) {
            for (let j = 0; j < inputDim; j++) {
                // Box-Muller for Gaussian
                const u1 = random();
                const u2 = random();
                const gaussian = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
                result[i] += gaussian * scale * embedding[j];
            }
        }

        return this.normalize(result);
    }

    // =========================================================================
    // SEMANTIC ANALYSIS
    // =========================================================================

    /**
     * Analyze text and return semantic 4D vector
     */
    async analyzeSemantics(text: string): Promise<{
        vector: Vector4D;
        dominantEmotion: string;
        confidence: number;
        emotionScores: Record<string, number>;
    }> {
        let semanticVector: Vector4D;

        if (this.config.voyageApiKey) {
            const embedding = await this.getEmbedding(text);
            semanticVector = this.projectTo4D(embedding);
        } else {
            // Keyword-based fallback
            semanticVector = this.keywordAnalysis(text);
        }

        // Match against emotion archetypes
        const emotionScores: Record<string, number> = {};
        let maxScore = -Infinity;
        let dominantEmotion = 'neutral';

        for (const [name, vector] of this.archetypeVectors) {
            const score = this.cosineSimilarity(semanticVector, vector);
            emotionScores[name] = score;

            if (score > maxScore) {
                maxScore = score;
                dominantEmotion = name;
            }
        }

        return {
            vector: semanticVector,
            dominantEmotion,
            confidence: Math.max(0, maxScore),
            emotionScores
        };
    }

    /**
     * Keyword-based fallback analysis
     */
    private keywordAnalysis(text: string): Vector4D {
        const lower = text.toLowerCase();
        const result: Vector4D = [0, 0, 0, 0];
        let matchCount = 0;

        for (const archetype of this.config.emotionArchetypes) {
            for (const keyword of archetype.keywords) {
                if (lower.includes(keyword)) {
                    const vector = this.archetypeVectors.get(archetype.name);
                    if (vector) {
                        result[0] += vector[0];
                        result[1] += vector[1];
                        result[2] += vector[2];
                        result[3] += vector[3];
                        matchCount++;
                    }
                }
            }
        }

        if (matchCount > 0) {
            return this.normalize([
                result[0] / matchCount,
                result[1] / matchCount,
                result[2] / matchCount,
                result[3] / matchCount
            ]);
        }

        return [0, 0, 0, 0];
    }

    // =========================================================================
    // BRIDGE OPERATIONS
    // =========================================================================

    /**
     * Create combined semantic-musical state
     */
    async createState(
        semanticDescription: string,
        chord?: string[]
    ): Promise<SemanticMusicalState> {
        // Get semantic analysis
        const semantic = await this.analyzeSemantics(semanticDescription);

        // Get musical geometry
        let musicalVector: Vector4D = [0, 0, 0, 0];
        let chordGeometry: ChordGeometry | undefined;
        let tension = 0;

        if (chord) {
            chordGeometry = this.musicDomain.chordToPolytope(chord);
            musicalVector = chordGeometry.centroid;
            tension = chordGeometry.tension;
        }

        // Blend vectors
        const w = this.config.semanticWeight;
        const combinedVector: Vector4D = [
            semantic.vector[0] * w + musicalVector[0] * (1 - w),
            semantic.vector[1] * w + musicalVector[1] * (1 - w),
            semantic.vector[2] * w + musicalVector[2] * (1 - w),
            semantic.vector[3] * w + musicalVector[3] * (1 - w)
        ];

        // Calculate harmonic alignment
        const alignment = chord
            ? this.calculateAlignment(semantic.dominantEmotion, chordGeometry!)
            : 0;

        return {
            semanticVector: semantic.vector,
            semanticDescription,
            dominantEmotion: semantic.dominantEmotion,
            emotionConfidence: semantic.confidence,
            musicalVector,
            currentChord: chordGeometry,
            tension,
            combinedVector: this.normalize(combinedVector),
            harmonicAlignment: alignment
        };
    }

    /**
     * Calculate how well a chord aligns with an emotion
     */
    private calculateAlignment(emotion: string, chord: ChordGeometry): number {
        const archetype = this.config.emotionArchetypes.find(a => a.name === emotion);
        if (!archetype) return 0.5;

        let score = 1.0;

        // Check tension alignment
        const [minTension, maxTension] = archetype.musicalFeatures.tensionRange;
        if (chord.tension < minTension) {
            score -= (minTension - chord.tension);
        } else if (chord.tension > maxTension) {
            score -= (chord.tension - maxTension);
        }

        // Check mode alignment (simplified - based on whether chord contains minor/major 3rd)
        // This would need more sophisticated analysis in production

        return Math.max(0, Math.min(1, score));
    }

    /**
     * Suggest chords that align with a semantic description
     */
    async suggestChords(
        semanticDescription: string,
        count: number = 4
    ): Promise<HarmonicSuggestion[]> {
        const semantic = await this.analyzeSemantics(semanticDescription);
        const archetype = this.config.emotionArchetypes.find(
            a => a.name === semantic.dominantEmotion
        );

        if (!archetype) {
            return [];
        }

        // Generate candidate chords based on emotion characteristics
        const candidates: HarmonicSuggestion[] = [];

        // Major chords
        const majorChords = [
            { name: 'C', notes: ['C', 'E', 'G'] },
            { name: 'F', notes: ['F', 'A', 'C'] },
            { name: 'G', notes: ['G', 'B', 'D'] },
            { name: 'D', notes: ['D', 'F#', 'A'] },
            { name: 'A', notes: ['A', 'C#', 'E'] },
            { name: 'E', notes: ['E', 'G#', 'B'] },
        ];

        // Minor chords
        const minorChords = [
            { name: 'Am', notes: ['A', 'C', 'E'] },
            { name: 'Dm', notes: ['D', 'F', 'A'] },
            { name: 'Em', notes: ['E', 'G', 'B'] },
            { name: 'Bm', notes: ['B', 'D', 'F#'] },
            { name: 'F#m', notes: ['F#', 'A', 'C#'] },
            { name: 'C#m', notes: ['C#', 'E', 'G#'] },
        ];

        // Select chord pool based on preferred mode
        let pool: typeof majorChords;
        if (archetype.musicalFeatures.preferredMode === 'major') {
            pool = majorChords;
        } else if (archetype.musicalFeatures.preferredMode === 'minor') {
            pool = minorChords;
        } else {
            pool = [...majorChords, ...minorChords];
        }

        // Score each chord
        for (const { name, notes } of pool) {
            const geom = this.musicDomain.chordToPolytope(notes);
            const alignment = this.calculateAlignment(semantic.dominantEmotion, geom);

            // Check tension range
            const [minT, maxT] = archetype.musicalFeatures.tensionRange;
            const tensionFit = geom.tension >= minT && geom.tension <= maxT;

            candidates.push({
                chord: notes,
                reason: `${name} (${tensionFit ? 'optimal' : 'acceptable'} tension: ${geom.tension.toFixed(2)})`,
                alignmentScore: alignment,
                tensionChange: geom.tension
            });
        }

        // Sort by alignment and return top N
        return candidates
            .sort((a, b) => b.alignmentScore - a.alignmentScore)
            .slice(0, count);
    }

    /**
     * Suggest a progression that follows an emotional arc
     */
    async suggestProgression(
        emotionalArc: string[],  // e.g., ['peaceful', 'tense', 'triumphant']
        chordsPerEmotion: number = 2
    ): Promise<{ emotion: string; chords: HarmonicSuggestion[] }[]> {
        const result: { emotion: string; chords: HarmonicSuggestion[] }[] = [];

        for (const emotion of emotionalArc) {
            const chords = await this.suggestChords(emotion, chordsPerEmotion);
            result.push({ emotion, chords });
        }

        return result;
    }

    // =========================================================================
    // UTILITY
    // =========================================================================

    private normalize(v: Vector4D): Vector4D {
        const mag = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3]);
        if (mag === 0) return [0, 0, 0, 0];
        return [v[0]/mag, v[1]/mag, v[2]/mag, v[3]/mag];
    }

    private cosineSimilarity(a: Vector4D, b: Vector4D): number {
        const dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
        const magA = Math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2] + a[3]*a[3]);
        const magB = Math.sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2] + b[3]*b[3]);
        if (magA === 0 || magB === 0) return 0;
        return dot / (magA * magB);
    }

    // =========================================================================
    // GETTERS
    // =========================================================================

    getMusicDomain(): MusicGeometryDomain {
        return this.musicDomain;
    }

    getArchetypeVectors(): Map<string, Vector4D> {
        return new Map(this.archetypeVectors);
    }

    isInitialized(): boolean {
        return this.initialized;
    }
}

// =============================================================================
// FACTORY
// =============================================================================

/**
 * Create bridge with Voyage API
 */
export function createSemanticHarmonicBridge(
    voyageApiKey: string,
    semanticWeight: number = 0.3
): SemanticHarmonicBridge {
    return new SemanticHarmonicBridge(
        new MusicGeometryDomain(),
        { voyageApiKey, semanticWeight }
    );
}

/**
 * Create bridge without API (keyword-based fallback)
 */
export function createOfflineBridge(
    semanticWeight: number = 0.3
): SemanticHarmonicBridge {
    return new SemanticHarmonicBridge(
        new MusicGeometryDomain(),
        { semanticWeight }
    );
}

export default SemanticHarmonicBridge;
