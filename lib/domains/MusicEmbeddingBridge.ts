/**
 * MusicEmbeddingBridge - project external embeddings (e.g., Voyage) into 4D vectors.
 *
 * This adapter bridges neural embeddings into the CPE geometry using HDCEncoder.
 */

import { Vector4D } from '../../types/index.js';
import { HDCEncoder } from '../encoding/HDCEncoder.js';

export class MusicEmbeddingBridge {
    private readonly _encoder: HDCEncoder;

    constructor(encoder: HDCEncoder = new HDCEncoder()) {
        this._encoder = encoder;
    }

    embeddingToVector4D(embedding: Float32Array | number[]): Vector4D {
        return this._encoder.embeddingToForce(embedding).linear;
    }

    chordEmbeddingToVector4D(embeddings: Array<Float32Array | number[]>): Vector4D {
        if (embeddings.length === 0) {
            return [0, 0, 0, 0];
        }
        const accum: Vector4D = [0, 0, 0, 0];
        for (const embedding of embeddings) {
            const vec = this.embeddingToVector4D(embedding);
            accum[0] += vec[0];
            accum[1] += vec[1];
            accum[2] += vec[2];
            accum[3] += vec[3];
        }
        return [
            accum[0] / embeddings.length,
            accum[1] / embeddings.length,
            accum[2] / embeddings.length,
            accum[3] / embeddings.length
        ];
    }

    async textToVector4DWithVoyage(
        text: string,
        apiKey: string,
        model = 'voyage-3'
    ): Promise<Vector4D> {
        const encoder = new HDCEncoder({
            inputDimension: 1024,
            embeddingAPI: {
                provider: 'voyage',
                apiKey,
                model
            }
        });
        const result = await encoder.encodeTextAsync(text);
        return result.force.linear;
    }
}

export const musicEmbeddingBridge = new MusicEmbeddingBridge();
