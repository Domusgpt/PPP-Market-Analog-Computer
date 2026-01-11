import { execFileSync } from 'node:child_process';
import { MusicGeometryDomain } from '../lib/domains/MusicGeometryDomain.js';

const apiKey = process.env.VOYAGE_API_KEY;

if (!apiKey) {
    console.error('VOYAGE_API_KEY is required to run this demo.');
    process.exit(1);
}

const domain = new MusicGeometryDomain();
const sample = 'bright minor pentatonic riff with warm timbre';

const response = execFileSync('curl', [
    '-sS',
    '-X',
    'POST',
    'https://api.voyageai.com/v1/embeddings',
    '-H',
    'Content-Type: application/json',
    '-H',
    `Authorization: Bearer ${apiKey}`,
    '-d',
    JSON.stringify({ model: 'voyage-3', input: sample, input_type: 'document' })
], { encoding: 'utf8' });

const payload = JSON.parse(response);
const embedding = payload?.data?.[0]?.embedding;
if (!embedding) {
    console.error('Voyage response missing embedding:', payload);
    process.exit(1);
}

const vector = domain.embeddingToVector4D(embedding);
console.log('Voyage embedding vector:', vector);
