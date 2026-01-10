import { MusicGeometryDomain } from '../lib/domains/MusicGeometryDomain.js';

const apiKey = process.env.VOYAGE_API_KEY;

if (!apiKey) {
    console.error('VOYAGE_API_KEY is required to run this demo.');
    process.exit(1);
}

const domain = new MusicGeometryDomain();
const sample = 'bright minor pentatonic riff with warm timbre';

const vector = await domain.textToVector4DWithVoyage(sample, apiKey);
console.log('Voyage embedding vector:', vector);
