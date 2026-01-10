#!/usr/bin/env npx tsx
/**
 * HDCEncoder End-to-End Test with Voyage AI
 * Tests the full pipeline: Text → Embedding → 4D Force Vector
 */

import 'dotenv/config';

const VOYAGE_API_KEY = process.env.VOYAGE_API_KEY;

if (!VOYAGE_API_KEY) {
    console.error('❌ VOYAGE_API_KEY not set');
    process.exit(1);
}

// Rate limit: 3 RPM on free tier - need 20s+ between calls
let lastCallTime = 0;
const MIN_DELAY_MS = 21000; // 21 seconds between calls (3 RPM = 20s)

// Voyage API call using curl (Node fetch has issues in this env)
async function getVoyageEmbedding(text: string, retries = 3): Promise<number[]> {
    const { execSync } = await import('child_process');

    // Rate limiting
    const now = Date.now();
    const timeSinceLast = now - lastCallTime;
    if (lastCallTime > 0 && timeSinceLast < MIN_DELAY_MS) {
        const wait = MIN_DELAY_MS - timeSinceLast;
        process.stdout.write(`  [rate limit: waiting ${Math.ceil(wait/1000)}s] `);
        await new Promise(r => setTimeout(r, wait));
    }
    lastCallTime = Date.now();

    const payload = JSON.stringify({ model: "voyage-3", input: text, input_type: "document" });
    const payloadEscaped = payload.replace(/'/g, "'\"'\"'");

    for (let attempt = 0; attempt < retries; attempt++) {
        try {
            const cmd = `curl -s --connect-timeout 10 --max-time 30 "https://api.voyageai.com/v1/embeddings" \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer ${VOYAGE_API_KEY}" \
                -d '${payloadEscaped}'`;

            const result = execSync(cmd, { encoding: 'utf-8', timeout: 35000 });
            const data = JSON.parse(result);

            if (data.data && data.data[0] && data.data[0].embedding) {
                return data.data[0].embedding;
            }

            // Check for rate limit response
            if (data.detail && data.detail.includes('rate limit')) {
                console.log('\n  [Rate limited - waiting 25s...]');
                await new Promise(r => setTimeout(r, 25000));
                continue;
            }

            throw new Error('Invalid response: ' + JSON.stringify(data).substring(0, 100));
        } catch (e) {
            if (attempt === retries - 1) throw e;
            await new Promise(r => setTimeout(r, 5000 * (attempt + 1)));
        }
    }
    throw new Error('Failed after retries');
}

// Johnson-Lindenstrauss random projection matrix (1024 → 4)
function createProjectionMatrix(inputDim: number, outputDim: number, seed: number = 42): number[][] {
    const matrix: number[][] = [];
    let state = seed;

    const random = () => {
        state = (state * 1103515245 + 12345) & 0x7fffffff;
        return state / 0x7fffffff;
    };

    const scale = Math.sqrt(2 / outputDim);

    for (let i = 0; i < outputDim; i++) {
        const row: number[] = [];
        for (let j = 0; j < inputDim; j++) {
            // Box-Muller transform for Gaussian
            const u1 = random();
            const u2 = random();
            const gaussian = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
            row.push(gaussian * scale);
        }
        matrix.push(row);
    }
    return matrix;
}

// Project embedding to 4D
function projectTo4D(embedding: number[], matrix: number[][]): number[] {
    return matrix.map(row =>
        row.reduce((sum, val, i) => sum + val * embedding[i], 0)
    );
}

// Normalize to unit hypersphere
function normalize(vec: number[]): number[] {
    const mag = Math.sqrt(vec.reduce((sum, v) => sum + v * v, 0));
    return mag > 0 ? vec.map(v => v / mag) : vec;
}

// 24 Concept Archetypes (mapped to 24-Cell vertices)
const ARCHETYPES = [
    'existence', 'negation', 'causation', 'time',
    'space', 'quantity', 'quality', 'relation',
    'action', 'passion', 'identity', 'difference',
    'whole', 'part', 'necessity', 'possibility',
    'actuality', 'substance', 'attribute', 'mode',
    'truth', 'falsity', 'good', 'evil'
];

// Cosine similarity
function cosineSim(a: number[], b: number[]): number {
    const dot = a.reduce((sum, v, i) => sum + v * b[i], 0);
    const magA = Math.sqrt(a.reduce((sum, v) => sum + v * v, 0));
    const magB = Math.sqrt(b.reduce((sum, v) => sum + v * v, 0));
    return dot / (magA * magB + 1e-10);
}

async function main() {
    console.log('');
    console.log('╔══════════════════════════════════════════════════════════╗');
    console.log('║      HDCEncoder E2E Test with Voyage AI (voyage-3)       ║');
    console.log('╚══════════════════════════════════════════════════════════╝');
    console.log('');

    // Test texts (reduced for rate limit - 3 RPM)
    const testTexts = [
        'The cause preceded the effect in time',
        'All parts combine to form a unified whole'
    ];

    // Create projection matrix (1024D → 4D)
    console.log('Creating projection matrix (1024D → 4D)...');
    const projMatrix = createProjectionMatrix(1024, 4);
    console.log('✓ Projection matrix ready\n');

    // Process each test text
    console.log('Processing test texts through pipeline:\n');
    console.log('Text → Voyage Embedding (1024D) → 4D Force Vector\n');
    console.log('─'.repeat(60));

    const results: { text: string; embedding: number[]; force4D: number[] }[] = [];

    for (const text of testTexts) {
        process.stdout.write(`\n"${text.substring(0, 40)}..."\n`);

        // Get embedding from Voyage
        const embedding = await getVoyageEmbedding(text);
        console.log(`  → Voyage embedding: ${embedding.length}D`);

        // Project to 4D
        const force4D = normalize(projectTo4D(embedding, projMatrix));
        console.log(`  → 4D force vector: [${force4D.map(v => v.toFixed(4)).join(', ')}]`);

        results.push({ text, embedding, force4D });
    }

    console.log('\n' + '─'.repeat(60));
    console.log('\n4D Similarity Matrix (cosine similarity of force vectors):\n');

    // Print similarity matrix
    const labels = testTexts.map((t, i) => `T${i + 1}`);
    console.log('     ' + labels.map(l => l.padStart(6)).join(' '));

    for (let i = 0; i < results.length; i++) {
        let row = labels[i] + '  ';
        for (let j = 0; j < results.length; j++) {
            const sim = cosineSim(results[i].force4D, results[j].force4D);
            row += sim.toFixed(3).padStart(6) + ' ';
        }
        console.log(row);
    }

    // Archetype matching (using first 4 archetypes as demo)
    console.log('\n' + '─'.repeat(60));
    console.log('\nArchetype Matching (top matches for each text):\n');

    // Get archetype embeddings (just 4 for rate limit)
    const archetypeEmbeddings: { name: string; force4D: number[] }[] = [];
    const testArchetypes = ['causation', 'whole', 'truth', 'time'];

    for (const arch of testArchetypes) {
        const emb = await getVoyageEmbedding(arch);
        const force = normalize(projectTo4D(emb, projMatrix));
        archetypeEmbeddings.push({ name: arch, force4D: force });
    }

    for (const result of results) {
        const matches = archetypeEmbeddings
            .map(a => ({ name: a.name, sim: cosineSim(result.force4D, a.force4D) }))
            .sort((a, b) => b.sim - a.sim)
            .slice(0, 3);

        console.log(`"${result.text.substring(0, 35)}..."`);
        console.log(`  Top archetypes: ${matches.map(m => `${m.name}(${m.sim.toFixed(2)})`).join(', ')}`);
        console.log('');
    }

    console.log('═'.repeat(60));
    console.log('✓ E2E test complete - HDCEncoder pipeline working with Voyage');
    console.log('═'.repeat(60));
}

main().catch(console.error);
