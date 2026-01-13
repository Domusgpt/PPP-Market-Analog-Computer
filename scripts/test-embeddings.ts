#!/usr/bin/env npx tsx
/**
 * HDCEncoder Embedding API Test Script
 *
 * Tests the embedding integrations with real API calls.
 *
 * Setup:
 * 1. Copy .env.example to .env
 * 2. Add your API keys:
 *    - GOOGLE_API_KEY from https://aistudio.google.com/app/apikey
 *    - VOYAGE_API_KEY from https://dash.voyageai.com/api-keys
 * 3. Run: npm run test:embeddings
 */

import * as dotenv from 'dotenv';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Load environment variables
dotenv.config({ path: path.join(__dirname, '..', '.env') });

// Test configuration
const TEST_TEXTS = [
    "reasoning about causality",
    "the dog chased the cat",
    "machine learning algorithms",
    "quantum entanglement phenomena"
];

// Colors for terminal output
const colors = {
    reset: '\x1b[0m',
    green: '\x1b[32m',
    red: '\x1b[31m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    cyan: '\x1b[36m',
    dim: '\x1b[2m'
};

function log(message: string, color: keyof typeof colors = 'reset') {
    console.log(`${colors[color]}${message}${colors.reset}`);
}

function logSection(title: string) {
    console.log('\n' + '='.repeat(60));
    log(title, 'cyan');
    console.log('='.repeat(60));
}

function logResult(label: string, success: boolean, details?: string) {
    const icon = success ? '✓' : '✗';
    const color = success ? 'green' : 'red';
    log(`  ${icon} ${label}`, color);
    if (details) {
        log(`    ${details}`, 'dim');
    }
}

// Direct API test functions (bypassing module imports for now)
interface EmbeddingResult {
    embedding: number[];
    model?: string;
}

async function testGeminiAPI(apiKey: string, text: string): Promise<EmbeddingResult> {
    const model = 'text-embedding-004';
    const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${model}:embedContent?key=${apiKey}`;

    const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model: `models/${model}`,
            content: { parts: [{ text }] },
            taskType: 'RETRIEVAL_DOCUMENT'
        })
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Gemini API error: ${response.status} - ${errorText}`);
    }

    const data = await response.json();
    return {
        embedding: data.embedding?.values || data.embedding,
        model
    };
}

async function testVoyageAPI(apiKey: string, text: string): Promise<EmbeddingResult> {
    const model = 'voyage-3';
    const endpoint = 'https://api.voyageai.com/v1/embeddings';

    const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify({
            model,
            input: text,
            input_type: 'document'
        })
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Voyage API error: ${response.status} - ${errorText}`);
    }

    const data = await response.json();
    return {
        embedding: data.data[0].embedding,
        model
    };
}

async function testGemini() {
    logSection('Testing Google Gemini Embeddings');

    const apiKey = process.env.GOOGLE_API_KEY;
    if (!apiKey || apiKey === 'your_gemini_api_key_here') {
        log('⚠ GOOGLE_API_KEY not set. Skipping Gemini tests.', 'yellow');
        log('  Get your key at: https://aistudio.google.com/app/apikey', 'dim');
        return false;
    }

    log(`API Key: ${apiKey.substring(0, 8)}...${apiKey.substring(apiKey.length - 4)}`, 'dim');

    let allPassed = true;
    for (const text of TEST_TEXTS) {
        const start = Date.now();
        try {
            const result = await testGeminiAPI(apiKey, text);
            const time = Date.now() - start;

            if (!result.embedding || result.embedding.length === 0) {
                throw new Error('Empty embedding returned');
            }

            logResult(
                `"${text.substring(0, 30)}..."`,
                true,
                `${time}ms, ${result.embedding.length} dims`
            );
        } catch (error) {
            const time = Date.now() - start;
            logResult(
                `"${text.substring(0, 30)}..."`,
                false,
                error instanceof Error ? error.message : String(error)
            );
            allPassed = false;
        }
    }

    if (allPassed) {
        const result = await testGeminiAPI(apiKey, TEST_TEXTS[0]);
        log('\nSample embedding (first 5 values):', 'blue');
        log(`  [${result.embedding.slice(0, 5).map(v => v.toFixed(4)).join(', ')}...]`, 'dim');
        log(`  Dimensions: ${result.embedding.length}`, 'dim');
    }

    return allPassed;
}

async function testVoyage() {
    logSection('Testing Voyage AI (Anthropic-recommended) Embeddings');

    const apiKey = process.env.VOYAGE_API_KEY;
    if (!apiKey || apiKey === 'your_voyage_api_key_here') {
        log('⚠ VOYAGE_API_KEY not set. Skipping Voyage tests.', 'yellow');
        log('  Get your key at: https://dash.voyageai.com/api-keys', 'dim');
        return false;
    }

    log(`API Key: ${apiKey.substring(0, 8)}...${apiKey.substring(apiKey.length - 4)}`, 'dim');

    let allPassed = true;
    for (const text of TEST_TEXTS) {
        const start = Date.now();
        try {
            const result = await testVoyageAPI(apiKey, text);
            const time = Date.now() - start;

            if (!result.embedding || result.embedding.length === 0) {
                throw new Error('Empty embedding returned');
            }

            logResult(
                `"${text.substring(0, 30)}..."`,
                true,
                `${time}ms, ${result.embedding.length} dims`
            );
        } catch (error) {
            const time = Date.now() - start;
            logResult(
                `"${text.substring(0, 30)}..."`,
                false,
                error instanceof Error ? error.message : String(error)
            );
            allPassed = false;
        }
    }

    if (allPassed) {
        const result = await testVoyageAPI(apiKey, TEST_TEXTS[0]);
        log('\nSample embedding (first 5 values):', 'blue');
        log(`  [${result.embedding.slice(0, 5).map(v => v.toFixed(4)).join(', ')}...]`, 'dim');
        log(`  Dimensions: ${result.embedding.length}`, 'dim');
    }

    return allPassed;
}

async function testSemanticSimilarity() {
    logSection('Testing Semantic Similarity');

    const apiKey = process.env.GOOGLE_API_KEY;
    if (!apiKey || apiKey === 'your_gemini_api_key_here') {
        log('⚠ Skipping (requires GOOGLE_API_KEY)', 'yellow');
        return false;
    }

    const pairs = [
        { a: 'dog', b: 'canine', expected: 'similar' },
        { a: 'dog', b: 'economics', expected: 'different' },
        { a: 'happy', b: 'joyful', expected: 'similar' },
        { a: 'hot', b: 'cold', expected: 'opposite' }
    ];

    let allPassed = true;
    for (const pair of pairs) {
        try {
            const resultA = await testGeminiAPI(apiKey, pair.a);
            const resultB = await testGeminiAPI(apiKey, pair.b);

            // Compute cosine similarity
            let dotProduct = 0;
            let magA = 0;
            let magB = 0;
            const len = Math.min(resultA.embedding.length, resultB.embedding.length);

            for (let i = 0; i < len; i++) {
                dotProduct += resultA.embedding[i] * resultB.embedding[i];
                magA += resultA.embedding[i] * resultA.embedding[i];
                magB += resultB.embedding[i] * resultB.embedding[i];
            }
            const similarity = dotProduct / (Math.sqrt(magA) * Math.sqrt(magB));

            const isSimilar = similarity > 0.7;
            const matchesExpected = pair.expected === 'similar' ? isSimilar :
                pair.expected === 'different' ? similarity < 0.5 :
                    similarity > 0.3 && similarity < 0.7; // opposite

            logResult(
                `"${pair.a}" vs "${pair.b}"`,
                matchesExpected,
                `similarity: ${(similarity * 100).toFixed(1)}% (expected: ${pair.expected})`
            );

            if (!matchesExpected) allPassed = false;
        } catch (error) {
            logResult(
                `"${pair.a}" vs "${pair.b}"`,
                false,
                error instanceof Error ? error.message : String(error)
            );
            allPassed = false;
        }
    }

    return allPassed;
}

async function main() {
    console.log('\n' + '╔' + '═'.repeat(58) + '╗');
    console.log('║' + ' '.repeat(15) + 'HDCEncoder Embedding Test Suite' + ' '.repeat(11) + '║');
    console.log('╚' + '═'.repeat(58) + '╝');

    const results: { name: string; passed: boolean | null }[] = [];

    // Test Gemini
    results.push({ name: 'Google Gemini', passed: await testGemini() });

    // Test Voyage/Anthropic
    results.push({ name: 'Voyage (Anthropic)', passed: await testVoyage() });

    // Test semantic similarity
    results.push({ name: 'Semantic Similarity', passed: await testSemanticSimilarity() });

    // Summary
    logSection('Test Summary');
    for (const result of results) {
        if (result.passed === null) {
            log(`  ○ ${result.name} (skipped)`, 'yellow');
        } else {
            logResult(result.name, result.passed);
        }
    }

    const testedCount = results.filter(r => r.passed !== null).length;
    const passedCount = results.filter(r => r.passed === true).length;

    console.log('\n' + '-'.repeat(60));
    if (testedCount === 0) {
        log('No API keys configured. Set up your .env file to run tests.', 'yellow');
    } else if (passedCount === testedCount) {
        log(`All ${testedCount} test suites passed!`, 'green');
    } else {
        log(`${passedCount}/${testedCount} test suites passed`, 'yellow');
    }

    // Environment check
    console.log('\n' + '-'.repeat(60));
    log('Environment:', 'blue');
    log(`  GOOGLE_API_KEY: ${process.env.GOOGLE_API_KEY ? '✓ set' : '✗ not set'}`, process.env.GOOGLE_API_KEY ? 'green' : 'red');
    log(`  VOYAGE_API_KEY: ${process.env.VOYAGE_API_KEY ? '✓ set' : '✗ not set'}`, process.env.VOYAGE_API_KEY ? 'green' : 'red');

    if (!process.env.GOOGLE_API_KEY || !process.env.VOYAGE_API_KEY) {
        console.log('\n' + '-'.repeat(60));
        log('Setup Instructions:', 'yellow');
        log('  1. cp .env.example .env', 'dim');
        log('  2. Get your API keys:', 'dim');
        log('     • Gemini: https://aistudio.google.com/app/apikey', 'cyan');
        log('     • Voyage: https://dash.voyageai.com/api-keys', 'cyan');
        log('  3. Edit .env with your keys', 'dim');
        log('  4. Run: npm run test:embeddings', 'dim');
    }

    console.log();
}

main().catch(console.error);
