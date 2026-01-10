#!/usr/bin/env npx ts-node
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
 * 3. Run: npx ts-node scripts/test-embeddings.ts
 */

import * as dotenv from 'dotenv';
import * as path from 'path';

// Load environment variables
dotenv.config({ path: path.join(__dirname, '..', '.env') });

import {
    HDCEncoder,
    createGeminiEncoder,
    createAnthropicEncoder,
    createAPIEncoder,
    EmbeddingProvider
} from '../lib/encoding/HDCEncoder.js';

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

async function testProvider(
    name: string,
    encoder: HDCEncoder,
    testText: string
): Promise<{ success: boolean; time: number; error?: string }> {
    const start = Date.now();
    try {
        const result = await encoder.encodeTextAsync(testText);
        const time = Date.now() - start;

        // Validate result
        if (!result.force) {
            return { success: false, time, error: 'No force returned' };
        }
        if (!result.force.linear || result.force.linear.length !== 4) {
            return { success: false, time, error: 'Invalid linear force dimensions' };
        }
        if (!result.activatedConcepts || result.activatedConcepts.length === 0) {
            return { success: false, time, error: 'No concept activations' };
        }

        return { success: true, time };
    } catch (error) {
        const time = Date.now() - start;
        return {
            success: false,
            time,
            error: error instanceof Error ? error.message : String(error)
        };
    }
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

    const encoder = createGeminiEncoder(apiKey);
    log(`Encoder created with ${encoder.config.inputDimension}D input`, 'dim');

    let allPassed = true;
    for (const text of TEST_TEXTS) {
        const result = await testProvider('Gemini', encoder, text);
        logResult(
            `"${text.substring(0, 30)}..."`,
            result.success,
            result.success ? `${result.time}ms` : result.error
        );
        if (!result.success) allPassed = false;
    }

    if (allPassed) {
        // Show sample output
        const sampleResult = await encoder.encodeTextAsync(TEST_TEXTS[0]);
        log('\nSample output:', 'blue');
        log(`  Force: [${sampleResult.force.linear.map(v => v.toFixed(3)).join(', ')}]`, 'dim');
        log(`  Top concepts: ${sampleResult.activatedConcepts.slice(0, 3).map(c =>
            `${encoder.getArchetype(c.index)?.label}(${(c.weight * 100).toFixed(1)}%)`
        ).join(', ')}`, 'dim');
        log(`  Confidence: ${(sampleResult.confidence * 100).toFixed(1)}%`, 'dim');
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

    const encoder = createAnthropicEncoder(apiKey);
    log(`Encoder created with ${encoder.config.inputDimension}D input`, 'dim');

    let allPassed = true;
    for (const text of TEST_TEXTS) {
        const result = await testProvider('Voyage', encoder, text);
        logResult(
            `"${text.substring(0, 30)}..."`,
            result.success,
            result.success ? `${result.time}ms` : result.error
        );
        if (!result.success) allPassed = false;
    }

    if (allPassed) {
        // Show sample output
        const sampleResult = await encoder.encodeTextAsync(TEST_TEXTS[0]);
        log('\nSample output:', 'blue');
        log(`  Force: [${sampleResult.force.linear.map(v => v.toFixed(3)).join(', ')}]`, 'dim');
        log(`  Top concepts: ${sampleResult.activatedConcepts.slice(0, 3).map(c =>
            `${encoder.getArchetype(c.index)?.label}(${(c.weight * 100).toFixed(1)}%)`
        ).join(', ')}`, 'dim');
        log(`  Confidence: ${(sampleResult.confidence * 100).toFixed(1)}%`, 'dim');
    }

    return allPassed;
}

async function testHashFallback() {
    logSection('Testing Hash-based Fallback (No API)');

    const encoder = new HDCEncoder();
    log(`Encoder created with ${encoder.config.inputDimension}D input (hash-based)`, 'dim');

    let allPassed = true;
    for (const text of TEST_TEXTS) {
        const start = Date.now();
        try {
            const result = encoder.encodeText(text);
            const time = Date.now() - start;
            logResult(
                `"${text.substring(0, 30)}..."`,
                true,
                `${time}ms (sync)`
            );
        } catch (error) {
            logResult(
                `"${text.substring(0, 30)}..."`,
                false,
                error instanceof Error ? error.message : String(error)
            );
            allPassed = false;
        }
    }

    // Show attention visualization
    const viz = encoder.getAttentionVisualization(TEST_TEXTS[0]);
    if (viz) {
        log('\nAttention visualization:', 'blue');
        log(`  Tokens: ${viz.tokens.join(', ')}`, 'dim');
        log(`  Contributions: ${viz.contributions.map(c => c.toFixed(3)).join(', ')}`, 'dim');
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

    const encoder = createGeminiEncoder(apiKey);

    const pairs = [
        { a: 'dog', b: 'canine', expected: 'similar' },
        { a: 'dog', b: 'economics', expected: 'different' },
        { a: 'happy', b: 'joyful', expected: 'similar' },
        { a: 'hot', b: 'cold', expected: 'different' }
    ];

    for (const pair of pairs) {
        const resultA = await encoder.encodeTextAsync(pair.a);
        const resultB = await encoder.encodeTextAsync(pair.b);

        // Compute cosine similarity of forces
        const forceA = resultA.force.linear;
        const forceB = resultB.force.linear;

        let dotProduct = 0;
        let magA = 0;
        let magB = 0;
        for (let i = 0; i < 4; i++) {
            dotProduct += forceA[i] * forceB[i];
            magA += forceA[i] * forceA[i];
            magB += forceB[i] * forceB[i];
        }
        const similarity = dotProduct / (Math.sqrt(magA) * Math.sqrt(magB));

        const isSimilar = similarity > 0.5;
        const matchesExpected = (pair.expected === 'similar') === isSimilar;

        logResult(
            `"${pair.a}" vs "${pair.b}"`,
            matchesExpected,
            `similarity: ${(similarity * 100).toFixed(1)}% (expected: ${pair.expected})`
        );
    }

    return true;
}

async function main() {
    console.log('\n' + '╔' + '═'.repeat(58) + '╗');
    console.log('║' + ' '.repeat(15) + 'HDCEncoder Embedding Test Suite' + ' '.repeat(11) + '║');
    console.log('╚' + '═'.repeat(58) + '╝');

    const results: { name: string; passed: boolean }[] = [];

    // Test hash fallback first (always works)
    results.push({ name: 'Hash Fallback', passed: await testHashFallback() });

    // Test Gemini
    results.push({ name: 'Google Gemini', passed: await testGemini() });

    // Test Voyage/Anthropic
    results.push({ name: 'Voyage (Anthropic)', passed: await testVoyage() });

    // Test semantic similarity
    results.push({ name: 'Semantic Similarity', passed: await testSemanticSimilarity() });

    // Summary
    logSection('Test Summary');
    for (const result of results) {
        logResult(result.name, result.passed);
    }

    const passedCount = results.filter(r => r.passed).length;
    const totalCount = results.length;

    console.log('\n' + '-'.repeat(60));
    if (passedCount === totalCount) {
        log(`All ${totalCount} test suites passed!`, 'green');
    } else {
        log(`${passedCount}/${totalCount} test suites passed`, 'yellow');
    }

    // Environment check
    console.log('\n' + '-'.repeat(60));
    log('Environment:', 'blue');
    log(`  GOOGLE_API_KEY: ${process.env.GOOGLE_API_KEY ? 'set' : 'not set'}`, 'dim');
    log(`  VOYAGE_API_KEY: ${process.env.VOYAGE_API_KEY ? 'set' : 'not set'}`, 'dim');

    if (!process.env.GOOGLE_API_KEY || !process.env.VOYAGE_API_KEY) {
        console.log('\n' + '-'.repeat(60));
        log('To enable all tests, set up your API keys:', 'yellow');
        log('  1. cp .env.example .env', 'dim');
        log('  2. Edit .env with your keys:', 'dim');
        log('     - Gemini: https://aistudio.google.com/app/apikey', 'dim');
        log('     - Voyage: https://dash.voyageai.com/api-keys', 'dim');
    }

    console.log();
}

main().catch(console.error);
