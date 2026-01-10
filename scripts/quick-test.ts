/**
 * Quick test of the hash-based encoder (no API needed)
 */
import { HDCEncoder } from '../lib/encoding/HDCEncoder.js';

const encoder = new HDCEncoder();

console.log('Testing hash-based encoder (no API):\n');

const texts = [
    "reasoning about causality",
    "the dog chased the cat"
];

for (const text of texts) {
    const result = encoder.encodeText(text);
    const forceStr = result.force.linear.map(v => v.toFixed(3)).join(', ');
    const conceptsStr = result.activatedConcepts.slice(0,3).map(c =>
        encoder.getArchetype(c.index)?.label
    ).join(', ');

    console.log(`"${text}"`);
    console.log(`  Force: [${forceStr}]`);
    console.log(`  Top concepts: ${conceptsStr}`);
    console.log(`  Confidence: ${(result.confidence * 100).toFixed(1)}%`);
    console.log();
}

// Test attention
const viz = encoder.getAttentionVisualization("the quick brown fox");
if (viz) {
    console.log('Attention weights for "the quick brown fox":');
    console.log(`  Tokens: ${viz.tokens.join(', ')}`);
    console.log(`  Contributions: ${viz.contributions.map(c => c.toFixed(3)).join(', ')}`);
}
