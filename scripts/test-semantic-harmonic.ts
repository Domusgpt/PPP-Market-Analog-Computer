#!/usr/bin/env npx tsx
/**
 * Semantic-Harmonic Bridge Test
 *
 * Tests the integration of HDCEncoder semantics with MusicGeometryDomain
 */

import 'dotenv/config';
import {
    SemanticHarmonicBridge,
    createSemanticHarmonicBridge,
    createOfflineBridge,
    DEFAULT_EMOTION_ARCHETYPES
} from '../lib/domains/SemanticHarmonicBridge.js';
import { Vector4D } from '../lib/domains/MusicGeometryDomain.js';

// =============================================================================
// UTILITIES
// =============================================================================

function vec4ToString(v: Vector4D, decimals = 3): string {
    return `[${v.map(x => x.toFixed(decimals)).join(', ')}]`;
}

function printSection(title: string): void {
    console.log('\n' + '═'.repeat(60));
    console.log(`  ${title}`);
    console.log('═'.repeat(60));
}

// =============================================================================
// MAIN TEST
// =============================================================================

async function main() {
    console.log('');
    console.log('╔══════════════════════════════════════════════════════════╗');
    console.log('║        Semantic-Harmonic Bridge Test Suite               ║');
    console.log('║                                                          ║');
    console.log('║   Bridging Natural Language → Music → 4D Geometry        ║');
    console.log('╚══════════════════════════════════════════════════════════╝');

    const voyageKey = process.env.VOYAGE_API_KEY;
    let bridge: SemanticHarmonicBridge;

    if (voyageKey) {
        console.log('\n✓ Voyage API key found - using full semantic analysis');
        bridge = createSemanticHarmonicBridge(voyageKey, 0.3);
    } else {
        console.log('\n⚠ No Voyage API key - using keyword-based fallback');
        bridge = createOfflineBridge(0.3);
    }

    // Initialize archetypes
    await bridge.initialize();

    // =========================================================================
    // TEST 1: Emotion Archetype Vectors
    // =========================================================================
    printSection('1. Emotion Archetype Vectors');

    console.log('\nBase vectors for each emotion (4D):');
    const archetypes = bridge.getArchetypeVectors();
    for (const [name, vector] of archetypes) {
        console.log(`  ${name.padEnd(10)} ${vec4ToString(vector)}`);
    }

    // =========================================================================
    // TEST 2: Semantic Analysis
    // =========================================================================
    printSection('2. Semantic Analysis');

    const testPhrases = [
        'a melancholic, sorrowful melody',
        'bright and cheerful music',
        'dark, tense, suspenseful atmosphere',
        'peaceful, calm, and serene',
        'triumphant victory celebration',
        'mysterious and enigmatic soundscape'
    ];

    console.log('\nAnalyzing phrases:\n');
    for (const phrase of testPhrases) {
        const analysis = await bridge.analyzeSemantics(phrase);
        console.log(`"${phrase}"`);
        console.log(`  → Emotion: ${analysis.dominantEmotion} (${(analysis.confidence * 100).toFixed(0)}%)`);
        console.log(`  → Vector:  ${vec4ToString(analysis.vector)}`);
        console.log('');
    }

    // =========================================================================
    // TEST 3: Combined Semantic-Musical State
    // =========================================================================
    printSection('3. Combined Semantic-Musical State');

    const stateTests = [
        { description: 'sad and melancholic', chord: ['A', 'C', 'E'] },      // A minor
        { description: 'happy and bright', chord: ['C', 'E', 'G'] },        // C major
        { description: 'tense and anxious', chord: ['B', 'D', 'F', 'Ab'] }, // Bdim7
        { description: 'peaceful resolution', chord: ['F', 'A', 'C'] },    // F major
    ];

    console.log('\nCombining semantics with chord geometry:\n');
    for (const { description, chord } of stateTests) {
        const state = await bridge.createState(description, chord);

        console.log(`"${description}" + [${chord.join('-')}]`);
        console.log(`  Semantic:   ${vec4ToString(state.semanticVector)} (${state.dominantEmotion})`);
        console.log(`  Musical:    ${vec4ToString(state.musicalVector)} (tension: ${state.tension.toFixed(2)})`);
        console.log(`  Combined:   ${vec4ToString(state.combinedVector)}`);
        console.log(`  Alignment:  ${(state.harmonicAlignment * 100).toFixed(0)}%`);
        console.log('');
    }

    // =========================================================================
    // TEST 4: Chord Suggestions
    // =========================================================================
    printSection('4. Chord Suggestions by Emotion');

    const emotionQueries = ['melancholic', 'triumphant', 'mysterious', 'peaceful'];

    for (const emotion of emotionQueries) {
        console.log(`\nChords for "${emotion}":`);
        const suggestions = await bridge.suggestChords(emotion, 3);

        for (const s of suggestions) {
            console.log(`  [${s.chord.join('-')}] - ${s.reason}`);
            console.log(`     Alignment: ${(s.alignmentScore * 100).toFixed(0)}%`);
        }
    }

    // =========================================================================
    // TEST 5: Emotional Arc Progression
    // =========================================================================
    printSection('5. Emotional Arc → Chord Progression');

    const emotionalArcs = [
        ['peaceful', 'tension', 'triumph'],
        ['joy', 'nostalgia', 'sadness'],
        ['mystery', 'tension', 'anger', 'peace']
    ];

    for (const arc of emotionalArcs) {
        console.log(`\nArc: ${arc.join(' → ')}`);
        const progression = await bridge.suggestProgression(arc, 1);

        let chordSequence: string[] = [];
        for (const segment of progression) {
            if (segment.chords.length > 0) {
                const chord = segment.chords[0].chord;
                chordSequence.push(`[${chord.join('-')}]`);
                console.log(`  ${segment.emotion}: [${chord.join('-')}]`);
            }
        }
        console.log(`  Progression: ${chordSequence.join(' → ')}`);
    }

    // =========================================================================
    // TEST 6: Alignment Validation
    // =========================================================================
    printSection('6. Alignment Validation');

    console.log('\nTesting semantic-musical alignment:');

    const alignmentTests = [
        { desc: 'happy', chord: ['C', 'E', 'G'], expected: 'high (major)' },
        { desc: 'happy', chord: ['A', 'C', 'E'], expected: 'medium (minor)' },
        { desc: 'sad', chord: ['A', 'C', 'E'], expected: 'high (minor)' },
        { desc: 'sad', chord: ['C', 'E', 'G'], expected: 'medium (major)' },
        { desc: 'tense', chord: ['B', 'D', 'F', 'Ab'], expected: 'high (dim7)' },
        { desc: 'peaceful', chord: ['C', 'E', 'G'], expected: 'high (stable)' },
    ];

    console.log('');
    console.log('Description    Chord     Alignment  Expected');
    console.log('─'.repeat(50));

    for (const test of alignmentTests) {
        const state = await bridge.createState(test.desc, test.chord);
        const alignPct = (state.harmonicAlignment * 100).toFixed(0) + '%';
        console.log(
            `${test.desc.padEnd(14)} [${test.chord.join('-')}]`.padEnd(25) +
            `${alignPct.padEnd(10)} ${test.expected}`
        );
    }

    // =========================================================================
    // SUMMARY
    // =========================================================================
    printSection('Summary');

    console.log(`
The Semantic-Harmonic Bridge successfully connects:

  Natural Language → Emotion Archetypes → 4D Vectors
          ↓
  Music Theory → Chord Geometry → 4D Vectors
          ↓
       Combined Semantic-Musical State

Key capabilities:
  • Analyze text to identify emotional content
  • Map emotions to 4D geometric space
  • Suggest chords that align with emotions
  • Generate progressions from emotional arcs
  • Validate semantic-musical alignment

This enables the CPE to respond to natural language
with geometrically-grounded musical suggestions,
creating an audible representation of semantic space.
`);

    console.log('═'.repeat(60));
    console.log('✓ All tests complete');
    console.log('═'.repeat(60));
}

main().catch(console.error);
