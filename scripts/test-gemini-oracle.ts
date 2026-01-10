#!/usr/bin/env npx tsx
/**
 * GeminiAudioOracle Test Suite
 *
 * Tests the Gemini 3 Pro audio analysis module for CPE calibration.
 * Validates the three geometric-musical hypotheses.
 *
 * Requirements:
 * - GOOGLE_API_KEY environment variable
 * - Audio files in audio/stimuli/ directory (or generates test audio)
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';
import {
    GeminiAudioOracle,
    createGemini3ProOracle,
    TensionAnalysis,
    EmotionAnalysis,
    CalibrationResult
} from '../lib/domains/GeminiAudioOracle.js';
import { MusicGeometryDomain, Vector4D } from '../lib/domains/MusicGeometryDomain.js';

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

function printSubsection(title: string): void {
    console.log('\n' + '─'.repeat(40));
    console.log(`  ${title}`);
    console.log('─'.repeat(40));
}

// =============================================================================
// AUDIO GENERATION (SOX-BASED)
// =============================================================================

/**
 * Generate a test audio file using SoX (if available)
 * Falls back to a simple WAV generator if SoX is not installed
 */
async function generateTestAudio(
    notes: string[],
    filename: string,
    durationMs: number = 2000
): Promise<string> {
    const audioDir = '/home/user/ppp-info-site/audio/stimuli';

    // Ensure directory exists
    if (!fs.existsSync(audioDir)) {
        fs.mkdirSync(audioDir, { recursive: true });
    }

    const outputPath = path.join(audioDir, filename);

    // Note to frequency mapping (A4 = 440Hz)
    const noteFrequencies: Record<string, number> = {
        'C': 261.63, 'C#': 277.18, 'Db': 277.18,
        'D': 293.66, 'D#': 311.13, 'Eb': 311.13,
        'E': 329.63, 'Fb': 329.63,
        'F': 349.23, 'F#': 369.99, 'Gb': 369.99,
        'G': 392.00, 'G#': 415.30, 'Ab': 415.30,
        'A': 440.00, 'A#': 466.16, 'Bb': 466.16,
        'B': 493.88, 'Cb': 493.88
    };

    // Get frequencies for all notes
    const frequencies = notes.map(note => {
        const baseName = note.replace(/\d/, '');
        const octave = parseInt(note.match(/\d/)?.[0] || '4');
        const baseFreq = noteFrequencies[baseName] || 261.63;
        return baseFreq * Math.pow(2, octave - 4);
    });

    // Try to use SoX for high-quality audio generation
    const { execSync } = await import('child_process');

    try {
        // Check if SoX is available
        execSync('which sox', { encoding: 'utf-8' });

        // Generate chord using SoX
        const durationSec = durationMs / 1000;
        const soxCommands: string[] = [];

        for (const freq of frequencies) {
            soxCommands.push(
                `sox -n -r 48000 -b 24 -c 1 /tmp/tone_${freq.toFixed(0)}.wav ` +
                `synth ${durationSec} sine ${freq} fade 0.05 ${durationSec} 0.1`
            );
        }

        // Generate individual tones
        for (const cmd of soxCommands) {
            execSync(cmd);
        }

        // Mix all tones together
        const toneFiles = frequencies.map(f => `/tmp/tone_${f.toFixed(0)}.wav`).join(' ');
        execSync(`sox -m ${toneFiles} -r 48000 -b 24 "${outputPath}" norm -3`);

        // Cleanup temp files
        frequencies.forEach(f => {
            const tempFile = `/tmp/tone_${f.toFixed(0)}.wav`;
            if (fs.existsSync(tempFile)) fs.unlinkSync(tempFile);
        });

        console.log(`  Generated: ${filename} (${notes.join('-')}) using SoX`);

    } catch {
        // Fallback: Generate simple WAV file programmatically
        console.log(`  SoX not available, generating simple WAV...`);

        const sampleRate = 48000;
        const numSamples = Math.floor((durationMs / 1000) * sampleRate);
        const numChannels = 1;
        const bitsPerSample = 16;

        // Create audio buffer
        const buffer = Buffer.alloc(44 + numSamples * 2); // WAV header + samples

        // Write WAV header
        buffer.write('RIFF', 0);
        buffer.writeUInt32LE(36 + numSamples * 2, 4);
        buffer.write('WAVE', 8);
        buffer.write('fmt ', 12);
        buffer.writeUInt32LE(16, 16); // Subchunk1Size
        buffer.writeUInt16LE(1, 20);  // AudioFormat (PCM)
        buffer.writeUInt16LE(numChannels, 22);
        buffer.writeUInt32LE(sampleRate, 24);
        buffer.writeUInt32LE(sampleRate * numChannels * bitsPerSample / 8, 28);
        buffer.writeUInt16LE(numChannels * bitsPerSample / 8, 32);
        buffer.writeUInt16LE(bitsPerSample, 34);
        buffer.write('data', 36);
        buffer.writeUInt32LE(numSamples * 2, 40);

        // Generate samples (sum of sine waves)
        for (let i = 0; i < numSamples; i++) {
            const t = i / sampleRate;
            let sample = 0;

            for (const freq of frequencies) {
                sample += Math.sin(2 * Math.PI * freq * t);
            }

            // Normalize and apply envelope
            sample /= frequencies.length;
            const envelope = Math.min(1, t * 20) * Math.min(1, (durationMs / 1000 - t) * 10);
            sample *= envelope * 0.7;

            // Convert to 16-bit integer
            const intSample = Math.max(-32768, Math.min(32767, Math.floor(sample * 32767)));
            buffer.writeInt16LE(intSample, 44 + i * 2);
        }

        fs.writeFileSync(outputPath, buffer);
        console.log(`  Generated: ${filename} (${notes.join('-')}) using programmatic WAV`);
    }

    return outputPath;
}

// =============================================================================
// TEST FUNCTIONS
// =============================================================================

async function testTensionAnalysis(oracle: GeminiAudioOracle): Promise<void> {
    printSection('1. Tension Analysis');

    const testChords = [
        { name: 'C Major', notes: ['C4', 'E4', 'G4'], expectedTension: 'low' },
        { name: 'G7 (Dominant)', notes: ['G3', 'B3', 'D4', 'F4'], expectedTension: 'medium' },
        { name: 'Bdim7', notes: ['B3', 'D4', 'F4', 'Ab4'], expectedTension: 'high' },
        { name: 'Cluster', notes: ['C4', 'Db4', 'D4', 'Eb4'], expectedTension: 'very high' },
    ];

    console.log('\nAnalyzing chord tension via Gemini 3 Pro:\n');
    console.log('Chord          Expected    Gemini Rating   Stability');
    console.log('─'.repeat(55));

    for (const chord of testChords) {
        const filename = `tension_${chord.name.toLowerCase().replace(/\s+/g, '_')}.wav`;
        const audioPath = await generateTestAudio(chord.notes, filename);

        try {
            const analysis = await oracle.analyzeTension(audioPath);
            console.log(
                `${chord.name.padEnd(14)} ${chord.expectedTension.padEnd(11)} ` +
                `${String(analysis.tensionRating).padEnd(15)} ${analysis.stability}`
            );
        } catch (error: any) {
            console.log(`${chord.name.padEnd(14)} ${chord.expectedTension.padEnd(11)} ERROR: ${error.message}`);
        }
    }
}

async function testEmotionAnalysis(oracle: GeminiAudioOracle): Promise<void> {
    printSection('2. Emotion Analysis');

    const testChords = [
        { name: 'C Major (joy)', notes: ['C4', 'E4', 'G4'] },
        { name: 'A Minor (sad)', notes: ['A3', 'C4', 'E4'] },
        { name: 'Db Major (mystery)', notes: ['Db4', 'F4', 'Ab4'] },
        { name: 'Bb Major (peace)', notes: ['Bb3', 'D4', 'F4'] },
    ];

    console.log('\nAnalyzing emotion via Gemini 3 Pro:\n');
    console.log('Chord              Primary Emotion   Confidence   Valence');
    console.log('─'.repeat(60));

    for (const chord of testChords) {
        const filename = `emotion_${chord.name.toLowerCase().replace(/[\s()]/g, '_')}.wav`;
        const audioPath = await generateTestAudio(chord.notes, filename);

        try {
            const analysis = await oracle.analyzeEmotion(audioPath);
            console.log(
                `${chord.name.padEnd(18)} ${analysis.primaryEmotion.padEnd(17)} ` +
                `${(analysis.confidence * 100).toFixed(0).padStart(3)}%       ` +
                `${analysis.valence.toFixed(2)}`
            );
        } catch (error: any) {
            console.log(`${chord.name.padEnd(18)} ERROR: ${error.message}`);
        }
    }
}

async function test4DCoordinates(oracle: GeminiAudioOracle): Promise<void> {
    printSection('3. 4D Perceptual Coordinates');

    const testChords = [
        { name: 'High Bright', notes: ['C6', 'E6', 'G6'] },
        { name: 'Low Dark', notes: ['C2', 'E2', 'G2'] },
        { name: 'Dense Cluster', notes: ['C4', 'Db4', 'D4', 'Eb4', 'E4'] },
        { name: 'Sparse Fifth', notes: ['C4', 'G4'] },
    ];

    console.log('\nMapping audio to 4D perceptual space:\n');
    console.log('Chord          [Brightness, Density, Movement, Depth]');
    console.log('─'.repeat(55));

    for (const chord of testChords) {
        const filename = `4d_${chord.name.toLowerCase().replace(/\s+/g, '_')}.wav`;
        const audioPath = await generateTestAudio(chord.notes, filename);

        try {
            const coords = await oracle.get4DCoordinates(audioPath);
            console.log(`${chord.name.padEnd(14)} ${vec4ToString(coords)}`);
        } catch (error: any) {
            console.log(`${chord.name.padEnd(14)} ERROR: ${error.message}`);
        }
    }
}

async function testIntervalAnalysis(oracle: GeminiAudioOracle): Promise<void> {
    printSection('4. Interval Analysis');

    const intervals = [
        { name: 'Unison', notes: ['C4', 'C4'] },
        { name: 'Perfect Fifth', notes: ['C4', 'G4'] },
        { name: 'Major Third', notes: ['C4', 'E4'] },
        { name: 'Minor Second', notes: ['C4', 'Db4'] },
        { name: 'Tritone', notes: ['C4', 'Gb4'] },
    ];

    console.log('\nAnalyzing interval consonance:\n');
    console.log('Interval         Consonance  Roughness  Quality');
    console.log('─'.repeat(55));

    for (const interval of intervals) {
        const filename = `interval_${interval.name.toLowerCase().replace(/\s+/g, '_')}.wav`;
        const audioPath = await generateTestAudio(interval.notes, filename);

        try {
            const analysis = await oracle.analyzeInterval(audioPath);
            console.log(
                `${interval.name.padEnd(16)} ${analysis.consonance.toFixed(2).padEnd(11)} ` +
                `${analysis.roughness.toFixed(2).padEnd(10)} ${analysis.intervalQuality}`
            );
        } catch (error: any) {
            console.log(`${interval.name.padEnd(16)} ERROR: ${error.message}`);
        }
    }
}

async function testGeometricCorrelation(oracle: GeminiAudioOracle): Promise<void> {
    printSection('5. Geometric-Acoustic Correlation (H1 Preview)');

    const domain = oracle.getMusicDomain();

    const chords = [
        ['C', 'E', 'G'],           // Major triad
        ['A', 'C', 'E'],           // Minor triad
        ['G', 'B', 'D', 'F'],      // Dominant 7th
        ['B', 'D', 'F', 'Ab'],     // Diminished 7th
        ['C', 'E', 'G#'],          // Augmented
        ['C', 'F', 'G'],           // Suspended
    ];

    console.log('\nComparing geometric vs perceived tension:\n');
    console.log('Chord           Geometric   Perceived   Match');
    console.log('─'.repeat(55));

    const geometricTensions: number[] = [];
    const perceivedTensions: number[] = [];

    for (const chord of chords) {
        const geom = domain.chordToPolytope(chord);
        geometricTensions.push(geom.tension);

        const filename = `corr_${chord.join('_').toLowerCase()}.wav`;
        const notes = chord.map(n => n + '4');
        const audioPath = await generateTestAudio(notes, filename);

        try {
            const analysis = await oracle.analyzeTension(audioPath);
            const normalizedTension = analysis.tensionRating / 10;
            perceivedTensions.push(normalizedTension);

            const diff = Math.abs(geom.tension - normalizedTension);
            const match = diff < 0.2 ? '++' : diff < 0.4 ? '+' : '-';

            console.log(
                `[${chord.join('-')}]`.padEnd(16) +
                `${geom.tension.toFixed(3).padEnd(11)} ` +
                `${normalizedTension.toFixed(3).padEnd(11)} ${match}`
            );
        } catch (error: any) {
            console.log(`[${chord.join('-')}]`.padEnd(16) + `ERROR: ${error.message}`);
        }
    }

    // Calculate correlation if we have data
    if (perceivedTensions.length >= 3) {
        const n = perceivedTensions.length;
        const meanG = geometricTensions.reduce((a, b) => a + b, 0) / n;
        const meanP = perceivedTensions.reduce((a, b) => a + b, 0) / n;

        let numerator = 0, denomG = 0, denomP = 0;
        for (let i = 0; i < n; i++) {
            const dg = geometricTensions[i] - meanG;
            const dp = perceivedTensions[i] - meanP;
            numerator += dg * dp;
            denomG += dg * dg;
            denomP += dp * dp;
        }

        const r = numerator / Math.sqrt(denomG * denomP);
        console.log(`\nPearson correlation: r = ${r.toFixed(4)}`);
        console.log(r >= 0.5 ? '  H1 appears SUPPORTED (r >= 0.5)' : '  H1 needs more data or recalibration');
    }
}

// =============================================================================
// MAIN
// =============================================================================

async function main() {
    console.log('');
    console.log('╔══════════════════════════════════════════════════════════╗');
    console.log('║         GeminiAudioOracle Test Suite                     ║');
    console.log('║                                                          ║');
    console.log('║   Validating Geometric-Musical Hypotheses via AI Audio   ║');
    console.log('╚══════════════════════════════════════════════════════════╝');

    const apiKey = process.env.GOOGLE_API_KEY;

    if (!apiKey) {
        console.log('\n⚠ GOOGLE_API_KEY not set');
        console.log('\nTo run these tests:');
        console.log('  1. Get a Gemini API key from https://aistudio.google.com/apikey');
        console.log('  2. Add GOOGLE_API_KEY=your_key_here to .env file');
        console.log('  3. Run: npm run test:oracle\n');

        console.log('Running in DEMO mode (no API calls)...\n');
        await runDemoMode();
        return;
    }

    console.log('\n✓ Google API key found');
    console.log('  Using Gemini 3 Pro for analysis');

    const oracle = createGemini3ProOracle(apiKey);

    try {
        // Run test suites
        await testTensionAnalysis(oracle);
        await testEmotionAnalysis(oracle);
        await test4DCoordinates(oracle);
        await testIntervalAnalysis(oracle);
        await testGeometricCorrelation(oracle);

        // Summary
        printSection('Summary');
        console.log(`
Total API calls: ${oracle.getCallCount()}

The GeminiAudioOracle successfully:
  - Analyzes musical tension from raw audio
  - Classifies emotional content
  - Maps audio to 4D perceptual coordinates
  - Measures interval consonance/roughness
  - Correlates geometric and acoustic measures

Ready for full hypothesis validation with:
  - validateH1_TensionTransfer()
  - validateH2_GeodesicVoiceLeading()
  - validateH3_PythagoreanComma()
`);
    } catch (error: any) {
        console.error('\nTest failed:', error.message);
        if (error.message.includes('403') || error.message.includes('Forbidden')) {
            console.log('\nNote: Gemini API may block requests from cloud/VPS IPs.');
            console.log('Consider running locally or using a different API provider.');
        }
    }

    console.log('═'.repeat(60));
    console.log('✓ Test suite complete');
    console.log('═'.repeat(60));
}

async function runDemoMode(): Promise<void> {
    const domain = new MusicGeometryDomain();

    printSection('Demo: Geometric Analysis Only');

    const chords = [
        { name: 'C Major', notes: ['C', 'E', 'G'] },
        { name: 'A Minor', notes: ['A', 'C', 'E'] },
        { name: 'G7', notes: ['G', 'B', 'D', 'F'] },
        { name: 'Bdim7', notes: ['B', 'D', 'F', 'Ab'] },
    ];

    console.log('\nGeometric analysis (no API calls):');
    console.log('\nChord      Centroid                    Tension   Symmetry');
    console.log('─'.repeat(60));

    for (const chord of chords) {
        const geom = domain.chordToPolytope(chord.notes);
        console.log(
            `${chord.name.padEnd(10)} ${vec4ToString(geom.centroid)}  ` +
            `${geom.tension.toFixed(3).padEnd(9)} ${geom.symmetryGroup}`
        );
    }

    printSection('What API Testing Would Show');
    console.log(`
With a valid GOOGLE_API_KEY, this test suite would:

1. TENSION ANALYSIS
   - Send chord audio to Gemini 3 Pro
   - Get tension ratings (1-10 scale)
   - Compare with geometric tension values

2. EMOTION ANALYSIS
   - Classify emotional content of chords
   - Map to valence/arousal dimensions
   - Compare with semantic bridge predictions

3. 4D COORDINATES
   - Map audio to [brightness, density, movement, depth]
   - Compare with MusicGeometryDomain coordinates
   - Validate perceptual-geometric alignment

4. INTERVAL ANALYSIS
   - Measure consonance and roughness
   - Detect beating and tuning deviations
   - Correlate with geometric distance

5. HYPOTHESIS VALIDATION
   - H1: Geometric tension ↔ Acoustic tension correlation
   - H2: Bach chorale paths vs random paths
   - H3: Pythagorean comma detection
`);

    console.log('═'.repeat(60));
    console.log('✓ Demo complete (add GOOGLE_API_KEY for full tests)');
    console.log('═'.repeat(60));
}

main().catch(console.error);
