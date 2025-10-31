#!/usr/bin/env node
import { readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { resolve, basename } from 'node:path';

import { SonicGeometryEngine } from './SonicGeometryEngine.js';

const printUsage = () => {
    console.error('Usage: node scripts/headlessRunner.js --input <file> [--out <file>] [--limit <n>]');
    console.error('       --input / -i   Path to a recorder export, single frame, or frame array.');
    console.error('       --out / -o     Optional output file for telemetry JSON (defaults to stdout).');
    console.error('       --limit / -l   Optional maximum number of frames to process.');
};

const parseArgs = (argv) => {
    const options = {};
    for (let index = 0; index < argv.length; index += 1) {
        const token = argv[index];
        switch (token) {
            case '--input':
            case '-i':
                options.input = argv[index + 1];
                index += 1;
                break;
            case '--out':
            case '-o':
                options.out = argv[index + 1];
                index += 1;
                break;
            case '--limit':
            case '-l':
                options.limit = Number(argv[index + 1]);
                index += 1;
                break;
            case '--help':
            case '-h':
                options.help = true;
                break;
            default:
                if (!token.startsWith('-') && !options.input) {
                    options.input = token;
                }
                break;
        }
    }
    return options;
};

const sanitizeValues = (values = [], limit = 32) => {
    if (!values || typeof values.length !== 'number') {
        return [];
    }
    const length = Math.min(limit, Math.max(0, Number(values.length) || 0));
    const result = new Array(length);
    for (let index = 0; index < length; index += 1) {
        const numeric = Number(values[index]);
        result[index] = Number.isFinite(numeric) ? Math.min(Math.max(numeric, 0), 1) : 0;
    }
    return result;
};

const coerceFrameArray = (payload) => {
    if (Array.isArray(payload)) {
        return payload;
    }
    if (payload && typeof payload === 'object') {
        if (Array.isArray(payload.frames)) {
            return payload.frames;
        }
        if (Array.isArray(payload.records)) {
            return payload.records;
        }
        if (payload.values || payload.data || payload.channels) {
            return [payload];
        }
    }
    return [];
};

const resolveValues = (frame) => {
    if (Array.isArray(frame.values)) {
        return frame.values;
    }
    if (Array.isArray(frame.data)) {
        return frame.data;
    }
    if (Array.isArray(frame.channels)) {
        return frame.channels;
    }
    if (Array.isArray(frame.mapped)) {
        return frame.mapped;
    }
    return [];
};

const resolveUniforms = (frame) => {
    if (frame.uniforms && typeof frame.uniforms === 'object') {
        return frame.uniforms;
    }
    if (frame.visualUniforms && typeof frame.visualUniforms === 'object') {
        return frame.visualUniforms;
    }
    return null;
};

const resolveDerivedUniforms = (frame) => {
    if (frame.derivedUniforms && typeof frame.derivedUniforms === 'object') {
        return frame.derivedUniforms;
    }
    return null;
};

const buildTransport = (frame, index, frameCount) => {
    const progress = frameCount > 1 ? index / (frameCount - 1) : 0;
    if (frame.transport && typeof frame.transport === 'object') {
        return {
            playing: frame.transport.playing !== false,
            progress: Number.isFinite(frame.transport.progress) ? frame.transport.progress : progress,
            mode: typeof frame.transport.mode === 'string' ? frame.transport.mode : 'headless',
            frameIndex: Number.isFinite(frame.transport.frameIndex) ? frame.transport.frameIndex : index,
            frameCount: Number.isFinite(frame.transport.frameCount) ? frame.transport.frameCount : frameCount,
            loop: Boolean(frame.transport.loop)
        };
    }
    return {
        playing: true,
        progress,
        mode: 'headless',
        frameIndex: index,
        frameCount,
        loop: false
    };
};

const main = async () => {
    const options = parseArgs(process.argv.slice(2));
    if (options.help || !options.input) {
        printUsage();
        process.exit(options.help ? 0 : 1);
    }
    const inputPath = resolve(process.cwd(), options.input);
    if (!existsSync(inputPath)) {
        console.error(`Input file not found: ${inputPath}`);
        process.exit(1);
    }

    const raw = JSON.parse(await readFile(inputPath, 'utf8'));
    const frames = coerceFrameArray(raw);
    if (!frames.length) {
        console.error('No frames detected in input. Provide a recorder export or array of frames.');
        process.exit(1);
    }

    const limit = Number.isFinite(options.limit) && options.limit > 0
        ? Math.min(frames.length, Math.floor(options.limit))
        : frames.length;

    const engine = new SonicGeometryEngine({ outputMode: 'analysis', contextFactory: null });
    await engine.enable();

    const processed = [];
    for (let index = 0; index < limit; index += 1) {
        const frame = frames[index];
        const transport = buildTransport(frame, index, frames.length);
        const uniforms = resolveUniforms(frame);
        const derivedUniforms = resolveDerivedUniforms(frame);
        const values = sanitizeValues(resolveValues(frame), engine.channelLimit || 32);
        const metadata = {
            transport,
            visualUniforms: uniforms,
            derivedUniforms,
            progress: transport.progress
        };
        const analysis = engine.updateFromData(values, metadata);
        if (!analysis) {
            continue;
        }
        processed.push({
            index,
            source: frame.source || transport.mode || 'headless',
            timestamp: Number.isFinite(frame.timestamp) ? frame.timestamp : null,
            elapsed: Number.isFinite(frame.elapsed) ? frame.elapsed : null,
            summary: analysis.summary,
            transport: analysis.transport,
            analysis,
            signal: engine.getLastSignal(),
            transduction: engine.getLastTransduction(),
            manifold: engine.getLastManifold(),
            topology: engine.getLastTopology(),
            continuum: engine.getLastContinuum(),
            lattice: engine.getLastLattice(),
            constellation: engine.getLastConstellation(),
            performance: engine.getPerformanceMetrics()
        });
    }

    const payload = {
        meta: {
            generatedAt: new Date().toISOString(),
            input: basename(inputPath),
            frameCount: processed.length,
            originalFrames: frames.length,
            outputMode: engine.getOutputMode(),
            performance: engine.getPerformanceMetrics()
        },
        frames: processed
    };

    if (options.out) {
        const outputPath = resolve(process.cwd(), options.out);
        await writeFile(outputPath, JSON.stringify(payload, null, 2));
        console.log(`Headless run complete – wrote ${processed.length} frames to ${outputPath}`);
    } else {
        console.log(JSON.stringify(payload, null, 2));
    }
};

main().catch((error) => {
    console.error('Headless runner failed:', error);
    process.exit(1);
});
