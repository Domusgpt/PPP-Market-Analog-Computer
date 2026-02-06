#!/usr/bin/env node
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance as nodePerformance } from 'node:perf_hooks';
import { gzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';

import { CalibrationToolkit } from './CalibrationToolkit.js';
import { CalibrationDatasetBuilder } from './CalibrationDatasetBuilder.js';
import { CalibrationInsightEngine } from './CalibrationInsightEngine.js';
import { SonicGeometryEngine } from './SonicGeometryEngine.js';
import { DataMapper } from './DataMapper.js';
import { defaultMapping } from './defaultMapping.js';
import { DATA_CHANNEL_COUNT } from './constants.js';

if (typeof globalThis.performance === 'undefined' || !globalThis.performance) {
    globalThis.performance = nodePerformance;
}

const moduleDir = fileURLToPath(new URL('.', import.meta.url));
const projectRoot = resolve(moduleDir, '..');

const toRelativePath = (absolutePath) => {
    const candidate = relative(projectRoot, absolutePath);
    if (!candidate || candidate.startsWith('..')) {
        return absolutePath;
    }
    return candidate.replace(/\\/g, '/');
};

const clamp = (value, min, max) => {
    if (!Number.isFinite(value)) {
        return min;
    }
    if (value < min) {
        return min;
    }
    if (value > max) {
        return max;
    }
    return value;
};

const parseArgs = (argv) => {
    const options = {
        out: 'dist/calibration/ppp-calibration-dataset.json',
        summary: 'samples/calibration/ppp-calibration-dataset-summary.json',
        insights: 'samples/calibration/ppp-calibration-insights.json',
        includeSamples: true
    };
    for (let index = 0; index < argv.length; index += 1) {
        const token = argv[index];
        switch (token) {
            case '--out':
            case '-o':
                options.out = argv[index + 1];
                index += 1;
                break;
            case '--summary':
            case '-s':
                options.summary = argv[index + 1];
                index += 1;
                break;
            case '--insights':
                options.insights = argv[index + 1];
                index += 1;
                break;
            case '--no-samples':
                options.includeSamples = false;
                break;
            case '--help':
            case '-h':
                options.help = true;
                break;
            default:
                break;
        }
    }
    return options;
};

const sanitizeValues = (values = []) => {
    if (!values || typeof values.length !== 'number') {
        return [];
    }
    const length = Math.min(DATA_CHANNEL_COUNT, Math.max(0, Number(values.length) || 0));
    const sanitized = new Array(length);
    for (let index = 0; index < length; index += 1) {
        const numeric = Number(values[index]);
        sanitized[index] = Number.isFinite(numeric) ? clamp(numeric, 0, 1) : 0;
    }
    return sanitized;
};

const encodeOverlay = ({
    values = [],
    uniforms = {},
    sequenceId = 'sequence',
    label = 'Sequence',
    step = 0,
    totalSteps = 1,
    summary = ''
} = {}) => {
    if (!values.length) {
        return null;
    }
    const width = 320;
    const height = 180;
    const plotHeight = height - 60;
    const plotWidth = width - 40;
    const points = values
        .map((value, index) => {
            const x = 20 + (plotWidth * (values.length === 1 ? 0.5 : index / (values.length - 1)));
            const y = 30 + plotHeight - clamp(value, 0, 1) * plotHeight;
            return `${x.toFixed(1)},${y.toFixed(1)}`;
        })
        .join(' ');
    const progress = totalSteps > 1 ? (step / (totalSteps - 1)) : 0;
    const uniformKeys = Object.keys(uniforms)
        .filter((key) => key.startsWith('u_'))
        .slice(0, 4);
    const uniformText = uniformKeys
        .map((key) => `${key.replace('u_', '')}:${Number(uniforms[key]).toFixed(2)}`)
        .join(' · ');
    const svg = `<?xml version="1.0" encoding="UTF-8"?>\n` +
        `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">` +
        `<rect width="100%" height="100%" fill="#05060a"/>` +
        `<g transform="translate(0,20)" stroke="#1f8ff0" stroke-width="2" fill="none">` +
        `<polyline points="${points}" />` +
        `</g>` +
        `<text x="20" y="28" fill="#8ad3ff" font-family="monospace" font-size="14">${label}</text>` +
        `<text x="20" y="48" fill="#ffffff" font-family="monospace" font-size="12">` +
        `Frame ${step + 1}/${totalSteps} · ${(progress * 100).toFixed(1)}%</text>` +
        `<text x="20" y="68" fill="#9ff17d" font-family="monospace" font-size="12">${uniformText}</text>` +
        `<text x="20" y="160" fill="#d0d7ff" font-family="monospace" font-size="12">${summary}</text>` +
        `<text x="20" y="178" fill="#5c6b89" font-family="monospace" font-size="10">${sequenceId}</text>` +
        `</svg>`;
    return `data:image/svg+xml;base64,${Buffer.from(svg).toString('base64')}`;
};

const main = async () => {
    const options = parseArgs(process.argv.slice(2));
    if (options.help) {
        console.log('Usage: node scripts/buildCalibrationDataset.js [--out <file>] [--summary <file>]');
        console.log('       --insights <file>      Optional insights JSON path.');
        console.log('       --no-samples           Omit sample payloads from written manifest.');
        console.log('Paths resolve relative to the repository root unless absolute.');
        return;
    }

    const engine = new SonicGeometryEngine({ outputMode: 'analysis', contextFactory: null });
    await engine.enable();

    let lastFrameState = null;
    const dataMapper = new DataMapper({ mapping: defaultMapping, smoothing: 0.12 });
    const writeJson = async (targetPath, payload) => {
        const json = `${JSON.stringify(payload, null, 2)}\n`;
        const shouldGzip = targetPath.endsWith('.gz');
        if (shouldGzip) {
            const buffer = gzipSync(json, { mtime: 0 });
            await writeFile(targetPath, buffer);
            return {
                format: 'json.gz',
                bytes: buffer.length,
                sha256: createHash('sha256').update(buffer).digest('hex'),
                contentSha256: createHash('sha256').update(json).digest('hex')
            };
        }

        const buffer = Buffer.from(json, 'utf8');
        await writeFile(targetPath, buffer);
        const digest = createHash('sha256').update(buffer).digest('hex');
        return {
            format: 'json',
            bytes: buffer.length,
            sha256: digest,
            contentSha256: digest
        };
    };

    const applyDataArray = (values, {
        source = 'calibration',
        uniformOverride = null,
        playbackFrame = null,
        transportOverride = null,
        analysisMetadata = null
    } = {}) => {
        const normalized = sanitizeValues(values);
        dataMapper.updateData(normalized);
        const derivedUniforms = dataMapper.getUniformSnapshot();
        const uniforms = uniformOverride && typeof uniformOverride === 'object'
            ? { ...derivedUniforms, ...uniformOverride }
            : derivedUniforms;
        const transport = {
            playing: true,
            loop: false,
            progress: playbackFrame && Number.isFinite(playbackFrame.progress)
                ? clamp(playbackFrame.progress, 0, 1)
                : 0,
            frameIndex: transportOverride && Number.isFinite(transportOverride.frameIndex)
                ? transportOverride.frameIndex
                : (playbackFrame && Number.isFinite(playbackFrame.index) ? playbackFrame.index : 0),
            frameCount: transportOverride && Number.isFinite(transportOverride.frameCount)
                ? transportOverride.frameCount
                : (playbackFrame && Number.isFinite(playbackFrame.count) ? playbackFrame.count : normalized.length),
            mode: typeof source === 'string' ? source : 'calibration'
        };
        if (transportOverride && typeof transportOverride === 'object') {
            if (Number.isFinite(transportOverride.progress)) {
                transport.progress = clamp(transportOverride.progress, 0, 1);
            }
            if (typeof transportOverride.mode === 'string') {
                transport.mode = transportOverride.mode;
            }
            transport.playing = Boolean(transportOverride.playing);
            transport.loop = Boolean(transportOverride.loop);
        }
        const metadata = {
            source: transport.mode,
            transport,
            visualUniforms: uniforms,
            derivedUniforms,
            playbackFrame,
            timestamp: performance.now()
        };
        if (analysisMetadata && typeof analysisMetadata === 'object') {
            Object.assign(metadata, analysisMetadata);
        }
        const analysis = engine.updateFromData(normalized, metadata);
        lastFrameState = {
            values: normalized,
            uniforms,
            sequenceId: analysisMetadata?.calibration?.sequenceId || transport.mode,
            label: analysisMetadata?.calibration?.sequenceId || 'calibration-sequence',
            step: analysisMetadata?.calibration?.step || 0,
            totalSteps: analysisMetadata?.calibration?.totalSteps || normalized.length,
            summary: analysis?.summary || ''
        };
        return uniforms;
    };

    const captureFrame = () => {
        if (!lastFrameState) {
            return null;
        }
        return encodeOverlay(lastFrameState);
    };

    const getSonicAnalysis = () => engine.getLastAnalysis();

    const statusMessages = [];
    const toolkit = new CalibrationToolkit({
        applyDataArray,
        captureFrame,
        getSonicAnalysis,
        onStatus: (text) => {
            statusMessages.push(text);
            console.log(`[Toolkit] ${text}`);
        }
    });

    const builder = new CalibrationDatasetBuilder({
        toolkit,
        onStatus: (text) => console.log(`[Dataset] ${text}`),
        metadata: {
            release: 'ppp-calibration-reference-1',
            environment: 'headless-node',
            generator: 'buildCalibrationDataset.js'
        }
    });

    const manifest = await builder.runPlan();
    if (!manifest) {
        console.error('Calibration dataset plan did not produce a manifest.');
        process.exit(1);
    }

    const summaryManifest = builder.getLastManifest({ includeSamples: false });
    const insightEngine = new CalibrationInsightEngine();
    const insights = insightEngine.analyzeManifest(manifest);
    const narrative = insightEngine.generateNarrative(insights, { maxItems: 10 });
    const datasetReference = {
        release: summaryManifest?.metadata?.release || 'ppp-calibration-reference-1',
        artifact: null
    };

    const insightsPayload = {
        generatedAt: new Date().toISOString(),
        dataset: datasetReference,
        summary: insights,
        narrative
    };

    const resolvedOut = resolve(projectRoot, options.out);
    const resolvedSummary = resolve(projectRoot, options.summary);
    const resolvedInsights = options.insights
        ? resolve(projectRoot, options.insights)
        : null;


    await mkdir(dirname(resolvedOut), { recursive: true });
    await mkdir(dirname(resolvedSummary), { recursive: true });
    if (resolvedInsights) {
        await mkdir(dirname(resolvedInsights), { recursive: true });
    }

    const manifestToWrite = options.includeSamples ? manifest : summaryManifest;
    const manifestArtifact = await writeJson(resolvedOut, manifestToWrite);

    if (!summaryManifest.metadata || typeof summaryManifest.metadata !== 'object') {
        summaryManifest.metadata = {};
    }

    summaryManifest.metadata.artifact = {
        path: toRelativePath(resolvedOut),
        includesSamples: Boolean(options.includeSamples),
        format: manifestArtifact.format,
        bytes: manifestArtifact.bytes,
        sha256: manifestArtifact.sha256,
        contentSha256: manifestArtifact.contentSha256
    };

    datasetReference.artifact = summaryManifest.metadata.artifact;

    summaryManifest.outputs = {
        manifest: summaryManifest.metadata.artifact,
        summary: {
            path: toRelativePath(resolvedSummary),
            format: 'json'
        },
        insights: resolvedInsights
            ? {
                path: toRelativePath(resolvedInsights),
                format: 'json'
            }
            : null
    };

    if (statusMessages.length) {
        summaryManifest.statusLog = [...statusMessages];
    }

    await writeJson(resolvedSummary, summaryManifest);

    if (resolvedInsights) {
        await writeJson(resolvedInsights, insightsPayload);
    }

    console.log('Calibration dataset manifest written to', resolvedOut);
    console.log('Calibration dataset summary written to', resolvedSummary);
    if (resolvedInsights) {
        console.log('Calibration dataset insights written to', resolvedInsights);
    }

    console.log('--- Dataset Summary ---');
    console.log(`Sequences: ${summaryManifest?.totals?.sequenceCount ?? 'n/a'}`);
    console.log(`Frames: ${summaryManifest?.totals?.sampleCount ?? 'n/a'}`);
    console.log(`Dataset Score: ${Number.isFinite(summaryManifest?.score) ? summaryManifest.score.toFixed(4) : 'n/a'}`);
    const delta = summaryManifest?.parity?.visualContinuumDelta?.average;
    if (Number.isFinite(delta)) {
        console.log(`Visual/Continuum Δ Average: ${delta.toFixed(4)}`);
    }
    const gate = summaryManifest?.parity?.carrierGateRatio?.average;
    if (Number.isFinite(gate)) {
        console.log(`Carrier Gate Ratio Average: ${gate.toFixed(4)}`);
    }
    if (narrative.length) {
        console.log('Narrative Highlights:');
        narrative.forEach((line) => console.log(` • ${line}`));
    }
};

main().catch((error) => {
    console.error('Calibration dataset build failed.', error);
    process.exit(1);
});
