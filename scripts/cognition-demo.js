import { VisualCognitionEngine } from './visualCognitionEngine.js';

const createDemo = () => {
    const canvas = document.getElementById('cognitionCanvas');
    const telemetryOutput = document.getElementById('cognitionTelemetry');
    const topologyOutput = document.getElementById('cognitionTopology');
    const startButton = document.getElementById('cognitionStart');
    const stopButton = document.getElementById('cognitionStop');
    const speedControl = document.getElementById('cognitionSpeed');
    const speedValue = document.getElementById('cognitionSpeedValue');

    if (!canvas || !telemetryOutput || !topologyOutput) {
        return;
    }

    const context = canvas.getContext('2d');
    const engine = new VisualCognitionEngine();
    const topology = engine.exportTopology();
    topologyOutput.textContent = JSON.stringify(topology, null, 2);

    let animationFrame = null;
    let lastTime = 0;
    let isRunning = false;

    const resizeCanvas = () => {
        const rect = canvas.getBoundingClientRect();
        const ratio = window.devicePixelRatio || 1;
        canvas.width = rect.width * ratio;
        canvas.height = rect.height * ratio;
        context.setTransform(ratio, 0, 0, ratio, 0, 0);
    };

    const update = (timestamp) => {
        const delta = lastTime ? (timestamp - lastTime) / 1000 : 0;
        lastTime = timestamp;
        const speed = parseFloat(speedControl.value);
        engine.updateRotation({
            xy: delta * 0.3 * speed,
            xz: delta * 0.2 * speed,
            xw: delta * 0.4 * speed,
            yz: delta * 0.25 * speed,
            yw: delta * 0.18 * speed,
            zw: delta * 0.35 * speed
        });
        engine.updateStateFromChannels([
            0.5 + 0.5 * Math.sin(timestamp * 0.0006),
            0.5 + 0.5 * Math.cos(timestamp * 0.0004),
            0.5 + 0.5 * Math.sin(timestamp * 0.0003),
            0.5 + 0.5 * Math.cos(timestamp * 0.0005)
        ]);
        const { telemetry } = engine.renderToCanvas(canvas, context);
        telemetryOutput.textContent = JSON.stringify(telemetry, null, 2);
        animationFrame = requestAnimationFrame(update);
    };

    const start = () => {
        if (isRunning) {
            return;
        }
        isRunning = true;
        lastTime = 0;
        resizeCanvas();
        animationFrame = requestAnimationFrame(update);
    };

    const stop = () => {
        if (!isRunning) {
            return;
        }
        isRunning = false;
        cancelAnimationFrame(animationFrame);
    };

    startButton?.addEventListener('click', start);
    stopButton?.addEventListener('click', stop);
    speedControl?.addEventListener('input', () => {
        speedValue.textContent = `${speedControl.value}×`;
    });
    window.addEventListener('resize', () => {
        if (isRunning) {
            resizeCanvas();
        }
    });

    speedValue.textContent = `${speedControl.value}×`;
    start();
};

window.addEventListener('DOMContentLoaded', createDemo);
