import { clampValue, lerp } from './utils.js';
import { DATA_CHANNEL_COUNT } from './constants.js';

const DEFAULT_THEME = {
    backgroundTop: 'rgba(8, 18, 40, 0.95)',
    backgroundBottom: 'rgba(2, 8, 24, 0.95)',
    grid: 'rgba(124, 216, 255, 0.14)',
    bar: 'rgba(79, 185, 255, 0.82)',
    barHighlight: 'rgba(255, 204, 102, 0.9)',
    barOver: 'rgba(255, 99, 132, 0.85)',
    sparkLine: 'rgba(124, 216, 255, 0.75)',
    sparkFill: 'rgba(124, 216, 255, 0.12)',
    textPrimary: 'rgba(216, 236, 255, 0.95)',
    textSecondary: 'rgba(180, 220, 255, 0.7)'
};

const adjustAlpha = (color, factor) => {
    if (typeof color !== 'string') {
        return color;
    }
    const match = color.match(/rgba\(([^)]+),\s*([0-9]*\.?[0-9]+)\)/i);
    if (!match) {
        return color;
    }
    const base = match[1];
    const alpha = clampValue(parseFloat(match[2]) * factor, 0, 1);
    return `rgba(${base}, ${alpha.toFixed(2)})`;
};

const DEFAULT_HIGHLIGHT_INDICES = [9, 10, 11, 12, 13, 14];

const sanitizeIndices = (indices, max) => {
    if (!indices) {
        return [];
    }
    if (typeof indices === 'number' && Number.isFinite(indices)) {
        return [Math.max(0, Math.min(max - 1, Math.floor(indices)))];
    }
    if (Array.isArray(indices) || ArrayBuffer.isView(indices)) {
        return Array.from(indices)
            .map((value) => Math.max(0, Math.min(max - 1, Math.floor(Number(value)))))
            .filter((value) => Number.isFinite(value));
    }
    if (indices instanceof Set) {
        return Array.from(indices)
            .map((value) => Math.max(0, Math.min(max - 1, Math.floor(Number(value)))))
            .filter((value) => Number.isFinite(value));
    }
    return [];
};

export class ChannelMonitor {
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.ctx = canvas && typeof canvas.getContext === 'function' ? canvas.getContext('2d') : null;
        this.enabled = Boolean(this.ctx);
        this.channelCount = Math.min(
            DATA_CHANNEL_COUNT,
            Math.max(1, Math.floor(options.channelCount ?? DATA_CHANNEL_COUNT))
        );
        this.smoothing = clampValue(typeof options.smoothing === 'number' ? options.smoothing : 0.35, 0, 1);
        this.theme = { ...DEFAULT_THEME, ...(options.theme || {}) };
        this.setValueRange(options.valueRange);

        this.displayValues = new Float32Array(this.channelCount);
        this.overHigh = new Array(this.channelCount).fill(false);
        this.overLow = new Array(this.channelCount).fill(false);

        this.historyLength = Math.max(0, Math.floor(options.historyLength ?? 120));
        this.history = this.historyLength > 1 && this.enabled
            ? Array.from({ length: this.channelCount }, () => new Float32Array(this.historyLength))
            : null;
        this.historyIndex = 0;
        this.historyCount = 0;
        this.summary = null;

        this.highlight = new Set();
        this.setHighlightIndices(options.highlightIndices ?? DEFAULT_HIGHLIGHT_INDICES);

        if (this.enabled) {
            this.resize();
            this.render();
        }
    }

    setHighlightIndices(indices) {
        this.highlight.clear();
        const sanitized = sanitizeIndices(indices, this.channelCount);
        sanitized.forEach((value) => this.highlight.add(value));
        this.render();
    }

    setSmoothing(value) {
        this.smoothing = clampValue(typeof value === 'number' ? value : this.smoothing, 0, 1);
    }

    setValueRange(range) {
        if (Array.isArray(range) && range.length === 2 && range.every((value) => Number.isFinite(value))) {
            const [a, b] = range;
            this.rangeMin = Math.min(a, b);
            this.rangeMax = Math.max(a, b);
        } else {
            this.rangeMin = 0;
            this.rangeMax = 1;
        }
        this.rangeDenom = this.rangeMax - this.rangeMin !== 0 ? this.rangeMax - this.rangeMin : 1;
        if (this.summary) {
            this.render();
        }
    }

    getValueRange() {
        return [this.rangeMin, this.rangeMax];
    }

    resize() {
        if (!this.enabled) {
            return;
        }
        const dpr = window.devicePixelRatio || 1;
        const width = this.canvas.clientWidth || 320;
        const height = this.canvas.clientHeight || 140;
        const internalWidth = Math.round(width * dpr);
        const internalHeight = Math.round(height * dpr);
        if (this.canvas.width !== internalWidth || this.canvas.height !== internalHeight) {
            this.canvas.width = internalWidth;
            this.canvas.height = internalHeight;
        }
        this.ctx.setTransform(1, 0, 0, 1, 0, 0);
        this.ctx.scale(dpr, dpr);
        this.internalWidth = width;
        this.internalHeight = height;
        this.render();
    }

    update(values = []) {
        if (!this.enabled) {
            return null;
        }
        const source = Array.isArray(values) || (values && typeof values.length === 'number') ? values : [];
        const length = Math.min(source.length, this.channelCount);
        let sum = 0;
        let maxValue = -Infinity;
        let maxIndex = 0;
        for (let i = 0; i < this.channelCount; i += 1) {
            const raw = i < length ? Number(source[i]) : 0;
            const normalized = clampValue((raw - this.rangeMin) / this.rangeDenom, 0, 1);
            const eased = lerp(this.displayValues[i], normalized, this.smoothing);
            this.displayValues[i] = eased;
            sum += eased;
            if (eased > maxValue) {
                maxValue = eased;
                maxIndex = i;
            }
            this.overHigh[i] = raw > this.rangeMax;
            this.overLow[i] = raw < this.rangeMin;
            if (this.history) {
                this.history[i][this.historyIndex] = eased;
            }
        }
        if (this.history) {
            this.historyIndex = (this.historyIndex + 1) % this.historyLength;
            this.historyCount = Math.min(this.historyCount + 1, this.historyLength);
        }
        this.summary = {
            average: this.channelCount ? sum / this.channelCount : 0,
            maxIndex,
            maxValue: maxValue > 0 ? maxValue : 0,
            sampleCount: length
        };
        this.render();
        return this.getSummary();
    }

    getSummary() {
        if (!this.summary) {
            return null;
        }
        return { ...this.summary };
    }

    render() {
        if (!this.enabled) {
            return;
        }
        const ctx = this.ctx;
        const width = this.internalWidth || this.canvas.width;
        const height = this.internalHeight || this.canvas.height;
        ctx.clearRect(0, 0, width, height);

        const gradient = ctx.createLinearGradient(0, 0, 0, height);
        gradient.addColorStop(0, this.theme.backgroundTop);
        gradient.addColorStop(1, this.theme.backgroundBottom);
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);

        const topMargin = 32;
        const bottomMargin = 20;
        const chartHeight = Math.max(0, height - topMargin - bottomMargin);

        ctx.strokeStyle = this.theme.grid;
        ctx.lineWidth = 1;
        const gridLines = 4;
        for (let i = 0; i <= gridLines; i += 1) {
            const y = topMargin + (chartHeight * i) / gridLines;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }

        const spacing = Math.max(2, width / (this.channelCount * 3 + 4));
        const availableWidth = width - spacing * (this.channelCount + 1);
        const barWidth = Math.max(2, availableWidth / this.channelCount);

        const highlightIndices = Array.from(this.highlight).filter((index) => index < this.channelCount);
        const highlightSet = new Set(highlightIndices);

        for (let i = 0; i < this.channelCount; i += 1) {
            const value = this.displayValues[i];
            const barHeight = value * chartHeight;
            const x = spacing + i * (barWidth + spacing);
            const y = topMargin + chartHeight - barHeight;
            const isHighlighted = highlightSet.has(i);
            const isOver = this.overHigh[i] || this.overLow[i];

            let fill = this.theme.bar;
            if (isOver) {
                fill = this.theme.barOver;
            } else if (isHighlighted) {
                fill = this.theme.barHighlight;
            }

            if (barHeight <= 0) {
                ctx.fillStyle = adjustAlpha(fill, 0.4);
                ctx.fillRect(x, topMargin + chartHeight - 1, barWidth, 1);
                continue;
            }

            const barGradient = ctx.createLinearGradient(x, y, x, y + barHeight);
            barGradient.addColorStop(0, fill);
            barGradient.addColorStop(1, adjustAlpha(fill, 0.6));
            ctx.fillStyle = barGradient;
            ctx.fillRect(x, y, barWidth, barHeight);

            if (barHeight > 3) {
                ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
                ctx.fillRect(x, y, barWidth, Math.min(3, barHeight));
            }
        }

        if (this.history && this.historyCount > 1) {
            const count = this.historyCount;
            const sparkHeight = topMargin - 12;
            if (sparkHeight > 4) {
                ctx.beginPath();
                for (let i = 0; i < count; i += 1) {
                    const pointer = (this.historyIndex - count + i + this.historyLength) % this.historyLength;
                    let aggregate = 0;
                    for (let channel = 0; channel < this.channelCount; channel += 1) {
                        aggregate += this.history[channel][pointer];
                    }
                    const average = aggregate / this.channelCount;
                    const x = 8 + ((width - 16) * i) / Math.max(1, count - 1);
                    const y = sparkHeight + 8 - average * sparkHeight;
                    if (i === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                }
                ctx.strokeStyle = this.theme.sparkLine;
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
        }

        const summary = this.summary || { average: 0, maxIndex: 0, maxValue: 0, sampleCount: 0 };
        const rangeLabel = `${this.rangeMin.toFixed(2)}–${this.rangeMax.toFixed(2)}`;
        ctx.fillStyle = this.theme.textPrimary;
        ctx.font = '11px "Inter", "Segoe UI", sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(
            `Avg ${summary.average.toFixed(2)} · Max C${summary.maxIndex}=${summary.maxValue.toFixed(2)} · Range ${rangeLabel}`,
            8,
            14
        );

        ctx.fillStyle = this.theme.textSecondary;
        const highlightLabel = highlightIndices.length
            ? `Highlights: ${highlightIndices.slice(0, 8).join(', ')}${highlightIndices.length > 8 ? '…' : ''}`
            : 'No highlight channels';
        ctx.fillText(highlightLabel, 8, height - 6);

        ctx.textAlign = 'right';
        ctx.fillText('1.0', width - 8, topMargin + 4);
        ctx.fillText('0.0', width - 8, topMargin + chartHeight + 14);
        ctx.textAlign = 'left';
    }
}
