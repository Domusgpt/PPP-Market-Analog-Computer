const cloneArray = (value) => {
    if (Array.isArray(value)) {
        return value.slice();
    }
    if (ArrayBuffer.isView(value)) {
        return Array.from(value);
    }
    if (value && typeof value.length === 'number') {
        try {
            return Array.from(value);
        } catch (error) {
            return [];
        }
    }
    return [];
};

const cloneUniformSnapshot = (uniforms) => {
    if (!uniforms || typeof uniforms !== 'object') {
        return {};
    }
    return Object.fromEntries(
        Object.entries(uniforms).map(([key, value]) => {
            if (Array.isArray(value)) {
                return [key, value.slice()];
            }
            if (ArrayBuffer.isView(value)) {
                return [key, Array.from(value)];
            }
            if (value && typeof value === 'object') {
                return [key, cloneUniformSnapshot(value)];
            }
            return [key, value];
        })
    );
};

const DEFAULT_FRAME_INTERVAL = 140;

const computeProgress = (elapsed, duration, index, frameCount) => {
    if (!frameCount) {
        return 0;
    }
    if (frameCount === 1) {
        return 1;
    }
    if (Number.isFinite(duration) && duration > 0 && Number.isFinite(elapsed)) {
        return Math.min(1, Math.max(0, elapsed / duration));
    }
    if (Number.isFinite(index) && frameCount > 1) {
        return Math.min(1, Math.max(0, index / (frameCount - 1)));
    }
    return 0;
};

const sanitizeFrames = (frames = []) => {
    const sanitized = [];
    let lastElapsed = 0;
    frames.forEach((frame, index) => {
        if (!frame || typeof frame !== 'object') {
            return;
        }
        const elapsed = Number.isFinite(frame.elapsed)
            ? frame.elapsed
            : index === 0
                ? 0
                : lastElapsed + DEFAULT_FRAME_INTERVAL;
        const data = cloneArray(frame.data);
        const uniforms = frame.uniforms && typeof frame.uniforms === 'object'
            ? cloneUniformSnapshot(frame.uniforms)
            : undefined;
        sanitized.push({
            index,
            elapsed,
            data,
            uniforms
        });
        lastElapsed = elapsed;
    });
    return sanitized;
};

const buildEmptyStatus = () => ({
    state: 'empty',
    frameCount: 0,
    currentIndex: -1,
    playing: false,
    loop: false,
    speed: 1,
    duration: 0,
    progress: 0,
    currentElapsed: 0,
    sourceLabel: ''
});

export class DataPlayer {
    constructor({ onFrame, onStatusChange } = {}) {
        this.frames = [];
        this.currentIndex = -1;
        this.timeoutId = null;
        this.playing = false;
        this.loop = false;
        this.speed = 1;
        this.duration = 0;
        this.onFrame = typeof onFrame === 'function' ? onFrame : null;
        this.onStatusChange = typeof onStatusChange === 'function' ? onStatusChange : null;
        this.sourceLabel = '';
        this.lastStatus = buildEmptyStatus();
    }

    loadFrames(frames, { label } = {}) {
        this.clear();
        const sanitized = sanitizeFrames(frames);
        this.frames = sanitized;
        this.duration = sanitized.length
            ? sanitized[sanitized.length - 1].elapsed
            : 0;
        this.sourceLabel = typeof label === 'string' ? label : '';
        this.#notifyStatus();
        return this.frames.length;
    }

    loadFromRecorderExport(payload, { label } = {}) {
        if (!payload) {
            return 0;
        }
        let frames = [];
        if (Array.isArray(payload)) {
            frames = payload;
        } else if (Array.isArray(payload.records)) {
            frames = payload.records;
        } else if (Array.isArray(payload.frames)) {
            frames = payload.frames;
        }
        return this.loadFrames(frames, { label });
    }

    hasFrames() {
        return this.frames.length > 0;
    }

    getFrame(index) {
        if (!this.hasFrames()) {
            return null;
        }
        const clamped = Math.max(0, Math.min(this.frames.length - 1, Math.floor(index)));
        const frame = this.frames[clamped];
        return this.#clonePublicFrame(frame);
    }

    getCurrentFrame() {
        if (!this.hasFrames() || this.currentIndex < 0 || this.currentIndex >= this.frames.length) {
            return null;
        }
        return this.#clonePublicFrame(this.frames[this.currentIndex]);
    }

    play(options = {}) {
        if (!this.hasFrames()) {
            this.#notifyStatus();
            return false;
        }

        if (typeof options.speed === 'number' && options.speed > 0) {
            this.setSpeed(options.speed);
        }
        if (typeof options.loop === 'boolean') {
            this.setLoop(options.loop);
        }

        if (options.restart === true) {
            this.currentIndex = -1;
        }

        if (this.playing) {
            this.#notifyStatus();
            return true;
        }

        this.playing = true;
        const nextIndex = this.currentIndex < 0 || this.currentIndex >= this.frames.length - 1
            ? 0
            : this.currentIndex + 1;
        this.#emitFrame(nextIndex, 'play');
        this.#scheduleNextFrame(nextIndex + 1);
        return true;
    }

    pause() {
        if (this.timeoutId !== null) {
            clearTimeout(this.timeoutId);
            this.timeoutId = null;
        }
        const wasPlaying = this.playing;
        this.playing = false;
        if (wasPlaying) {
            this.#notifyStatus();
        }
        return this.getStatus();
    }

    stop({ reset = true } = {}) {
        const status = this.pause();
        if (reset) {
            this.currentIndex = -1;
            this.#notifyStatus();
        }
        return status;
    }

    seek(index) {
        if (!this.hasFrames()) {
            return null;
        }
        this.pause();
        const clamped = Math.max(0, Math.min(this.frames.length - 1, Math.floor(index)));
        return this.#emitFrame(clamped, 'seek');
    }

    seekToElapsed(elapsed) {
        if (!this.hasFrames()) {
            return null;
        }
        const target = Number.isFinite(elapsed) ? Math.max(0, elapsed) : 0;
        this.pause();
        const index = this.#findFrameIndexByElapsed(target);
        return this.#emitFrame(index, 'seek');
    }

    seekToProgress(progress) {
        if (!this.hasFrames()) {
            return null;
        }
        const normalized = Number.isFinite(progress) ? Math.min(1, Math.max(0, progress)) : 0;
        if (this.frames.length <= 1) {
            return this.seek(0);
        }
        if (this.duration > 0) {
            return this.seekToElapsed(normalized * this.duration);
        }
        const lastIndex = this.frames.length - 1;
        return this.seek(normalized * lastIndex);
    }

    step(direction = 1) {
        if (!this.hasFrames()) {
            return null;
        }
        this.pause();
        const delta = direction >= 0 ? 1 : -1;
        let nextIndex;
        if (this.currentIndex < 0) {
            nextIndex = delta >= 0 ? 0 : this.frames.length - 1;
        } else {
            nextIndex = this.currentIndex + delta;
            if (nextIndex < 0) {
                nextIndex = 0;
            }
            if (nextIndex >= this.frames.length) {
                nextIndex = this.frames.length - 1;
            }
        }
        return this.#emitFrame(nextIndex, 'step');
    }

    setLoop(value) {
        this.loop = Boolean(value);
        this.#notifyStatus();
        return this.loop;
    }

    setSpeed(value) {
        if (Number.isFinite(value) && value > 0) {
            this.speed = value;
            if (this.playing) {
                if (this.timeoutId !== null) {
                    clearTimeout(this.timeoutId);
                    this.timeoutId = null;
                }
                const nextIndex = this.currentIndex + 1;
                this.#scheduleNextFrame(nextIndex);
            }
            this.#notifyStatus();
        }
        return this.speed;
    }

    clear() {
        this.pause();
        this.frames = [];
        this.currentIndex = -1;
        this.duration = 0;
        this.sourceLabel = '';
        this.#notifyStatus();
    }

    getStatus() {
        return { ...this.lastStatus };
    }

    getDuration() {
        return this.duration;
    }

    #scheduleNextFrame(targetIndex) {
        if (!this.playing) {
            return;
        }
        if (!this.hasFrames()) {
            this.pause();
            return;
        }
        if (targetIndex >= this.frames.length) {
            if (this.loop) {
                const delay = this.#computeDelay(this.frames.length - 1, 0);
                this.timeoutId = setTimeout(() => {
                    if (!this.playing) {
                        return;
                    }
                    this.#emitFrame(0, 'loop');
                    this.#scheduleNextFrame(1);
                }, this.#normalizeDelay(delay));
            } else {
                this.pause();
            }
            return;
        }
        const delay = this.#computeDelay(this.currentIndex, targetIndex);
        this.timeoutId = setTimeout(() => {
            if (!this.playing) {
                return;
            }
            this.#emitFrame(targetIndex, 'play');
            this.#scheduleNextFrame(targetIndex + 1);
        }, this.#normalizeDelay(delay));
    }

    #computeDelay(prevIndex, nextIndex) {
        if (nextIndex <= 0) {
            return 0;
        }
        const prevElapsed = prevIndex >= 0 && prevIndex < this.frames.length
            ? this.frames[prevIndex].elapsed
            : 0;
        const nextElapsed = nextIndex < this.frames.length
            ? this.frames[nextIndex].elapsed
            : prevElapsed;
        const delta = Math.max(0, nextElapsed - prevElapsed);
        return this.speed !== 0 ? delta / this.speed : delta;
    }

    #normalizeDelay(value) {
        if (!Number.isFinite(value) || value <= 0) {
            return 0;
        }
        return value;
    }

    #emitFrame(index, reason) {
        if (index < 0 || index >= this.frames.length) {
            return null;
        }
        const frame = this.frames[index];
        this.currentIndex = index;
        const payload = this.#clonePublicFrame(frame);
        if (typeof reason === 'string') {
            payload.reason = reason;
        }
        if (this.onFrame) {
            this.onFrame(payload, reason);
        }
        this.#notifyStatus();
        return payload;
    }

    #clonePublicFrame(frame) {
        if (!frame) {
            return null;
        }
        const data = cloneArray(frame.data);
        const uniforms = frame.uniforms ? cloneUniformSnapshot(frame.uniforms) : undefined;
        const progress = computeProgress(frame.elapsed, this.duration, frame.index, this.frames.length);
        return {
            index: frame.index,
            frameCount: this.frames.length,
            elapsed: frame.elapsed,
            duration: this.duration,
            data,
            progress,
            ...(uniforms ? { uniforms } : {}),
            sourceLabel: this.sourceLabel || ''
        };
    }

    #buildStatus() {
        if (!this.hasFrames()) {
            return buildEmptyStatus();
        }
        const currentFrame = this.currentIndex >= 0 && this.currentIndex < this.frames.length
            ? this.frames[this.currentIndex]
            : null;
        let state = 'loaded';
        if (this.playing) {
            state = 'playing';
        } else if (this.currentIndex < 0) {
            state = 'loaded';
        } else if (this.currentIndex >= this.frames.length - 1) {
            state = 'finished';
        } else {
            state = 'paused';
        }
        const progress = currentFrame
            ? computeProgress(currentFrame.elapsed, this.duration, this.currentIndex, this.frames.length)
            : 0;
        return {
            state,
            frameCount: this.frames.length,
            currentIndex: this.currentIndex,
            playing: this.playing,
            loop: this.loop,
            speed: this.speed,
            duration: this.duration,
            progress,
            currentElapsed: currentFrame ? currentFrame.elapsed : 0,
            sourceLabel: this.sourceLabel || ''
        };
    }

    #notifyStatus() {
        const status = this.#buildStatus();
        this.lastStatus = status;
        if (this.onStatusChange) {
            this.onStatusChange({ ...status });
        }
    }

    #findFrameIndexByElapsed(targetElapsed) {
        if (!this.hasFrames()) {
            return 0;
        }
        if (!Number.isFinite(targetElapsed) || targetElapsed <= 0) {
            return 0;
        }
        if (this.duration > 0 && targetElapsed >= this.duration) {
            return this.frames.length - 1;
        }
        for (let index = 0; index < this.frames.length; index += 1) {
            const frame = this.frames[index];
            if (frame.elapsed >= targetElapsed) {
                return index;
            }
        }
        return this.frames.length - 1;
    }
}
