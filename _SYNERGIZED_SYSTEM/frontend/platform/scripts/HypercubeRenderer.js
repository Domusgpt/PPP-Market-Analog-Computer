import { fragmentShaderSource, vertexShaderSource } from './shaders.js';
import {
    DATA_CHANNEL_COUNT,
    DEFAULT_BACKGROUND_COLOR,
    DEFAULT_PRIMARY_COLOR,
    DEFAULT_SECONDARY_COLOR
} from './constants.js';

class ShaderProgram {
    constructor(gl, vertexSource, fragmentSource) {
        this.gl = gl;
        this.program = this.#createProgram(vertexSource, fragmentSource);
    }

    use() {
        this.gl.useProgram(this.program);
    }

    getAttribLocation(name) {
        return this.gl.getAttribLocation(this.program, name);
    }

    getUniformLocation(name) {
        return this.gl.getUniformLocation(this.program, name);
    }

    #compileShader(type, source) {
        const shader = this.gl.createShader(type);
        if (!shader) {
            throw new Error('Unable to create shader.');
        }
        this.gl.shaderSource(shader, source);
        this.gl.compileShader(shader);
        if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
            const log = this.gl.getShaderInfoLog(shader) || 'Unknown shader compilation error.';
            this.gl.deleteShader(shader);
            throw new Error(log.trim());
        }
        return shader;
    }

    #createProgram(vertexSource, fragmentSource) {
        const vertexShader = this.#compileShader(this.gl.VERTEX_SHADER, vertexSource);
        const fragmentShader = this.#compileShader(this.gl.FRAGMENT_SHADER, fragmentSource);
        const program = this.gl.createProgram();
        if (!program) {
            throw new Error('Unable to create shader program.');
        }
        this.gl.attachShader(program, vertexShader);
        this.gl.attachShader(program, fragmentShader);
        this.gl.linkProgram(program);
        if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
            const log = this.gl.getProgramInfoLog(program) || 'Unknown program linking error.';
            this.gl.deleteProgram(program);
            throw new Error(log.trim());
        }
        this.gl.deleteShader(vertexShader);
        this.gl.deleteShader(fragmentShader);
        return program;
    }
}

export class HypercubeRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        const gl = canvas.getContext('webgl');
        if (!gl) {
            throw new Error('WebGL is not available in this environment.');
        }
        this.gl = gl;
        this.shaderProgram = new ShaderProgram(gl, vertexShaderSource, fragmentShaderSource);
        this.shaderProgram.use();
        this.positionLocation = this.shaderProgram.getAttribLocation('a_position');
        this.uniformLocations = {};
        this.#collectUniformLocations([
            'u_time', 'u_resolution', 'u_dimension', 'u_morphFactor', 'u_colorShift', 'u_patternIntensity',
            'u_gridDensity', 'u_lineThickness', 'u_shellWidth', 'u_tetraThickness', 'u_universeModifier',
            'u_glitchIntensity', 'u_rotXY', 'u_rotXZ', 'u_rotXW', 'u_rotYZ', 'u_rotYW', 'u_rotZW',
            'u_primaryColor', 'u_secondaryColor', 'u_backgroundColor', 'u_dataChannels'
        ]);
        this.#createFullscreenTriangle();
        gl.disable(gl.DEPTH_TEST);
        gl.disable(gl.CULL_FACE);
        gl.clearColor(0.01, 0.01, 0.04, 1.0);
        this.uniformState = {
            u_time: 0,
            u_resolution: new Float32Array([canvas.width || 1, canvas.height || 1]),
            u_dimension: 4.0,
            u_morphFactor: 0.6,
            u_colorShift: 0.0,
            u_patternIntensity: 2.0,
            u_gridDensity: 5.0,
            u_lineThickness: 0.08,
            u_shellWidth: 0.25,
            u_tetraThickness: 0.18,
            u_universeModifier: 1.0,
            u_glitchIntensity: 0.05,
            u_rotXY: 0.0,
            u_rotXZ: 0.0,
            u_rotXW: 0.0,
            u_rotYZ: 0.0,
            u_rotYW: 0.0,
            u_rotZW: 0.0,
            u_primaryColor: new Float32Array(DEFAULT_PRIMARY_COLOR),
            u_secondaryColor: new Float32Array(DEFAULT_SECONDARY_COLOR),
            u_backgroundColor: new Float32Array(DEFAULT_BACKGROUND_COLOR),
            u_dataChannels: new Float32Array(DATA_CHANNEL_COUNT)
        };
        this.uniformState.u_dataChannels.fill(0.5);
        this.startTime = performance.now();
        this.render = this.render.bind(this);
        this.resizeObserver = new ResizeObserver(() => this.resizeCanvas());
        this.resizeObserver.observe(canvas);
        this.resizeCanvas();
        requestAnimationFrame(this.render);
    }

    setUniformState(updates) {
        Object.entries(updates).forEach(([key, value]) => {
            if (!(key in this.uniformState)) {
                return;
            }
            const current = this.uniformState[key];
            if (current instanceof Float32Array) {
                this.#assignToFloatArray(current, value);
            } else if (value instanceof Float32Array) {
                this.uniformState[key] = new Float32Array(value);
            } else if (Array.isArray(value)) {
                this.uniformState[key] = value.slice();
            } else if (typeof value === 'number') {
                this.uniformState[key] = value;
            }
        });
    }

    getUniformState() {
        const snapshot = {};
        Object.entries(this.uniformState).forEach(([key, value]) => {
            if (Array.isArray(value)) {
                snapshot[key] = value.slice();
            } else if (value instanceof Float32Array) {
                snapshot[key] = Array.from(value);
            } else {
                snapshot[key] = value;
            }
        });
        return snapshot;
    }

    resizeCanvas() {
        const dpr = window.devicePixelRatio || 1;
        const displayWidth = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
        const displayHeight = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
        if (this.canvas.width !== displayWidth || this.canvas.height !== displayHeight) {
            this.canvas.width = displayWidth;
            this.canvas.height = displayHeight;
        }
        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        const resolution = this.uniformState.u_resolution;
        if (resolution instanceof Float32Array && resolution.length >= 2) {
            resolution[0] = this.canvas.width;
            resolution[1] = this.canvas.height;
        } else {
            this.uniformState.u_resolution = [this.canvas.width, this.canvas.height];
        }
    }

    render(now) {
        const elapsedSeconds = (now - this.startTime) * 0.001;
        this.uniformState.u_time = elapsedSeconds;
        this.resizeCanvas();
        this.gl.clear(this.gl.COLOR_BUFFER_BIT);
        this.shaderProgram.use();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.fullscreenBuffer);
        this.gl.enableVertexAttribArray(this.positionLocation);
        this.gl.vertexAttribPointer(this.positionLocation, 2, this.gl.FLOAT, false, 0, 0);
        this.#uploadUniforms();
        this.gl.drawArrays(this.gl.TRIANGLES, 0, 3);
        requestAnimationFrame(this.render);
    }

    dispose() {
        this.resizeObserver.disconnect();
    }

    #collectUniformLocations(names) {
        names.forEach((name) => {
            this.uniformLocations[name] = this.shaderProgram.getUniformLocation(name);
        });
    }

    #createFullscreenTriangle() {
        const vertices = new Float32Array([
            -1, -1,
             3, -1,
            -1,  3
        ]);
        this.fullscreenBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.fullscreenBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, vertices, this.gl.STATIC_DRAW);
    }

    #uploadUniforms() {
        Object.entries(this.uniformState).forEach(([name, value]) => {
            const location = this.uniformLocations[name];
            if (location === null || location === undefined) {
                return;
            }
            if (typeof value === 'number') {
                this.gl.uniform1f(location, value);
            } else if (Array.isArray(value)) {
                this.#uploadArrayUniform(location, value);
            } else if (value instanceof Float32Array) {
                this.#uploadArrayUniform(location, value);
            }
        });
    }

    #uploadArrayUniform(location, value) {
        const length = value.length;
        const typed = value instanceof Float32Array ? value : new Float32Array(value);
        if (length === 2) {
            this.gl.uniform2fv(location, typed);
        } else if (length === 3) {
            this.gl.uniform3fv(location, typed);
        } else if (length === 4) {
            this.gl.uniform4fv(location, typed);
        } else {
            this.gl.uniform1fv(location, typed);
        }
    }

    #assignToFloatArray(target, source) {
        const length = target.length;
        if (source instanceof Float32Array) {
            const copyLength = Math.min(length, source.length);
            target.set(source.subarray(0, copyLength));
            if (copyLength < length) {
                target.fill(0, copyLength);
            }
        } else if (Array.isArray(source)) {
            const copyLength = Math.min(length, source.length);
            for (let i = 0; i < copyLength; i += 1) {
                target[i] = Number.isFinite(source[i]) ? source[i] : 0;
            }
            if (copyLength < length) {
                target.fill(0, copyLength);
            }
        } else if (typeof source === 'number') {
            target.fill(source);
        }
    }

    getCanvas() {
        return this.canvas;
    }

    captureFrame({ format = 'image/png', quality = 0.92 } = {}) {
        if (!this.canvas || typeof this.canvas.toDataURL !== 'function') {
            return null;
        }
        try {
            return this.canvas.toDataURL(format, quality);
        } catch (error) {
            console.warn('HypercubeRenderer captureFrame failed', error);
            return null;
        }
    }
}
