import { DATA_CHANNEL_COUNT } from './constants.js';

export const vertexShaderSource = /* glsl */ `
    attribute vec2 a_position;

    void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
    }
`;

export const fragmentShaderSource = /* glsl */ `
    precision highp float;

    #define DATA_CHANNEL_COUNT ${DATA_CHANNEL_COUNT}

    uniform vec2 u_resolution;
    uniform float u_time;
    uniform float u_dimension;
    uniform float u_morphFactor;
    uniform float u_colorShift;
    uniform float u_patternIntensity;
    uniform float u_gridDensity;
    uniform float u_lineThickness;
    uniform float u_shellWidth;
    uniform float u_tetraThickness;
    uniform float u_universeModifier;
    uniform float u_glitchIntensity;

    uniform float u_rotXY;
    uniform float u_rotXZ;
    uniform float u_rotXW;
    uniform float u_rotYZ;
    uniform float u_rotYW;
    uniform float u_rotZW;

    uniform vec3 u_primaryColor;
    uniform vec3 u_secondaryColor;
    uniform vec3 u_backgroundColor;

    uniform float u_dataChannels[DATA_CHANNEL_COUNT];

    mat4 rotXY(float angle) {
        float c = cos(angle);
        float s = sin(angle);
        return mat4(
            c,  s,  0.0, 0.0,
           -s,  c,  0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        );
    }

    mat4 rotXZ(float angle) {
        float c = cos(angle);
        float s = sin(angle);
        return mat4(
            c,  0.0, s,   0.0,
            0.0, 1.0, 0.0, 0.0,
           -s,  0.0, c,   0.0,
            0.0, 0.0, 0.0, 1.0
        );
    }

    mat4 rotXW(float angle) {
        float c = cos(angle);
        float s = sin(angle);
        return mat4(
            c,  0.0, 0.0, s,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
           -s,  0.0, 0.0, c
        );
    }

    mat4 rotYZ(float angle) {
        float c = cos(angle);
        float s = sin(angle);
        return mat4(
            1.0, 0.0, 0.0, 0.0,
            0.0, c,   s,   0.0,
            0.0, -s,  c,   0.0,
            0.0, 0.0, 0.0, 1.0
        );
    }

    mat4 rotYW(float angle) {
        float c = cos(angle);
        float s = sin(angle);
        return mat4(
            1.0, 0.0, 0.0, 0.0,
            0.0, c,   0.0, s,
            0.0, 0.0, 1.0, 0.0,
            0.0, -s,  0.0, c
        );
    }

    mat4 rotZW(float angle) {
        float c = cos(angle);
        float s = sin(angle);
        return mat4(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, c,   s,
            0.0, 0.0, -s,  c
        );
    }

    vec4 apply6DRotation(vec4 p) {
        p = rotXY(u_rotXY) * p;
        p = rotXZ(u_rotXZ) * p;
        p = rotXW(u_rotXW) * p;
        p = rotYZ(u_rotYZ) * p;
        p = rotYW(u_rotYW) * p;
        p = rotZW(u_rotZW) * p;
        return p;
    }

    vec3 project4Dto3D(vec4 p) {
        float baseDistance = 2.6 + u_dimension * 0.35 + u_dataChannels[8] * 0.35;
        float denom = baseDistance - p.w;
        float factor = baseDistance / max(0.25, denom);
        return p.xyz * factor;
    }

    float tetrahedralField(vec3 p) {
        const float invSqrt3 = 0.5773502692;
        float plane0 = abs(p.x + p.y + p.z) * invSqrt3;
        float plane1 = abs(p.x + p.y - p.z) * invSqrt3;
        float plane2 = abs(p.x - p.y + p.z) * invSqrt3;
        float plane3 = abs(-p.x + p.y + p.z) * invSqrt3;
        float d = min(min(plane0, plane1), min(plane2, plane3));
        return 1.0 - smoothstep(u_tetraThickness, u_tetraThickness + 0.08, d);
    }

    void main() {
        vec2 uv = (gl_FragCoord.xy / u_resolution.xy) * 2.0 - 1.0;
        uv.x *= u_resolution.x / max(u_resolution.y, 1.0);

        vec2 glitch = vec2(
            sin(u_time * (9.0 + u_dataChannels[20] * 2.5) + uv.y * (16.0 + u_dataChannels[21] * 4.0) + u_dataChannels[5] * 6.28318),
            cos(u_time * (7.0 + u_dataChannels[20] * 1.5) + uv.x * (13.0 + u_dataChannels[21] * 3.5) - u_dataChannels[6] * 6.28318)
        ) * (0.035 * u_glitchIntensity);
        uv += glitch;

        float dynamicDepth = sin(u_time * 0.27 + uv.x * (u_patternIntensity + u_dataChannels[24] * 0.8) + u_dataChannels[2] * 6.28318);
        float dynamicWidth = cos(u_time * 0.21 + uv.y * (u_patternIntensity + u_dataChannels[25] * 0.8) + u_dataChannels[3] * 6.28318);
        float widthMod = max(0.15, 0.8 + u_morphFactor * 0.6 + u_dataChannels[16] * 0.4);
        float depthMod = max(0.15, 0.9 + u_morphFactor * 0.4 + u_dataChannels[17] * 0.35);
        float wOffset = u_dataChannels[18] * 0.8;

        vec4 p4 = vec4(
            uv * (1.1 + u_morphFactor * 0.55 + u_dataChannels[0] * 0.25),
            dynamicWidth * widthMod,
            dynamicDepth * depthMod
        );
        p4.w += wOffset;
        p4.xy += u_dataChannels[19] * vec2(0.35, -0.28);

        vec4 rotated = apply6DRotation(p4);
        vec3 p = project4Dto3D(rotated);

        float dimensionBlend = clamp((u_dimension - 3.0) * 0.6, 0.0, 1.0);
        float radialTarget = mix(0.85, 1.45, dimensionBlend) + u_dataChannels[10] * 0.25 + u_dataChannels[20] * 0.35;
        float radialBand = smoothstep(u_shellWidth, 0.0, abs(length(p) - radialTarget));

        float gridScale = u_gridDensity + u_dataChannels[9] * 5.0 + u_dataChannels[21] * 4.0;
        vec3 cell = fract(
            p * gridScale + vec3(
                u_time * 0.05 + u_dataChannels[22] * 0.12,
                u_time * 0.035 - u_dataChannels[22] * 0.08,
                u_time * -0.04 + u_dataChannels[22] * 0.1
            )
        ) - 0.5;
        float cellDistance = min(min(abs(cell.x), abs(cell.y)), abs(cell.z));
        float lattice = smoothstep(u_lineThickness, 0.0, cellDistance);

        float tetra = tetrahedralField(p);

        float baseMorph = mix(lattice, radialBand, clamp(u_morphFactor, 0.0, 1.0));
        float tetraInfluence = clamp(u_dataChannels[11] * 0.5 + 0.5 + u_dataChannels[23] * 0.5, 0.0, 1.0);
        float structure = mix(baseMorph, max(baseMorph, tetra), tetraInfluence);

        float xFrequency = u_patternIntensity + 1.0 + u_dataChannels[12] * 4.0;
        float yFrequency = u_patternIntensity + 0.5 + u_dataChannels[25] * 2.0;
        float phaseShift = u_dataChannels[24] * 3.14159265;
        float temporalShift = u_dataChannels[26];
        float wave = abs(
            sin(p.x * xFrequency + u_time * (0.6 + u_dataChannels[13]) + phaseShift) *
            cos(p.y * yFrequency - u_time * (0.45 + temporalShift))
        );
        float exponentMod = mix(0.65, 1.6, clamp(0.5 + 0.5 * u_dataChannels[27], 0.0, 1.0));
        float pattern = pow(wave * (0.55 + structure * 0.45) + 0.0005, exponentMod / max(0.25, u_universeModifier));
        float highlight = clamp(pattern + structure * 0.35, 0.0, 1.0);

        float glitchBand = sin(u_time * (12.0 + u_dataChannels[20] * 8.0) + p.z * 18.0 + u_dataChannels[14] * 6.28318) * u_glitchIntensity;
        float luminance = clamp(highlight + glitchBand, 0.0, 1.0);
        luminance = clamp(luminance + u_colorShift, 0.0, 1.0);

        float accentBlend = clamp(0.5 + 0.5 * u_dataChannels[28], 0.0, 1.0);
        vec3 accentColor = mix(u_primaryColor, u_secondaryColor, accentBlend);
        float tertiaryBlend = clamp(0.5 + 0.5 * u_dataChannels[29], 0.0, 1.0);
        vec3 tertiaryColor = mix(vec3(0.08, 0.1, 0.24), accentColor, tertiaryBlend);
        float haloStrength = clamp(0.5 + 0.5 * u_dataChannels[30], 0.0, 1.0);
        float tertiaryWeight = clamp(0.5 + 0.5 * u_dataChannels[31], 0.0, 1.0);
        vec3 gradient = mix(u_primaryColor, u_secondaryColor, luminance);
        vec3 color = mix(u_backgroundColor, gradient, clamp(structure + pattern, 0.0, 1.0));
        color = mix(color, accentColor, 0.12 + 0.25 * haloStrength * (1.0 - structure));
        float accentLift = clamp(0.5 + 0.5 * u_dataChannels[15], 0.0, 1.0);
        color += (accentColor - u_primaryColor) * (0.18 * accentLift * luminance);
        color += tertiaryColor * (0.08 + 0.18 * tertiaryWeight) * luminance;
        color = clamp(color, 0.0, 1.0);

        gl_FragColor = vec4(color, 1.0);
    }
`;
