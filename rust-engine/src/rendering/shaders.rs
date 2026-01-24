//! WGSL Shader source code for the rendering pipeline

/// Main vertex shader for polytope rendering
pub const VERTEX_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    time: f32,
    point_size: f32,
    edge_width: f32,
    alpha_blend: f32,
    resolution: vec2<f32>,
    _padding: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) w_depth: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) w_depth: f32,
    @location(2) world_pos: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Apply view-projection transform
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);

    // Pass through color with alpha modulation based on w-depth
    let depth_alpha = 1.0 - abs(in.w_depth) * 0.2;
    out.color = vec4<f32>(in.color.rgb, in.color.a * depth_alpha * uniforms.alpha_blend);

    out.w_depth = in.w_depth;
    out.world_pos = in.position;

    return out;
}
"#;

/// Main fragment shader for polytope rendering
pub const FRAGMENT_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    time: f32,
    point_size: f32,
    edge_width: f32,
    alpha_blend: f32,
    resolution: vec2<f32>,
    _padding: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct FragmentInput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) w_depth: f32,
    @location(2) world_pos: vec3<f32>,
}

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    // Base color with depth-based shading
    var color = in.color;

    // Add subtle glow effect based on w-depth
    let glow = exp(-abs(in.w_depth) * 2.0) * 0.3;
    color = vec4<f32>(color.rgb + vec3<f32>(glow), color.a);

    // Gamma correction
    let gamma = 2.2;
    color = vec4<f32>(pow(color.rgb, vec3<f32>(1.0 / gamma)), color.a);

    return color;
}
"#;

/// Point rendering vertex shader (for vertices as points)
#[allow(dead_code)]
pub const POINT_VERTEX_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    time: f32,
    point_size: f32,
    edge_width: f32,
    alpha_blend: f32,
    resolution: vec2<f32>,
    _padding: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) w_depth: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @builtin(point_size) point_size: f32,
    @location(0) color: vec4<f32>,
    @location(1) w_depth: f32,
}

@vertex
fn vs_point(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);

    // Scale point size by depth
    let depth_scale = 1.0 / (1.0 + abs(in.w_depth) * 0.5);
    out.point_size = uniforms.point_size * depth_scale;

    // Color with pulsing effect
    let pulse = sin(uniforms.time * 2.0 + in.w_depth * 3.0) * 0.1 + 0.9;
    out.color = vec4<f32>(in.color.rgb * pulse, in.color.a * uniforms.alpha_blend);

    out.w_depth = in.w_depth;

    return out;
}
"#;

/// Point fragment shader with circular points and glow
#[allow(dead_code)]
pub const POINT_FRAGMENT_SHADER: &str = r#"
struct FragmentInput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) w_depth: f32,
}

@fragment
fn fs_point(in: FragmentInput) -> @location(0) vec4<f32> {
    // Soft circular point with glow
    let dist = length(vec2<f32>(0.5, 0.5) - fract(in.clip_position.xy / 8.0));
    let alpha = smoothstep(0.5, 0.3, dist);

    // Add core brightness
    let core = smoothstep(0.3, 0.0, dist) * 0.5;

    var color = in.color;
    color = vec4<f32>(color.rgb + vec3<f32>(core), color.a * alpha);

    return color;
}
"#;

/// Post-processing shader for pixel-level analog computation
#[allow(dead_code)]
pub const POSTPROCESS_SHADER: &str = r#"
@group(0) @binding(0)
var input_texture: texture_2d<f32>;
@group(0) @binding(1)
var texture_sampler: sampler;

struct PostUniforms {
    resolution: vec2<f32>,
    time: f32,
    blend_mode: u32,
    threshold: f32,
    glow_intensity: f32,
    _padding: vec2<f32>,
}

@group(0) @binding(2)
var<uniform> post_uniforms: PostUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Full-screen triangle
@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    // Generate fullscreen triangle vertices
    let x = f32((vertex_index & 1u) << 2u) - 1.0;
    let y = f32((vertex_index & 2u) << 1u) - 1.0;

    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);

    return out;
}

@fragment
fn fs_postprocess(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = textureSample(input_texture, texture_sampler, in.uv);

    // Apply blend mode effects
    var color = texel;

    // Mode 0: Pass-through
    // Mode 1: Threshold (binary)
    // Mode 2: Glow enhancement
    // Mode 3: Edge detection

    if post_uniforms.blend_mode == 1u {
        // Threshold mode - create binary pattern
        let luminance = dot(color.rgb, vec3<f32>(0.299, 0.587, 0.114));
        let binary = step(post_uniforms.threshold, luminance);
        color = vec4<f32>(vec3<f32>(binary), color.a);
    } else if post_uniforms.blend_mode == 2u {
        // Glow enhancement
        let glow = post_uniforms.glow_intensity;

        // Sample neighbors for blur
        let pixel_size = 1.0 / post_uniforms.resolution;
        var blur = vec4<f32>(0.0);
        for (var dx = -2; dx <= 2; dx++) {
            for (var dy = -2; dy <= 2; dy++) {
                let offset = vec2<f32>(f32(dx), f32(dy)) * pixel_size;
                blur += textureSample(input_texture, texture_sampler, in.uv + offset);
            }
        }
        blur = blur / 25.0;

        color = mix(color, blur, glow) + color * glow * 0.5;
    } else if post_uniforms.blend_mode == 3u {
        // Sobel edge detection
        let pixel_size = 1.0 / post_uniforms.resolution;

        let tl = textureSample(input_texture, texture_sampler, in.uv + vec2<f32>(-1.0, -1.0) * pixel_size).r;
        let t  = textureSample(input_texture, texture_sampler, in.uv + vec2<f32>( 0.0, -1.0) * pixel_size).r;
        let tr = textureSample(input_texture, texture_sampler, in.uv + vec2<f32>( 1.0, -1.0) * pixel_size).r;
        let l  = textureSample(input_texture, texture_sampler, in.uv + vec2<f32>(-1.0,  0.0) * pixel_size).r;
        let r  = textureSample(input_texture, texture_sampler, in.uv + vec2<f32>( 1.0,  0.0) * pixel_size).r;
        let bl = textureSample(input_texture, texture_sampler, in.uv + vec2<f32>(-1.0,  1.0) * pixel_size).r;
        let b  = textureSample(input_texture, texture_sampler, in.uv + vec2<f32>( 0.0,  1.0) * pixel_size).r;
        let br = textureSample(input_texture, texture_sampler, in.uv + vec2<f32>( 1.0,  1.0) * pixel_size).r;

        let gx = -tl - 2.0*l - bl + tr + 2.0*r + br;
        let gy = -tl - 2.0*t - tr + bl + 2.0*b + br;
        let edge = sqrt(gx*gx + gy*gy);

        color = vec4<f32>(vec3<f32>(edge), 1.0);
    }

    return color;
}
"#;

/// Compute shader for pixel-rule analog processing
#[allow(dead_code)]
pub const COMPUTE_SHADER: &str = r#"
@group(0) @binding(0)
var input_texture: texture_2d<f32>;
@group(0) @binding(1)
var output_texture: texture_storage_2d<rgba8unorm, write>;

struct ComputeUniforms {
    resolution: vec2<u32>,
    time: f32,
    rule_id: u32,
}

@group(0) @binding(2)
var<uniform> compute_uniforms: ComputeUniforms;

@compute @workgroup_size(8, 8, 1)
fn cs_pixel_rule(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = compute_uniforms.resolution;

    if global_id.x >= dims.x || global_id.y >= dims.y {
        return;
    }

    let coord = vec2<i32>(global_id.xy);
    let pixel = textureLoad(input_texture, coord, 0);

    var output = pixel;

    // Apply pixel rules
    if compute_uniforms.rule_id == 1u {
        // Conway's Game of Life style rule
        var neighbors = 0u;
        for (var dx = -1; dx <= 1; dx++) {
            for (var dy = -1; dy <= 1; dy++) {
                if dx == 0 && dy == 0 { continue; }
                let nc = coord + vec2<i32>(dx, dy);
                if nc.x >= 0 && nc.x < i32(dims.x) && nc.y >= 0 && nc.y < i32(dims.y) {
                    let n = textureLoad(input_texture, nc, 0);
                    if n.r > 0.5 { neighbors++; }
                }
            }
        }

        let alive = pixel.r > 0.5;
        let new_alive = (alive && (neighbors == 2u || neighbors == 3u)) || (!alive && neighbors == 3u);
        output = vec4<f32>(vec3<f32>(f32(new_alive)), 1.0);

    } else if compute_uniforms.rule_id == 2u {
        // Diffusion rule - spread activation
        var sum = vec4<f32>(0.0);
        var count = 0.0;
        for (var dx = -1; dx <= 1; dx++) {
            for (var dy = -1; dy <= 1; dy++) {
                let nc = coord + vec2<i32>(dx, dy);
                if nc.x >= 0 && nc.x < i32(dims.x) && nc.y >= 0 && nc.y < i32(dims.y) {
                    sum += textureLoad(input_texture, nc, 0);
                    count += 1.0;
                }
            }
        }
        output = sum / count;

    } else if compute_uniforms.rule_id == 3u {
        // Interference pattern enhancement
        let luminance = dot(pixel.rgb, vec3<f32>(0.299, 0.587, 0.114));
        let enhanced = pow(luminance, 0.5) * 1.2;
        output = vec4<f32>(pixel.rgb * enhanced / max(luminance, 0.001), pixel.a);
    }

    textureStore(output_texture, coord, output);
}
"#;

/// Get combined shader module source
pub fn get_render_shader_source() -> String {
    format!("{}\n{}", VERTEX_SHADER, FRAGMENT_SHADER)
}

/// Get point shader source
#[allow(dead_code)]
pub fn get_point_shader_source() -> String {
    format!("{}\n{}", POINT_VERTEX_SHADER, POINT_FRAGMENT_SHADER)
}

// ============================================================
// MARKET LARYNX / CRASH VOID SHADERS (Harmonic Alpha)
// ============================================================

/// Uniforms for market-aware rendering
#[allow(dead_code)]
pub const MARKET_UNIFORMS: &str = r#"
struct MarketUniforms {
    // Base uniforms
    view_proj: mat4x4<f32>,
    time: f32,
    point_size: f32,
    edge_width: f32,
    alpha_blend: f32,
    resolution: vec2<f32>,

    // Market Larynx data
    tension: f32,                // 0.0 = calm, 1.0 = crash
    consonance: f32,             // Musical consonance (inverse of tension)
    gamma_active: f32,           // 1.0 if crash event is active
    crash_probability: f32,      // TDA-derived crash probability

    // Sonification frequencies (for visual resonance)
    freq_fundamental: f32,
    freq_harmonic1: f32,
    freq_harmonic2: f32,

    // Regime (encoded as float for shader)
    // 0=Bull, 1=MildBull, 2=Neutral, 3=MildBear, 4=Bear, 5=CrashRisk, 6=GammaEvent
    regime: f32,

    _padding: vec2<f32>,
}
"#;

/// Market-aware vertex shader with tension-based displacement
#[allow(dead_code)]
pub const MARKET_VERTEX_SHADER: &str = r#"
struct MarketUniforms {
    view_proj: mat4x4<f32>,
    time: f32,
    point_size: f32,
    edge_width: f32,
    alpha_blend: f32,
    resolution: vec2<f32>,
    tension: f32,
    consonance: f32,
    gamma_active: f32,
    crash_probability: f32,
    freq_fundamental: f32,
    freq_harmonic1: f32,
    freq_harmonic2: f32,
    regime: f32,
    _padding: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: MarketUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) w_depth: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) w_depth: f32,
    @location(2) world_pos: vec3<f32>,
    @location(3) tension: f32,
    @location(4) crash_phase: f32,
}

// Noise function for displacement
fn hash(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    return mix(
        mix(mix(hash(i + vec3<f32>(0.0, 0.0, 0.0)), hash(i + vec3<f32>(1.0, 0.0, 0.0)), u.x),
            mix(hash(i + vec3<f32>(0.0, 1.0, 0.0)), hash(i + vec3<f32>(1.0, 1.0, 0.0)), u.x), u.y),
        mix(mix(hash(i + vec3<f32>(0.0, 0.0, 1.0)), hash(i + vec3<f32>(1.0, 0.0, 1.0)), u.x),
            mix(hash(i + vec3<f32>(0.0, 1.0, 1.0)), hash(i + vec3<f32>(1.0, 1.0, 1.0)), u.x), u.y), u.z);
}

@vertex
fn vs_market(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    var pos = in.position;
    let t = uniforms.tension;
    let time = uniforms.time;

    // === TENSION-BASED VERTEX DISPLACEMENT ===

    // Low tension: smooth, harmonious movement
    // High tension: chaotic displacement, structural breakdown

    // Harmonic oscillation based on frequencies from market state
    let freq1 = uniforms.freq_fundamental / 110.0; // Normalize around A2
    let freq2 = uniforms.freq_harmonic1 / 165.0;
    let freq3 = uniforms.freq_harmonic2 / 220.0;

    // Phase offset per vertex for wave-like motion
    let phase_offset = in.w_depth * 3.14159 + length(pos) * 0.5;

    // Consonant motion (low tension): smooth sinusoidal
    let consonant_motion = vec3<f32>(
        sin(time * freq1 + phase_offset) * 0.05,
        cos(time * freq2 + phase_offset * 1.3) * 0.05,
        sin(time * freq3 + phase_offset * 0.7) * 0.05
    ) * uniforms.consonance;

    // Dissonant motion (high tension): noise-based jitter
    let noise_pos = pos * 3.0 + vec3<f32>(time * 2.0);
    let dissonant_motion = vec3<f32>(
        noise(noise_pos) - 0.5,
        noise(noise_pos + vec3<f32>(17.0)) - 0.5,
        noise(noise_pos + vec3<f32>(31.0)) - 0.5
    ) * t * 0.3;

    // Combine motions
    pos = pos + consonant_motion + dissonant_motion;

    // === CRASH VOID (GAMMA EVENT) - DIMENSIONAL COLLAPSE ===
    if uniforms.gamma_active > 0.5 {
        // Pull vertices toward the origin (geometric collapse)
        let collapse_factor = sin(time * 5.0) * 0.3 + 0.7;
        let collapse_strength = uniforms.gamma_active;

        // 4D → 3D collapse: attenuate based on w_depth
        let w_attenuation = 1.0 - abs(in.w_depth) * collapse_strength * 0.5;

        // Spiral inward motion
        let angle = time * 3.0 + atan2(pos.y, pos.x);
        let radius = length(pos.xy) * collapse_factor;
        pos.x = cos(angle) * radius * w_attenuation;
        pos.y = sin(angle) * radius * w_attenuation;
        pos.z = pos.z * collapse_factor * w_attenuation;
    }

    // === CRASH PROBABILITY DISTORTION ===
    // As crash probability increases, geometry becomes increasingly unstable
    let prob = uniforms.crash_probability;
    if prob > 0.3 {
        let instability = (prob - 0.3) / 0.7; // 0 to 1 range
        let shatter_offset = vec3<f32>(
            noise(pos * 10.0 + time) - 0.5,
            noise(pos * 10.0 + time + vec3<f32>(100.0)) - 0.5,
            noise(pos * 10.0 + time + vec3<f32>(200.0)) - 0.5
        ) * instability * 0.2;
        pos = pos + shatter_offset;
    }

    // Transform to clip space
    out.clip_position = uniforms.view_proj * vec4<f32>(pos, 1.0);

    // Pass data to fragment shader
    out.w_depth = in.w_depth;
    out.world_pos = pos;
    out.tension = t;
    out.crash_phase = uniforms.gamma_active;

    // === COLOR MODULATION BASED ON MARKET REGIME ===
    var color = in.color;

    // Regime-based color tinting
    let regime = uniforms.regime;
    if regime < 1.0 {
        // Bull: Green tint
        color = vec4<f32>(color.r * 0.7, color.g * 1.3, color.b * 0.7, color.a);
    } else if regime < 3.0 {
        // Neutral/MildBull: Yellow tint
        color = vec4<f32>(color.r * 1.1, color.g * 1.1, color.b * 0.7, color.a);
    } else if regime < 5.0 {
        // Bear/MildBear: Orange/Red tint
        let bear_factor = (regime - 3.0) / 2.0;
        color = vec4<f32>(color.r * (1.0 + bear_factor * 0.5), color.g * (1.0 - bear_factor * 0.3), color.b * 0.6, color.a);
    } else {
        // CrashRisk/GammaEvent: Intense red with pulsing
        let pulse = sin(time * 10.0) * 0.3 + 0.7;
        color = vec4<f32>(pulse, color.g * 0.2, color.b * 0.2, color.a);
    }

    // Gamma event: dramatic color shift
    if uniforms.gamma_active > 0.5 {
        let flash = sin(time * 15.0) * 0.5 + 0.5;
        color = mix(color, vec4<f32>(1.0, 0.0, 1.0, 1.0), flash * 0.5);
    }

    // Apply base alpha modulation
    let depth_alpha = 1.0 - abs(in.w_depth) * 0.2;
    out.color = vec4<f32>(color.rgb, color.a * depth_alpha * uniforms.alpha_blend);

    return out;
}
"#;

/// Market-aware fragment shader with crash void effects
#[allow(dead_code)]
pub const MARKET_FRAGMENT_SHADER: &str = r#"
struct MarketUniforms {
    view_proj: mat4x4<f32>,
    time: f32,
    point_size: f32,
    edge_width: f32,
    alpha_blend: f32,
    resolution: vec2<f32>,
    tension: f32,
    consonance: f32,
    gamma_active: f32,
    crash_probability: f32,
    freq_fundamental: f32,
    freq_harmonic1: f32,
    freq_harmonic2: f32,
    regime: f32,
    _padding: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: MarketUniforms;

struct FragmentInput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) w_depth: f32,
    @location(2) world_pos: vec3<f32>,
    @location(3) tension: f32,
    @location(4) crash_phase: f32,
}

// Fractal noise for void rendering
fn fbm(p: vec3<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    var pp = p;

    for (var i = 0; i < 4; i++) {
        value += amplitude * (sin(dot(pp, vec3<f32>(1.0, 1.7, 2.3))) * 0.5 + 0.5);
        pp = pp * 2.0;
        amplitude *= 0.5;
    }
    return value;
}

@fragment
fn fs_market(in: FragmentInput) -> @location(0) vec4<f32> {
    var color = in.color;
    let t = uniforms.tension;
    let time = uniforms.time;

    // === GLOW EFFECT (stronger with tension) ===
    let base_glow = exp(-abs(in.w_depth) * 2.0) * 0.3;
    let tension_glow = t * 0.4;
    let glow = base_glow + tension_glow;

    color = vec4<f32>(color.rgb + vec3<f32>(glow), color.a);

    // === CRASH VOID VISUAL EFFECT ===
    if uniforms.gamma_active > 0.5 {
        // Create "void" effect - darker center, bright edges
        let dist_from_center = length(in.world_pos);
        let void_edge = smoothstep(0.5, 1.5, dist_from_center);

        // Void core: dark with occasional flashes
        let flash = max(0.0, sin(time * 20.0 + dist_from_center * 5.0) - 0.9) * 10.0;
        let void_color = vec3<f32>(0.1 + flash * 0.3, 0.0, 0.2 + flash * 0.5);

        // Event horizon: bright ring
        let horizon_width = 0.3;
        let horizon_intensity = smoothstep(1.0 - horizon_width, 1.0, dist_from_center) *
                                smoothstep(1.0 + horizon_width, 1.0, dist_from_center);
        let horizon_color = vec3<f32>(1.0, 0.3, 0.8) * horizon_intensity * 2.0;

        // Mix void effect with base color
        color = vec4<f32>(
            mix(void_color, color.rgb, void_edge) + horizon_color,
            color.a
        );

        // Add dimensional fracture lines
        let fracture = fbm(in.world_pos * 5.0 + vec3<f32>(time * 0.5));
        if fracture > 0.7 {
            color = vec4<f32>(color.rgb + vec3<f32>(0.5, 0.2, 0.8), color.a);
        }
    }

    // === TENSION-BASED INTERFERENCE PATTERNS ===
    if t > 0.5 {
        let interference = sin(length(in.world_pos) * 20.0 - time * 5.0) *
                          sin(dot(in.world_pos, vec3<f32>(1.0, 1.0, 1.0)) * 15.0 - time * 3.0);
        let pattern_intensity = (t - 0.5) * 2.0 * 0.2;
        color = vec4<f32>(color.rgb + vec3<f32>(interference * pattern_intensity), color.a);
    }

    // === CRASH PROBABILITY VISUAL WARNING ===
    let prob = uniforms.crash_probability;
    if prob > 0.5 {
        // Red warning overlay that increases with probability
        let warning_intensity = (prob - 0.5) * 2.0;
        let warning_pulse = sin(time * 3.0) * 0.3 + 0.7;
        let warning_color = vec3<f32>(1.0, 0.0, 0.0) * warning_intensity * warning_pulse * 0.3;
        color = vec4<f32>(color.rgb + warning_color, color.a);
    }

    // === GAMMA CORRECTION ===
    let gamma = 2.2;
    color = vec4<f32>(pow(color.rgb, vec3<f32>(1.0 / gamma)), color.a);

    // Clamp output
    color = clamp(color, vec4<f32>(0.0), vec4<f32>(1.0));

    return color;
}
"#;

/// Post-processing shader for market visualization overlay
#[allow(dead_code)]
pub const MARKET_POSTPROCESS_SHADER: &str = r#"
@group(0) @binding(0)
var input_texture: texture_2d<f32>;
@group(0) @binding(1)
var texture_sampler: sampler;

struct MarketPostUniforms {
    resolution: vec2<f32>,
    time: f32,
    tension: f32,
    gamma_active: f32,
    crash_probability: f32,
    _padding: vec2<f32>,
}

@group(0) @binding(2)
var<uniform> post_uniforms: MarketPostUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_market_post(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32((vertex_index & 1u) << 2u) - 1.0;
    let y = f32((vertex_index & 2u) << 1u) - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_market_post(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = textureSample(input_texture, texture_sampler, in.uv);
    let time = post_uniforms.time;
    let tension = post_uniforms.tension;

    // === VIGNETTE (stronger with tension) ===
    let center = vec2<f32>(0.5, 0.5);
    let dist = length(in.uv - center);
    let vignette_base = 0.3;
    let vignette_tension = tension * 0.4;
    let vignette = smoothstep(0.8, 0.4, dist * (1.0 + vignette_base + vignette_tension));
    color = vec4<f32>(color.rgb * vignette, color.a);

    // === CHROMATIC ABERRATION (increases with tension) ===
    if tension > 0.3 {
        let aberration_amount = (tension - 0.3) * 0.01;
        let r = textureSample(input_texture, texture_sampler, in.uv + vec2<f32>(aberration_amount, 0.0)).r;
        let b = textureSample(input_texture, texture_sampler, in.uv - vec2<f32>(aberration_amount, 0.0)).b;
        color = vec4<f32>(r, color.g, b, color.a);
    }

    // === GAMMA EVENT: SCREEN SHAKE EFFECT ===
    if post_uniforms.gamma_active > 0.5 {
        let shake_amount = 0.01;
        let shake_offset = vec2<f32>(
            sin(time * 50.0) * shake_amount,
            cos(time * 47.0) * shake_amount
        );
        color = textureSample(input_texture, texture_sampler, in.uv + shake_offset);

        // Flash effect
        let flash = max(0.0, sin(time * 10.0) - 0.8) * 5.0;
        color = vec4<f32>(color.rgb + vec3<f32>(flash * 0.3), color.a);
    }

    // === CRASH WARNING BORDER ===
    let prob = post_uniforms.crash_probability;
    if prob > 0.6 {
        let border_width = 0.02 + prob * 0.03;
        let border_dist = min(min(in.uv.x, 1.0 - in.uv.x), min(in.uv.y, 1.0 - in.uv.y));
        if border_dist < border_width {
            let pulse = sin(time * 5.0) * 0.5 + 0.5;
            let border_alpha = (1.0 - border_dist / border_width) * pulse * (prob - 0.6) / 0.4;
            color = vec4<f32>(mix(color.rgb, vec3<f32>(1.0, 0.0, 0.0), border_alpha), color.a);
        }
    }

    // === SCANLINES (subtle market data aesthetic) ===
    let scanline = sin(in.uv.y * post_uniforms.resolution.y * 1.0) * 0.02;
    color = vec4<f32>(color.rgb - vec3<f32>(scanline), color.a);

    return color;
}
"#;

/// Get market-aware shader module source
#[allow(dead_code)]
pub fn get_market_shader_source() -> String {
    format!("{}\n{}", MARKET_VERTEX_SHADER, MARKET_FRAGMENT_SHADER)
}

/// Get market post-processing shader source
#[allow(dead_code)]
pub fn get_market_postprocess_source() -> String {
    MARKET_POSTPROCESS_SHADER.to_string()
}
