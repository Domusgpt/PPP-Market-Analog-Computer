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
