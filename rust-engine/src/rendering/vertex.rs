//! Vertex structures for GPU rendering

use bytemuck::{Pod, Zeroable};

/// Vertex data for rendering points and edges
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    /// Position in 3D (after 4D→3D projection)
    pub position: [f32; 3],
    /// RGBA color
    pub color: [f32; 4],
    /// Original 4D w-coordinate (for depth/effects)
    pub w_depth: f32,
}

impl Vertex {
    pub fn new(position: [f32; 3], color: [f32; 4]) -> Self {
        Self {
            position,
            color,
            w_depth: 0.0,
        }
    }

    pub fn with_depth(mut self, w: f32) -> Self {
        self.w_depth = w;
        self
    }

    /// Get the vertex buffer layout for WebGPU
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // Position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Color
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // W depth
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 7]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

/// Uniform data passed to shaders
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Uniforms {
    /// View-projection matrix
    pub view_proj: [[f32; 4]; 4],
    /// Time (for animations)
    pub time: f32,
    /// Point size
    pub point_size: f32,
    /// Edge width
    pub edge_width: f32,
    /// Alpha blend factor
    pub alpha_blend: f32,
    /// Screen resolution
    pub resolution: [f32; 2],
    /// Padding for alignment
    pub _padding: [f32; 2],
}

impl Default for Uniforms {
    fn default() -> Self {
        Self {
            view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            time: 0.0,
            point_size: 8.0,
            edge_width: 2.0,
            alpha_blend: 0.8,
            resolution: [1280.0, 720.0],
            _padding: [0.0, 0.0],
        }
    }
}

/// Instance data for instanced rendering
#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Instance {
    /// Model matrix (for per-instance transforms)
    pub model: [[f32; 4]; 4],
    /// Instance color tint
    pub color_tint: [f32; 4],
}

#[allow(dead_code)]
impl Instance {
    pub fn identity() -> Self {
        Self {
            model: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            color_tint: [1.0, 1.0, 1.0, 1.0],
        }
    }

    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Instance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // Model matrix (4 vec4s)
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Color tint
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}
