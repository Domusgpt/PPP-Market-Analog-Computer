//! WebGPU Render Engine
//!
//! The main rendering engine that handles GPU resource management,
//! pipeline creation, and frame rendering.

use super::{Vertex, PixelRule, shaders};
use super::vertex::Uniforms;
use crate::geometry::GeometryRenderData;
use crate::EngineConfig;
use wgpu::util::DeviceExt;
use anyhow::Result;
use bytemuck;

/// Render configuration options
#[derive(Debug, Clone)]
pub struct RenderConfig {
    pub width: u32,
    pub height: u32,
    pub clear_color: wgpu::Color,
    pub point_size: f32,
    pub edge_width: f32,
    pub enable_antialiasing: bool,
    pub enable_depth_test: bool,
    pub pixel_rules: Vec<PixelRule>,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            clear_color: wgpu::Color {
                r: 0.02,
                g: 0.02,
                b: 0.05,
                a: 1.0,
            },
            point_size: 8.0,
            edge_width: 2.0,
            enable_antialiasing: true,
            enable_depth_test: true,
            pixel_rules: vec![PixelRule::AlphaBlend],
        }
    }
}

impl From<&EngineConfig> for RenderConfig {
    fn from(config: &EngineConfig) -> Self {
        Self {
            width: config.width,
            height: config.height,
            ..Default::default()
        }
    }
}

/// The main WebGPU render engine
pub struct RenderEngine {
    #[allow(dead_code)]
    instance: wgpu::Instance,
    #[allow(dead_code)]
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: RenderConfig,

    // Pipelines
    line_pipeline: wgpu::RenderPipeline,
    #[allow(dead_code)]
    point_pipeline: Option<wgpu::RenderPipeline>,

    // Buffers
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,

    // Current state
    uniforms: Uniforms,
    vertex_count: usize,
    index_count: usize,
    time: f32,

    // Offscreen render target for pixel processing
    #[allow(dead_code)]
    render_texture: Option<wgpu::Texture>,
    #[allow(dead_code)]
    render_texture_view: Option<wgpu::TextureView>,
}

impl RenderEngine {
    /// Create a new render engine
    pub async fn new(engine_config: &EngineConfig) -> Result<Self> {
        let config = RenderConfig::from(engine_config);

        // Create instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("Failed to find suitable GPU adapter"))?;

        log::info!("Using GPU adapter: {:?}", adapter.get_info().name);

        // Request device
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Geometric Cognition Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| anyhow::anyhow!("Failed to request device: {:?}", e))?;

        // Create uniform buffer
        let uniforms = Uniforms {
            resolution: [config.width as f32, config.height as f32],
            point_size: config.point_size,
            edge_width: config.edge_width,
            ..Default::default()
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Uniform Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Create shader module
        let shader_source = shaders::get_render_shader_source();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Main Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create line render pipeline
        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Line Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: if config.enable_depth_test {
                Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                })
            } else {
                None
            },
            multisample: wgpu::MultisampleState {
                count: if config.enable_antialiasing { 4 } else { 1 },
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        // Create vertex and index buffers (initial allocation)
        let max_vertices = 10000;
        let max_indices = 50000;

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: (max_vertices * std::mem::size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Index Buffer"),
            size: (max_indices * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            config,
            line_pipeline,
            point_pipeline: None,
            vertex_buffer,
            index_buffer,
            uniform_buffer,
            uniform_bind_group,
            uniforms,
            vertex_count: 0,
            index_count: 0,
            time: 0.0,
            render_texture: None,
            render_texture_view: None,
        })
    }

    /// Update geometry data on the GPU
    pub fn update_geometry(&mut self, render_data: &GeometryRenderData) {
        // Build vertices and indices from all layers
        let mut vertices = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        for layer in &render_data.layers {
            let base_index = vertices.len() as u32;

            // Add vertices
            for (i, pos) in layer.vertices_3d.iter().enumerate() {
                let color = layer.colors.get(i).copied().unwrap_or([1.0, 1.0, 1.0, layer.opacity]);
                vertices.push(Vertex::new(*pos, color));
            }

            // Add edge indices
            for (a, b) in &layer.edges {
                indices.push(base_index + *a as u32);
                indices.push(base_index + *b as u32);
            }
        }

        // Update buffers
        if !vertices.is_empty() {
            self.queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
            self.vertex_count = vertices.len();
        }

        if !indices.is_empty() {
            self.queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&indices));
            self.index_count = indices.len();
        }

        // Update uniforms with view-projection matrix
        self.uniforms.view_proj = mat4_multiply(&render_data.view_matrix, &render_data.projection_matrix);
    }

    /// Render to texture (for headless operation)
    pub fn render(&mut self, render_data: &GeometryRenderData) -> Result<()> {
        // Update geometry
        self.update_geometry(render_data);

        // Update time
        self.time += 1.0 / 60.0;
        self.uniforms.time = self.time;

        // Update uniform buffer
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.uniforms]));

        // Create output texture for offscreen rendering
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Render Target"),
            size: wgpu::Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create depth texture
        let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.config.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: if self.config.enable_depth_test {
                    Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    })
                } else {
                    None
                },
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.line_pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

            if self.index_count > 0 {
                render_pass.draw_indexed(0..self.index_count as u32, 0, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        Ok(())
    }

    /// Get current frame time
    pub fn time(&self) -> f32 {
        self.time
    }

    /// Get device reference
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get queue reference
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Resize the render target
    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width;
        self.config.height = height;
        self.uniforms.resolution = [width as f32, height as f32];
    }
}

/// Matrix multiplication helper
fn mat4_multiply(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_config_default() {
        let config = RenderConfig::default();
        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
    }

    #[test]
    fn test_mat4_multiply_identity() {
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let result = mat4_multiply(&identity, &identity);
        assert_eq!(result, identity);
    }
}
