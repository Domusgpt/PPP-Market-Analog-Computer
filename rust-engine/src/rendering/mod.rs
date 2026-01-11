//! Rendering Module - WebGPU-based Visualization Pipeline
//!
//! This module handles GPU-accelerated rendering of 4D polytope projections,
//! including the pixel-level analog computation through shader-based blending.

mod engine;
mod shaders;
mod pixel_rules;
mod vertex;

pub use engine::{RenderEngine, RenderConfig};
pub use pixel_rules::PixelRule;
pub use vertex::Vertex;

/// Render statistics
#[derive(Debug, Clone, Default)]
pub struct RenderStats {
    pub frame_time_ms: f64,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub draw_calls: usize,
}
