//! # Geometric Cognition Engine
//!
//! A GPU-accelerated engine for analog cognitive computation using high-dimensional
//! geometry and visual processing. This engine encodes data as configurations of
//! 4D polytopes (24-cell, 600-cell, 120-cell) and projects them to 2D images,
//! effectively using rendering as computation.
//!
//! ## Architecture
//!
//! The engine is organized into distinct modules forming a processing pipeline:
//! ```text
//! Input Data → 4D Geometric Encoding → 3D Projection → 2D Rendering → Output Analysis
//! ```
//!
//! ## Key Features
//!
//! - **Mathematically-Rigorous Geometry**: Accurate models of 4D polytopes with
//!   quaternion-based rotations and E₈ lattice projections
//! - **Triadic Dialectic Reasoning**: 24-cell Trinity decomposition (Alpha/Beta/Gamma)
//!   for thesis-antithesis-synthesis logic
//! - **Pixel-Rule Analog Processing**: GPU blending as implicit computation
//! - **Homological Signal Extraction**: Topological feature detection via Betti numbers

pub mod geometry;
pub mod cognitive;
pub mod rendering;
pub mod pipeline;
pub mod analysis;

#[cfg(feature = "web")]
pub mod web;

// Re-export commonly used types
pub use geometry::{
    Vec4, Quaternion, Polytope4D,
    Cell24, Cell600, Cell120,
    ProjectionMode, GeometryMode,
    TrinityComponent,
};
pub use cognitive::{
    TrinityState, DialecticEngine, CognitiveRule,
};
pub use rendering::{
    RenderEngine, RenderConfig, PixelRule,
};
pub use pipeline::{
    DataIngestion, DataMapper, OutputChannel,
};
pub use analysis::{
    HomologyAnalyzer, BettiNumbers, TopologicalSignal,
};

/// Golden ratio φ = (1 + √5) / 2
pub const PHI: f64 = 1.618033988749895;

/// Inverse golden ratio φ⁻¹ = φ - 1
pub const PHI_INV: f64 = 0.6180339887498949;

/// Engine configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EngineConfig {
    /// Target frames per second
    pub target_fps: u32,
    /// Number of data input channels
    pub data_channels: usize,
    /// Enable 600-cell expanded mode
    pub expanded_mode: bool,
    /// Enable E₈ dual-layer scaling
    pub e8_layer_enabled: bool,
    /// Pixel processing rules to apply
    pub pixel_rules: Vec<String>,
    /// Window dimensions
    pub width: u32,
    pub height: u32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            target_fps: 60,
            data_channels: 64,
            expanded_mode: false,
            e8_layer_enabled: true,
            pixel_rules: vec!["alpha_blend".to_string()],
            width: 1280,
            height: 720,
        }
    }
}

/// Main engine state container
pub struct GeometricCognitionEngine {
    config: EngineConfig,
    geometry_core: geometry::GeometryCore,
    cognitive_layer: cognitive::CognitiveLayer,
    render_engine: Option<rendering::RenderEngine>,
    data_pipeline: pipeline::DataPipeline,
    analyzer: analysis::Analyzer,
    frame_count: u64,
}

impl GeometricCognitionEngine {
    /// Create a new engine instance with the given configuration
    pub fn new(config: EngineConfig) -> Self {
        log::info!("Initializing Geometric Cognition Engine");

        Self {
            geometry_core: geometry::GeometryCore::new(&config),
            cognitive_layer: cognitive::CognitiveLayer::new(),
            render_engine: None,
            data_pipeline: pipeline::DataPipeline::new(config.data_channels),
            analyzer: analysis::Analyzer::new(),
            frame_count: 0,
            config,
        }
    }

    /// Initialize the rendering subsystem (requires GPU)
    pub async fn init_rendering(&mut self) -> anyhow::Result<()> {
        let render_engine = rendering::RenderEngine::new(&self.config).await?;
        self.render_engine = Some(render_engine);
        Ok(())
    }

    /// Process one simulation frame
    pub fn update(&mut self, delta_time: f64) {
        // 1. Ingest external data and map to geometric parameters
        let data_state = self.data_pipeline.process();

        // 2. Update geometry based on data and cognitive rules
        self.geometry_core.apply_data_mapping(&data_state);
        self.cognitive_layer.process(&mut self.geometry_core, delta_time);

        // 3. Project 4D → 3D
        self.geometry_core.project_all();

        self.frame_count += 1;
    }

    /// Render current state to the screen
    pub fn render(&mut self) -> anyhow::Result<()> {
        if let Some(ref mut renderer) = self.render_engine {
            let geometry_data = self.geometry_core.get_render_data();
            renderer.render(&geometry_data)?;
        }
        Ok(())
    }

    /// Analyze current state and extract signals
    pub fn analyze(&mut self) -> analysis::AnalysisResult {
        self.analyzer.analyze(&self.geometry_core)
    }

    /// Get current frame count
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get mutable reference to data pipeline for external data injection
    pub fn data_pipeline_mut(&mut self) -> &mut pipeline::DataPipeline {
        &mut self.data_pipeline
    }

    /// Get reference to geometry core for inspection
    pub fn geometry(&self) -> &geometry::GeometryCore {
        &self.geometry_core
    }

    /// Get mutable reference to cognitive layer for rule modification
    pub fn cognitive_layer_mut(&mut self) -> &mut cognitive::CognitiveLayer {
        &mut self.cognitive_layer
    }

    /// Get reference to cognitive layer
    pub fn cognitive_layer(&self) -> &cognitive::CognitiveLayer {
        &self.cognitive_layer
    }

    /// Get mutable reference to geometry core
    pub fn geometry_mut(&mut self) -> &mut geometry::GeometryCore {
        &mut self.geometry_core
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let config = EngineConfig::default();
        let engine = GeometricCognitionEngine::new(config);
        assert_eq!(engine.frame_count(), 0);
    }

    #[test]
    fn test_golden_ratio() {
        assert!((PHI * PHI_INV - 1.0).abs() < 1e-10);
        assert!((PHI - PHI_INV - 1.0).abs() < 1e-10);
    }
}

#[cfg(test)]
mod render_test {
    use crate::*;
    use crate::geometry::GeometryCore;
    
    #[test]
    fn test_actual_vertex_output() {
        let config = EngineConfig::default();
        let core = GeometryCore::new(&config);
        
        let render_data = core.get_render_data();
        
        println!("Number of layers: {}", render_data.layers.len());
        for (i, layer) in render_data.layers.iter().enumerate() {
            println!("Layer {}: {} vertices, {} edges", i, layer.vertices_3d.len(), layer.edges.len());
            if !layer.vertices_3d.is_empty() {
                println!("  First vertex: {:?}", layer.vertices_3d[0]);
                println!("  Last vertex: {:?}", layer.vertices_3d[layer.vertices_3d.len()-1]);
                // Check for NaN or Inf
                for (j, v) in layer.vertices_3d.iter().enumerate() {
                    if !v[0].is_finite() || !v[1].is_finite() || !v[2].is_finite() {
                        println!("  WARNING: Non-finite vertex at {}: {:?}", j, v);
                    }
                }
            }
        }
    }
}
