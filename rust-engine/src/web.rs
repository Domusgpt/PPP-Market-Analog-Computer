//! WebAssembly bindings for the Geometric Cognition Engine
//!
//! This module exposes the engine to JavaScript via wasm-bindgen.

#![cfg(feature = "web")]

use wasm_bindgen::prelude::*;

use crate::{EngineConfig, GeometricCognitionEngine};
use crate::geometry::{GeometryMode, TrinityComponent};

/// Initialize panic hook for better error messages in browser console
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    console_log("Geometric Cognition Engine WASM initialized");
}

/// Log to browser console
#[wasm_bindgen]
pub fn console_log(s: &str) {
    web_sys::console::log_1(&s.into());
}

/// The web-facing engine wrapper
#[wasm_bindgen]
pub struct WebEngine {
    engine: GeometricCognitionEngine,
    canvas_id: String,
    running: bool,
    frame_count: u64,
}

#[wasm_bindgen]
impl WebEngine {
    /// Create a new WebEngine instance
    #[wasm_bindgen(constructor)]
    pub fn new(canvas_id: &str) -> Result<WebEngine, JsValue> {
        let config = EngineConfig {
            width: 800,
            height: 600,
            target_fps: 60,
            data_channels: 64,
            expanded_mode: false,
            e8_layer_enabled: true,
            pixel_rules: vec!["alpha_blend".to_string()],
        };

        let engine = GeometricCognitionEngine::new(config);

        Ok(WebEngine {
            engine,
            canvas_id: canvas_id.to_string(),
            running: false,
            frame_count: 0,
        })
    }

    /// Create with expanded 600-cell mode
    #[wasm_bindgen]
    pub fn new_expanded(canvas_id: &str) -> Result<WebEngine, JsValue> {
        let config = EngineConfig {
            width: 800,
            height: 600,
            target_fps: 60,
            data_channels: 64,
            expanded_mode: true,
            e8_layer_enabled: true,
            pixel_rules: vec!["alpha_blend".to_string()],
        };

        let engine = GeometricCognitionEngine::new(config);

        Ok(WebEngine {
            engine,
            canvas_id: canvas_id.to_string(),
            running: false,
            frame_count: 0,
        })
    }

    /// Update the simulation by one frame
    #[wasm_bindgen]
    pub fn update(&mut self, delta_time: f64) {
        self.engine.update(delta_time);
        self.frame_count += 1;
    }

    /// Inject data into a specific channel (0-63)
    #[wasm_bindgen]
    pub fn inject_channel(&mut self, channel: usize, value: f64) {
        self.engine.data_pipeline_mut().inject(channel, value);
    }

    /// Inject data into multiple channels from an array
    #[wasm_bindgen]
    pub fn inject_channels(&mut self, values: &[f64]) {
        self.engine.data_pipeline_mut().inject_all(values);
    }

    /// Set rotation speeds for all 6 planes (XY, XZ, XW, YZ, YW, ZW)
    #[wasm_bindgen]
    pub fn set_rotation_speeds(&mut self, xy: f64, xz: f64, xw: f64, yz: f64, yw: f64, zw: f64) {
        self.engine.geometry_mut().set_rotation_speed([xy, xz, xw, yz, yw, zw]);
    }

    /// Get current frame count
    #[wasm_bindgen]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get Betti numbers as JSON string
    #[wasm_bindgen]
    pub fn get_betti_numbers(&mut self) -> String {
        let analysis = self.engine.analyze();
        serde_json::json!({
            "b0": analysis.betti.b0,
            "b1": analysis.betti.b1,
            "b2": analysis.betti.b2,
            "b3": analysis.betti.b3,
            "euler": analysis.betti.euler_characteristic()
        }).to_string()
    }

    /// Get detected patterns as JSON array
    #[wasm_bindgen]
    pub fn get_patterns(&mut self) -> String {
        let analysis = self.engine.analyze();
        serde_json::to_string(&analysis.patterns).unwrap_or_else(|_| "[]".to_string())
    }

    /// Get full analysis as JSON
    #[wasm_bindgen]
    pub fn get_analysis(&mut self) -> String {
        let analysis = self.engine.analyze();
        serde_json::json!({
            "betti": {
                "b0": analysis.betti.b0,
                "b1": analysis.betti.b1,
                "b2": analysis.betti.b2,
            },
            "patterns": analysis.patterns,
            "metrics": {
                "vertex_count": analysis.metrics.vertex_count,
                "active_vertices": analysis.metrics.active_vertices,
                "cluster_count": analysis.metrics.cluster_count,
                "symmetry_score": analysis.metrics.symmetry_score,
                "coherence_score": analysis.metrics.coherence_score,
                "complexity": analysis.metrics.complexity,
            }
        }).to_string()
    }

    /// Get Trinity state as JSON
    #[wasm_bindgen]
    pub fn get_trinity_state(&self) -> String {
        let state = self.engine.cognitive_layer().trinity_state();
        serde_json::json!({
            "alpha": {
                "level": state.alpha.level,
                "dominant": state.alpha.dominant
            },
            "beta": {
                "level": state.beta.level,
                "dominant": state.beta.dominant
            },
            "gamma": {
                "level": state.gamma.level,
                "dominant": state.gamma.dominant
            },
            "tension": state.dialectic_tension(),
            "coherence": state.synthesis_coherence(),
            "resonant": state.is_resonant()
        }).to_string()
    }

    /// Get projected vertices for rendering (as flat f32 array: x,y,z,r,g,b,a,...)
    #[wasm_bindgen]
    pub fn get_vertices(&self) -> Vec<f32> {
        let render_data = self.engine.geometry().get_render_data();
        let mut result = Vec::new();

        for layer in &render_data.layers {
            for (i, pos) in layer.vertices_3d.iter().enumerate() {
                let color = layer.colors.get(i).copied().unwrap_or([1.0, 1.0, 1.0, 1.0]);
                result.push(pos[0]);
                result.push(pos[1]);
                result.push(pos[2]);
                result.push(color[0]);
                result.push(color[1]);
                result.push(color[2]);
                result.push(color[3]);
            }
        }

        result
    }

    /// Get edges as flat u32 array of index pairs
    #[wasm_bindgen]
    pub fn get_edges(&self) -> Vec<u32> {
        let render_data = self.engine.geometry().get_render_data();
        let mut result = Vec::new();
        let mut offset = 0u32;

        for layer in &render_data.layers {
            for (a, b) in &layer.edges {
                result.push(offset + *a as u32);
                result.push(offset + *b as u32);
            }
            offset += layer.vertices_3d.len() as u32;
        }

        result
    }

    /// Get view matrix as flat f32 array (4x4 = 16 elements)
    #[wasm_bindgen]
    pub fn get_view_matrix(&self) -> Vec<f32> {
        let render_data = self.engine.geometry().get_render_data();
        render_data.view_matrix.iter().flatten().copied().collect()
    }

    /// Get projection matrix as flat f32 array
    #[wasm_bindgen]
    pub fn get_projection_matrix(&self) -> Vec<f32> {
        let render_data = self.engine.geometry().get_render_data();
        render_data.projection_matrix.iter().flatten().copied().collect()
    }

    /// Set geometry mode: "core24", "trinity", "expanded600", "full120", "e8dual"
    #[wasm_bindgen]
    pub fn set_mode(&mut self, mode: &str) {
        let mode = match mode {
            "core24" => GeometryMode::Core24Cell,
            "trinity" => GeometryMode::Trinity,
            "expanded600" => GeometryMode::Expanded600,
            "full120" => GeometryMode::Full120,
            "e8dual" => GeometryMode::E8DualLayer,
            _ => GeometryMode::Trinity,
        };
        self.engine.geometry_mut().set_mode(mode);
    }

    /// Get current mode as string
    #[wasm_bindgen]
    pub fn get_mode(&self) -> String {
        match self.engine.geometry().mode() {
            GeometryMode::Core24Cell => "core24".to_string(),
            GeometryMode::Trinity => "trinity".to_string(),
            GeometryMode::Expanded600 => "expanded600".to_string(),
            GeometryMode::Full120 => "full120".to_string(),
            GeometryMode::E8DualLayer => "e8dual".to_string(),
        }
    }

    /// Check if synthesis is currently detected
    #[wasm_bindgen]
    pub fn is_synthesis_detected(&self) -> bool {
        self.engine.geometry().check_synthesis().is_some()
    }

    /// Get dialectic distance between Alpha and Beta
    #[wasm_bindgen]
    pub fn get_dialectic_distance(&self) -> f64 {
        self.engine.geometry().dialectic_distance(
            TrinityComponent::Alpha,
            TrinityComponent::Beta,
        )
    }
}

/// Get engine version
#[wasm_bindgen]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Get engine description
#[wasm_bindgen]
pub fn get_description() -> String {
    "Geometric Cognition Engine: GPU-accelerated analog computation using 4D polytopes".to_string()
}
