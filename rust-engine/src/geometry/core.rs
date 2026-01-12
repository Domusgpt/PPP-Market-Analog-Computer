//! Geometry Core - Central management of all geometric structures
//!
//! The GeometryCore handles:
//! - Multiple polytope instances (24-cell, 600-cell, 120-cell)
//! - E₈ dual-layer scaling
//! - Data mapping to geometric parameters
//! - Projection pipeline coordination

use super::{
    Rotation4D,
    Cell24, Cell600, Cell120, TrinityComponent,
    Projector, Projected3D,
    Polytope4D,
};
use crate::{EngineConfig, PHI};
use serde::{Serialize, Deserialize};

/// Operating mode for the geometry core
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GeometryMode {
    /// Core 24-cell only
    Core24Cell,
    /// 24-cell with Trinity decomposition visible
    Trinity,
    /// Expanded 600-cell mode
    Expanded600,
    /// Full 120-cell (highest detail)
    Full120,
    /// E₈ dual-layer (two φ-scaled 600-cells)
    E8DualLayer,
}

impl Default for GeometryMode {
    fn default() -> Self {
        Self::Trinity
    }
}

/// Render data for a single polytope layer
#[derive(Debug, Clone)]
pub struct PolytopeRenderData {
    pub vertices_3d: Vec<[f32; 3]>,
    pub colors: Vec<[f32; 4]>,
    pub edges: Vec<(usize, usize)>,
    pub layer_id: usize,
    pub opacity: f32,
}

/// Complete render data from geometry core
#[derive(Debug, Clone)]
pub struct GeometryRenderData {
    pub layers: Vec<PolytopeRenderData>,
    pub view_matrix: [[f32; 4]; 4],
    pub projection_matrix: [[f32; 4]; 4],
}

/// State snapshot for external analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometryState {
    pub mode: GeometryMode,
    pub cell24_rotation: [f64; 4], // Quaternion components
    pub cell600_rotation: [f64; 4],
    pub trinity_activations: [f64; 3], // Alpha, Beta, Gamma
    pub synthesis_detected: bool,
    pub vertex_count: usize,
}

/// The central geometry management structure
pub struct GeometryCore {
    /// Primary 24-cell
    cell24: Cell24,
    /// Expanded 600-cell (lazy-initialized)
    cell600: Option<Cell600>,
    /// Full 120-cell (lazy-initialized)
    cell120: Option<Cell120>,
    /// E₈ second layer (φ-scaled 600-cell)
    e8_layer: Option<Cell600>,
    /// Current operating mode
    mode: GeometryMode,
    /// Projector for 4D → 3D → 2D
    projector: Projector,
    /// Cached 3D projections
    projected_3d: Vec<Projected3D>,
    /// Global rotation animation speed (radians per second)
    rotation_speed: [f64; 6], // XY, XZ, XW, YZ, YW, ZW planes
    /// Enable E₈ dual-layer
    e8_enabled: bool,
}

impl GeometryCore {
    /// Create a new geometry core with given configuration
    pub fn new(config: &EngineConfig) -> Self {
        let mode = if config.expanded_mode {
            GeometryMode::Expanded600
        } else {
            GeometryMode::Trinity
        };

        let mut core = Self {
            cell24: Cell24::new(),
            cell600: None,
            cell120: None,
            e8_layer: None,
            mode,
            projector: Projector::default(),
            projected_3d: Vec::new(),
            rotation_speed: [0.0; 6],
            e8_enabled: config.e8_layer_enabled,
        };

        // Initialize expanded structures if needed
        if config.expanded_mode {
            core.ensure_600cell();
        }

        if config.e8_layer_enabled {
            core.ensure_e8_layer();
        }

        core
    }

    /// Get current geometry mode
    pub fn mode(&self) -> GeometryMode {
        self.mode
    }

    /// Set geometry mode
    pub fn set_mode(&mut self, mode: GeometryMode) {
        self.mode = mode;

        // Initialize required structures
        match mode {
            GeometryMode::Expanded600 | GeometryMode::E8DualLayer => {
                self.ensure_600cell();
            }
            GeometryMode::Full120 => {
                self.ensure_120cell();
            }
            _ => {}
        }

        if mode == GeometryMode::E8DualLayer {
            self.ensure_e8_layer();
        }
    }

    /// Ensure 600-cell is initialized
    fn ensure_600cell(&mut self) {
        if self.cell600.is_none() {
            self.cell600 = Some(Cell600::new());
        }
    }

    /// Ensure 120-cell is initialized
    fn ensure_120cell(&mut self) {
        if self.cell120.is_none() {
            self.cell120 = Some(Cell120::new());
        }
    }

    /// Ensure E₈ layer is initialized
    fn ensure_e8_layer(&mut self) {
        if self.e8_layer.is_none() {
            self.ensure_600cell();
            if let Some(ref cell600) = self.cell600 {
                self.e8_layer = Some(cell600.phi_scaled_copy());
            }
        }
    }

    /// Get reference to primary 24-cell
    pub fn cell24(&self) -> &Cell24 {
        &self.cell24
    }

    /// Get mutable reference to primary 24-cell
    pub fn cell24_mut(&mut self) -> &mut Cell24 {
        &mut self.cell24
    }

    /// Get reference to 600-cell (if initialized)
    pub fn cell600(&self) -> Option<&Cell600> {
        self.cell600.as_ref()
    }

    /// Get reference to 120-cell (if initialized)
    pub fn cell120(&self) -> Option<&Cell120> {
        self.cell120.as_ref()
    }

    /// Set rotation speeds for all planes (radians per second)
    pub fn set_rotation_speed(&mut self, speeds: [f64; 6]) {
        self.rotation_speed = speeds;
    }

    /// Apply data mapping to geometric parameters
    pub fn apply_data_mapping(&mut self, data: &crate::pipeline::DataState) {
        // Map data channels to rotation angles
        if data.channels.len() >= 6 {
            let scale = std::f64::consts::PI; // Map [0,1] to [0, π]
            let angles = [
                data.channels[0] * scale,
                data.channels[1] * scale,
                data.channels[2] * scale,
                data.channels[3] * scale,
                data.channels[4] * scale,
                data.channels[5] * scale,
            ];

            // Compose rotations in all 6 planes
            let mut rotation = Rotation4D::identity();
            rotation = rotation.compose(Rotation4D::simple_xy(angles[0]));
            rotation = rotation.compose(Rotation4D::simple_xz(angles[1]));
            rotation = rotation.compose(Rotation4D::simple_xw(angles[2]));
            rotation = rotation.compose(Rotation4D::simple_yz(angles[3]));
            rotation = rotation.compose(Rotation4D::simple_yw(angles[4]));
            rotation = rotation.compose(Rotation4D::simple_zw(angles[5]));

            self.cell24.set_rotation(rotation);

            if let Some(ref mut cell600) = self.cell600 {
                cell600.set_rotation(rotation);
            }

            if let Some(ref mut e8_layer) = self.e8_layer {
                // E₈ layer rotates slightly differently (offset phase)
                let phase_offset = std::f64::consts::PI / 12.0;
                let mut e8_rotation = Rotation4D::identity();
                e8_rotation = e8_rotation.compose(Rotation4D::simple_xy(angles[0] + phase_offset));
                e8_rotation = e8_rotation.compose(Rotation4D::simple_xz(angles[1] + phase_offset));
                e8_rotation = e8_rotation.compose(Rotation4D::simple_xw(angles[2]));
                e8_rotation = e8_rotation.compose(Rotation4D::simple_yz(angles[3]));
                e8_rotation = e8_rotation.compose(Rotation4D::simple_yw(angles[4] + phase_offset));
                e8_rotation = e8_rotation.compose(Rotation4D::simple_zw(angles[5]));
                e8_layer.set_rotation(e8_rotation);
            }
        }

        // Map additional channels to Trinity component activations
        if data.channels.len() >= 9 {
            // Could be used to highlight specific components
            // Channels 6, 7, 8 -> Alpha, Beta, Gamma activation
        }
    }

    /// Update animation by delta time (seconds)
    pub fn update_animation(&mut self, delta: f64) {
        // Apply continuous rotation based on rotation_speed
        let delta_rotation = Rotation4D::simple_xy(self.rotation_speed[0] * delta)
            .compose(Rotation4D::simple_xz(self.rotation_speed[1] * delta))
            .compose(Rotation4D::simple_xw(self.rotation_speed[2] * delta))
            .compose(Rotation4D::simple_yz(self.rotation_speed[3] * delta))
            .compose(Rotation4D::simple_yw(self.rotation_speed[4] * delta))
            .compose(Rotation4D::simple_zw(self.rotation_speed[5] * delta));

        self.cell24.rotate(delta_rotation);

        if let Some(ref mut cell600) = self.cell600 {
            cell600.rotate(delta_rotation);
        }

        if let Some(ref mut cell120) = self.cell120 {
            cell120.rotate(delta_rotation);
        }

        if let Some(ref mut e8_layer) = self.e8_layer {
            // E₈ layer rotates at golden-ratio speed
            let e8_delta = Rotation4D::simple_xy(self.rotation_speed[0] * delta * PHI)
                .compose(Rotation4D::simple_zw(self.rotation_speed[5] * delta * PHI));
            e8_layer.rotate(e8_delta);
        }
    }

    /// Project all active geometry to 3D
    pub fn project_all(&mut self) {
        self.projected_3d.clear();

        match self.mode {
            GeometryMode::Core24Cell | GeometryMode::Trinity => {
                let colors = if self.mode == GeometryMode::Trinity {
                    Some(self.cell24.vertex_colors())
                } else {
                    None
                };

                self.projected_3d = self.projector.project_all_to_3d(
                    self.cell24.transformed_vertices(),
                    colors.as_deref(),
                );
            }

            GeometryMode::Expanded600 => {
                if let Some(ref cell600) = self.cell600 {
                    self.projected_3d = self.projector.project_all_to_3d(
                        cell600.transformed_vertices(),
                        None,
                    );
                }
            }

            GeometryMode::Full120 => {
                if let Some(ref cell120) = self.cell120 {
                    self.projected_3d = self.projector.project_all_to_3d(
                        cell120.transformed_vertices(),
                        None,
                    );
                }
            }

            GeometryMode::E8DualLayer => {
                // First layer
                if let Some(ref cell600) = self.cell600 {
                    self.projected_3d = self.projector.project_all_to_3d(
                        cell600.transformed_vertices(),
                        None,
                    );
                }
                // Second layer (will be rendered separately)
            }
        }
    }

    /// Get render data for the GPU
    pub fn get_render_data(&self) -> GeometryRenderData {
        let mut layers = Vec::new();

        match self.mode {
            GeometryMode::Core24Cell => {
                layers.push(self.create_render_layer(&self.cell24, 0, 1.0, None));
            }

            GeometryMode::Trinity => {
                // Render each Trinity component separately with its color
                layers.push(self.create_render_layer(
                    &self.cell24, 0, 1.0,
                    Some(self.cell24.vertex_colors())
                ));
            }

            GeometryMode::Expanded600 => {
                if let Some(ref cell600) = self.cell600 {
                    layers.push(self.create_render_layer(cell600, 0, 1.0, None));
                }
            }

            GeometryMode::Full120 => {
                if let Some(ref cell120) = self.cell120 {
                    layers.push(self.create_render_layer(cell120, 0, 1.0, None));
                }
            }

            GeometryMode::E8DualLayer => {
                // Layer 1: primary 600-cell
                if let Some(ref cell600) = self.cell600 {
                    layers.push(self.create_render_layer(cell600, 0, 0.8, None));
                }
                // Layer 2: φ-scaled 600-cell
                if let Some(ref e8_layer) = self.e8_layer {
                    let golden_color = vec![[1.0, 0.84, 0.0, 0.5]; e8_layer.vertex_count()];
                    layers.push(self.create_render_layer(
                        e8_layer, 1, 0.5, Some(golden_color)
                    ));
                }
            }
        }

        GeometryRenderData {
            layers,
            view_matrix: self.projector.view_matrix(),
            projection_matrix: self.projector.projection_matrix(),
        }
    }

    /// Create render layer for a polytope
    fn create_render_layer<P: Polytope4D>(
        &self,
        polytope: &P,
        layer_id: usize,
        opacity: f32,
        colors: Option<Vec<[f32; 4]>>,
    ) -> PolytopeRenderData {
        let vertices = polytope.transformed_vertices();
        let default_color = [1.0, 1.0, 1.0, opacity];

        let vertices_3d: Vec<[f32; 3]> = vertices.iter()
            .map(|v| {
                let p = self.projector.project_4d_to_3d(*v);
                [p[0] as f32, p[1] as f32, p[2] as f32]
            })
            .collect();

        let colors = colors.unwrap_or_else(|| {
            vec![default_color; vertices.len()]
        });

        let edges: Vec<(usize, usize)> = polytope.edges()
            .iter()
            .map(|e| e.vertices())
            .collect();

        PolytopeRenderData {
            vertices_3d,
            colors,
            edges,
            layer_id,
            opacity,
        }
    }

    /// Get current state for analysis
    pub fn state(&self) -> GeometryState {
        let rotation = self.cell24.rotation();
        GeometryState {
            mode: self.mode,
            cell24_rotation: [
                rotation.left.w,
                rotation.left.x,
                rotation.left.y,
                rotation.left.z,
            ],
            cell600_rotation: self.cell600.as_ref().map(|c| {
                let r = c.rotation();
                [r.left.w, r.left.x, r.left.y, r.left.z]
            }).unwrap_or([1.0, 0.0, 0.0, 0.0]),
            trinity_activations: [1.0, 1.0, 1.0], // Could be computed from data
            synthesis_detected: self.cell24.check_synthesis().is_some(),
            vertex_count: self.active_vertex_count(),
        }
    }

    /// Get total active vertex count
    fn active_vertex_count(&self) -> usize {
        match self.mode {
            GeometryMode::Core24Cell | GeometryMode::Trinity => 24,
            GeometryMode::Expanded600 | GeometryMode::E8DualLayer => {
                self.cell600.as_ref().map(|c| c.vertex_count()).unwrap_or(0)
            }
            GeometryMode::Full120 => {
                self.cell120.as_ref().map(|c| c.vertex_count()).unwrap_or(0)
            }
        }
    }

    /// Check for synthesis conditions in the Trinity
    pub fn check_synthesis(&self) -> Option<Vec<usize>> {
        self.cell24.check_synthesis()
    }

    /// Get dialectic distance between components
    pub fn dialectic_distance(&self, a: TrinityComponent, b: TrinityComponent) -> f64 {
        self.cell24.dialectic_distance(a, b)
    }

    /// Get projector for external modification
    pub fn projector_mut(&mut self) -> &mut Projector {
        &mut self.projector
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geometry_core_creation() {
        let config = EngineConfig::default();
        let core = GeometryCore::new(&config);
        assert_eq!(core.mode(), GeometryMode::Trinity);
    }

    #[test]
    fn test_geometry_mode_switching() {
        let config = EngineConfig::default();
        let mut core = GeometryCore::new(&config);

        core.set_mode(GeometryMode::Expanded600);
        assert!(core.cell600().is_some());

        core.set_mode(GeometryMode::Full120);
        assert!(core.cell120().is_some());
    }

    #[test]
    fn test_render_data_generation() {
        let config = EngineConfig::default();
        let mut core = GeometryCore::new(&config);
        core.project_all();

        let render_data = core.get_render_data();
        assert!(!render_data.layers.is_empty());
        assert_eq!(render_data.layers[0].vertices_3d.len(), 24);
    }
}
