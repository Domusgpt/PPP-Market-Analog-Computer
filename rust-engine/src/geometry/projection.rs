//! Projection operations from 4D to 3D to 2D
//!
//! The projection pipeline transforms high-dimensional polytope data into
//! visualizable 2D images while preserving structural information.

use super::Vec4;
use serde::{Serialize, Deserialize};

/// Available projection methods
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ProjectionMode {
    /// Drop one axis (simple orthographic)
    Orthographic { drop_axis: usize },
    /// Perspective projection from 4D with focal distance
    Perspective { focal_distance: f64 },
    /// Stereographic projection from 4D hypersphere
    Stereographic,
    /// Schlegel diagram (project from cell center)
    Schlegel { center: usize },
}

impl Default for ProjectionMode {
    fn default() -> Self {
        Self::Perspective { focal_distance: 3.0 }
    }
}

/// A projected 3D vertex with metadata
#[derive(Debug, Clone, Copy)]
pub struct Projected3D {
    pub position: [f64; 3],
    pub depth: f64,        // Original W coordinate for depth sorting
    pub original_index: usize,
    pub color: [f32; 4],
}

/// A projected 2D vertex (final screen space)
#[derive(Debug, Clone, Copy)]
pub struct Projected2D {
    pub position: [f64; 2],
    pub depth: f64,
    pub original_index: usize,
    pub color: [f32; 4],
}

/// Projector handles the 4D → 3D → 2D projection pipeline
#[derive(Debug, Clone)]
pub struct Projector {
    /// 4D → 3D projection mode
    pub mode_4d_to_3d: ProjectionMode,
    /// Camera position in 3D space
    pub camera_position: [f64; 3],
    /// Camera target (look-at point)
    pub camera_target: [f64; 3],
    /// Camera up vector
    pub camera_up: [f64; 3],
    /// Field of view in radians
    pub fov: f64,
    /// Aspect ratio (width / height)
    pub aspect: f64,
    /// Near clipping plane
    pub near: f64,
    /// Far clipping plane
    pub far: f64,
}

impl Default for Projector {
    fn default() -> Self {
        Self {
            mode_4d_to_3d: ProjectionMode::default(),
            camera_position: [0.0, 0.0, 5.0],
            camera_target: [0.0, 0.0, 0.0],
            camera_up: [0.0, 1.0, 0.0],
            fov: std::f64::consts::PI / 4.0, // 45 degrees
            aspect: 16.0 / 9.0,
            near: 0.1,
            far: 100.0,
        }
    }
}

impl Projector {
    /// Create a new projector with given settings
    pub fn new(mode: ProjectionMode) -> Self {
        Self {
            mode_4d_to_3d: mode,
            ..Default::default()
        }
    }

    /// Set the camera view parameters
    pub fn set_camera(&mut self, position: [f64; 3], target: [f64; 3], up: [f64; 3]) {
        self.camera_position = position;
        self.camera_target = target;
        self.camera_up = up;
    }

    /// Project a 4D point to 3D
    pub fn project_4d_to_3d(&self, point: Vec4) -> [f64; 3] {
        match self.mode_4d_to_3d {
            ProjectionMode::Orthographic { drop_axis } => {
                point.project_drop_axis(drop_axis)
            }
            ProjectionMode::Perspective { focal_distance } => {
                point.perspective_project(focal_distance)
            }
            ProjectionMode::Stereographic => {
                point.stereographic_project()
            }
            ProjectionMode::Schlegel { center: _ } => {
                // Simplified Schlegel: project from a point "outside" the polytope
                let focal = 2.5;
                point.perspective_project(focal)
            }
        }
    }

    /// Project a 4D point to 3D with metadata
    pub fn project_4d_to_3d_full(
        &self,
        point: Vec4,
        index: usize,
        color: [f32; 4]
    ) -> Projected3D {
        let position = self.project_4d_to_3d(point);
        Projected3D {
            position,
            depth: point.w,
            original_index: index,
            color,
        }
    }

    /// Project all 4D vertices to 3D
    pub fn project_all_to_3d(
        &self,
        vertices: &[Vec4],
        colors: Option<&[[f32; 4]]>
    ) -> Vec<Projected3D> {
        let default_color = [1.0, 1.0, 1.0, 1.0];

        vertices.iter().enumerate().map(|(i, v)| {
            let color = colors.map(|c| c[i]).unwrap_or(default_color);
            self.project_4d_to_3d_full(*v, i, color)
        }).collect()
    }

    /// Project a 3D point to 2D screen coordinates
    pub fn project_3d_to_2d(&self, point: [f64; 3]) -> [f64; 2] {
        // Build view matrix components
        let forward = normalize_3d(sub_3d(self.camera_target, self.camera_position));
        let right = normalize_3d(cross_3d(forward, self.camera_up));
        let up = cross_3d(right, forward);

        // Transform to camera space
        let rel = sub_3d(point, self.camera_position);
        let view_x = dot_3d(rel, right);
        let view_y = dot_3d(rel, up);
        let view_z = dot_3d(rel, forward);

        // Perspective projection
        if view_z.abs() < 0.001 {
            return [0.0, 0.0];
        }

        let scale = (self.fov / 2.0).tan();
        let x = view_x / (view_z * scale * self.aspect);
        let y = view_y / (view_z * scale);

        [x, y]
    }

    /// Project a Projected3D to 2D
    pub fn project_3d_to_2d_full(&self, point: &Projected3D) -> Projected2D {
        let position = self.project_3d_to_2d(point.position);
        Projected2D {
            position,
            depth: point.depth,
            original_index: point.original_index,
            color: point.color,
        }
    }

    /// Full pipeline: 4D → 2D
    pub fn project_4d_to_2d(
        &self,
        vertices: &[Vec4],
        colors: Option<&[[f32; 4]]>
    ) -> Vec<Projected2D> {
        let projected_3d = self.project_all_to_3d(vertices, colors);
        projected_3d.iter().map(|p| self.project_3d_to_2d_full(p)).collect()
    }

    /// Get view matrix as 4x4 array (for GPU)
    pub fn view_matrix(&self) -> [[f32; 4]; 4] {
        let forward = normalize_3d(sub_3d(self.camera_target, self.camera_position));
        let right = normalize_3d(cross_3d(forward, self.camera_up));
        let up = cross_3d(right, forward);

        let tx = -dot_3d(right, self.camera_position);
        let ty = -dot_3d(up, self.camera_position);
        let tz = dot_3d(forward, self.camera_position);

        [
            [right[0] as f32, up[0] as f32, -forward[0] as f32, 0.0],
            [right[1] as f32, up[1] as f32, -forward[1] as f32, 0.0],
            [right[2] as f32, up[2] as f32, -forward[2] as f32, 0.0],
            [tx as f32, ty as f32, tz as f32, 1.0],
        ]
    }

    /// Get projection matrix as 4x4 array (for GPU)
    pub fn projection_matrix(&self) -> [[f32; 4]; 4] {
        let f = 1.0 / (self.fov / 2.0).tan();
        let nf = 1.0 / (self.near - self.far);

        [
            [(f / self.aspect) as f32, 0.0, 0.0, 0.0],
            [0.0, f as f32, 0.0, 0.0],
            [0.0, 0.0, ((self.far + self.near) * nf) as f32, -1.0],
            [0.0, 0.0, ((2.0 * self.far * self.near) * nf) as f32, 0.0],
        ]
    }

    /// Get combined view-projection matrix
    pub fn view_projection_matrix(&self) -> [[f32; 4]; 4] {
        let view = self.view_matrix();
        let proj = self.projection_matrix();
        mat4_mul(&view, &proj)
    }
}

// Helper functions for 3D vector math

fn sub_3d(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot_3d(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross_3d(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize_3d(v: [f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-10 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn mat4_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
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

/// Multi-view projector for generating multiple simultaneous projections
#[derive(Debug, Clone)]
pub struct MultiViewProjector {
    pub projectors: Vec<Projector>,
}

impl MultiViewProjector {
    /// Create with orthographic projections along all 4 axes
    pub fn orthographic_all_axes() -> Self {
        Self {
            projectors: (0..4).map(|axis| {
                Projector::new(ProjectionMode::Orthographic { drop_axis: axis })
            }).collect()
        }
    }

    /// Create with perspective from multiple angles
    pub fn multi_perspective() -> Self {
        let mut projectors = Vec::new();

        for &focal in &[2.0, 3.0, 5.0] {
            let mut proj = Projector::new(ProjectionMode::Perspective {
                focal_distance: focal
            });

            // Vary camera positions
            proj.camera_position = [0.0, 0.0, focal * 2.0];
            projectors.push(proj);
        }

        Self { projectors }
    }

    /// Project vertices through all projectors
    pub fn project_all(
        &self,
        vertices: &[Vec4],
        colors: Option<&[[f32; 4]]>
    ) -> Vec<Vec<Projected2D>> {
        self.projectors.iter()
            .map(|p| p.project_4d_to_2d(vertices, colors))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orthographic_projection() {
        let projector = Projector::new(ProjectionMode::Orthographic { drop_axis: 3 });
        let point = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let projected = projector.project_4d_to_3d(point);

        assert_eq!(projected[0], 1.0);
        assert_eq!(projected[1], 2.0);
        assert_eq!(projected[2], 3.0);
    }

    #[test]
    fn test_perspective_projection() {
        let projector = Projector::new(ProjectionMode::Perspective { focal_distance: 2.0 });
        let point = Vec4::new(1.0, 1.0, 1.0, 0.0);
        let projected = projector.project_4d_to_3d(point);

        // With w=0 and focal=2, scale should be 2/2 = 1
        assert!((projected[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_view_matrix() {
        let projector = Projector::default();
        let view = projector.view_matrix();

        // View matrix should be valid
        assert!(view[0][0].is_finite());
        assert!(view[3][3].abs() > 0.5); // Should be ~1.0
    }
}
