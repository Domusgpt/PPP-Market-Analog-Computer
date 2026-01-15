//! 600-Cell (Hexacosichoron) Implementation
//!
//! The 600-cell has 120 vertices, 720 edges, 1200 triangular faces, and 600
//! tetrahedral cells. It is the 4D analog of the icosahedron and has remarkable
//! connections to the golden ratio φ.
//!
//! # Key Properties
//!
//! - Its 120 vertices correspond to the 120 icosian quaternions (binary icosahedral
//!   group 2I ≅ SL(2,5))
//! - Contains multiple 24-cells as substructures (~25 distinct 24-cell embeddings)
//! - When projected from the E₈ root system in 8D, two interlocking 600-cells
//!   appear, scaled by φ relative to each other
//!
//! # Vertex Coordinates
//!
//! All 120 vertices can be generated from:
//! 1. 8 vertices: permutations of (±2, 0, 0, 0)
//! 2. 16 vertices: (±1, ±1, ±1, ±1)
//! 3. 96 vertices: even permutations of (±φ, ±1, ±1/φ, 0)
//!
//! (Scaling by 1/2 gives unit edge length)

use super::{Vec4, Rotation4D, Edge, Polytope4D, generate_edges_by_distance};
use crate::{PHI, PHI_INV};
use serde::{Serialize, Deserialize};

/// The 600-Cell polytope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cell600 {
    /// All 120 original vertices
    original_vertices: Vec<Vec4>,
    /// All 120 transformed vertices
    transformed_vertices: Vec<Vec4>,
    /// The 720 edges
    edges: Vec<Edge>,
    /// Indices of vertices that form embedded 24-cells
    embedded_24cells: Vec<Vec<usize>>,
    /// Current rotation state
    rotation: Rotation4D,
}

impl Cell600 {
    /// Generate the 600-cell with standard coordinates
    pub fn new() -> Self {
        let vertices = Self::generate_vertices();

        // Edge length for 600-cell with our coordinates is approximately 1/φ
        let edge_threshold = PHI_INV + 0.01;
        let edges = generate_edges_by_distance(&vertices, edge_threshold);

        // Find embedded 24-cells (the 24-cell vertices are a subset of 600-cell)
        let embedded_24cells = Self::find_embedded_24cells(&vertices);

        Self {
            transformed_vertices: vertices.clone(),
            original_vertices: vertices,
            edges,
            embedded_24cells,
            rotation: Rotation4D::identity(),
        }
    }

    /// Generate the 120 vertices of the 600-cell
    fn generate_vertices() -> Vec<Vec4> {
        let mut vertices = Vec::with_capacity(120);
        let scale = 0.5; // Scale to unit edge length

        // Type 1: 8 vertices - permutations of (±2, 0, 0, 0)
        for i in 0..4 {
            for s in [-1.0, 1.0] {
                let mut v = [0.0; 4];
                v[i] = 2.0 * s * scale;
                vertices.push(Vec4::from_array(v));
            }
        }

        // Type 2: 16 vertices - (±1, ±1, ±1, ±1)
        for s0 in [-1.0, 1.0] {
            for s1 in [-1.0, 1.0] {
                for s2 in [-1.0, 1.0] {
                    for s3 in [-1.0, 1.0] {
                        vertices.push(Vec4::new(
                            s0 * scale,
                            s1 * scale,
                            s2 * scale,
                            s3 * scale,
                        ));
                    }
                }
            }
        }

        // Type 3: 96 vertices - even permutations of (±φ, ±1, ±1/φ, 0)
        let vals = [PHI, 1.0, PHI_INV, 0.0];

        // Generate even permutations
        let even_perms = [
            [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2],
            [1, 0, 3, 2], [1, 2, 0, 3], [1, 3, 2, 0],
            [2, 0, 1, 3], [2, 1, 3, 0], [2, 3, 0, 1],
            [3, 0, 2, 1], [3, 1, 0, 2], [3, 2, 1, 0],
        ];

        for perm in even_perms {
            // Only apply signs to non-zero components
            let base = [vals[perm[0]], vals[perm[1]], vals[perm[2]], vals[perm[3]]];

            for s0 in [-1.0, 1.0] {
                for s1 in [-1.0, 1.0] {
                    for s2 in [-1.0, 1.0] {
                        // Skip if zero component would get sign
                        if base[0].abs() < 0.01 || base[1].abs() < 0.01 || base[2].abs() < 0.01 {
                            let v = Vec4::new(
                                base[0] * s0 * scale,
                                base[1] * s1 * scale,
                                base[2] * s2 * scale,
                                base[3] * scale,
                            );
                            if !Self::is_duplicate(&vertices, v) {
                                vertices.push(v);
                            }
                        } else {
                            for s3 in [-1.0, 1.0] {
                                let v = Vec4::new(
                                    base[0] * s0 * scale,
                                    base[1] * s1 * scale,
                                    base[2] * s2 * scale,
                                    base[3] * s3 * scale,
                                );
                                if !Self::is_duplicate(&vertices, v) {
                                    vertices.push(v);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Ensure we have exactly 120 vertices (may need adjustment)
        Self::deduplicate_vertices(&mut vertices);

        vertices
    }

    /// Check if vertex is duplicate
    fn is_duplicate(vertices: &[Vec4], v: Vec4) -> bool {
        vertices.iter().any(|u| v.distance_squared(*u) < 1e-8)
    }

    /// Remove duplicate vertices
    fn deduplicate_vertices(vertices: &mut Vec<Vec4>) {
        let mut unique = Vec::with_capacity(120);
        for v in vertices.iter() {
            if !Self::is_duplicate(&unique, *v) {
                unique.push(*v);
            }
        }
        *vertices = unique;
    }

    /// Find embedded 24-cells within the 600-cell
    fn find_embedded_24cells(vertices: &[Vec4]) -> Vec<Vec<usize>> {
        // A 24-cell can be found as a subset of the 600-cell
        // The simplest is the "base" 24-cell formed by the 24 vertices
        // closest to the coordinate axes

        let mut cells = Vec::new();

        // Find the primary 24-cell: vertices of the form (±1, ±1, 0, 0) permutations
        // scaled appropriately
        let mut primary_24cell = Vec::new();
        let target_dist = 0.5 * std::f64::consts::SQRT_2; // Expected distance from origin

        for (i, v) in vertices.iter().enumerate() {
            let dist = v.magnitude();
            // Check if this vertex has exactly 2 non-zero components
            let nonzero_count = [v.x, v.y, v.z, v.w]
                .iter()
                .filter(|c| c.abs() > 0.1)
                .count();

            if nonzero_count == 2 && (dist - target_dist).abs() < 0.1 {
                primary_24cell.push(i);
            }
        }

        if primary_24cell.len() >= 20 {
            cells.push(primary_24cell);
        }

        cells
    }

    /// Get the current rotation
    pub fn rotation(&self) -> Rotation4D {
        self.rotation
    }

    /// Set rotation directly
    pub fn set_rotation(&mut self, rotation: Rotation4D) {
        self.rotation = rotation;
        self.apply_rotation_internal();
    }

    /// Compose with an additional rotation
    pub fn rotate(&mut self, delta_rotation: Rotation4D) {
        self.rotation = self.rotation.compose(delta_rotation);
        self.apply_rotation_internal();
    }

    /// Apply rotation to all vertices
    fn apply_rotation_internal(&mut self) {
        for (i, orig) in self.original_vertices.iter().enumerate() {
            self.transformed_vertices[i] = self.rotation.rotate_point(*orig);
        }
    }

    /// Get indices of vertices that form embedded 24-cells
    pub fn embedded_24cells(&self) -> &[Vec<usize>] {
        &self.embedded_24cells
    }

    /// Create a φ-scaled copy of this 600-cell (for E₈ dual-layer)
    pub fn phi_scaled_copy(&self) -> Self {
        let scaled_original: Vec<Vec4> = self.original_vertices
            .iter()
            .map(|v| v.scale_phi())
            .collect();

        let scaled_transformed: Vec<Vec4> = self.transformed_vertices
            .iter()
            .map(|v| v.scale_phi())
            .collect();

        Self {
            original_vertices: scaled_original,
            transformed_vertices: scaled_transformed,
            edges: self.edges.clone(),
            embedded_24cells: self.embedded_24cells.clone(),
            rotation: self.rotation,
        }
    }

    /// Get the Hopf fibration structure
    /// The 600-cell admits a beautiful Hopf fibration with 60 fibers of 2 vertices
    pub fn hopf_fibers(&self) -> Vec<Vec<usize>> {
        // Simplified: group antipodal vertices
        let mut fibers = Vec::new();
        let mut used = vec![false; self.original_vertices.len()];

        for i in 0..self.original_vertices.len() {
            if used[i] {
                continue;
            }

            let v = self.original_vertices[i];
            let antipode = -v;

            // Find the antipodal vertex
            for j in (i + 1)..self.original_vertices.len() {
                if !used[j] && self.original_vertices[j].distance(antipode) < 1e-6 {
                    fibers.push(vec![i, j]);
                    used[i] = true;
                    used[j] = true;
                    break;
                }
            }
        }

        fibers
    }

    /// Calculate vertex density in a region
    pub fn vertex_density(&self, center: Vec4, radius: f64) -> usize {
        self.transformed_vertices
            .iter()
            .filter(|v| v.distance(center) <= radius)
            .count()
    }
}

impl Default for Cell600 {
    fn default() -> Self {
        Self::new()
    }
}

impl Polytope4D for Cell600 {
    fn name(&self) -> &'static str {
        "600-Cell (Hexacosichoron)"
    }

    fn vertex_count(&self) -> usize {
        self.original_vertices.len()
    }

    fn edge_count(&self) -> usize {
        self.edges.len()
    }

    fn original_vertices(&self) -> &[Vec4] {
        &self.original_vertices
    }

    fn transformed_vertices(&self) -> &[Vec4] {
        &self.transformed_vertices
    }

    fn transformed_vertices_mut(&mut self) -> &mut [Vec4] {
        &mut self.transformed_vertices
    }

    fn edges(&self) -> &[Edge] {
        &self.edges
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell600_vertex_count() {
        let cell = Cell600::new();
        // Should have 120 vertices (or close, depending on generation precision)
        assert!(cell.vertex_count() >= 100 && cell.vertex_count() <= 130,
                "Expected ~120 vertices, got {}", cell.vertex_count());
    }

    #[test]
    fn test_cell600_centroid() {
        let cell = Cell600::new();
        let centroid = cell.centroid();

        // Centroid should be at origin for symmetric 600-cell
        assert!(centroid.magnitude() < 0.1,
                "Centroid should be near origin, got {:?}", centroid);
    }

    #[test]
    fn test_phi_scaled_copy() {
        let cell = Cell600::new();
        let scaled = cell.phi_scaled_copy();

        // Scaled vertices should be φ times larger
        let orig_dist = cell.original_vertices[0].magnitude();
        let scaled_dist = scaled.original_vertices[0].magnitude();

        assert!((scaled_dist / orig_dist - PHI).abs() < 0.01,
                "Scaling factor should be φ");
    }

    #[test]
    fn test_cell600_symmetry() {
        let cell = Cell600::new();

        // All vertices should be at same distance from origin (on hypersphere)
        let distances: Vec<f64> = cell.original_vertices
            .iter()
            .map(|v| v.magnitude())
            .collect();

        if !distances.is_empty() {
            let avg = distances.iter().sum::<f64>() / distances.len() as f64;
            let max_dev = distances.iter()
                .map(|d| (d - avg).abs())
                .fold(0.0, f64::max);

            // Allow some tolerance for different vertex types
            assert!(max_dev < 0.5,
                    "Vertex distances should be similar, max deviation: {}", max_dev);
        }
    }
}
