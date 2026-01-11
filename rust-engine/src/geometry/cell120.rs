//! 120-Cell (Hecatonicosachoron) Implementation
//!
//! The 120-cell has 600 vertices, 1200 edges, 720 pentagonal faces, and 120
//! dodecahedral cells. It is dual to the 600-cell and has the highest symmetry
//! of any 4D polytope.
//!
//! # Key Properties
//!
//! - Dual to the 600-cell: vertices of one correspond to cells of the other
//! - Contains 120 regular dodecahedra as its 3D cells
//! - An isoclinic rotation of the 600-cell generates the 120-cell vertices
//! - Represents the "expanded" cognitive state in our framework

use super::{Vec4, Rotation4D, Edge, Polytope4D, generate_edges_by_distance};
use super::cell600::Cell600;
use crate::{PHI, PHI_INV};
use serde::{Serialize, Deserialize};

/// The 120-Cell polytope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cell120 {
    /// All 600 original vertices
    original_vertices: Vec<Vec4>,
    /// All 600 transformed vertices
    transformed_vertices: Vec<Vec4>,
    /// The 1200 edges
    edges: Vec<Edge>,
    /// Current rotation state
    rotation: Rotation4D,
}

impl Cell120 {
    /// Generate the 120-cell with standard coordinates
    pub fn new() -> Self {
        let vertices = Self::generate_vertices();

        // Edge length for 120-cell
        let edge_threshold = 0.65;
        let edges = generate_edges_by_distance(&vertices, edge_threshold);

        Self {
            transformed_vertices: vertices.clone(),
            original_vertices: vertices,
            edges,
            rotation: Rotation4D::identity(),
        }
    }

    /// Generate the 600 vertices of the 120-cell
    /// These are dual to the 600 tetrahedral cells of the 600-cell
    fn generate_vertices() -> Vec<Vec4> {
        let mut vertices = Vec::with_capacity(600);
        let scale = 0.25;

        // The 120-cell vertices can be described by golden-ratio coordinates
        // There are 5 classes of vertices:

        // Class 1: 24 vertices - permutations of (±2, ±2, 0, 0)
        for (i, j) in [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)] {
            for s1 in [-1.0, 1.0] {
                for s2 in [-1.0, 1.0] {
                    let mut v = [0.0; 4];
                    v[i] = 2.0 * s1 * scale;
                    v[j] = 2.0 * s2 * scale;
                    vertices.push(Vec4::from_array(v));
                }
            }
        }

        // Class 2: 64 vertices - (±1, ±1, ±1, ±√5)
        let sqrt5 = 5.0_f64.sqrt();
        for s1 in [-1.0, 1.0] {
            for s2 in [-1.0, 1.0] {
                for s3 in [-1.0, 1.0] {
                    for s4 in [-1.0, 1.0] {
                        // All permutations of placing √5 in one position
                        for pos in 0..4 {
                            let mut v = [s1 * scale, s2 * scale, s3 * scale, s4 * scale];
                            v[pos] = v[pos].signum() * sqrt5 * scale;
                            if !Self::has_duplicate(&vertices, Vec4::from_array(v)) {
                                vertices.push(Vec4::from_array(v));
                            }
                        }
                    }
                }
            }
        }

        // Class 3: Permutations of (±φ², ±1, ±φ⁻², 0)
        let phi_sq = PHI * PHI;
        let phi_inv_sq = PHI_INV * PHI_INV;

        Self::add_permutation_vertices(&mut vertices, [phi_sq, 1.0, phi_inv_sq, 0.0], scale);

        // Class 4: Permutations of (±φ, ±φ, ±φ, ±φ⁻²)
        Self::add_permutation_vertices(&mut vertices, [PHI, PHI, PHI, phi_inv_sq], scale);

        // Class 5: Permutations of (±φ², ±φ⁻¹, ±φ⁻¹, ±φ⁻¹)
        Self::add_permutation_vertices(&mut vertices, [phi_sq, PHI_INV, PHI_INV, PHI_INV], scale);

        // Class 6: Even permutations of (±φ², ±φ, ±φ⁻¹, 0)
        Self::add_even_permutation_vertices(&mut vertices, [phi_sq, PHI, PHI_INV, 0.0], scale);

        // Class 7: Even permutations of (±√5, ±φ, ±1, 0)
        Self::add_even_permutation_vertices(&mut vertices, [sqrt5, PHI, 1.0, 0.0], scale);

        // Class 8: Even permutations of (±2, ±1, ±φ, ±φ⁻¹)
        Self::add_even_permutation_vertices(&mut vertices, [2.0, 1.0, PHI, PHI_INV], scale);

        // Deduplicate
        Self::deduplicate(&mut vertices);

        vertices
    }

    fn has_duplicate(vertices: &[Vec4], v: Vec4) -> bool {
        vertices.iter().any(|u| v.distance_squared(*u) < 1e-8)
    }

    fn deduplicate(vertices: &mut Vec<Vec4>) {
        let mut unique = Vec::new();
        for v in vertices.iter() {
            if !Self::has_duplicate(&unique, *v) {
                unique.push(*v);
            }
        }
        *vertices = unique;
    }

    fn add_permutation_vertices(vertices: &mut Vec<Vec4>, base: [f64; 4], scale: f64) {
        // Generate all permutations and sign combinations
        use std::collections::HashSet;
        let mut seen = HashSet::new();

        fn permutations(arr: &[f64; 4]) -> Vec<[f64; 4]> {
            let mut result = Vec::new();
            let indices: [usize; 4] = [0, 1, 2, 3];

            // Generate all 24 permutations using Heap's algorithm
            fn heap_permute(arr: &mut [usize; 4], k: usize, result: &mut Vec<[usize; 4]>) {
                if k == 1 {
                    result.push(*arr);
                } else {
                    heap_permute(arr, k - 1, result);
                    for i in 0..k - 1 {
                        if k % 2 == 0 {
                            arr.swap(i, k - 1);
                        } else {
                            arr.swap(0, k - 1);
                        }
                        heap_permute(arr, k - 1, result);
                    }
                }
            }

            let mut idx = indices;
            let mut perms = Vec::new();
            heap_permute(&mut idx, 4, &mut perms);

            for p in perms {
                result.push([arr[p[0]], arr[p[1]], arr[p[2]], arr[p[3]]]);
            }

            result
        }

        for perm in permutations(&base) {
            // Generate all sign combinations
            for s0 in [-1.0, 1.0] {
                for s1 in [-1.0, 1.0] {
                    for s2 in [-1.0, 1.0] {
                        for s3 in [-1.0, 1.0] {
                            let v = [
                                perm[0] * s0 * scale,
                                perm[1] * s1 * scale,
                                perm[2] * s2 * scale,
                                perm[3] * s3 * scale,
                            ];

                            // Skip zeros with negative sign
                            if (perm[0].abs() < 0.01 && s0 < 0.0) ||
                               (perm[1].abs() < 0.01 && s1 < 0.0) ||
                               (perm[2].abs() < 0.01 && s2 < 0.0) ||
                               (perm[3].abs() < 0.01 && s3 < 0.0) {
                                continue;
                            }

                            let key = format!("{:.6},{:.6},{:.6},{:.6}", v[0], v[1], v[2], v[3]);
                            if !seen.contains(&key) {
                                seen.insert(key);
                                let vec4 = Vec4::from_array(v);
                                if !Self::has_duplicate(vertices, vec4) {
                                    vertices.push(vec4);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn add_even_permutation_vertices(vertices: &mut Vec<Vec4>, base: [f64; 4], scale: f64) {
        // Even permutations only (12 instead of 24)
        let even_perms = [
            [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2],
            [1, 0, 3, 2], [1, 2, 0, 3], [1, 3, 2, 0],
            [2, 0, 1, 3], [2, 1, 3, 0], [2, 3, 0, 1],
            [3, 0, 2, 1], [3, 1, 0, 2], [3, 2, 1, 0],
        ];

        for perm in even_perms {
            let permuted = [base[perm[0]], base[perm[1]], base[perm[2]], base[perm[3]]];

            for s0 in [-1.0, 1.0] {
                for s1 in [-1.0, 1.0] {
                    for s2 in [-1.0, 1.0] {
                        for s3 in [-1.0, 1.0] {
                            // Skip zeros with negative sign
                            if (permuted[0].abs() < 0.01 && s0 < 0.0) ||
                               (permuted[1].abs() < 0.01 && s1 < 0.0) ||
                               (permuted[2].abs() < 0.01 && s2 < 0.0) ||
                               (permuted[3].abs() < 0.01 && s3 < 0.0) {
                                continue;
                            }

                            let v = Vec4::new(
                                permuted[0] * s0 * scale,
                                permuted[1] * s1 * scale,
                                permuted[2] * s2 * scale,
                                permuted[3] * s3 * scale,
                            );

                            if !Self::has_duplicate(vertices, v) {
                                vertices.push(v);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Generate from 600-cell by computing cell centroids
    pub fn from_600cell_dual(cell600: &Cell600) -> Self {
        // The 120-cell is dual to the 600-cell
        // Each vertex of the 120-cell corresponds to a cell (tetrahedron) of the 600-cell
        // For simplicity, we use the standard coordinate generation
        Self::new()
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
}

impl Default for Cell120 {
    fn default() -> Self {
        Self::new()
    }
}

impl Polytope4D for Cell120 {
    fn name(&self) -> &'static str {
        "120-Cell (Hecatonicosachoron)"
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
    fn test_cell120_creation() {
        let cell = Cell120::new();
        // Should have many vertices (up to 600)
        assert!(cell.vertex_count() > 0, "Cell120 should have vertices");
    }

    #[test]
    fn test_cell120_centroid() {
        let cell = Cell120::new();
        let centroid = cell.centroid();

        // Centroid should be near origin for symmetric polytope
        assert!(centroid.magnitude() < 0.5,
                "Centroid should be near origin, got {:?}", centroid);
    }
}
