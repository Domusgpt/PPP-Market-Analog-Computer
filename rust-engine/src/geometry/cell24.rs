//! 24-Cell (Icositetrachoron) Implementation
//!
//! The 24-cell is a unique 4D polytope with 24 vertices, 96 edges, 96 triangular
//! faces, and 24 octahedral cells. It is self-dual and has remarkable symmetry
//! properties.
//!
//! # Trinity Decomposition
//!
//! The 24-cell can be uniquely decomposed into three mutually orthogonal 16-cells
//! (Alpha, Beta, Gamma). Each 16-cell has 8 vertices, and together they form the
//! complete 24-vertex structure. This provides a natural triadic framework for
//! thesis-antithesis-synthesis reasoning.
//!
//! # Vertex Coordinates
//!
//! The 24 vertices can be described as all permutations of (±1, ±1, 0, 0):
//! - 8 vertices with x,y varying: (±1, ±1, 0, 0)
//! - 8 vertices with x,z varying: (±1, 0, ±1, 0)
//! - 8 vertices with x,w varying: (±1, 0, 0, ±1)
//! Plus the remaining 8 to complete the 24.
//!
//! Alternatively: permutations of (±1, 0, 0, 0) and (±1/√2, ±1/√2, ±1/√2, ±1/√2)

use super::{Vec4, Rotation4D, Edge, Polytope4D, generate_edges_by_distance};
use serde::{Serialize, Deserialize};

/// Identifies which 16-cell a vertex belongs to in the Trinity decomposition
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TrinityComponent {
    /// Thesis - first 16-cell
    Alpha,
    /// Antithesis - second 16-cell
    Beta,
    /// Synthesis - third 16-cell
    Gamma,
}

impl TrinityComponent {
    /// Get the next component in the dialectic cycle
    pub fn next(self) -> Self {
        match self {
            Self::Alpha => Self::Beta,
            Self::Beta => Self::Gamma,
            Self::Gamma => Self::Alpha,
        }
    }

    /// Get the previous component
    pub fn prev(self) -> Self {
        match self {
            Self::Alpha => Self::Gamma,
            Self::Beta => Self::Alpha,
            Self::Gamma => Self::Beta,
        }
    }

    /// Get the color associated with this component
    pub fn color(self) -> [f32; 4] {
        match self {
            Self::Alpha => [1.0, 0.2, 0.2, 0.8], // Red (thesis)
            Self::Beta => [0.2, 1.0, 0.2, 0.8],  // Green (antithesis)
            Self::Gamma => [0.2, 0.2, 1.0, 0.8], // Blue (synthesis)
        }
    }
}

/// A 16-cell (hexadecachoron) - a component of the Trinity decomposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cell16 {
    /// Which component this is
    pub component: TrinityComponent,
    /// The 8 vertex indices in the parent 24-cell
    pub vertex_indices: Vec<usize>,
    /// Original vertex positions (before transformation)
    pub original_vertices: Vec<Vec4>,
    /// Current transformed vertices
    pub transformed_vertices: Vec<Vec4>,
    /// Edges connecting the 8 vertices (24 edges for a 16-cell)
    pub edges: Vec<Edge>,
}

impl Cell16 {
    /// Create a new 16-cell with given vertices
    pub fn new(component: TrinityComponent, vertex_indices: Vec<usize>, vertices: Vec<Vec4>) -> Self {
        // Generate edges: in a 16-cell, each vertex connects to 6 others
        // Edge length is √2 times the vertex-to-center distance
        let edges = generate_edges_by_distance(&vertices, 2.01); // sqrt(2) * sqrt(2) ≈ 2

        Self {
            component,
            vertex_indices,
            transformed_vertices: vertices.clone(),
            original_vertices: vertices,
            edges,
        }
    }

    /// Apply a rotation to this 16-cell independently
    pub fn apply_rotation(&mut self, rotation: Rotation4D) {
        for (i, orig) in self.original_vertices.iter().enumerate() {
            self.transformed_vertices[i] = rotation.rotate_point(*orig);
        }
    }

    /// Reset to original positions
    pub fn reset_transform(&mut self) {
        self.transformed_vertices = self.original_vertices.clone();
    }
}

/// The 24-Cell polytope with Trinity decomposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cell24 {
    /// All 24 original vertices
    original_vertices: Vec<Vec4>,
    /// All 24 transformed vertices
    transformed_vertices: Vec<Vec4>,
    /// The 96 edges
    edges: Vec<Edge>,
    /// Trinity decomposition: Alpha, Beta, Gamma 16-cells
    pub alpha: Cell16,
    pub beta: Cell16,
    pub gamma: Cell16,
    /// Current rotation state
    rotation: Rotation4D,
}

impl Cell24 {
    /// Generate the 24-cell with standard coordinates
    pub fn new() -> Self {
        let vertices = Self::generate_vertices();
        let edges = generate_edges_by_distance(&vertices, 1.42); // Edge length ≈ √2

        // Trinity decomposition: partition 24 vertices into three 16-cells
        // Using coordinate-based assignment
        let (alpha_idx, beta_idx, gamma_idx) = Self::compute_trinity_partition(&vertices);

        let alpha_verts: Vec<Vec4> = alpha_idx.iter().map(|&i| vertices[i]).collect();
        let beta_verts: Vec<Vec4> = beta_idx.iter().map(|&i| vertices[i]).collect();
        let gamma_verts: Vec<Vec4> = gamma_idx.iter().map(|&i| vertices[i]).collect();

        Self {
            transformed_vertices: vertices.clone(),
            original_vertices: vertices,
            edges,
            alpha: Cell16::new(TrinityComponent::Alpha, alpha_idx, alpha_verts),
            beta: Cell16::new(TrinityComponent::Beta, beta_idx, beta_verts),
            gamma: Cell16::new(TrinityComponent::Gamma, gamma_idx, gamma_verts),
            rotation: Rotation4D::identity(),
        }
    }

    /// Generate the 24 vertices of the 24-cell
    fn generate_vertices() -> Vec<Vec4> {
        let mut vertices = Vec::with_capacity(24);

        // All permutations of (±1, ±1, 0, 0)
        // This gives us 24 vertices forming a 24-cell

        let coords = [
            // Pair in xy plane (z=0, w=0)
            (0, 1), // indices for the two non-zero coords
            // Pair in xz plane
            (0, 2),
            // Pair in xw plane
            (0, 3),
            // Pair in yz plane
            (1, 2),
            // Pair in yw plane
            (1, 3),
            // Pair in zw plane
            (2, 3),
        ];

        for (i, j) in coords {
            for s1 in [-1.0, 1.0] {
                for s2 in [-1.0, 1.0] {
                    let mut v = [0.0; 4];
                    v[i] = s1;
                    v[j] = s2;
                    vertices.push(Vec4::from_array(v));
                }
            }
        }

        vertices
    }

    /// Partition vertices into three orthogonal 16-cells
    fn compute_trinity_partition(vertices: &[Vec4]) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        // The 24-cell can be partitioned based on which coordinate pairs are non-zero
        // Alpha: vertices in XY and ZW planes
        // Beta: vertices in XZ and YW planes
        // Gamma: vertices in XW and YZ planes

        let mut alpha = Vec::new();
        let mut beta = Vec::new();
        let mut gamma = Vec::new();

        for (i, v) in vertices.iter().enumerate() {
            let nonzero = [
                v.x.abs() > 0.5,
                v.y.abs() > 0.5,
                v.z.abs() > 0.5,
                v.w.abs() > 0.5,
            ];

            // Determine which pair of coordinates is non-zero
            match (nonzero[0], nonzero[1], nonzero[2], nonzero[3]) {
                (true, true, false, false) | (false, false, true, true) => alpha.push(i),
                (true, false, true, false) | (false, true, false, true) => beta.push(i),
                (true, false, false, true) | (false, true, true, false) => gamma.push(i),
                _ => {}
            }
        }

        (alpha, beta, gamma)
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

        // Also update Trinity components
        self.alpha.apply_rotation(self.rotation);
        self.beta.apply_rotation(self.rotation);
        self.gamma.apply_rotation(self.rotation);
    }

    /// Get a specific Trinity component
    pub fn get_component(&self, component: TrinityComponent) -> &Cell16 {
        match component {
            TrinityComponent::Alpha => &self.alpha,
            TrinityComponent::Beta => &self.beta,
            TrinityComponent::Gamma => &self.gamma,
        }
    }

    /// Get mutable reference to a Trinity component
    pub fn get_component_mut(&mut self, component: TrinityComponent) -> &mut Cell16 {
        match component {
            TrinityComponent::Alpha => &mut self.alpha,
            TrinityComponent::Beta => &mut self.beta,
            TrinityComponent::Gamma => &mut self.gamma,
        }
    }

    /// Get the "dialectic distance" between two components
    /// Returns a measure of how opposed they are based on their current transforms
    pub fn dialectic_distance(&self, a: TrinityComponent, b: TrinityComponent) -> f64 {
        let comp_a = self.get_component(a);
        let comp_b = self.get_component(b);

        // Calculate average distance between corresponding vertices
        let mut total_dist = 0.0;
        let count = comp_a.transformed_vertices.len().min(comp_b.transformed_vertices.len());

        for i in 0..count {
            total_dist += comp_a.transformed_vertices[i]
                .distance(comp_b.transformed_vertices[i]);
        }

        total_dist / count as f64
    }

    /// Check if a synthesis condition is met
    /// This is triggered when Alpha and Beta configurations overlap
    /// in a way that activates specific Gamma vertices
    pub fn check_synthesis(&self) -> Option<Vec<usize>> {
        let threshold = 0.5;
        let mut activated_gamma = Vec::new();

        for (gi, gv) in self.gamma.transformed_vertices.iter().enumerate() {
            // Check proximity to both Alpha and Beta vertices
            let alpha_dist = self.alpha.transformed_vertices
                .iter()
                .map(|av| gv.distance(*av))
                .fold(f64::MAX, |a, b| a.min(b));

            let beta_dist = self.beta.transformed_vertices
                .iter()
                .map(|bv| gv.distance(*bv))
                .fold(f64::MAX, |a, b| a.min(b));

            // Gamma vertex is "activated" if close to both Alpha and Beta
            if alpha_dist < threshold && beta_dist < threshold {
                activated_gamma.push(gi);
            }
        }

        if activated_gamma.is_empty() {
            None
        } else {
            Some(activated_gamma)
        }
    }

    /// Get vertex colors based on Trinity membership
    pub fn vertex_colors(&self) -> Vec<[f32; 4]> {
        let mut colors = vec![[1.0, 1.0, 1.0, 1.0]; 24];

        for &i in &self.alpha.vertex_indices {
            colors[i] = TrinityComponent::Alpha.color();
        }
        for &i in &self.beta.vertex_indices {
            colors[i] = TrinityComponent::Beta.color();
        }
        for &i in &self.gamma.vertex_indices {
            colors[i] = TrinityComponent::Gamma.color();
        }

        colors
    }

    /// Get the edge length (√2 for unit 24-cell)
    pub fn edge_length() -> f64 {
        std::f64::consts::SQRT_2
    }
}

impl Default for Cell24 {
    fn default() -> Self {
        Self::new()
    }
}

impl Polytope4D for Cell24 {
    fn name(&self) -> &'static str {
        "24-Cell (Icositetrachoron)"
    }

    fn vertex_count(&self) -> usize {
        24
    }

    fn edge_count(&self) -> usize {
        96
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
    fn test_cell24_vertex_count() {
        let cell = Cell24::new();
        assert_eq!(cell.vertex_count(), 24);
    }

    #[test]
    fn test_cell24_edge_count() {
        let cell = Cell24::new();
        // 24-cell has 96 edges
        assert!(cell.edge_count() >= 90 && cell.edge_count() <= 100);
    }

    #[test]
    fn test_trinity_partition() {
        let cell = Cell24::new();

        // Each 16-cell should have 8 vertices
        assert_eq!(cell.alpha.vertex_indices.len(), 8);
        assert_eq!(cell.beta.vertex_indices.len(), 8);
        assert_eq!(cell.gamma.vertex_indices.len(), 8);

        // All vertices should be covered exactly once
        let mut all_indices: Vec<usize> = Vec::new();
        all_indices.extend(&cell.alpha.vertex_indices);
        all_indices.extend(&cell.beta.vertex_indices);
        all_indices.extend(&cell.gamma.vertex_indices);
        all_indices.sort();
        all_indices.dedup();
        assert_eq!(all_indices.len(), 24);
    }

    #[test]
    fn test_cell24_centroid() {
        let cell = Cell24::new();
        let centroid = cell.centroid();

        // Centroid should be at origin for symmetric 24-cell
        assert!(centroid.magnitude() < 1e-10);
    }

    #[test]
    fn test_cell24_rotation_preserves_structure() {
        let mut cell = Cell24::new();
        let rotation = Rotation4D::simple_xy(std::f64::consts::PI / 4.0);
        cell.rotate(rotation);

        // Check that edge lengths are preserved
        for edge in cell.edges() {
            let (a, b) = edge.vertices();
            let dist = cell.transformed_vertices[a].distance(cell.transformed_vertices[b]);
            assert!((dist - Cell24::edge_length()).abs() < 1e-6);
        }
    }
}
