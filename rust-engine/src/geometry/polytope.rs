//! Base polytope traits and structures for 4D polytopes

use super::{Vec4, Quaternion, Rotation4D, VertexState};
use serde::{Serialize, Deserialize};

/// An edge connecting two vertices by index
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Edge(pub usize, pub usize);

impl Edge {
    pub fn new(a: usize, b: usize) -> Self {
        // Normalize edge direction (smaller index first)
        if a <= b {
            Self(a, b)
        } else {
            Self(b, a)
        }
    }

    pub fn vertices(&self) -> (usize, usize) {
        (self.0, self.1)
    }
}

/// A face (typically triangular or higher polygon) by vertex indices
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Face(pub Vec<usize>);

impl Face {
    pub fn new(indices: Vec<usize>) -> Self {
        Self(indices)
    }

    pub fn triangle(a: usize, b: usize, c: usize) -> Self {
        Self(vec![a, b, c])
    }

    pub fn quad(a: usize, b: usize, c: usize, d: usize) -> Self {
        Self(vec![a, b, c, d])
    }

    pub fn indices(&self) -> &[usize] {
        &self.0
    }
}

/// Trait for 4D polytopes
pub trait Polytope4D: Send + Sync {
    /// Name of this polytope type
    fn name(&self) -> &'static str;

    /// Number of vertices
    fn vertex_count(&self) -> usize;

    /// Number of edges
    fn edge_count(&self) -> usize;

    /// Get the original (untransformed) vertices
    fn original_vertices(&self) -> &[Vec4];

    /// Get the current transformed vertices
    fn transformed_vertices(&self) -> &[Vec4];

    /// Get mutable access to transformed vertices
    fn transformed_vertices_mut(&mut self) -> &mut [Vec4];

    /// Get edges as pairs of vertex indices
    fn edges(&self) -> &[Edge];

    /// Get faces (if any)
    fn faces(&self) -> &[Face] {
        &[]
    }

    /// Apply a 4D rotation to all vertices
    fn apply_rotation(&mut self, rotation: Rotation4D) {
        let original = self.original_vertices().to_vec();
        let transformed = self.transformed_vertices_mut();

        for (i, orig) in original.iter().enumerate() {
            transformed[i] = rotation.rotate_point(*orig);
        }
    }

    /// Apply quaternion rotation (simpler rotation, left multiplication)
    fn apply_quaternion(&mut self, q: Quaternion) {
        let original = self.original_vertices().to_vec();
        let transformed = self.transformed_vertices_mut();

        for (i, orig) in original.iter().enumerate() {
            // Treat Vec4 as quaternion and multiply
            let p = Quaternion::new(orig.w, orig.x, orig.y, orig.z);
            let result = q.mul(p).mul(q.conjugate());
            transformed[i] = Vec4::new(result.x, result.y, result.z, result.w);
        }
    }

    /// Scale all transformed vertices by a factor
    fn apply_scale(&mut self, factor: f64) {
        for v in self.transformed_vertices_mut() {
            *v = *v * factor;
        }
    }

    /// Translate all transformed vertices
    fn apply_translation(&mut self, offset: Vec4) {
        for v in self.transformed_vertices_mut() {
            *v = *v + offset;
        }
    }

    /// Reset transformed vertices to original positions
    fn reset_transform(&mut self) {
        let original = self.original_vertices().to_vec();
        let transformed = self.transformed_vertices_mut();
        for (i, orig) in original.iter().enumerate() {
            transformed[i] = *orig;
        }
    }

    /// Calculate the centroid of the polytope
    fn centroid(&self) -> Vec4 {
        let verts = self.transformed_vertices();
        let n = verts.len() as f64;
        let sum = verts.iter().fold(Vec4::zero(), |acc, v| acc + *v);
        sum * (1.0 / n)
    }

    /// Calculate the bounding box
    fn bounding_box(&self) -> (Vec4, Vec4) {
        let verts = self.transformed_vertices();
        let mut min = Vec4::new(f64::MAX, f64::MAX, f64::MAX, f64::MAX);
        let mut max = Vec4::new(f64::MIN, f64::MIN, f64::MIN, f64::MIN);

        for v in verts {
            min = min.min(*v);
            max = max.max(*v);
        }

        (min, max)
    }

    /// Find the nearest vertex to a given point
    fn nearest_vertex(&self, point: Vec4) -> (usize, f64) {
        let verts = self.transformed_vertices();
        let mut best_idx = 0;
        let mut best_dist = f64::MAX;

        for (i, v) in verts.iter().enumerate() {
            let d = point.distance_squared(*v);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }

        (best_idx, best_dist.sqrt())
    }

    /// Get vertex states with colors and activations
    fn vertex_states(&self) -> Vec<VertexState> {
        self.transformed_vertices()
            .iter()
            .map(|&v| VertexState::new(v))
            .collect()
    }
}

/// A concrete implementation of a mutable polytope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcretePolytope {
    name: &'static str,
    original: Vec<Vec4>,
    transformed: Vec<Vec4>,
    edges: Vec<Edge>,
    faces: Vec<Face>,
}

impl ConcretePolytope {
    /// Create a new polytope from vertices and edges
    pub fn new(name: &'static str, vertices: Vec<Vec4>, edges: Vec<Edge>) -> Self {
        Self {
            name,
            transformed: vertices.clone(),
            original: vertices,
            edges,
            faces: Vec::new(),
        }
    }

    /// Create with faces
    pub fn with_faces(mut self, faces: Vec<Face>) -> Self {
        self.faces = faces;
        self
    }
}

impl Polytope4D for ConcretePolytope {
    fn name(&self) -> &'static str {
        self.name
    }

    fn vertex_count(&self) -> usize {
        self.original.len()
    }

    fn edge_count(&self) -> usize {
        self.edges.len()
    }

    fn original_vertices(&self) -> &[Vec4] {
        &self.original
    }

    fn transformed_vertices(&self) -> &[Vec4] {
        &self.transformed
    }

    fn transformed_vertices_mut(&mut self) -> &mut [Vec4] {
        &mut self.transformed
    }

    fn edges(&self) -> &[Edge] {
        &self.edges
    }

    fn faces(&self) -> &[Face] {
        &self.faces
    }
}

/// Generate edges for a set of vertices based on distance threshold
pub fn generate_edges_by_distance(vertices: &[Vec4], threshold: f64) -> Vec<Edge> {
    let mut edges = Vec::new();
    let threshold_sq = threshold * threshold;

    for i in 0..vertices.len() {
        for j in (i + 1)..vertices.len() {
            let dist_sq = vertices[i].distance_squared(vertices[j]);
            if dist_sq <= threshold_sq + 1e-10 {
                edges.push(Edge::new(i, j));
            }
        }
    }

    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_normalization() {
        let e1 = Edge::new(5, 3);
        let e2 = Edge::new(3, 5);
        assert_eq!(e1, e2);
        assert_eq!(e1.vertices(), (3, 5));
    }

    #[test]
    fn test_concrete_polytope() {
        let vertices = vec![
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
        ];
        let edges = vec![Edge::new(0, 1), Edge::new(1, 2), Edge::new(2, 0)];

        let poly = ConcretePolytope::new("test", vertices, edges);
        assert_eq!(poly.vertex_count(), 3);
        assert_eq!(poly.edge_count(), 3);
    }
}
