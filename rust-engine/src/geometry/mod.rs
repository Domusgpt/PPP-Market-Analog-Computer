//! # Geometry Core Module
//!
//! High-dimensional geometry processing including 4D polytopes, quaternion rotations,
//! and projection operations from E₈ → 4D → 3D → 2D.

mod vector;
mod quaternion;
mod polytope;
mod cell24;
mod cell600;
mod cell120;
mod projection;
mod core;

pub use vector::Vec4;
pub use quaternion::{Quaternion, Rotation4D, icosian_quaternions};
pub use polytope::{Polytope4D, Edge, Face, generate_edges_by_distance};
pub use cell24::Cell24;
pub use cell600::Cell600;
pub use cell120::Cell120;
pub use projection::{ProjectionMode, Projector, Projected3D, Projected2D};
pub use core::{GeometryCore, GeometryMode, GeometryRenderData, PolytopeRenderData, GeometryState};
pub use cell24::TrinityComponent;

/// Vertex activation state for computing which vertices are "active"
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VertexState {
    pub position: Vec4,
    pub activation: f64,
    pub color: [f32; 4],
}

impl VertexState {
    pub fn new(position: Vec4) -> Self {
        Self {
            position,
            activation: 1.0,
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }

    pub fn with_color(mut self, color: [f32; 4]) -> Self {
        self.color = color;
        self
    }

    pub fn with_activation(mut self, activation: f64) -> Self {
        self.activation = activation;
        self
    }
}
