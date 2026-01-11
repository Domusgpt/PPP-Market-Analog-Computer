//! Trinity State Management
//!
//! Tracks the state of the Alpha/Beta/Gamma 16-cell components within the 24-cell
//! and manages the dialectic relationships between them.

use crate::geometry::{GeometryCore, TrinityComponent, Vec4};
use serde::{Serialize, Deserialize};
use std::collections::VecDeque;

/// Activation level for each Trinity component
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ComponentActivation {
    /// Raw activation value (0.0 to 1.0)
    pub level: f64,
    /// Activation velocity (rate of change)
    pub velocity: f64,
    /// Whether this component is currently "dominant"
    pub dominant: bool,
}

/// Record of a synthesis event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisRecord {
    /// Frame number when synthesis occurred
    pub frame: u64,
    /// Which Gamma vertices were activated
    pub activated_vertices: Vec<usize>,
    /// Alpha/Beta distance at time of synthesis
    pub dialectic_distance: f64,
    /// Resulting "insight" or pattern detected
    pub pattern_id: Option<u32>,
}

/// The Trinity state tracker
#[derive(Debug, Clone)]
pub struct TrinityState {
    /// Alpha (thesis) activation
    pub alpha: ComponentActivation,
    /// Beta (antithesis) activation
    pub beta: ComponentActivation,
    /// Gamma (synthesis) activation
    pub gamma: ComponentActivation,

    /// Centroid positions of each component
    pub alpha_centroid: Vec4,
    pub beta_centroid: Vec4,
    pub gamma_centroid: Vec4,

    /// Distance metrics between components
    pub alpha_beta_distance: f64,
    pub alpha_gamma_distance: f64,
    pub beta_gamma_distance: f64,

    /// History of synthesis events
    pub synthesis_history: VecDeque<SynthesisRecord>,
    /// Maximum history size
    max_history: usize,

    /// Current frame counter
    frame: u64,
}

impl TrinityState {
    pub fn new() -> Self {
        Self {
            alpha: ComponentActivation::default(),
            beta: ComponentActivation::default(),
            gamma: ComponentActivation::default(),
            alpha_centroid: Vec4::zero(),
            beta_centroid: Vec4::zero(),
            gamma_centroid: Vec4::zero(),
            alpha_beta_distance: 0.0,
            alpha_gamma_distance: 0.0,
            beta_gamma_distance: 0.0,
            synthesis_history: VecDeque::new(),
            max_history: 100,
            frame: 0,
        }
    }

    /// Update state from geometry core
    pub fn update_from_geometry(&mut self, geometry: &GeometryCore) {
        self.frame += 1;

        let cell24 = geometry.cell24();

        // Calculate centroids
        self.alpha_centroid = Self::centroid(&cell24.alpha.transformed_vertices);
        self.beta_centroid = Self::centroid(&cell24.beta.transformed_vertices);
        self.gamma_centroid = Self::centroid(&cell24.gamma.transformed_vertices);

        // Calculate distances
        let prev_ab = self.alpha_beta_distance;
        self.alpha_beta_distance = geometry.dialectic_distance(
            TrinityComponent::Alpha,
            TrinityComponent::Beta
        );
        self.alpha_gamma_distance = geometry.dialectic_distance(
            TrinityComponent::Alpha,
            TrinityComponent::Gamma
        );
        self.beta_gamma_distance = geometry.dialectic_distance(
            TrinityComponent::Beta,
            TrinityComponent::Gamma
        );

        // Update velocities
        self.alpha.velocity = self.alpha_beta_distance - prev_ab;

        // Determine dominance based on centroid distances from origin
        let alpha_mag = self.alpha_centroid.magnitude();
        let beta_mag = self.beta_centroid.magnitude();
        let gamma_mag = self.gamma_centroid.magnitude();

        let min_mag = alpha_mag.min(beta_mag).min(gamma_mag);
        self.alpha.dominant = (alpha_mag - min_mag).abs() < 0.01;
        self.beta.dominant = (beta_mag - min_mag).abs() < 0.01;
        self.gamma.dominant = (gamma_mag - min_mag).abs() < 0.01;

        // Set activation levels based on distances to other components
        self.alpha.level = 1.0 / (1.0 + self.alpha_beta_distance);
        self.beta.level = 1.0 / (1.0 + self.alpha_beta_distance);
        self.gamma.level = 1.0 / (1.0 + self.alpha_gamma_distance + self.beta_gamma_distance);
    }

    /// Record a synthesis event
    pub fn record_synthesis(&mut self, activated: &[usize]) {
        let record = SynthesisRecord {
            frame: self.frame,
            activated_vertices: activated.to_vec(),
            dialectic_distance: self.alpha_beta_distance,
            pattern_id: self.detect_pattern(activated),
        };

        self.synthesis_history.push_back(record);

        // Trim history
        while self.synthesis_history.len() > self.max_history {
            self.synthesis_history.pop_front();
        }
    }

    /// Attempt to identify a pattern from activated vertices
    fn detect_pattern(&self, activated: &[usize]) -> Option<u32> {
        // Pattern detection based on which vertices are active
        // This is a placeholder for more sophisticated pattern recognition
        match activated.len() {
            0 => None,
            1 => Some(1), // Single vertex pattern
            2 => Some(2), // Edge pattern
            3 => Some(3), // Triangle pattern
            4 => Some(4), // Tetrahedron pattern
            _ => Some(100 + activated.len() as u32),
        }
    }

    /// Get the "tension" between thesis and antithesis
    pub fn dialectic_tension(&self) -> f64 {
        // Tension is high when Alpha and Beta are activated but distant
        let activation_product = self.alpha.level * self.beta.level;
        activation_product * self.alpha_beta_distance
    }

    /// Get the "coherence" of the synthesis
    pub fn synthesis_coherence(&self) -> f64 {
        // Coherence is high when Gamma is well-activated and equidistant from both
        let distance_balance = 1.0 / (1.0 + (self.alpha_gamma_distance - self.beta_gamma_distance).abs());
        self.gamma.level * distance_balance
    }

    /// Check if the state is in a "resonance" condition
    pub fn is_resonant(&self) -> bool {
        // Resonance occurs when all three components are similarly activated
        let avg = (self.alpha.level + self.beta.level + self.gamma.level) / 3.0;
        let variance = (
            (self.alpha.level - avg).powi(2) +
            (self.beta.level - avg).powi(2) +
            (self.gamma.level - avg).powi(2)
        ) / 3.0;

        variance < 0.01 && avg > 0.5
    }

    /// Get synthesis rate (syntheses per frame)
    pub fn synthesis_rate(&self, window: usize) -> f64 {
        let recent: Vec<_> = self.synthesis_history.iter()
            .rev()
            .take(window)
            .collect();

        if recent.is_empty() || self.frame == 0 {
            return 0.0;
        }

        let oldest_frame = recent.last().map(|r| r.frame).unwrap_or(0);
        let frame_span = (self.frame - oldest_frame).max(1) as f64;

        recent.len() as f64 / frame_span
    }

    /// Calculate centroid of a set of vertices
    fn centroid(vertices: &[Vec4]) -> Vec4 {
        if vertices.is_empty() {
            return Vec4::zero();
        }

        let sum = vertices.iter().fold(Vec4::zero(), |acc, v| acc + *v);
        sum * (1.0 / vertices.len() as f64)
    }

    /// Get summary statistics
    pub fn summary(&self) -> TrinityStateSummary {
        TrinityStateSummary {
            alpha_level: self.alpha.level,
            beta_level: self.beta.level,
            gamma_level: self.gamma.level,
            tension: self.dialectic_tension(),
            coherence: self.synthesis_coherence(),
            resonant: self.is_resonant(),
            synthesis_count: self.synthesis_history.len(),
        }
    }
}

impl Default for TrinityState {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics for external reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrinityStateSummary {
    pub alpha_level: f64,
    pub beta_level: f64,
    pub gamma_level: f64,
    pub tension: f64,
    pub coherence: f64,
    pub resonant: bool,
    pub synthesis_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trinity_state_creation() {
        let state = TrinityState::new();
        assert_eq!(state.frame, 0);
        assert!(state.synthesis_history.is_empty());
    }

    #[test]
    fn test_pattern_detection() {
        let state = TrinityState::new();

        assert_eq!(state.detect_pattern(&[]), None);
        assert_eq!(state.detect_pattern(&[0]), Some(1));
        assert_eq!(state.detect_pattern(&[0, 1]), Some(2));
        assert_eq!(state.detect_pattern(&[0, 1, 2, 3]), Some(4));
    }
}
