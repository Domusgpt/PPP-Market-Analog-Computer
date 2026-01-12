//! Dialectic Engine - Thesis-Antithesis-Synthesis Processing
//!
//! The dialectic engine manages the core reasoning process where opposing
//! states (Alpha/Beta) are resolved through synthesis (Gamma).

use crate::geometry::{GeometryCore, TrinityComponent, Vec4};
use serde::{Serialize, Deserialize};

/// Configuration for dialectic processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialecticConfig {
    /// Proximity threshold for synthesis detection
    pub synthesis_threshold: f64,
    /// Minimum tension required for synthesis to occur
    pub min_tension: f64,
    /// Whether to allow multi-vertex synthesis
    pub multi_vertex_synthesis: bool,
    /// Decay rate for activation (per frame)
    pub activation_decay: f64,
}

impl Default for DialecticConfig {
    fn default() -> Self {
        Self {
            synthesis_threshold: 0.5,
            min_tension: 0.1,
            multi_vertex_synthesis: true,
            activation_decay: 0.01,
        }
    }
}

/// Result of a dialectic step
#[derive(Debug, Clone)]
pub struct DialecticResult {
    /// Whether synthesis occurred
    pub synthesis_occurred: bool,
    /// Activated Gamma vertex indices
    pub activated_gamma: Vec<usize>,
    /// Current dialectic tension
    pub tension: f64,
    /// Synthesis "strength" (0.0 to 1.0)
    pub synthesis_strength: f64,
    /// Phase of the dialectic cycle (0.0 to 1.0)
    pub cycle_phase: f64,
}

/// The dialectic engine processes thesis-antithesis-synthesis cycles
#[derive(Debug, Clone)]
pub struct DialecticEngine {
    /// Configuration
    config: DialecticConfig,
    /// Current Alpha (thesis) activation per vertex
    alpha_activation: Vec<f64>,
    /// Current Beta (antithesis) activation per vertex
    beta_activation: Vec<f64>,
    /// Current Gamma (synthesis) activation per vertex
    gamma_activation: Vec<f64>,
    /// Phase accumulator for cycle tracking
    phase: f64,
    /// Total syntheses completed
    synthesis_count: u64,
}

impl DialecticEngine {
    pub fn new() -> Self {
        Self {
            config: DialecticConfig::default(),
            alpha_activation: vec![1.0; 8], // 8 vertices per 16-cell
            beta_activation: vec![1.0; 8],
            gamma_activation: vec![0.0; 8],
            phase: 0.0,
            synthesis_count: 0,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: DialecticConfig) -> Self {
        Self {
            config,
            ..Self::new()
        }
    }

    /// Detect synthesis conditions in the geometry
    pub fn detect_synthesis(&mut self, geometry: &GeometryCore) -> Option<Vec<usize>> {
        let cell24 = geometry.cell24();

        // Check for synthesis via geometric overlap
        let synthesis_result = cell24.check_synthesis();

        if let Some(ref activated) = synthesis_result {
            // Update activation levels
            for &idx in activated {
                if idx < self.gamma_activation.len() {
                    self.gamma_activation[idx] = 1.0;
                }
            }
            self.synthesis_count += 1;
        }

        synthesis_result
    }

    /// Run one step of the dialectic process
    pub fn step(&mut self, geometry: &GeometryCore, delta_time: f64) -> DialecticResult {
        // Decay activations
        for a in &mut self.alpha_activation {
            *a = (*a - self.config.activation_decay * delta_time).max(0.0);
        }
        for a in &mut self.beta_activation {
            *a = (*a - self.config.activation_decay * delta_time).max(0.0);
        }
        for a in &mut self.gamma_activation {
            *a = (*a - self.config.activation_decay * delta_time).max(0.0);
        }

        // Update phase
        self.phase = (self.phase + delta_time * 0.1) % 1.0;

        // Calculate tension
        let tension = self.calculate_tension(geometry);

        // Check for synthesis
        let synthesis_result = if tension > self.config.min_tension {
            self.detect_synthesis(geometry)
        } else {
            None
        };

        let synthesis_occurred = synthesis_result.is_some();
        let activated_gamma = synthesis_result.unwrap_or_default();

        // Calculate synthesis strength
        let synthesis_strength = if synthesis_occurred {
            activated_gamma.len() as f64 / 8.0
        } else {
            0.0
        };

        DialecticResult {
            synthesis_occurred,
            activated_gamma,
            tension,
            synthesis_strength,
            cycle_phase: self.phase,
        }
    }

    /// Calculate dialectic tension between Alpha and Beta
    fn calculate_tension(&self, geometry: &GeometryCore) -> f64 {
        let _cell24 = geometry.cell24();

        // Base tension from geometric distance
        let geometric_distance = geometry.dialectic_distance(
            TrinityComponent::Alpha,
            TrinityComponent::Beta,
        );

        // Modulate by activation levels
        let alpha_sum: f64 = self.alpha_activation.iter().sum();
        let beta_sum: f64 = self.beta_activation.iter().sum();
        let activation_factor = (alpha_sum * beta_sum) / 64.0; // Normalized

        geometric_distance * activation_factor
    }

    /// Set Alpha (thesis) state from external data
    pub fn set_alpha_state(&mut self, activations: &[f64]) {
        let n = activations.len().min(self.alpha_activation.len());
        self.alpha_activation[..n].copy_from_slice(&activations[..n]);
    }

    /// Set Beta (antithesis) state from external data
    pub fn set_beta_state(&mut self, activations: &[f64]) {
        let n = activations.len().min(self.beta_activation.len());
        self.beta_activation[..n].copy_from_slice(&activations[..n]);
    }

    /// Get current Gamma (synthesis) activations
    pub fn gamma_state(&self) -> &[f64] {
        &self.gamma_activation
    }

    /// Reset all activations
    pub fn reset(&mut self) {
        self.alpha_activation.fill(1.0);
        self.beta_activation.fill(1.0);
        self.gamma_activation.fill(0.0);
        self.phase = 0.0;
    }

    /// Get the number of completed syntheses
    pub fn synthesis_count(&self) -> u64 {
        self.synthesis_count
    }

    /// Get current phase
    pub fn phase(&self) -> f64 {
        self.phase
    }

    /// Update configuration
    pub fn set_config(&mut self, config: DialecticConfig) {
        self.config = config;
    }

    /// Get configuration
    pub fn config(&self) -> &DialecticConfig {
        &self.config
    }

    /// Manually trigger synthesis for specific vertices
    pub fn force_synthesis(&mut self, vertices: &[usize]) {
        for &idx in vertices {
            if idx < self.gamma_activation.len() {
                self.gamma_activation[idx] = 1.0;
            }
        }
        self.synthesis_count += 1;
    }

    /// Get the "Phillips Synthesis" - the visual overlap pattern
    /// This represents the geometric intersection of Alpha and Beta projections
    pub fn phillips_synthesis(&self, geometry: &GeometryCore) -> Vec<(Vec4, f64)> {
        let cell24 = geometry.cell24();

        let mut synthesis_points = Vec::new();

        // For each Alpha vertex, check proximity to Beta vertices
        for (_ai, av) in cell24.alpha.transformed_vertices.iter().enumerate() {
            for (_bi, bv) in cell24.beta.transformed_vertices.iter().enumerate() {
                let dist = av.distance(*bv);

                if dist < self.config.synthesis_threshold * 2.0 {
                    // Midpoint represents synthesis
                    let midpoint = av.lerp(*bv, 0.5);
                    let strength = 1.0 - (dist / (self.config.synthesis_threshold * 2.0));

                    synthesis_points.push((midpoint, strength));
                }
            }
        }

        synthesis_points
    }
}

impl Default for DialecticEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// A dialectic proposition that can be processed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposition {
    /// Thesis representation (maps to Alpha vertices)
    pub thesis: Vec<f64>,
    /// Antithesis representation (maps to Beta vertices)
    pub antithesis: Vec<f64>,
    /// Optional label
    pub label: Option<String>,
}

impl Proposition {
    /// Create a new proposition
    pub fn new(thesis: Vec<f64>, antithesis: Vec<f64>) -> Self {
        Self {
            thesis,
            antithesis,
            label: None,
        }
    }

    /// Create with label
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Apply this proposition to a dialectic engine
    pub fn apply(&self, engine: &mut DialecticEngine) {
        engine.set_alpha_state(&self.thesis);
        engine.set_beta_state(&self.antithesis);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EngineConfig;

    #[test]
    fn test_dialectic_engine_creation() {
        let engine = DialecticEngine::new();
        assert_eq!(engine.synthesis_count(), 0);
        assert_eq!(engine.phase(), 0.0);
    }

    #[test]
    fn test_dialectic_config() {
        let config = DialecticConfig {
            synthesis_threshold: 0.3,
            ..Default::default()
        };
        let engine = DialecticEngine::with_config(config);
        assert!((engine.config().synthesis_threshold - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_proposition() {
        let prop = Proposition::new(
            vec![1.0, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 1.0],
        ).with_label("test");

        assert_eq!(prop.label, Some("test".to_string()));
    }
}
