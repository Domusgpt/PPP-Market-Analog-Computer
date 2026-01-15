//! Cognitive Layer - Triadic Dialectic Reasoning & Rule Engine
//!
//! This module implements the higher-level cognition logic using the 24-cell
//! Trinity decomposition for thesis-antithesis-synthesis reasoning.

mod trinity;
mod rules;
mod dialectic;

pub use trinity::TrinityState;
pub use rules::{CognitiveRule, RuleEngine};
pub use dialectic::DialecticEngine;

use crate::geometry::GeometryCore;

/// The cognitive layer manages reasoning over geometric states
pub struct CognitiveLayer {
    /// Trinity state tracker
    trinity_state: TrinityState,
    /// Dialectic engine for synthesis detection
    dialectic: DialecticEngine,
    /// Rule engine for procedural logic
    rules: RuleEngine,
    /// Whether automatic synthesis is enabled
    auto_synthesis: bool,
}

impl CognitiveLayer {
    pub fn new() -> Self {
        Self {
            trinity_state: TrinityState::new(),
            dialectic: DialecticEngine::new(),
            rules: RuleEngine::new(),
            auto_synthesis: true,
        }
    }

    /// Process one cognitive step
    pub fn process(&mut self, geometry: &mut GeometryCore, delta_time: f64) {
        // Update Trinity state from geometry
        self.trinity_state.update_from_geometry(geometry);

        // Run dialectic detection
        if self.auto_synthesis {
            if let Some(synthesis) = self.dialectic.detect_synthesis(geometry) {
                self.handle_synthesis(geometry, synthesis);
            }
        }

        // Execute rules
        self.rules.execute(geometry, &self.trinity_state, delta_time);

        // Update animation
        geometry.update_animation(delta_time);
    }

    /// Handle detected synthesis
    fn handle_synthesis(&mut self, _geometry: &mut GeometryCore, activated: Vec<usize>) {
        log::debug!("Synthesis detected: {} Gamma vertices activated", activated.len());

        // Record the synthesis event
        self.trinity_state.record_synthesis(&activated);

        // Could trigger mode transitions, feedback loops, etc.
    }

    /// Enable/disable automatic synthesis detection
    pub fn set_auto_synthesis(&mut self, enabled: bool) {
        self.auto_synthesis = enabled;
    }

    /// Add a cognitive rule
    pub fn add_rule(&mut self, rule: CognitiveRule) {
        self.rules.add_rule(rule);
    }

    /// Get current Trinity state
    pub fn trinity_state(&self) -> &TrinityState {
        &self.trinity_state
    }

    /// Get the dialectic engine
    pub fn dialectic(&self) -> &DialecticEngine {
        &self.dialectic
    }

    /// Get mutable dialectic engine
    pub fn dialectic_mut(&mut self) -> &mut DialecticEngine {
        &mut self.dialectic
    }
}

impl Default for CognitiveLayer {
    fn default() -> Self {
        Self::new()
    }
}
