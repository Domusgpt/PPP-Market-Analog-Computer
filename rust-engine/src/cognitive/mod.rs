//! Cognitive Layer - Triadic Dialectic Reasoning & Rule Engine
//!
//! This module implements the higher-level cognition logic using the 24-cell
//! Trinity decomposition for thesis-antithesis-synthesis reasoning.
//!
//! ## Harmonic Alpha Extension
//! The Market Larynx module extends this system to financial market analysis,
//! mapping price/sentiment dynamics to the dialectic framework.

mod trinity;
mod rules;
mod dialectic;
mod market_larynx;

pub use trinity::TrinityState;
pub use rules::{CognitiveRule, RuleEngine};
pub use dialectic::DialecticEngine;
pub use market_larynx::{
    MarketLarynx, MarketLarynxConfig, MarketLarynxResult,
    MarketRegime, GammaEvent, TopologicalFeature, MusicalInterval,
};

use crate::geometry::GeometryCore;

/// The cognitive layer manages reasoning over geometric states
pub struct CognitiveLayer {
    /// Trinity state tracker
    trinity_state: TrinityState,
    /// Dialectic engine for synthesis detection
    dialectic: DialecticEngine,
    /// Rule engine for procedural logic
    rules: RuleEngine,
    /// Market Larynx for financial market analysis (Harmonic Alpha)
    market_larynx: MarketLarynx,
    /// Whether automatic synthesis is enabled
    auto_synthesis: bool,
    /// Whether market analysis mode is enabled
    market_mode: bool,
}

impl CognitiveLayer {
    pub fn new() -> Self {
        Self {
            trinity_state: TrinityState::new(),
            dialectic: DialecticEngine::new(),
            rules: RuleEngine::new(),
            market_larynx: MarketLarynx::new(),
            auto_synthesis: true,
            market_mode: false,
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

        // Process market larynx if in market mode
        if self.market_mode {
            let market_result = self.market_larynx.step(delta_time);

            // If gamma event (crash) is detected, force synthesis
            if market_result.gamma_active {
                self.dialectic.force_synthesis(&[0, 1, 2, 3, 4, 5, 6, 7]);
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

    /// Enable/disable market analysis mode (Harmonic Alpha)
    pub fn set_market_mode(&mut self, enabled: bool) {
        self.market_mode = enabled;
    }

    /// Check if market mode is enabled
    pub fn is_market_mode(&self) -> bool {
        self.market_mode
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

    /// Get the market larynx
    pub fn market_larynx(&self) -> &MarketLarynx {
        &self.market_larynx
    }

    /// Get mutable market larynx
    pub fn market_larynx_mut(&mut self) -> &mut MarketLarynx {
        &mut self.market_larynx
    }

    /// Set market price (Thesis/Alpha) for Harmonic Alpha analysis
    pub fn set_market_price(&mut self, price: f64) {
        self.market_larynx.set_price(price);
    }

    /// Set market sentiment from embedding (Antithesis/Beta)
    pub fn set_market_sentiment_embedding(&mut self, embedding: &[f64]) {
        self.market_larynx.set_sentiment_from_embedding(embedding);
    }

    /// Set market sentiment directly (0-1 scale)
    pub fn set_market_sentiment(&mut self, sentiment: f64) {
        self.market_larynx.set_sentiment(sentiment);
    }

    /// Get current market tension
    pub fn market_tension(&self) -> f64 {
        self.market_larynx.tension()
    }

    /// Get current market regime
    pub fn market_regime(&self) -> MarketRegime {
        self.market_larynx.regime()
    }

    /// Check if market gamma (crash) event is active
    pub fn is_market_gamma_active(&self) -> bool {
        self.market_larynx.is_gamma_active()
    }
}

impl Default for CognitiveLayer {
    fn default() -> Self {
        Self::new()
    }
}
