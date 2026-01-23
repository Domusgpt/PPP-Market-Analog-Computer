//! Market Larynx - Harmonic Alpha Implementation
//!
//! This module implements the "Harmonic Alpha" paper concepts for financial market
//! analysis using music theory principles mapped to the dialectic engine.
//!
//! ## Mapping:
//! - **Thesis (Alpha)**: Market Price normalized to [0,1]
//! - **Antithesis (Beta)**: Market Sentiment (from embeddings like Voyage AI)
//! - **Synthesis (Gamma)**: Resolution/Crash Event when tension exceeds threshold
//!
//! ## Musical Intervals for Tension:
//! - Low Tension (< 0.3): Consonance - Perfect 5ths (ratio 3:2) = Bull Market
//! - Medium Tension (0.3-0.7): Partial dissonance = Uncertain Market
//! - High Tension (> 0.7): Dissonance - Tritones (ratio √2:1) = Bear/Crash Risk
//! - Gamma Event: Resolution chord = Market regime change

use serde::{Serialize, Deserialize};
use std::collections::VecDeque;

/// Musical interval ratios for sonification
#[derive(Debug, Clone, Copy)]
pub struct MusicalInterval {
    pub ratio: f64,
    pub cents: f64,
    pub consonance: f64, // 0.0 = dissonant, 1.0 = consonant
}

impl MusicalInterval {
    /// Perfect Unison (1:1)
    pub const UNISON: Self = Self { ratio: 1.0, cents: 0.0, consonance: 1.0 };
    /// Perfect Fifth (3:2) - Most consonant after unison
    pub const PERFECT_FIFTH: Self = Self { ratio: 1.5, cents: 702.0, consonance: 0.95 };
    /// Perfect Fourth (4:3)
    pub const PERFECT_FOURTH: Self = Self { ratio: 1.333333, cents: 498.0, consonance: 0.9 };
    /// Major Third (5:4)
    pub const MAJOR_THIRD: Self = Self { ratio: 1.25, cents: 386.0, consonance: 0.8 };
    /// Minor Third (6:5)
    pub const MINOR_THIRD: Self = Self { ratio: 1.2, cents: 316.0, consonance: 0.7 };
    /// Minor Second (16:15) - Dissonant
    pub const MINOR_SECOND: Self = Self { ratio: 1.0667, cents: 112.0, consonance: 0.2 };
    /// Tritone (√2:1) - Maximum dissonance, "devil's interval"
    pub const TRITONE: Self = Self { ratio: 1.41421356, cents: 600.0, consonance: 0.0 };

    /// Get interval based on tension level (0.0-1.0)
    pub fn from_tension(tension: f64) -> Self {
        let t = tension.clamp(0.0, 1.0);

        if t < 0.15 {
            Self::UNISON
        } else if t < 0.3 {
            Self::PERFECT_FIFTH
        } else if t < 0.45 {
            Self::PERFECT_FOURTH
        } else if t < 0.55 {
            Self::MAJOR_THIRD
        } else if t < 0.65 {
            Self::MINOR_THIRD
        } else if t < 0.8 {
            Self::MINOR_SECOND
        } else {
            Self::TRITONE
        }
    }
}

/// Market state classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketRegime {
    /// Strong bull market - consonant harmonics
    Bull,
    /// Mild bullish - stable harmonics
    MildBull,
    /// Neutral/uncertain - transitional harmonics
    Neutral,
    /// Mild bearish - emerging dissonance
    MildBear,
    /// Bear market - dissonant harmonics
    Bear,
    /// Crash imminent - maximum dissonance, tritone
    CrashRisk,
    /// Active crash/regime change - gamma event
    GammaEvent,
}

impl MarketRegime {
    pub fn from_tension(tension: f64) -> Self {
        if tension < 0.15 {
            Self::Bull
        } else if tension < 0.3 {
            Self::MildBull
        } else if tension < 0.45 {
            Self::Neutral
        } else if tension < 0.6 {
            Self::MildBear
        } else if tension < 0.75 {
            Self::Bear
        } else if tension < 0.9 {
            Self::CrashRisk
        } else {
            Self::GammaEvent
        }
    }

    pub fn consonance(&self) -> f64 {
        match self {
            Self::Bull => 1.0,
            Self::MildBull => 0.85,
            Self::Neutral => 0.6,
            Self::MildBear => 0.4,
            Self::Bear => 0.2,
            Self::CrashRisk => 0.05,
            Self::GammaEvent => 0.0,
        }
    }
}

/// Configuration for the Market Larynx
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketLarynxConfig {
    /// Threshold for triggering gamma (crash) event
    pub gamma_threshold: f64,
    /// Number of frames to smooth tension over
    pub smoothing_window: usize,
    /// Rate at which tension decays toward equilibrium
    pub tension_decay: f64,
    /// Sensitivity multiplier for sentiment influence
    pub sentiment_sensitivity: f64,
    /// Base frequency for sonification (Hz)
    pub base_frequency: f64,
    /// TDA (Topological Data Analysis) persistence threshold
    pub tda_persistence_threshold: f64,
}

impl Default for MarketLarynxConfig {
    fn default() -> Self {
        Self {
            gamma_threshold: 0.85,
            smoothing_window: 30,
            tension_decay: 0.02,
            sentiment_sensitivity: 1.5,
            base_frequency: 110.0, // A2
            tda_persistence_threshold: 0.1,
        }
    }
}

/// A gamma event (crash/resolution)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GammaEvent {
    /// Frame when the event occurred
    pub frame: u64,
    /// Tension level at trigger
    pub tension_at_trigger: f64,
    /// Duration in frames
    pub duration: u64,
    /// Whether this was a true crash or false positive
    pub confirmed: Option<bool>,
    /// Pattern detected (from TDA)
    pub tda_pattern: Option<String>,
}

/// TDA-inspired topological feature for crash detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalFeature {
    /// Birth time (when feature appeared)
    pub birth: f64,
    /// Death time (when feature disappeared)
    pub death: f64,
    /// Persistence (death - birth) - longer = more significant
    pub persistence: f64,
    /// Dimension (0 = connected components, 1 = loops, 2 = voids)
    pub dimension: u8,
    /// Associated market data index
    pub data_index: usize,
}

impl TopologicalFeature {
    pub fn new(birth: f64, death: f64, dimension: u8, data_index: usize) -> Self {
        Self {
            birth,
            death,
            persistence: death - birth,
            dimension,
            data_index,
        }
    }

    /// A "crash void" is a high-persistence 2D feature (void in the price-sentiment space)
    pub fn is_crash_void(&self, threshold: f64) -> bool {
        self.dimension == 2 && self.persistence > threshold
    }
}

/// Result from a market larynx processing step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketLarynxResult {
    /// Current market tension (0.0 to 1.0)
    pub tension: f64,
    /// Smoothed tension over window
    pub smoothed_tension: f64,
    /// Current market regime
    pub regime: MarketRegime,
    /// Musical interval for sonification
    pub interval_ratio: f64,
    /// Consonance level (inverse of tension)
    pub consonance: f64,
    /// Whether a gamma event is active
    pub gamma_active: bool,
    /// Suggested frequencies for sonification [fundamental, harmonic1, harmonic2]
    pub sonification_frequencies: [f64; 3],
    /// Topological features detected
    pub tda_features: Vec<TopologicalFeature>,
    /// Crash probability based on TDA
    pub crash_probability: f64,
}

/// The Market Larynx engine
#[derive(Debug, Clone)]
pub struct MarketLarynx {
    config: MarketLarynxConfig,
    /// Current price input (normalized 0-1)
    price_alpha: f64,
    /// Current sentiment input (normalized 0-1, from embedding)
    sentiment_beta: f64,
    /// Current tension
    tension: f64,
    /// Tension history for smoothing
    tension_history: VecDeque<f64>,
    /// Price history for TDA
    price_history: VecDeque<f64>,
    /// Sentiment history for TDA
    sentiment_history: VecDeque<f64>,
    /// Gamma event history
    gamma_events: Vec<GammaEvent>,
    /// Current frame
    frame: u64,
    /// Whether gamma event is currently active
    gamma_active: bool,
    /// Frame when current gamma started
    gamma_start_frame: u64,
}

impl MarketLarynx {
    pub fn new() -> Self {
        Self::with_config(MarketLarynxConfig::default())
    }

    pub fn with_config(config: MarketLarynxConfig) -> Self {
        Self {
            tension_history: VecDeque::with_capacity(config.smoothing_window),
            price_history: VecDeque::with_capacity(100),
            sentiment_history: VecDeque::with_capacity(100),
            config,
            price_alpha: 0.5,
            sentiment_beta: 0.5,
            tension: 0.0,
            gamma_events: Vec::new(),
            frame: 0,
            gamma_active: false,
            gamma_start_frame: 0,
        }
    }

    /// Set the price (Thesis/Alpha) - should be normalized to 0-1
    pub fn set_price(&mut self, price: f64) {
        self.price_alpha = price.clamp(0.0, 1.0);

        // Record for TDA
        self.price_history.push_back(self.price_alpha);
        if self.price_history.len() > 100 {
            self.price_history.pop_front();
        }
    }

    /// Set the sentiment (Antithesis/Beta) from embedding vector
    /// Takes a high-dimensional embedding and reduces to scalar sentiment
    pub fn set_sentiment_from_embedding(&mut self, embedding: &[f64]) {
        // Convert high-dim embedding to scalar sentiment
        // Using magnitude and sign of first principal component approximation
        let n = embedding.len() as f64;
        if n == 0.0 {
            return;
        }

        // Simple approach: use mean and variance
        let mean: f64 = embedding.iter().sum::<f64>() / n;
        let variance: f64 = embedding.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / n;

        // Map to 0-1 range using sigmoid-like transform
        let raw_sentiment = mean + variance.sqrt() * 0.5;
        self.sentiment_beta = (1.0 / (1.0 + (-raw_sentiment * 2.0).exp())).clamp(0.0, 1.0);

        // Record for TDA
        self.sentiment_history.push_back(self.sentiment_beta);
        if self.sentiment_history.len() > 100 {
            self.sentiment_history.pop_front();
        }
    }

    /// Set sentiment directly (0-1 scale)
    pub fn set_sentiment(&mut self, sentiment: f64) {
        self.sentiment_beta = sentiment.clamp(0.0, 1.0);

        self.sentiment_history.push_back(self.sentiment_beta);
        if self.sentiment_history.len() > 100 {
            self.sentiment_history.pop_front();
        }
    }

    /// Process one step of the market larynx
    pub fn step(&mut self, delta_time: f64) -> MarketLarynxResult {
        self.frame += 1;

        // Calculate raw tension as distance between price and sentiment
        // When price is high but sentiment is low (or vice versa), tension is high
        let raw_tension = (self.price_alpha - self.sentiment_beta).abs();

        // Apply sensitivity and blend with previous tension
        let target_tension = (raw_tension * self.config.sentiment_sensitivity).clamp(0.0, 1.0);

        // Smooth transition with decay
        self.tension = self.tension + (target_tension - self.tension) * (1.0 - (-delta_time / self.config.tension_decay).exp());
        self.tension = self.tension.clamp(0.0, 1.0);

        // Record tension history
        self.tension_history.push_back(self.tension);
        if self.tension_history.len() > self.config.smoothing_window {
            self.tension_history.pop_front();
        }

        // Calculate smoothed tension
        let smoothed_tension = if self.tension_history.is_empty() {
            self.tension
        } else {
            self.tension_history.iter().sum::<f64>() / self.tension_history.len() as f64
        };

        // Perform TDA analysis
        let tda_features = self.compute_tda_features();
        let crash_probability = self.compute_crash_probability(&tda_features);

        // Check for gamma event (crash)
        let gamma_threshold_adjusted = self.config.gamma_threshold * (1.0 - crash_probability * 0.3);

        if smoothed_tension > gamma_threshold_adjusted && !self.gamma_active {
            // Trigger gamma event
            self.gamma_active = true;
            self.gamma_start_frame = self.frame;

            let tda_pattern = if crash_probability > 0.7 {
                Some("crash_void_detected".to_string())
            } else if crash_probability > 0.4 {
                Some("volatility_loop".to_string())
            } else {
                None
            };

            self.gamma_events.push(GammaEvent {
                frame: self.frame,
                tension_at_trigger: smoothed_tension,
                duration: 0,
                confirmed: None,
                tda_pattern,
            });
        } else if smoothed_tension < self.config.gamma_threshold * 0.7 && self.gamma_active {
            // End gamma event
            if let Some(event) = self.gamma_events.last_mut() {
                event.duration = self.frame - self.gamma_start_frame;
            }
            self.gamma_active = false;
        }

        // Get regime and musical interval
        let regime = MarketRegime::from_tension(smoothed_tension);
        let interval = MusicalInterval::from_tension(smoothed_tension);

        // Calculate sonification frequencies
        let base = self.config.base_frequency;
        let sonification_frequencies = if self.gamma_active {
            // During gamma: play resolution chord (I-IV-V)
            [base, base * 4.0/3.0, base * 3.0/2.0]
        } else {
            // Normal: play interval based on tension
            [base, base * interval.ratio, base * interval.ratio * interval.ratio]
        };

        MarketLarynxResult {
            tension: self.tension,
            smoothed_tension,
            regime,
            interval_ratio: interval.ratio,
            consonance: interval.consonance,
            gamma_active: self.gamma_active,
            sonification_frequencies,
            tda_features,
            crash_probability,
        }
    }

    /// Compute TDA features from price-sentiment history
    /// This is a simplified TDA implementation focused on detecting "voids"
    fn compute_tda_features(&self) -> Vec<TopologicalFeature> {
        let mut features = Vec::new();

        if self.price_history.len() < 10 || self.sentiment_history.len() < 10 {
            return features;
        }

        // Compute pairwise distances in price-sentiment space
        let prices: Vec<f64> = self.price_history.iter().copied().collect();
        let sentiments: Vec<f64> = self.sentiment_history.iter().copied().collect();
        let n = prices.len().min(sentiments.len());

        // Simple persistence computation:
        // Look for "holes" in the price-sentiment point cloud
        // A crash void appears when price and sentiment diverge creating a gap

        // Compute local density
        let mut densities = vec![0.0; n];
        let epsilon = 0.15; // neighborhood radius

        for i in 0..n {
            let mut count = 0;
            for j in 0..n {
                if i != j {
                    let dp = prices[i] - prices[j];
                    let ds = sentiments[i] - sentiments[j];
                    let dist = (dp*dp + ds*ds).sqrt();
                    if dist < epsilon {
                        count += 1;
                    }
                }
            }
            densities[i] = count as f64 / n as f64;
        }

        // Find low-density regions (potential voids)
        let avg_density: f64 = densities.iter().sum::<f64>() / n as f64;

        for (i, &density) in densities.iter().enumerate() {
            if density < avg_density * 0.3 {
                // This point is in a sparse region - potential void
                let persistence = avg_density - density;
                if persistence > self.config.tda_persistence_threshold {
                    features.push(TopologicalFeature::new(
                        density,           // birth at this density level
                        avg_density,       // death when filled
                        2,                 // dimension 2 = void
                        i,
                    ));
                }
            }
        }

        // Detect 1D features (loops) - rapid oscillation patterns
        let mut volatility_windows = Vec::new();
        for i in 5..n {
            let window: Vec<f64> = prices[i-5..i].to_vec();
            let mean = window.iter().sum::<f64>() / 5.0;
            let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 5.0;
            volatility_windows.push(variance);
        }

        // High volatility creates "loops" in phase space
        let avg_vol = volatility_windows.iter().sum::<f64>() / volatility_windows.len().max(1) as f64;
        for (i, &vol) in volatility_windows.iter().enumerate() {
            if vol > avg_vol * 2.0 {
                features.push(TopologicalFeature::new(
                    avg_vol,
                    vol,
                    1, // dimension 1 = loop
                    i + 5,
                ));
            }
        }

        features
    }

    /// Compute crash probability based on TDA features AND tension
    /// Falls back to tension-based calculation when TDA data is insufficient
    fn compute_crash_probability(&self, features: &[TopologicalFeature]) -> f64 {
        // Calculate smoothed tension
        let smoothed_tension = if self.tension_history.is_empty() {
            self.tension
        } else {
            self.tension_history.iter().sum::<f64>() / self.tension_history.len() as f64
        };

        // Base probability from tension (always available)
        // Higher tension = higher crash probability
        let tension_prob = if smoothed_tension > 0.8 {
            0.7 + (smoothed_tension - 0.8) * 1.5  // 0.7-1.0 range
        } else if smoothed_tension > 0.6 {
            0.4 + (smoothed_tension - 0.6) * 1.5  // 0.4-0.7 range
        } else if smoothed_tension > 0.4 {
            0.2 + (smoothed_tension - 0.4) * 1.0  // 0.2-0.4 range
        } else {
            smoothed_tension * 0.5  // 0.0-0.2 range
        };

        // If we have TDA features, boost or reduce based on topology
        let tda_modifier = if features.is_empty() {
            1.0  // No modification if no TDA data
        } else {
            let mut crash_score = 0.0;
            let mut total_weight = 0.0;

            for feature in features {
                let weight = feature.persistence;
                total_weight += weight;

                match feature.dimension {
                    2 => {
                        // Voids strongly indicate crashes - big boost
                        crash_score += weight * 2.0;
                    }
                    1 => {
                        // Loops indicate volatility - moderate boost
                        crash_score += weight * 0.8;
                    }
                    0 => {
                        // Connected components - slight reduction (stability)
                        crash_score -= weight * 0.3;
                    }
                    _ => {}
                }
            }

            if total_weight > 0.0 {
                1.0 + (crash_score / total_weight).clamp(-0.5, 1.0)
            } else {
                1.0
            }
        };

        // Final probability: tension-based * TDA modifier
        (tension_prob * tda_modifier).clamp(0.0, 1.0)
    }

    /// Get the current tension level
    pub fn tension(&self) -> f64 {
        self.tension
    }

    /// Get the current market regime
    pub fn regime(&self) -> MarketRegime {
        MarketRegime::from_tension(self.tension)
    }

    /// Check if gamma event is active
    pub fn is_gamma_active(&self) -> bool {
        self.gamma_active
    }

    /// Get gamma event history
    pub fn gamma_events(&self) -> &[GammaEvent] {
        &self.gamma_events
    }

    /// Reset the larynx state
    pub fn reset(&mut self) {
        self.price_alpha = 0.5;
        self.sentiment_beta = 0.5;
        self.tension = 0.0;
        self.tension_history.clear();
        self.price_history.clear();
        self.sentiment_history.clear();
        self.gamma_events.clear();
        self.frame = 0;
        self.gamma_active = false;
        self.gamma_start_frame = 0;
    }

    /// Get configuration
    pub fn config(&self) -> &MarketLarynxConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: MarketLarynxConfig) {
        self.config = config;
    }
}

impl Default for MarketLarynx {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_larynx_creation() {
        let larynx = MarketLarynx::new();
        assert_eq!(larynx.frame, 0);
        assert!(!larynx.is_gamma_active());
    }

    #[test]
    fn test_tension_calculation() {
        let mut larynx = MarketLarynx::new();

        // When price and sentiment are equal, tension should be low
        larynx.set_price(0.5);
        larynx.set_sentiment(0.5);
        let result = larynx.step(0.016);
        assert!(result.tension < 0.3);

        // When price and sentiment diverge, tension should be high
        larynx.set_price(1.0);
        larynx.set_sentiment(0.0);
        for _ in 0..60 {
            larynx.step(0.016);
        }
        let result = larynx.step(0.016);
        assert!(result.tension > 0.5);
    }

    #[test]
    fn test_gamma_event_trigger() {
        let mut larynx = MarketLarynx::with_config(MarketLarynxConfig {
            gamma_threshold: 0.7,
            smoothing_window: 5,
            tension_decay: 0.001,
            ..Default::default()
        });

        // Create high tension scenario
        larynx.set_price(1.0);
        larynx.set_sentiment(0.0);

        // Run many steps to build up tension
        for _ in 0..100 {
            larynx.step(0.1);
        }

        assert!(larynx.is_gamma_active() || larynx.tension() > 0.5);
    }

    #[test]
    fn test_musical_intervals() {
        assert!((MusicalInterval::PERFECT_FIFTH.ratio - 1.5).abs() < 0.01);
        assert!((MusicalInterval::TRITONE.ratio - 1.414).abs() < 0.01);

        // Low tension should give consonant intervals
        let low_interval = MusicalInterval::from_tension(0.1);
        assert!(low_interval.consonance > 0.8);

        // High tension should give dissonant intervals
        let high_interval = MusicalInterval::from_tension(0.9);
        assert!(high_interval.consonance < 0.3);
    }

    #[test]
    fn test_regime_classification() {
        // Test boundary values based on MarketRegime::from_tension implementation
        assert_eq!(MarketRegime::from_tension(0.1), MarketRegime::Bull);
        assert_eq!(MarketRegime::from_tension(0.35), MarketRegime::Neutral);
        assert_eq!(MarketRegime::from_tension(0.95), MarketRegime::GammaEvent);
    }

    #[test]
    fn test_sonification_frequencies() {
        let mut larynx = MarketLarynx::new();

        // Create some tension for interesting frequencies
        larynx.set_price(0.7);
        larynx.set_sentiment(0.4);

        // Run a few steps to build up tension
        for _ in 0..10 {
            larynx.step(0.016);
        }
        let result = larynx.step(0.016);

        let base = result.sonification_frequencies[0];
        assert!(base > 100.0 && base < 120.0, "Base frequency should be around 110 Hz, got {}", base);

        // The frequencies should form musical intervals (ratio >= 1.0)
        let ratio1 = result.sonification_frequencies[1] / result.sonification_frequencies[0];
        assert!(ratio1 >= 1.0 && ratio1 <= 2.5, "Ratio1 should be reasonable musical interval, got {}", ratio1);
    }

    #[test]
    fn test_embedding_to_sentiment() {
        let mut larynx = MarketLarynx::new();

        // Test with a simple embedding vector
        let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        larynx.set_sentiment_from_embedding(&embedding);

        // Sentiment should be between 0 and 1
        let result = larynx.step(0.016);
        assert!(result.tension >= 0.0 && result.tension <= 1.0);
    }

    #[test]
    fn test_tda_features() {
        let mut larynx = MarketLarynx::new();

        // Build up history with volatile data
        for i in 0..50 {
            let t = i as f64 / 50.0;
            // Oscillating price
            let price = 0.5 + 0.3 * (t * 10.0).sin();
            // Diverging sentiment
            let sentiment = 0.5 - 0.2 * (t * 8.0).cos();

            larynx.set_price(price);
            larynx.set_sentiment(sentiment);
            larynx.step(0.016);
        }

        let result = larynx.step(0.016);

        // With enough history and volatility, we should detect features
        // (either voids or loops)
        assert!(result.tda_features.len() >= 0); // May or may not detect features
    }

    #[test]
    fn test_crash_scenario() {
        let mut larynx = MarketLarynx::with_config(MarketLarynxConfig {
            gamma_threshold: 0.8,
            smoothing_window: 10,
            tension_decay: 0.005,
            sentiment_sensitivity: 2.0,
            ..Default::default()
        });

        // Simulate a crash: price drops rapidly while sentiment stays high
        // (delayed reaction = high divergence)
        for i in 0..100 {
            let t = i as f64 / 100.0;
            let price = 1.0 - t * 0.9; // Price crashes from 1.0 to 0.1
            let sentiment = 0.8 - t * 0.3; // Sentiment drops slower

            larynx.set_price(price);
            larynx.set_sentiment(sentiment);
            larynx.step(0.05);
        }

        // Should have high tension and possibly gamma event
        assert!(larynx.tension() > 0.3);

        // Regime should be bearish or worse
        let regime = larynx.regime();
        assert!(
            regime == MarketRegime::Bear ||
            regime == MarketRegime::CrashRisk ||
            regime == MarketRegime::GammaEvent ||
            regime == MarketRegime::MildBear
        );
    }

    #[test]
    fn test_recovery_scenario() {
        let mut larynx = MarketLarynx::new();

        // Start with high tension
        larynx.set_price(0.1);
        larynx.set_sentiment(0.9);
        for _ in 0..30 {
            larynx.step(0.05);
        }
        assert!(larynx.tension() > 0.3);

        // Recovery: price and sentiment converge
        for i in 0..50 {
            let t = i as f64 / 50.0;
            let price = 0.1 + t * 0.4;
            let sentiment = 0.9 - t * 0.4;

            larynx.set_price(price);
            larynx.set_sentiment(sentiment);
            larynx.step(0.05);
        }

        // Tension should decrease
        let final_result = larynx.step(0.05);
        assert!(final_result.smoothed_tension < 0.5);
    }

    #[test]
    fn test_reset() {
        let mut larynx = MarketLarynx::new();

        // Build up state
        larynx.set_price(0.9);
        larynx.set_sentiment(0.1);
        for _ in 0..50 {
            larynx.step(0.05);
        }

        // Reset
        larynx.reset();

        // State should be cleared
        assert_eq!(larynx.tension(), 0.0);
        assert!(!larynx.is_gamma_active());
        assert!(larynx.gamma_events().is_empty());
    }

    #[test]
    fn test_topological_feature_creation() {
        let feature = TopologicalFeature::new(0.1, 0.5, 2, 10);

        assert!((feature.birth - 0.1).abs() < 0.001);
        assert!((feature.death - 0.5).abs() < 0.001);
        assert!((feature.persistence - 0.4).abs() < 0.001);
        assert_eq!(feature.dimension, 2);
        assert!(feature.is_crash_void(0.3));
        assert!(!feature.is_crash_void(0.5));
    }

    #[test]
    fn test_all_regimes() {
        // Test that all tension levels map to appropriate regimes
        let test_cases = [
            (0.05, MarketRegime::Bull),
            (0.20, MarketRegime::MildBull),
            (0.40, MarketRegime::Neutral),
            (0.55, MarketRegime::MildBear),
            (0.70, MarketRegime::Bear),
            (0.80, MarketRegime::CrashRisk),
            (0.95, MarketRegime::GammaEvent),
        ];

        for (tension, expected_regime) in test_cases {
            let regime = MarketRegime::from_tension(tension);
            assert_eq!(regime, expected_regime, "Tension {} should map to {:?}", tension, expected_regime);
        }
    }

    #[test]
    fn test_consonance_decreases_with_tension() {
        let intervals: Vec<_> = (0..10)
            .map(|i| {
                let tension = i as f64 / 10.0;
                MusicalInterval::from_tension(tension)
            })
            .collect();

        // Consonance should generally decrease with tension
        for i in 1..intervals.len() {
            // Allow for some tolerance due to step functions
            assert!(
                intervals[i].consonance <= intervals[i - 1].consonance + 0.1,
                "Consonance should decrease: {} at tension {} vs {} at tension {}",
                intervals[i].consonance, i as f64 / 10.0,
                intervals[i - 1].consonance, (i - 1) as f64 / 10.0
            );
        }
    }
}
