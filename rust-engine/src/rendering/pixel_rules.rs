//! Pixel-level analog computation rules
//!
//! These rules define how the GPU processes the rendered image to perform
//! analog computation through visual interference and blending.

use serde::{Serialize, Deserialize};

/// Available pixel processing rules
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PixelRule {
    /// No post-processing, pass through
    PassThrough,
    /// Alpha blending of overlapping layers
    AlphaBlend,
    /// Threshold to create binary pattern
    Threshold,
    /// Glow enhancement for overlapping regions
    GlowEnhance,
    /// Edge detection (Sobel)
    EdgeDetect,
    /// Cellular automaton (Conway-like)
    CellularAutomaton,
    /// Diffusion spreading
    Diffusion,
    /// Interference pattern enhancement
    InterferenceEnhance,
    /// Custom rule by ID
    Custom(u32),
}

impl PixelRule {
    /// Get the rule ID for shader dispatch
    pub fn shader_id(&self) -> u32 {
        match self {
            Self::PassThrough => 0,
            Self::AlphaBlend => 0, // Handled by blend state
            Self::Threshold => 1,
            Self::GlowEnhance => 2,
            Self::EdgeDetect => 3,
            Self::CellularAutomaton => 1, // Compute shader rule 1
            Self::Diffusion => 2,          // Compute shader rule 2
            Self::InterferenceEnhance => 3, // Compute shader rule 3
            Self::Custom(id) => *id,
        }
    }

    /// Whether this rule uses compute shaders
    pub fn uses_compute(&self) -> bool {
        matches!(
            self,
            Self::CellularAutomaton | Self::Diffusion | Self::InterferenceEnhance
        )
    }

    /// Whether this rule requires the post-process pass
    pub fn uses_postprocess(&self) -> bool {
        matches!(
            self,
            Self::Threshold | Self::GlowEnhance | Self::EdgeDetect
        )
    }

    /// Get description of what this rule does
    pub fn description(&self) -> &'static str {
        match self {
            Self::PassThrough => "No processing, render directly",
            Self::AlphaBlend => "Blend overlapping layers with alpha transparency",
            Self::Threshold => "Convert to binary pattern based on brightness threshold",
            Self::GlowEnhance => "Enhance brightness where layers overlap (Phillips synthesis)",
            Self::EdgeDetect => "Detect edges using Sobel operator",
            Self::CellularAutomaton => "Apply Conway's Game of Life rules to bright pixels",
            Self::Diffusion => "Spread activation to neighboring pixels",
            Self::InterferenceEnhance => "Enhance interference patterns from overlapping projections",
            Self::Custom(_) => "Custom user-defined rule",
        }
    }
}

impl Default for PixelRule {
    fn default() -> Self {
        Self::AlphaBlend
    }
}

/// Configuration for pixel processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PixelConfig {
    /// Primary rule to apply
    pub primary_rule: PixelRule,
    /// Secondary rules (applied in order)
    pub secondary_rules: Vec<PixelRule>,
    /// Threshold value for threshold rule (0.0 to 1.0)
    pub threshold: f32,
    /// Glow intensity for glow rule
    pub glow_intensity: f32,
    /// Number of iterations for iterative rules
    pub iterations: u32,
    /// Blend factor for rule output
    pub blend_factor: f32,
}

impl Default for PixelConfig {
    fn default() -> Self {
        Self {
            primary_rule: PixelRule::AlphaBlend,
            secondary_rules: vec![],
            threshold: 0.5,
            glow_intensity: 0.3,
            iterations: 1,
            blend_factor: 1.0,
        }
    }
}

impl PixelConfig {
    /// Create config for thesis-antithesis overlap detection
    pub fn for_synthesis_detection() -> Self {
        Self {
            primary_rule: PixelRule::AlphaBlend,
            secondary_rules: vec![PixelRule::GlowEnhance, PixelRule::Threshold],
            threshold: 0.6,
            glow_intensity: 0.5,
            iterations: 1,
            blend_factor: 1.0,
        }
    }

    /// Create config for pattern visualization
    pub fn for_visualization() -> Self {
        Self {
            primary_rule: PixelRule::GlowEnhance,
            secondary_rules: vec![],
            threshold: 0.5,
            glow_intensity: 0.4,
            iterations: 1,
            blend_factor: 1.0,
        }
    }

    /// Create config for analog computation
    pub fn for_computation() -> Self {
        Self {
            primary_rule: PixelRule::InterferenceEnhance,
            secondary_rules: vec![PixelRule::Diffusion],
            threshold: 0.5,
            glow_intensity: 0.3,
            iterations: 3,
            blend_factor: 0.8,
        }
    }
}

/// Result of pixel analysis
#[derive(Debug, Clone, Default)]
pub struct PixelAnalysis {
    /// Total brightness (sum of all pixels)
    pub total_brightness: f64,
    /// Average brightness
    pub average_brightness: f64,
    /// Maximum brightness
    pub max_brightness: f64,
    /// Number of pixels above threshold
    pub active_pixels: u32,
    /// Center of mass (brightness-weighted centroid)
    pub center_of_mass: [f64; 2],
    /// Detected overlap regions
    pub overlap_regions: Vec<OverlapRegion>,
}

/// An identified region of overlap between layers
#[derive(Debug, Clone)]
pub struct OverlapRegion {
    /// Center position (normalized 0-1)
    pub center: [f64; 2],
    /// Approximate radius
    pub radius: f64,
    /// Intensity (0-1)
    pub intensity: f64,
    /// Which layers are involved (by ID)
    pub layers: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_rule_shader_id() {
        assert_eq!(PixelRule::PassThrough.shader_id(), 0);
        assert_eq!(PixelRule::Threshold.shader_id(), 1);
        assert_eq!(PixelRule::Custom(42).shader_id(), 42);
    }

    #[test]
    fn test_pixel_config_presets() {
        let synthesis_config = PixelConfig::for_synthesis_detection();
        assert!(synthesis_config.secondary_rules.contains(&PixelRule::GlowEnhance));

        let viz_config = PixelConfig::for_visualization();
        assert_eq!(viz_config.primary_rule, PixelRule::GlowEnhance);
    }
}
