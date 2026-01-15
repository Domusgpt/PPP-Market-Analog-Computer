//! Data Mapper - Maps input channels to geometric parameters
//!
//! The mapper transforms raw data values into parameters that control
//! the geometric simulation: rotation angles, activations, scales, etc.

use serde::{Serialize, Deserialize};

/// Target for data mapping
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MappingTarget {
    /// Rotation in XY plane
    RotationXY,
    /// Rotation in XZ plane
    RotationXZ,
    /// Rotation in XW plane
    RotationXW,
    /// Rotation in YZ plane
    RotationYZ,
    /// Rotation in YW plane
    RotationYW,
    /// Rotation in ZW plane
    RotationZW,
    /// Alpha (thesis) activation level
    AlphaActivation,
    /// Beta (antithesis) activation level
    BetaActivation,
    /// Gamma (synthesis) activation level
    GammaActivation,
    /// Global scale
    Scale,
    /// E8 layer phase offset
    E8PhaseOffset,
    /// Projection focal distance
    FocalDistance,
    /// Blend alpha
    BlendAlpha,
    /// No mapping (ignore)
    None,
}

impl Default for MappingTarget {
    fn default() -> Self {
        Self::None
    }
}

/// Mapping function type
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MappingFunction {
    /// Linear mapping: output = input
    Linear,
    /// Quadratic: output = input²
    Quadratic,
    /// Square root: output = √input
    SquareRoot,
    /// Exponential: output = e^(input-0.5) normalized
    Exponential,
    /// Logarithmic: output = log(input+1) normalized
    Logarithmic,
    /// Sine: output = sin(input * π)
    Sine,
    /// Step function at threshold
    Step(f64),
    /// Smooth step (hermite interpolation)
    SmoothStep,
}

impl Default for MappingFunction {
    fn default() -> Self {
        Self::Linear
    }
}

impl MappingFunction {
    /// Apply the mapping function
    pub fn apply(&self, input: f64) -> f64 {
        let x = input.clamp(0.0, 1.0);

        match self {
            Self::Linear => x,
            Self::Quadratic => x * x,
            Self::SquareRoot => x.sqrt(),
            Self::Exponential => {
                let e = std::f64::consts::E;
                (e.powf(x - 0.5) - e.powf(-0.5)) / (e.powf(0.5) - e.powf(-0.5))
            }
            Self::Logarithmic => (x + 1.0).ln() / 2.0_f64.ln(),
            Self::Sine => (x * std::f64::consts::PI).sin(),
            Self::Step(threshold) => if x >= *threshold { 1.0 } else { 0.0 },
            Self::SmoothStep => x * x * (3.0 - 2.0 * x),
        }
    }
}

/// Configuration for a single mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MappingConfig {
    /// Source channel index
    pub source: usize,
    /// Target parameter
    pub target: MappingTarget,
    /// Mapping function
    pub function: MappingFunction,
    /// Output range minimum
    pub out_min: f64,
    /// Output range maximum
    pub out_max: f64,
    /// Invert the output
    pub invert: bool,
}

impl Default for MappingConfig {
    fn default() -> Self {
        Self {
            source: 0,
            target: MappingTarget::None,
            function: MappingFunction::Linear,
            out_min: 0.0,
            out_max: 1.0,
            invert: false,
        }
    }
}

impl MappingConfig {
    pub fn new(source: usize, target: MappingTarget) -> Self {
        Self {
            source,
            target,
            ..Default::default()
        }
    }

    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.out_min = min;
        self.out_max = max;
        self
    }

    pub fn with_function(mut self, func: MappingFunction) -> Self {
        self.function = func;
        self
    }

    /// Apply this mapping to an input value
    pub fn apply(&self, input: f64) -> f64 {
        let normalized = self.function.apply(input);
        let scaled = self.out_min + normalized * (self.out_max - self.out_min);
        if self.invert { self.out_max - scaled + self.out_min } else { scaled }
    }
}

/// The data mapper
pub struct DataMapper {
    mappings: Vec<MappingConfig>,
    output: Vec<f64>,
}

impl DataMapper {
    pub fn new(channel_count: usize) -> Self {
        // Create default mappings for first 6 channels to rotations
        let mut mappings = vec![
            MappingConfig::new(0, MappingTarget::RotationXY)
                .with_range(0.0, std::f64::consts::TAU),
            MappingConfig::new(1, MappingTarget::RotationXZ)
                .with_range(0.0, std::f64::consts::TAU),
            MappingConfig::new(2, MappingTarget::RotationXW)
                .with_range(0.0, std::f64::consts::TAU),
            MappingConfig::new(3, MappingTarget::RotationYZ)
                .with_range(0.0, std::f64::consts::TAU),
            MappingConfig::new(4, MappingTarget::RotationYW)
                .with_range(0.0, std::f64::consts::TAU),
            MappingConfig::new(5, MappingTarget::RotationZW)
                .with_range(0.0, std::f64::consts::TAU),
        ];

        // Add activation mappings if enough channels
        if channel_count > 6 {
            mappings.push(MappingConfig::new(6, MappingTarget::AlphaActivation));
            mappings.push(MappingConfig::new(7, MappingTarget::BetaActivation));
            mappings.push(MappingConfig::new(8, MappingTarget::GammaActivation));
        }

        Self {
            mappings,
            output: vec![0.0; channel_count],
        }
    }

    /// Map input values to output values
    pub fn map(&mut self, input: &[f64]) -> Vec<f64> {
        self.output.resize(input.len(), 0.0);

        for (i, v) in input.iter().enumerate() {
            // Find mapping for this channel
            if let Some(mapping) = self.mappings.iter().find(|m| m.source == i) {
                self.output[i] = mapping.apply(*v);
            } else {
                // Pass through if no mapping
                self.output[i] = *v;
            }
        }

        self.output.clone()
    }

    /// Add or replace a mapping
    pub fn set_mapping(&mut self, config: MappingConfig) {
        // Remove existing mapping for the same source
        self.mappings.retain(|m| m.source != config.source);
        self.mappings.push(config);
    }

    /// Remove mapping for a source channel
    pub fn remove_mapping(&mut self, source: usize) {
        self.mappings.retain(|m| m.source != source);
    }

    /// Get all mappings
    pub fn mappings(&self) -> &[MappingConfig] {
        &self.mappings
    }

    /// Get mapped value by target
    pub fn get_by_target(&self, target: MappingTarget, input: &[f64]) -> Option<f64> {
        self.mappings.iter()
            .find(|m| m.target == target)
            .and_then(|m| input.get(m.source))
            .map(|&v| {
                self.mappings.iter()
                    .find(|m| m.target == target)
                    .map(|m| m.apply(v))
                    .unwrap_or(v)
            })
    }

    /// Create a preset mapping configuration
    pub fn from_preset(preset: MapperPreset, channel_count: usize) -> Self {
        let mut mapper = Self::new(channel_count);

        match preset {
            MapperPreset::Default => {
                // Already configured in new()
            }
            MapperPreset::AudioReactive => {
                // Map frequency bands to different rotations
                mapper.set_mapping(MappingConfig::new(0, MappingTarget::RotationXY)
                    .with_range(0.0, std::f64::consts::PI)
                    .with_function(MappingFunction::SquareRoot));
                mapper.set_mapping(MappingConfig::new(1, MappingTarget::RotationZW)
                    .with_range(0.0, std::f64::consts::PI)
                    .with_function(MappingFunction::Quadratic));
                mapper.set_mapping(MappingConfig::new(2, MappingTarget::Scale)
                    .with_range(0.8, 1.2)
                    .with_function(MappingFunction::SmoothStep));
            }
            MapperPreset::IMUSensor => {
                // Map 6 DOF to all rotation planes
                for i in 0..6 {
                    let target = match i {
                        0 => MappingTarget::RotationXY,
                        1 => MappingTarget::RotationXZ,
                        2 => MappingTarget::RotationXW,
                        3 => MappingTarget::RotationYZ,
                        4 => MappingTarget::RotationYW,
                        _ => MappingTarget::RotationZW,
                    };
                    mapper.set_mapping(MappingConfig::new(i, target)
                        .with_range(-std::f64::consts::PI, std::f64::consts::PI));
                }
            }
            MapperPreset::Triadic => {
                // Focus on Alpha/Beta/Gamma activations
                mapper.set_mapping(MappingConfig::new(0, MappingTarget::AlphaActivation));
                mapper.set_mapping(MappingConfig::new(1, MappingTarget::BetaActivation));
                mapper.set_mapping(MappingConfig::new(2, MappingTarget::GammaActivation));
                mapper.set_mapping(MappingConfig::new(3, MappingTarget::RotationXY)
                    .with_range(0.0, std::f64::consts::TAU));
            }
        }

        mapper
    }
}

/// Preset configurations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MapperPreset {
    Default,
    AudioReactive,
    IMUSensor,
    Triadic,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mapping_function_linear() {
        let func = MappingFunction::Linear;
        assert!((func.apply(0.5) - 0.5).abs() < 0.01);
        assert!((func.apply(0.0) - 0.0).abs() < 0.01);
        assert!((func.apply(1.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_mapping_config() {
        let config = MappingConfig::new(0, MappingTarget::RotationXY)
            .with_range(0.0, std::f64::consts::PI);

        let result = config.apply(1.0);
        assert!((result - std::f64::consts::PI).abs() < 0.01);
    }

    #[test]
    fn test_data_mapper() {
        let mut mapper = DataMapper::new(10);
        let input = vec![0.5; 10];
        let output = mapper.map(&input);

        assert_eq!(output.len(), 10);
    }
}
