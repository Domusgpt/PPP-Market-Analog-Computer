//! Pattern Detection - Recognizing meaningful geometric configurations
//!
//! Detects known patterns in the polytope state that correspond to
//! specific cognitive or data conditions.

use crate::geometry::{GeometryCore, Vec4, Polytope4D};
use serde::{Serialize, Deserialize};

/// A detected pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    pub name: String,
    pub confidence: f64,
    pub description: String,
    pub location: Option<[f64; 4]>,
}

/// Pattern detector
pub struct PatternDetector {
    /// Tolerance for pattern matching
    pub tolerance: f64,
    /// Known pattern definitions
    patterns: Vec<PatternDefinition>,
}

/// Definition of a recognizable pattern
#[derive(Debug, Clone)]
struct PatternDefinition {
    name: String,
    description: String,
    detector: fn(&GeometryCore, f64) -> Option<f64>,
}

impl PatternDetector {
    pub fn new() -> Self {
        let mut detector = Self {
            tolerance: 0.1,
            patterns: Vec::new(),
        };

        // Register built-in patterns
        detector.register_builtin_patterns();

        detector
    }

    fn register_builtin_patterns(&mut self) {
        // Synthesis alignment pattern
        self.patterns.push(PatternDefinition {
            name: "synthesis_alignment".to_string(),
            description: "Alpha and Beta vertices align through Gamma".to_string(),
            detector: detect_synthesis_alignment,
        });

        // Resonance pattern
        self.patterns.push(PatternDefinition {
            name: "resonance".to_string(),
            description: "All Trinity components equally activated".to_string(),
            detector: detect_resonance,
        });

        // Symmetry pattern
        self.patterns.push(PatternDefinition {
            name: "high_symmetry".to_string(),
            description: "High rotational symmetry detected".to_string(),
            detector: detect_high_symmetry,
        });

        // Cluster formation
        self.patterns.push(PatternDefinition {
            name: "clustering".to_string(),
            description: "Vertices forming distinct clusters".to_string(),
            detector: detect_clustering,
        });

        // Expansion pattern
        self.patterns.push(PatternDefinition {
            name: "expansion".to_string(),
            description: "Polytope is expanding from center".to_string(),
            detector: detect_expansion,
        });

        // Contraction pattern
        self.patterns.push(PatternDefinition {
            name: "contraction".to_string(),
            description: "Polytope is contracting toward center".to_string(),
            detector: detect_contraction,
        });
    }

    /// Detect all matching patterns
    pub fn detect(&self, geometry: &GeometryCore) -> Vec<String> {
        let mut detected = Vec::new();

        for pattern in &self.patterns {
            if let Some(confidence) = (pattern.detector)(geometry, self.tolerance) {
                if confidence > 0.5 {
                    detected.push(pattern.name.clone());
                }
            }
        }

        detected
    }

    /// Detect with full pattern information
    pub fn detect_detailed(&self, geometry: &GeometryCore) -> Vec<Pattern> {
        let mut detected = Vec::new();

        for pattern in &self.patterns {
            if let Some(confidence) = (pattern.detector)(geometry, self.tolerance) {
                if confidence > 0.3 {
                    detected.push(Pattern {
                        name: pattern.name.clone(),
                        confidence,
                        description: pattern.description.clone(),
                        location: None,
                    });
                }
            }
        }

        detected
    }

    /// Set tolerance
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance;
    }
}

impl Default for PatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

// Pattern detection functions

fn detect_synthesis_alignment(geometry: &GeometryCore, tolerance: f64) -> Option<f64> {
    let cell24 = geometry.cell24();

    // Check if any Gamma vertices are close to being on the line between Alpha and Beta pairs
    let alpha_verts = &cell24.alpha.transformed_vertices;
    let beta_verts = &cell24.beta.transformed_vertices;
    let gamma_verts = &cell24.gamma.transformed_vertices;

    let mut alignment_score = 0.0;
    let mut count = 0;

    for av in alpha_verts {
        for bv in beta_verts {
            for gv in gamma_verts {
                // Check if gamma is on the line segment
                let ab = *bv - *av;
                let ag = *gv - *av;

                let ab_len_sq = ab.magnitude_squared();
                if ab_len_sq < 0.01 {
                    continue;
                }

                let t = ag.dot(ab) / ab_len_sq;
                if t > 0.0 && t < 1.0 {
                    let projection = *av + ab * t;
                    let dist = gv.distance(projection);

                    if dist < tolerance * 2.0 {
                        alignment_score += 1.0 - (dist / (tolerance * 2.0));
                        count += 1;
                    }
                }
            }
        }
    }

    if count > 0 {
        Some(alignment_score / count as f64)
    } else {
        Some(0.0)
    }
}

fn detect_resonance(geometry: &GeometryCore, tolerance: f64) -> Option<f64> {
    let cell24 = geometry.cell24();

    // Check if all three components have similar average distances from origin
    let alpha_dist: f64 = cell24.alpha.transformed_vertices.iter()
        .map(|v| v.magnitude())
        .sum::<f64>() / cell24.alpha.transformed_vertices.len() as f64;

    let beta_dist: f64 = cell24.beta.transformed_vertices.iter()
        .map(|v| v.magnitude())
        .sum::<f64>() / cell24.beta.transformed_vertices.len() as f64;

    let gamma_dist: f64 = cell24.gamma.transformed_vertices.iter()
        .map(|v| v.magnitude())
        .sum::<f64>() / cell24.gamma.transformed_vertices.len() as f64;

    let avg = (alpha_dist + beta_dist + gamma_dist) / 3.0;
    let variance = (
        (alpha_dist - avg).powi(2) +
        (beta_dist - avg).powi(2) +
        (gamma_dist - avg).powi(2)
    ) / 3.0;

    // Low variance = high resonance
    Some(1.0 / (1.0 + variance / tolerance))
}

fn detect_high_symmetry(geometry: &GeometryCore, _tolerance: f64) -> Option<f64> {
    let vertices = geometry.cell24().transformed_vertices();

    // Check rotational symmetry by looking at distance distribution from centroid
    let centroid = vertices.iter().fold(Vec4::zero(), |acc, v| acc + *v)
        * (1.0 / vertices.len() as f64);

    let distances: Vec<f64> = vertices.iter()
        .map(|v| v.distance(centroid))
        .collect();

    let avg_dist = distances.iter().sum::<f64>() / distances.len() as f64;
    let variance = distances.iter()
        .map(|d| (d - avg_dist).powi(2))
        .sum::<f64>() / distances.len() as f64;

    // Low variance in distances indicates high symmetry
    let symmetry_score = 1.0 / (1.0 + variance);
    Some(symmetry_score)
}

fn detect_clustering(geometry: &GeometryCore, _tolerance: f64) -> Option<f64> {
    let vertices = geometry.cell24().transformed_vertices();

    // Simple clustering detection: check if vertices group into distinct regions
    let mut pair_distances: Vec<f64> = Vec::new();

    for i in 0..vertices.len() {
        for j in (i + 1)..vertices.len() {
            pair_distances.push(vertices[i].distance(vertices[j]));
        }
    }

    if pair_distances.is_empty() {
        return Some(0.0);
    }

    // Check for bimodal distribution (clustered = many close pairs and many far pairs)
    let avg = pair_distances.iter().sum::<f64>() / pair_distances.len() as f64;
    let close_count = pair_distances.iter().filter(|&&d| d < avg * 0.5).count();
    let far_count = pair_distances.iter().filter(|&&d| d > avg * 1.5).count();

    let bimodality = (close_count + far_count) as f64 / pair_distances.len() as f64;
    Some(bimodality)
}

fn detect_expansion(geometry: &GeometryCore, _tolerance: f64) -> Option<f64> {
    let vertices = geometry.cell24().transformed_vertices();
    let original = geometry.cell24().original_vertices();

    // Compare current average distance from origin to original
    let current_avg: f64 = vertices.iter().map(|v| v.magnitude()).sum::<f64>()
        / vertices.len() as f64;
    let original_avg: f64 = original.iter().map(|v| v.magnitude()).sum::<f64>()
        / original.len() as f64;

    let ratio = current_avg / original_avg.max(0.001);

    if ratio > 1.1 {
        Some((ratio - 1.0).min(1.0))
    } else {
        Some(0.0)
    }
}

fn detect_contraction(geometry: &GeometryCore, _tolerance: f64) -> Option<f64> {
    let vertices = geometry.cell24().transformed_vertices();
    let original = geometry.cell24().original_vertices();

    let current_avg: f64 = vertices.iter().map(|v| v.magnitude()).sum::<f64>()
        / vertices.len() as f64;
    let original_avg: f64 = original.iter().map(|v| v.magnitude()).sum::<f64>()
        / original.len() as f64;

    let ratio = current_avg / original_avg.max(0.001);

    if ratio < 0.9 {
        Some((1.0 - ratio).min(1.0))
    } else {
        Some(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EngineConfig;

    #[test]
    fn test_pattern_detector_creation() {
        let detector = PatternDetector::new();
        assert!(!detector.patterns.is_empty());
    }

    #[test]
    fn test_pattern_detection() {
        let config = EngineConfig::default();
        let geometry = GeometryCore::new(&config);
        let detector = PatternDetector::new();

        let patterns = detector.detect(&geometry);
        // At least symmetry should be detected for a fresh 24-cell
        assert!(patterns.contains(&"high_symmetry".to_string()));
    }
}
