//! Metric Extraction - Numerical measures from geometric state
//!
//! Extracts quantitative metrics that can be used by external systems
//! for analysis, logging, or ML training.

use crate::geometry::{GeometryCore, Vec4, TrinityComponent};
use super::AnalysisMetrics;
use serde::{Serialize, Deserialize};

/// A metric signal for export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSignal {
    /// Signal name/type
    pub name: String,
    /// Signal value
    pub value: f64,
    /// Dimension (0 = component, 1 = loop, 2 = void)
    pub dimension: u8,
    /// Confidence in the signal
    pub confidence: f64,
}

/// Metric extractor
pub struct MetricExtractor {
    /// History of recent metric values for trend detection
    history: Vec<AnalysisMetrics>,
    max_history: usize,
}

impl MetricExtractor {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            max_history: 100,
        }
    }

    /// Extract all metrics from current geometry state
    pub fn extract(&mut self, geometry: &GeometryCore) -> AnalysisMetrics {
        let vertices = geometry.cell24().transformed_vertices();

        let metrics = AnalysisMetrics {
            vertex_count: vertices.len(),
            active_vertices: self.count_active(vertices, 0.5),
            cluster_count: self.estimate_clusters(vertices),
            symmetry_score: self.compute_symmetry(vertices),
            coherence_score: self.compute_coherence(geometry),
            complexity: self.compute_complexity(vertices),
        };

        // Store in history
        self.history.push(metrics.clone());
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }

        metrics
    }

    /// Count vertices above activation threshold
    fn count_active(&self, vertices: &[Vec4], threshold: f64) -> usize {
        // In this context, "active" means significant distance from origin
        vertices.iter()
            .filter(|v| v.magnitude() > threshold)
            .count()
    }

    /// Estimate number of clusters using simple spatial analysis
    fn estimate_clusters(&self, vertices: &[Vec4]) -> usize {
        if vertices.is_empty() {
            return 0;
        }

        // Simple grid-based clustering
        let grid_size = 4;
        let mut occupied = std::collections::HashSet::new();

        for v in vertices {
            // Map to grid cell
            let cell = (
                ((v.x + 2.0) * grid_size as f64 / 4.0) as i32,
                ((v.y + 2.0) * grid_size as f64 / 4.0) as i32,
                ((v.z + 2.0) * grid_size as f64 / 4.0) as i32,
            );
            occupied.insert(cell);
        }

        occupied.len()
    }

    /// Compute symmetry score based on distance distribution
    fn compute_symmetry(&self, vertices: &[Vec4]) -> f64 {
        if vertices.is_empty() {
            return 0.0;
        }

        let centroid = vertices.iter().fold(Vec4::zero(), |acc, v| acc + *v)
            * (1.0 / vertices.len() as f64);

        let distances: Vec<f64> = vertices.iter()
            .map(|v| v.distance(centroid))
            .collect();

        let avg = distances.iter().sum::<f64>() / distances.len() as f64;
        let variance = distances.iter()
            .map(|d| (d - avg).powi(2))
            .sum::<f64>() / distances.len() as f64;

        // Normalize to 0-1 range (low variance = high symmetry)
        1.0 / (1.0 + variance)
    }

    /// Compute coherence based on Trinity component alignment
    fn compute_coherence(&self, geometry: &GeometryCore) -> f64 {
        let ab_dist = geometry.dialectic_distance(TrinityComponent::Alpha, TrinityComponent::Beta);
        let ag_dist = geometry.dialectic_distance(TrinityComponent::Alpha, TrinityComponent::Gamma);
        let bg_dist = geometry.dialectic_distance(TrinityComponent::Beta, TrinityComponent::Gamma);

        // Coherence is high when distances are balanced
        let avg = (ab_dist + ag_dist + bg_dist) / 3.0;
        let variance = (
            (ab_dist - avg).powi(2) +
            (ag_dist - avg).powi(2) +
            (bg_dist - avg).powi(2)
        ) / 3.0;

        1.0 / (1.0 + variance)
    }

    /// Compute complexity measure
    fn compute_complexity(&self, vertices: &[Vec4]) -> f64 {
        if vertices.len() < 2 {
            return 0.0;
        }

        // Complexity based on pairwise distance entropy
        let mut distances = Vec::new();
        for i in 0..vertices.len() {
            for j in (i + 1)..vertices.len() {
                distances.push(vertices[i].distance(vertices[j]));
            }
        }

        if distances.is_empty() {
            return 0.0;
        }

        // Compute entropy-like measure
        let total: f64 = distances.iter().sum();
        let entropy: f64 = distances.iter()
            .map(|d| {
                let p = d / total;
                if p > 0.0 { -p * p.ln() } else { 0.0 }
            })
            .sum();

        // Normalize
        let max_entropy = (distances.len() as f64).ln();
        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        }
    }

    /// Get metric trend (positive = increasing, negative = decreasing)
    pub fn get_trend(&self, metric_name: &str) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }

        let recent: Vec<f64> = self.history.iter()
            .rev()
            .take(10)
            .map(|m| match metric_name {
                "symmetry" => m.symmetry_score,
                "coherence" => m.coherence_score,
                "complexity" => m.complexity,
                "clusters" => m.cluster_count as f64,
                _ => 0.0,
            })
            .collect();

        if recent.len() < 2 {
            return 0.0;
        }

        // Simple linear regression slope
        let n = recent.len() as f64;
        let sum_x: f64 = (0..recent.len()).map(|i| i as f64).sum();
        let sum_y: f64 = recent.iter().sum();
        let sum_xy: f64 = recent.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let sum_xx: f64 = (0..recent.len()).map(|i| (i as f64).powi(2)).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        slope
    }

    /// Get history
    pub fn history(&self) -> &[AnalysisMetrics] {
        &self.history
    }

    /// Clear history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }
}

impl Default for MetricExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EngineConfig;

    #[test]
    fn test_metric_extractor() {
        let config = EngineConfig::default();
        let geometry = GeometryCore::new(&config);
        let mut extractor = MetricExtractor::new();

        let metrics = extractor.extract(&geometry);

        assert_eq!(metrics.vertex_count, 24);
        assert!(metrics.symmetry_score > 0.0);
        assert!(metrics.coherence_score > 0.0);
    }

    #[test]
    fn test_trend_calculation() {
        let config = EngineConfig::default();
        let geometry = GeometryCore::new(&config);
        let mut extractor = MetricExtractor::new();

        // Extract multiple times
        for _ in 0..5 {
            extractor.extract(&geometry);
        }

        let trend = extractor.get_trend("symmetry");
        // Should be near zero for static geometry
        assert!(trend.abs() < 0.1);
    }
}
