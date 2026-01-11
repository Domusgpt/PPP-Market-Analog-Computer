//! Analysis Module - Homological Signal Extraction
//!
//! This module extracts topological features from the geometric state,
//! including Betti numbers, persistent homology, and pattern detection.

mod homology;
mod patterns;
mod metrics;

pub use homology::{HomologyAnalyzer, BettiNumbers};
pub use patterns::PatternDetector;
pub use metrics::{TopologicalSignal, MetricExtractor};

use crate::geometry::GeometryCore;

/// Analysis result container
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// Betti numbers of the current state
    pub betti: BettiNumbers,
    /// Detected topological signals
    pub signals: Vec<TopologicalSignal>,
    /// Pattern matches
    pub patterns: Vec<String>,
    /// Raw metrics
    pub metrics: AnalysisMetrics,
}

/// Numerical metrics from analysis
#[derive(Debug, Clone, Default)]
pub struct AnalysisMetrics {
    /// Total vertex count
    pub vertex_count: usize,
    /// Active vertex count (above threshold)
    pub active_vertices: usize,
    /// Cluster count
    pub cluster_count: usize,
    /// Symmetry score (0-1)
    pub symmetry_score: f64,
    /// Coherence score (0-1)
    pub coherence_score: f64,
    /// Complexity measure
    pub complexity: f64,
}

/// The main analyzer
pub struct Analyzer {
    homology: HomologyAnalyzer,
    patterns: PatternDetector,
    metrics: MetricExtractor,
}

impl Analyzer {
    pub fn new() -> Self {
        Self {
            homology: HomologyAnalyzer::new(),
            patterns: PatternDetector::new(),
            metrics: MetricExtractor::new(),
        }
    }

    /// Run full analysis on current geometry state
    pub fn analyze(&mut self, geometry: &GeometryCore) -> AnalysisResult {
        // Extract vertices for analysis
        let vertices = geometry.cell24().transformed_vertices();

        // Compute Betti numbers
        let betti = self.homology.compute_betti(vertices);

        // Detect topological signals
        let signals = self.homology.detect_signals(vertices);

        // Find patterns
        let patterns = self.patterns.detect(geometry);

        // Extract metrics
        let metrics = self.metrics.extract(geometry);

        AnalysisResult {
            betti,
            signals,
            patterns,
            metrics,
        }
    }

    /// Quick analysis (just Betti numbers)
    pub fn quick_analyze(&mut self, geometry: &GeometryCore) -> BettiNumbers {
        let vertices = geometry.cell24().transformed_vertices();
        self.homology.compute_betti(vertices)
    }

    /// Get homology analyzer for direct access
    pub fn homology(&self) -> &HomologyAnalyzer {
        &self.homology
    }

    /// Get pattern detector
    pub fn patterns(&self) -> &PatternDetector {
        &self.patterns
    }
}

impl Default for Analyzer {
    fn default() -> Self {
        Self::new()
    }
}
