//! Homological Analysis - Betti Numbers and Persistent Homology
//!
//! Computes topological invariants of the point cloud/image to detect
//! structural features like clusters, loops, and voids.

use crate::geometry::Vec4;
use serde::{Serialize, Deserialize};

/// Betti numbers characterizing the topology
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct BettiNumbers {
    /// β₀: Number of connected components (clusters)
    pub b0: usize,
    /// β₁: Number of 1-dimensional holes (loops)
    pub b1: usize,
    /// β₂: Number of 2-dimensional voids (cavities)
    pub b2: usize,
    /// β₃: Number of 3-dimensional voids (in 4D)
    pub b3: usize,
}

impl BettiNumbers {
    /// Total topological complexity
    pub fn total(&self) -> usize {
        self.b0 + self.b1 + self.b2 + self.b3
    }

    /// Euler characteristic (alternating sum)
    pub fn euler_characteristic(&self) -> i64 {
        self.b0 as i64 - self.b1 as i64 + self.b2 as i64 - self.b3 as i64
    }

    /// Is this topologically trivial (single connected component, no holes)?
    pub fn is_trivial(&self) -> bool {
        self.b0 == 1 && self.b1 == 0 && self.b2 == 0 && self.b3 == 0
    }
}

/// A topological signal detected in the data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalSignal {
    /// Signal type
    pub signal_type: SignalType,
    /// Persistence (birth - death time)
    pub persistence: f64,
    /// Location (approximate center)
    pub location: [f64; 4],
    /// Scale at which it appears
    pub scale: f64,
}

/// Types of topological signals
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalType {
    /// New cluster formed
    ClusterBirth,
    /// Clusters merged
    ClusterMerge,
    /// Loop formed
    LoopFormation,
    /// Loop closed/filled
    LoopFilled,
    /// Void formed
    VoidFormation,
    /// Void collapsed
    VoidCollapse,
    /// Symmetry detected
    SymmetryDetected,
}

/// Homological analyzer
pub struct HomologyAnalyzer {
    /// Distance threshold for connectivity
    pub distance_threshold: f64,
    /// Minimum persistence to report
    pub min_persistence: f64,
    /// Last computed Betti numbers
    last_betti: BettiNumbers,
}

impl HomologyAnalyzer {
    pub fn new() -> Self {
        Self {
            distance_threshold: 1.5,
            min_persistence: 0.1,
            last_betti: BettiNumbers::default(),
        }
    }

    /// Compute Betti numbers for a set of 4D points
    pub fn compute_betti(&mut self, points: &[Vec4]) -> BettiNumbers {
        if points.is_empty() {
            return BettiNumbers::default();
        }

        // Simplified Betti number computation using connectivity analysis
        // In a full implementation, we'd use persistent homology (e.g., Ripser)

        // β₀: Count connected components using union-find
        let b0 = self.count_components(points);

        // β₁: Estimate loops from graph cycles
        let b1 = self.estimate_loops(points);

        // β₂: Estimate voids (simplified)
        let b2 = self.estimate_voids(points);

        // β₃: 4D voids (very simplified)
        let b3 = 0;

        let betti = BettiNumbers { b0, b1, b2, b3 };
        self.last_betti = betti;
        betti
    }

    /// Count connected components using union-find
    fn count_components(&self, points: &[Vec4]) -> usize {
        let n = points.len();
        if n == 0 {
            return 0;
        }

        // Simple union-find
        let mut parent: Vec<usize> = (0..n).collect();

        fn find(parent: &mut [usize], i: usize) -> usize {
            if parent[i] != i {
                parent[i] = find(parent, parent[i]);
            }
            parent[i]
        }

        fn union(parent: &mut [usize], i: usize, j: usize) {
            let pi = find(parent, i);
            let pj = find(parent, j);
            if pi != pj {
                parent[pi] = pj;
            }
        }

        // Connect nearby points
        let threshold_sq = self.distance_threshold * self.distance_threshold;
        for i in 0..n {
            for j in (i + 1)..n {
                if points[i].distance_squared(points[j]) < threshold_sq {
                    union(&mut parent, i, j);
                }
            }
        }

        // Count unique roots
        let mut roots = std::collections::HashSet::new();
        for i in 0..n {
            roots.insert(find(&mut parent, i));
        }

        roots.len()
    }

    /// Estimate number of loops from graph structure
    fn estimate_loops(&self, points: &[Vec4]) -> usize {
        let n = points.len();
        if n < 3 {
            return 0;
        }

        // Build adjacency graph
        let threshold_sq = self.distance_threshold * self.distance_threshold;
        let mut edge_count = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                if points[i].distance_squared(points[j]) < threshold_sq {
                    edge_count += 1;
                }
            }
        }

        // From Euler's formula: V - E + F = 2 for connected planar graph
        // β₁ = E - V + 1 for a connected component
        // This is a rough estimate
        let components = self.count_components(points);
        if edge_count > n + components {
            edge_count - n - components + 1
        } else {
            0
        }
    }

    /// Estimate voids (simplified)
    fn estimate_voids(&self, points: &[Vec4]) -> usize {
        // This is a very rough estimate
        // Actual computation requires computing 2-simplices (triangles) and 3-simplices
        let n = points.len();
        if n < 4 {
            return 0;
        }

        // Check for approximately hollow structures
        let centroid = points.iter().fold(Vec4::zero(), |acc, p| acc + *p)
            * (1.0 / n as f64);

        let avg_dist: f64 = points.iter().map(|p| p.distance(centroid)).sum::<f64>() / n as f64;

        let dist_variance: f64 = points.iter()
            .map(|p| (p.distance(centroid) - avg_dist).powi(2))
            .sum::<f64>() / n as f64;

        // Low variance in distances to centroid suggests shell-like structure (void)
        if dist_variance < 0.1 * avg_dist * avg_dist && n >= 8 {
            1
        } else {
            0
        }
    }

    /// Detect significant topological signals (changes)
    pub fn detect_signals(&self, points: &[Vec4]) -> Vec<TopologicalSignal> {
        let mut signals = Vec::new();

        // Compare to last computed Betti numbers
        let current = self.compute_betti_without_update(points);

        // Detect changes
        if current.b0 > self.last_betti.b0 {
            signals.push(TopologicalSignal {
                signal_type: SignalType::ClusterBirth,
                persistence: 1.0,
                location: self.estimate_change_location(points, SignalType::ClusterBirth),
                scale: self.distance_threshold,
            });
        } else if current.b0 < self.last_betti.b0 {
            signals.push(TopologicalSignal {
                signal_type: SignalType::ClusterMerge,
                persistence: 1.0,
                location: self.estimate_change_location(points, SignalType::ClusterMerge),
                scale: self.distance_threshold,
            });
        }

        if current.b1 > self.last_betti.b1 {
            signals.push(TopologicalSignal {
                signal_type: SignalType::LoopFormation,
                persistence: 1.0,
                location: self.estimate_change_location(points, SignalType::LoopFormation),
                scale: self.distance_threshold,
            });
        } else if current.b1 < self.last_betti.b1 {
            signals.push(TopologicalSignal {
                signal_type: SignalType::LoopFilled,
                persistence: 1.0,
                location: self.estimate_change_location(points, SignalType::LoopFilled),
                scale: self.distance_threshold,
            });
        }

        if current.b2 > self.last_betti.b2 {
            signals.push(TopologicalSignal {
                signal_type: SignalType::VoidFormation,
                persistence: 1.0,
                location: self.estimate_change_location(points, SignalType::VoidFormation),
                scale: self.distance_threshold,
            });
        }

        signals
    }

    /// Compute Betti without updating last state
    fn compute_betti_without_update(&self, points: &[Vec4]) -> BettiNumbers {
        let b0 = self.count_components(points);
        let b1 = self.estimate_loops(points);
        let b2 = self.estimate_voids(points);
        BettiNumbers { b0, b1, b2, b3: 0 }
    }

    /// Estimate location of topological change
    fn estimate_change_location(&self, points: &[Vec4], _signal_type: SignalType) -> [f64; 4] {
        // Return centroid as approximation
        if points.is_empty() {
            return [0.0; 4];
        }

        let centroid = points.iter().fold(Vec4::zero(), |acc, p| acc + *p)
            * (1.0 / points.len() as f64);

        centroid.to_array()
    }

    /// Get last computed Betti numbers
    pub fn last_betti(&self) -> BettiNumbers {
        self.last_betti
    }

    /// Set distance threshold
    pub fn set_threshold(&mut self, threshold: f64) {
        self.distance_threshold = threshold;
    }
}

impl Default for HomologyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_betti_numbers_basic() {
        let betti = BettiNumbers { b0: 2, b1: 1, b2: 0, b3: 0 };
        assert_eq!(betti.total(), 3);
        assert_eq!(betti.euler_characteristic(), 1);
    }

    #[test]
    fn test_single_point() {
        let mut analyzer = HomologyAnalyzer::new();
        let points = vec![Vec4::new(0.0, 0.0, 0.0, 0.0)];
        let betti = analyzer.compute_betti(&points);

        assert_eq!(betti.b0, 1); // Single component
        assert_eq!(betti.b1, 0); // No loops
    }

    #[test]
    fn test_two_distant_points() {
        let mut analyzer = HomologyAnalyzer::new();
        analyzer.set_threshold(1.0);

        let points = vec![
            Vec4::new(0.0, 0.0, 0.0, 0.0),
            Vec4::new(10.0, 0.0, 0.0, 0.0), // Far apart
        ];
        let betti = analyzer.compute_betti(&points);

        assert_eq!(betti.b0, 2); // Two separate components
    }

    #[test]
    fn test_connected_points() {
        let mut analyzer = HomologyAnalyzer::new();
        analyzer.set_threshold(2.0);

        let points = vec![
            Vec4::new(0.0, 0.0, 0.0, 0.0),
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(2.0, 0.0, 0.0, 0.0),
        ];
        let betti = analyzer.compute_betti(&points);

        assert_eq!(betti.b0, 1); // Single connected component
    }
}
