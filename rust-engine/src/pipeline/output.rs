//! Output Channel - Data export and signal extraction
//!
//! Provides interfaces for extracting computed signals from the simulation
//! for external consumption by AI models, logging, or other systems.

use serde::{Serialize, Deserialize};
use std::collections::VecDeque;

/// Types of output signals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputSignal {
    /// Scalar value
    Scalar(f64),
    /// Vector (e.g., position, direction)
    Vector(Vec<f64>),
    /// Boolean flag
    Flag(bool),
    /// Discrete event
    Event(String),
    /// Complex structured data
    Structured(serde_json::Value),
}

/// A timestamped output record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputRecord {
    pub timestamp: f64,
    pub frame: u64,
    pub signals: Vec<(String, OutputSignal)>,
}

/// Output channel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Maximum records to keep in history
    pub max_history: usize,
    /// Whether to enable JSON export
    pub json_export: bool,
    /// Callback on new record (not serializable)
    #[serde(skip)]
    pub callback: Option<Box<dyn Fn(&OutputRecord) + Send + Sync>>,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            max_history: 1000,
            json_export: true,
            callback: None,
        }
    }
}

/// Output channel manager
pub struct OutputChannel {
    config: OutputConfig,
    history: VecDeque<OutputRecord>,
    current_frame: u64,
    current_signals: Vec<(String, OutputSignal)>,
}

impl OutputChannel {
    pub fn new() -> Self {
        Self {
            config: OutputConfig::default(),
            history: VecDeque::new(),
            current_frame: 0,
            current_signals: Vec::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: OutputConfig) -> Self {
        Self {
            config,
            history: VecDeque::new(),
            current_frame: 0,
            current_signals: Vec::new(),
        }
    }

    /// Record a scalar signal
    pub fn record_scalar(&mut self, name: impl Into<String>, value: f64) {
        self.current_signals.push((name.into(), OutputSignal::Scalar(value)));
    }

    /// Record a vector signal
    pub fn record_vector(&mut self, name: impl Into<String>, value: Vec<f64>) {
        self.current_signals.push((name.into(), OutputSignal::Vector(value)));
    }

    /// Record a flag signal
    pub fn record_flag(&mut self, name: impl Into<String>, value: bool) {
        self.current_signals.push((name.into(), OutputSignal::Flag(value)));
    }

    /// Record an event
    pub fn record_event(&mut self, name: impl Into<String>, event: impl Into<String>) {
        self.current_signals.push((name.into(), OutputSignal::Event(event.into())));
    }

    /// Record structured data
    pub fn record_structured(&mut self, name: impl Into<String>, data: serde_json::Value) {
        self.current_signals.push((name.into(), OutputSignal::Structured(data)));
    }

    /// Commit current frame's signals to history
    pub fn commit(&mut self) {
        if self.current_signals.is_empty() {
            return;
        }

        let record = OutputRecord {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0),
            frame: self.current_frame,
            signals: std::mem::take(&mut self.current_signals),
        };

        // Call callback if set
        if let Some(ref callback) = self.config.callback {
            callback(&record);
        }

        // Add to history
        self.history.push_back(record);

        // Trim history
        while self.history.len() > self.config.max_history {
            self.history.pop_front();
        }

        self.current_frame += 1;
    }

    /// Get history
    pub fn history(&self) -> &VecDeque<OutputRecord> {
        &self.history
    }

    /// Get the most recent record
    pub fn latest(&self) -> Option<&OutputRecord> {
        self.history.back()
    }

    /// Get records in a time range
    pub fn range(&self, start_time: f64, end_time: f64) -> Vec<&OutputRecord> {
        self.history.iter()
            .filter(|r| r.timestamp >= start_time && r.timestamp <= end_time)
            .collect()
    }

    /// Export history to JSON
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        let records: Vec<_> = self.history.iter().collect();
        serde_json::to_string_pretty(&records)
    }

    /// Clear history
    pub fn clear(&mut self) {
        self.history.clear();
        self.current_signals.clear();
    }

    /// Get signal by name from latest record
    pub fn get_signal(&self, name: &str) -> Option<&OutputSignal> {
        self.latest()
            .and_then(|r| r.signals.iter().find(|(n, _)| n == name))
            .map(|(_, s)| s)
    }

    /// Get scalar value by name
    pub fn get_scalar(&self, name: &str) -> Option<f64> {
        match self.get_signal(name) {
            Some(OutputSignal::Scalar(v)) => Some(*v),
            _ => None,
        }
    }

    /// Set callback for new records
    pub fn set_callback<F>(&mut self, callback: F)
    where
        F: Fn(&OutputRecord) + Send + Sync + 'static,
    {
        self.config.callback = Some(Box::new(callback));
    }

    /// Get statistics summary
    pub fn stats(&self) -> OutputStats {
        let scalar_count = self.history.iter()
            .flat_map(|r| r.signals.iter())
            .filter(|(_, s)| matches!(s, OutputSignal::Scalar(_)))
            .count();

        let event_count = self.history.iter()
            .flat_map(|r| r.signals.iter())
            .filter(|(_, s)| matches!(s, OutputSignal::Event(_)))
            .count();

        OutputStats {
            total_records: self.history.len(),
            total_signals: self.history.iter().map(|r| r.signals.len()).sum(),
            scalar_count,
            event_count,
            oldest_frame: self.history.front().map(|r| r.frame).unwrap_or(0),
            newest_frame: self.history.back().map(|r| r.frame).unwrap_or(0),
        }
    }
}

impl Default for OutputChannel {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about output history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputStats {
    pub total_records: usize,
    pub total_signals: usize,
    pub scalar_count: usize,
    pub event_count: usize,
    pub oldest_frame: u64,
    pub newest_frame: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_channel_creation() {
        let channel = OutputChannel::new();
        assert!(channel.history().is_empty());
    }

    #[test]
    fn test_recording_signals() {
        let mut channel = OutputChannel::new();

        channel.record_scalar("tension", 0.75);
        channel.record_flag("resonant", true);
        channel.commit();

        assert_eq!(channel.history().len(), 1);

        let scalar = channel.get_scalar("tension");
        assert!(scalar.is_some());
        assert!((scalar.unwrap() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_history_limit() {
        let config = OutputConfig {
            max_history: 5,
            ..Default::default()
        };
        let mut channel = OutputChannel::with_config(config);

        for i in 0..10 {
            channel.record_scalar("value", i as f64);
            channel.commit();
        }

        assert_eq!(channel.history().len(), 5);
    }

    #[test]
    fn test_json_export() {
        let mut channel = OutputChannel::new();
        channel.record_scalar("test", 1.0);
        channel.commit();

        let json = channel.export_json();
        assert!(json.is_ok());
    }
}
