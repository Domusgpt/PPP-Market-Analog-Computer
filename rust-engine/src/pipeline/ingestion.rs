//! Data Ingestion - Multi-channel input handling
//!
//! Supports multiple input sources: direct injection, file reading,
//! generated signals, and external streams.

use serde::{Serialize, Deserialize};
use std::collections::VecDeque;

/// Input source type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputSource {
    /// Direct value injection
    Direct,
    /// Sine wave generator
    SineWave { frequency: f64, phase: f64, amplitude: f64 },
    /// Noise generator
    Noise { seed: u64 },
    /// Constant value
    Constant(f64),
    /// External (to be connected)
    External(String),
}

impl Default for InputSource {
    fn default() -> Self {
        Self::Direct
    }
}

/// Configuration for a single input channel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelConfig {
    pub name: String,
    pub source: InputSource,
    pub min: f64,
    pub max: f64,
    pub smoothing: f64,
}

impl Default for ChannelConfig {
    fn default() -> Self {
        Self {
            name: "Channel".to_string(),
            source: InputSource::Direct,
            min: 0.0,
            max: 1.0,
            smoothing: 0.0,
        }
    }
}

/// Data ingestion handler
pub struct DataIngestion {
    channels: Vec<ChannelConfig>,
    values: Vec<f64>,
    smoothed: Vec<f64>,
    time: f64,
    history: Vec<VecDeque<f64>>,
    history_size: usize,
}

impl DataIngestion {
    pub fn new(channel_count: usize) -> Self {
        let mut channels = Vec::with_capacity(channel_count);
        let mut values = Vec::with_capacity(channel_count);
        let mut smoothed = Vec::with_capacity(channel_count);
        let mut history = Vec::with_capacity(channel_count);

        for i in 0..channel_count {
            let mut config = ChannelConfig::default();
            config.name = format!("Channel {}", i);

            // Set default sources for first 6 channels (rotation planes)
            if i < 6 {
                config.source = InputSource::SineWave {
                    frequency: 0.1 + i as f64 * 0.05,
                    phase: i as f64 * std::f64::consts::PI / 3.0,
                    amplitude: 1.0,
                };
            }

            channels.push(config);
            values.push(0.5);
            smoothed.push(0.5);
            history.push(VecDeque::new());
        }

        Self {
            channels,
            values,
            smoothed,
            time: 0.0,
            history,
            history_size: 100,
        }
    }

    /// Read current values from all channels
    pub fn read(&mut self) -> Vec<f64> {
        self.time += 1.0 / 60.0; // Assume 60 FPS

        for (i, channel) in self.channels.iter().enumerate() {
            let raw_value = match &channel.source {
                InputSource::Direct => self.values[i],

                InputSource::SineWave { frequency, phase, amplitude } => {
                    let t = self.time * frequency * 2.0 * std::f64::consts::PI + phase;
                    (t.sin() * amplitude + 1.0) / 2.0 // Normalize to 0-1
                }

                InputSource::Noise { seed } => {
                    // Simple LCG noise
                    let x = (self.time * 1000.0 + *seed as f64) as u64;
                    let noise = ((x.wrapping_mul(1103515245).wrapping_add(12345)) >> 16) as f64;
                    (noise % 1000.0) / 1000.0
                }

                InputSource::Constant(v) => *v,

                InputSource::External(_) => self.values[i],
            };

            // Apply smoothing
            if channel.smoothing > 0.0 {
                let alpha = 1.0 - channel.smoothing;
                self.smoothed[i] = self.smoothed[i] * (1.0 - alpha) + raw_value * alpha;
            } else {
                self.smoothed[i] = raw_value;
            }

            // Update history
            self.history[i].push_back(self.smoothed[i]);
            if self.history[i].len() > self.history_size {
                self.history[i].pop_front();
            }
        }

        self.smoothed.clone()
    }

    /// Inject a value directly into a channel
    pub fn inject(&mut self, channel: usize, value: f64) {
        if channel < self.values.len() {
            self.values[channel] = value.clamp(0.0, 1.0);
        }
    }

    /// Inject values into multiple channels
    pub fn inject_all(&mut self, values: &[f64]) {
        let n = values.len().min(self.values.len());
        for i in 0..n {
            self.values[i] = values[i].clamp(0.0, 1.0);
        }
    }

    /// Set channel configuration
    pub fn set_channel_config(&mut self, index: usize, config: ChannelConfig) {
        if index < self.channels.len() {
            self.channels[index] = config;
        }
    }

    /// Get channel configuration
    pub fn channel_config(&self, index: usize) -> Option<&ChannelConfig> {
        self.channels.get(index)
    }

    /// Get channel count
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Get history for a channel
    pub fn history(&self, channel: usize) -> Option<&VecDeque<f64>> {
        self.history.get(channel)
    }

    /// Reset all channels to default values
    pub fn reset(&mut self) {
        for v in &mut self.values {
            *v = 0.5;
        }
        for s in &mut self.smoothed {
            *s = 0.5;
        }
        for h in &mut self.history {
            h.clear();
        }
        self.time = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_ingestion_creation() {
        let ingestion = DataIngestion::new(64);
        assert_eq!(ingestion.channel_count(), 64);
    }

    #[test]
    fn test_data_injection() {
        let mut ingestion = DataIngestion::new(10);
        ingestion.inject(5, 0.75);
        assert!((ingestion.values[5] - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_sine_wave_source() {
        let mut ingestion = DataIngestion::new(1);
        ingestion.set_channel_config(0, ChannelConfig {
            name: "Test".to_string(),
            source: InputSource::SineWave {
                frequency: 1.0,
                phase: 0.0,
                amplitude: 1.0,
            },
            min: 0.0,
            max: 1.0,
            smoothing: 0.0,
        });

        // Read multiple frames
        for _ in 0..60 {
            let values = ingestion.read();
            assert!(values[0] >= 0.0 && values[0] <= 1.0);
        }
    }
}
