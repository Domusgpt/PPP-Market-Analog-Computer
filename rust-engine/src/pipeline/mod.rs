//! Data Pipeline Module
//!
//! Handles multi-channel data ingestion, mapping to geometric parameters,
//! and output extraction from the simulation state.

mod ingestion;
mod mapper;
mod output;

pub use ingestion::DataIngestion;
pub use mapper::DataMapper;
pub use output::OutputChannel;

/// Current data state after processing
#[derive(Debug, Clone)]
pub struct DataState {
    /// Normalized channel values (0.0 to 1.0)
    pub channels: Vec<f64>,
    /// Raw input values
    pub raw: Vec<f64>,
    /// Timestamp
    pub timestamp: f64,
    /// Frame number
    pub frame: u64,
}

impl DataState {
    pub fn new(channel_count: usize) -> Self {
        Self {
            channels: vec![0.5; channel_count],
            raw: vec![0.0; channel_count],
            timestamp: 0.0,
            frame: 0,
        }
    }
}

/// The main data pipeline
pub struct DataPipeline {
    ingestion: DataIngestion,
    mapper: DataMapper,
    output: OutputChannel,
    current_state: DataState,
}

impl DataPipeline {
    pub fn new(channel_count: usize) -> Self {
        Self {
            ingestion: DataIngestion::new(channel_count),
            mapper: DataMapper::new(channel_count),
            output: OutputChannel::new(),
            current_state: DataState::new(channel_count),
        }
    }

    /// Process one frame of data
    pub fn process(&mut self) -> DataState {
        // Get raw data from ingestion
        let raw = self.ingestion.read();

        // Map through the mapper
        let mapped = self.mapper.map(&raw);

        // Update current state
        self.current_state.raw = raw;
        self.current_state.channels = mapped;
        self.current_state.frame += 1;
        self.current_state.timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        self.current_state.clone()
    }

    /// Inject data directly
    pub fn inject(&mut self, channel: usize, value: f64) {
        self.ingestion.inject(channel, value);
    }

    /// Inject multiple channels
    pub fn inject_all(&mut self, values: &[f64]) {
        self.ingestion.inject_all(values);
    }

    /// Get the data mapper for configuration
    pub fn mapper_mut(&mut self) -> &mut DataMapper {
        &mut self.mapper
    }

    /// Get the output channel
    pub fn output(&self) -> &OutputChannel {
        &self.output
    }

    /// Get mutable output channel
    pub fn output_mut(&mut self) -> &mut OutputChannel {
        &mut self.output
    }

    /// Get current state
    pub fn current_state(&self) -> &DataState {
        &self.current_state
    }
}
