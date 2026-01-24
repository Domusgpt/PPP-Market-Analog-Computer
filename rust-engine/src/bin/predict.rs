//! Market Prediction CLI - Uses REAL MarketLarynx with live API data
//!
//! Usage:
//!   predict <btc_price> <fear_greed> <vix>
//!   predict 89776 24 15.5
//!
//! This is NOT a simulation - it runs the actual Rust MarketLarynx TDA implementation.

use geometric_cognition::cognitive::{
    MarketLarynx, MarketLarynxConfig, MusicalInterval,
};
use std::env;

fn normalize_price(price: f64) -> f64 {
    let min = 80000.0;
    let max = 110000.0;
    ((price - min) / (max - min)).clamp(0.0, 1.0)
}

fn normalize_sentiment(fear_greed: i32, vix: f64) -> f64 {
    let fg_norm = fear_greed as f64 / 100.0;
    let vix_norm = 1.0 - ((vix - 10.0) / 30.0).clamp(0.0, 1.0);
    fg_norm * 0.6 + vix_norm * 0.4
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse command line args or use defaults
    let (btc_price, fear_greed, vix) = if args.len() >= 4 {
        (
            args[1].parse::<f64>().unwrap_or(89776.0),
            args[2].parse::<i32>().unwrap_or(24),
            args[3].parse::<f64>().unwrap_or(15.5),
        )
    } else {
        eprintln!("Usage: predict <btc_price> <fear_greed> <vix>");
        eprintln!("Using defaults: 89776 24 15.5");
        (89776.0, 24, 15.5)
    };

    println!("======================================================================");
    println!("  HARMONIC ALPHA - REAL RUST ENGINE PREDICTION");
    println!("  Using actual MarketLarynx TDA implementation");
    println!("======================================================================\n");

    println!("[1] INPUT DATA:");
    println!("    BTC Price:       ${:.2}", btc_price);
    println!("    Fear & Greed:    {}", fear_greed);
    println!("    VIX:             {:.2}", vix);

    // Normalize inputs
    println!("\n[2] NORMALIZATION:");
    let price_alpha = normalize_price(btc_price);
    let sentiment_beta = normalize_sentiment(fear_greed, vix);
    println!("    Price Alpha:     {:.4}", price_alpha);
    println!("    Sentiment Beta:  {:.4}", sentiment_beta);
    println!("    Divergence:      {:.4} ({:.1}%)",
        (price_alpha - sentiment_beta).abs(),
        (price_alpha - sentiment_beta).abs() * 100.0);

    // Create REAL MarketLarynx instance
    println!("\n[3] REAL RUST MarketLarynx ENGINE:");
    let config = MarketLarynxConfig {
        gamma_threshold: 0.85,
        smoothing_window: 30,
        tension_decay: 0.02,
        sentiment_sensitivity: 1.5,
        base_frequency: 110.0,
        tda_persistence_threshold: 0.1,
    };
    let mut larynx = MarketLarynx::with_config(config);

    // Set market data
    larynx.set_price(price_alpha);
    larynx.set_sentiment(sentiment_beta);

    // Run 60 update cycles
    let mut result = larynx.step(1.0 / 60.0);
    for _ in 1..60 {
        result = larynx.step(1.0 / 60.0);
    }

    println!("    Tension:          {:.4}", result.tension);
    println!("    Smoothed Tension: {:.4}", result.smoothed_tension);
    println!("    Regime:           {:?}", result.regime);
    println!("    Gamma Active:     {}", result.gamma_active);
    println!("    TDA Features:     {} detected", result.tda_features.len());
    println!("    TDA Crash Prob:   {:.4} ({:.1}%)", result.crash_probability, result.crash_probability * 100.0);

    let interval = MusicalInterval::from_tension(result.smoothed_tension);
    println!("\n[4] MUSICAL MAPPING:");
    println!("    Interval Ratio:   {:.3}", interval.ratio);
    println!("    Consonance:       {:.2}", interval.consonance);
    println!("    Frequencies:      {:.1} Hz, {:.1} Hz, {:.1} Hz",
        result.sonification_frequencies[0],
        result.sonification_frequencies[1],
        result.sonification_frequencies[2]);

    // Determine risk level based on REAL TDA crash probability
    let crash_prob = result.crash_probability;
    let (risk_level, outlook) = if crash_prob > 0.7 {
        ("HIGH", "BEARISH - Significant correction likely")
    } else if crash_prob > 0.5 {
        ("ELEVATED", "CAUTIOUS - Increased downside risk")
    } else if crash_prob > 0.3 {
        ("MODERATE", "NEUTRAL - Watch for changes")
    } else {
        ("LOW", "STABLE - No immediate crash signals")
    };

    println!("\n======================================================================");
    println!("  PREDICTION (FROM REAL RUST TDA ENGINE)");
    println!("======================================================================");
    println!("\n  🎯 RISK LEVEL:  {}", risk_level);
    println!("  📊 OUTLOOK:     {}", outlook);
    println!("  📈 REGIME:      {:?}", result.regime);
    println!("  💥 CRASH PROB:  {:.1}%", crash_prob * 100.0);

    println!("\n======================================================================");
    println!("  Generated: {}", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"));
    println!("======================================================================");
}
