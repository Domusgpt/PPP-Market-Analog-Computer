//! Geometric Cognition Engine - CLI Application
//!
//! This is the main entry point for running the geometric cognition engine
//! as a standalone application with optional windowed display.

use geometric_cognition::{GeometricCognitionEngine, EngineConfig};

#[cfg(feature = "desktop")]
use winit::{
    event::{Event, WindowEvent, KeyEvent, ElementState},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
};

fn main() {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    log::info!("Geometric Cognition Engine v{}", env!("CARGO_PKG_VERSION"));
    log::info!("A GPU-accelerated engine for analog cognitive computation");
    log::info!("Using 4D polytopes: 24-cell, 600-cell, 120-cell");

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    if args.contains(&"--help".to_string()) || args.contains(&"-h".to_string()) {
        print_help();
        return;
    }

    // Create configuration
    let mut config = EngineConfig::default();

    // Check for expanded mode flag
    if args.contains(&"--expanded".to_string()) {
        config.expanded_mode = true;
        log::info!("Running in expanded 600-cell mode");
    }

    // Check for E8 layer flag
    if args.contains(&"--no-e8".to_string()) {
        config.e8_layer_enabled = false;
        log::info!("E₈ dual-layer disabled");
    }

    // Run the appropriate mode
    if args.contains(&"--headless".to_string()) {
        run_headless(config);
    } else if args.contains(&"--benchmark".to_string()) {
        run_benchmark(config);
    } else {
        #[cfg(feature = "desktop")]
        {
            run_windowed(config);
        }

        #[cfg(not(feature = "desktop"))]
        {
            log::info!("Desktop features not enabled. Running headless demo.");
            run_headless(config);
        }
    }
}

fn print_help() {
    println!("Geometric Cognition Engine");
    println!();
    println!("USAGE:");
    println!("    gce [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    -h, --help        Show this help message");
    println!("    --headless        Run without display (useful for testing)");
    println!("    --benchmark       Run performance benchmark");
    println!("    --expanded        Start in expanded 600-cell mode");
    println!("    --no-e8           Disable E₈ dual-layer scaling");
    println!();
    println!("DESCRIPTION:");
    println!("    This engine performs analog cognitive computation using");
    println!("    high-dimensional geometry and GPU-based visual processing.");
    println!();
    println!("    Key features:");
    println!("    - 24-cell Trinity decomposition (Alpha/Beta/Gamma)");
    println!("    - Thesis-antithesis-synthesis dialectic reasoning");
    println!("    - Pixel-rule analog computation via shaders");
    println!("    - Homological signal extraction (Betti numbers)");
    println!();
}

fn run_headless(config: EngineConfig) {
    log::info!("Running in headless mode");

    let mut engine = GeometricCognitionEngine::new(config);

    // Run simulation loop
    let delta_time = 1.0 / 60.0;

    log::info!("Starting simulation...");

    for frame in 0..300 {
        // Inject some test data
        let t = frame as f64 * delta_time;
        engine.data_pipeline_mut().inject(0, (t * 0.5).sin() * 0.5 + 0.5);
        engine.data_pipeline_mut().inject(1, (t * 0.7).cos() * 0.5 + 0.5);
        engine.data_pipeline_mut().inject(2, (t * 0.3).sin() * 0.5 + 0.5);

        // Update simulation
        engine.update(delta_time);

        // Analyze state periodically
        if frame % 60 == 0 {
            let analysis = engine.analyze();
            log::info!(
                "Frame {}: Betti=(β₀={}, β₁={}, β₂={}), Patterns={:?}",
                frame,
                analysis.betti.b0,
                analysis.betti.b1,
                analysis.betti.b2,
                analysis.patterns
            );
        }
    }

    log::info!("Simulation complete. Total frames: {}", engine.frame_count());
}

fn run_benchmark(config: EngineConfig) {
    log::info!("Running performance benchmark");

    let mut engine = GeometricCognitionEngine::new(config.clone());
    let delta_time = 1.0 / 60.0;

    let start = std::time::Instant::now();
    let iterations = 1000;

    for _ in 0..iterations {
        engine.update(delta_time);
    }

    let elapsed = start.elapsed();
    let avg_frame_time = elapsed.as_secs_f64() / iterations as f64;
    let fps = 1.0 / avg_frame_time;

    log::info!("Benchmark results:");
    log::info!("  Total time: {:.2?}", elapsed);
    log::info!("  Iterations: {}", iterations);
    log::info!("  Avg frame time: {:.4}ms", avg_frame_time * 1000.0);
    log::info!("  Theoretical FPS: {:.1}", fps);

    // Test with expanded mode
    let mut config_expanded = config.clone();
    config_expanded.expanded_mode = true;
    let mut engine_expanded = GeometricCognitionEngine::new(config_expanded);

    let start = std::time::Instant::now();
    for _ in 0..iterations {
        engine_expanded.update(delta_time);
    }
    let elapsed_expanded = start.elapsed();
    let fps_expanded = iterations as f64 / elapsed_expanded.as_secs_f64();

    log::info!("Expanded mode (600-cell):");
    log::info!("  Theoretical FPS: {:.1}", fps_expanded);
}

#[cfg(feature = "desktop")]
fn run_windowed(config: EngineConfig) {
    log::info!("Running in windowed mode");

    // Create event loop and window
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let window = WindowBuilder::new()
        .with_title("Geometric Cognition Engine")
        .with_inner_size(winit::dpi::LogicalSize::new(config.width, config.height))
        .build(&event_loop)
        .expect("Failed to create window");

    // Create engine
    let mut engine = GeometricCognitionEngine::new(config);

    // Initialize rendering (async)
    pollster::block_on(async {
        if let Err(e) = engine.init_rendering().await {
            log::error!("Failed to initialize rendering: {}", e);
            return;
        }

        log::info!("Rendering initialized successfully");
    });

    let delta_time = 1.0 / 60.0;
    let mut last_frame = std::time::Instant::now();
    let mut paused = false;

    // Event loop
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    elwt.exit();
                }

                WindowEvent::KeyboardInput {
                    event: KeyEvent {
                        physical_key: PhysicalKey::Code(key_code),
                        state: ElementState::Pressed,
                        ..
                    },
                    ..
                } => {
                    match key_code {
                        KeyCode::Escape => elwt.exit(),
                        KeyCode::Space => {
                            paused = !paused;
                            log::info!("Simulation {}", if paused { "paused" } else { "resumed" });
                        }
                        KeyCode::KeyR => {
                            log::info!("Resetting simulation");
                            // Reset logic would go here
                        }
                        KeyCode::KeyE => {
                            log::info!("Toggling expanded mode");
                            // Toggle mode logic
                        }
                        KeyCode::KeyA => {
                            let analysis = engine.analyze();
                            log::info!("Analysis: {:?}", analysis.betti);
                            log::info!("Patterns: {:?}", analysis.patterns);
                        }
                        _ => {}
                    }
                }

                WindowEvent::Resized(size) => {
                    log::debug!("Window resized to {}x{}", size.width, size.height);
                }

                WindowEvent::RedrawRequested => {
                    // Calculate delta time
                    let now = std::time::Instant::now();
                    let _actual_delta = now.duration_since(last_frame).as_secs_f64();
                    last_frame = now;

                    // Update and render
                    if !paused {
                        engine.update(delta_time);
                    }

                    if let Err(e) = engine.render() {
                        log::error!("Render error: {}", e);
                    }
                }

                _ => {}
            },

            Event::AboutToWait => {
                window.request_redraw();
            }

            _ => {}
        }
    }).expect("Event loop error");
}

#[cfg(not(feature = "desktop"))]
fn run_windowed(_config: EngineConfig) {
    log::warn!("Desktop features not compiled in");
    log::info!("Recompile with: cargo build --features desktop");
}
