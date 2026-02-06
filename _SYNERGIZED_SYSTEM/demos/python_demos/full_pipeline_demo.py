#!/usr/bin/env python3
"""
Full Pipeline Demo - Optical Kirigami Moiré Computation
=======================================================

Demonstrates the complete encode-compute-readout cycle:
1. Input injection into kirigami reservoir
2. Cascading dynamics (fading memory)
3. Moiré pattern generation
4. Feature extraction
5. Classification readout

This showcases the system as a Visual Modal Cognitive Machine
that performs computation via optical interference rather than
digital logic gates.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from optical_kirigami_moire import (
    OpticalKirigamiMoire,
    PipelineConfig,
    ComputationMode,
    RuleEnforcer,
    LogicPolarity,
)


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def demo_rule_enforcement():
    """Demonstrate the three rule sets."""
    print_header("RULE SET ENFORCEMENT")

    rules = RuleEnforcer(lattice_constant=1.0, wavelength=550.0)
    print(f"\nRule Enforcer: {rules}")
    print(f"Talbot Length Z_T = {rules.talbot_length:.4f} μm")

    # Rule 1: Angular Commensurability
    print("\n--- Rule Set 1: Angular Commensurability ---")
    print("Commensurate angles (m,n) -> θ:")
    for (m, n), data in rules.COMMENSURATE_TABLE.items():
        print(f"  ({m},{n}): {data['angle']:.3f}° - {data['mode']}")

    # Test angle enforcement
    test_angles = [5.0, 10.0, 15.0, 22.0]
    print("\nAngle enforcement (auto-snap to commensurate):")
    for angle in test_angles:
        lock = rules.enforce_angle(angle)
        print(f"  {angle:.1f}° → {lock.angle:.3f}° (mode: {lock.cognitive_mode})")

    # Rule 2: Trilatic Tilt Symmetry
    print("\n--- Rule Set 2: Trilatic Tilt Symmetry ---")
    print("Valid tilt axes: k × 60° for k ∈ {0,1,2,3,4,5}")

    for k in range(6):
        lock = rules.enforce_tilt(5.0, k)
        print(f"  Axis {k}: {lock.axis_angle}°, a_eff = {lock.effective_lattice_constant:.4f} μm")

    # Rule 3: Talbot Distance
    print("\n--- Rule Set 3: Talbot Distance (Integer Gap) ---")
    print("Talbot resonances:")

    for order in range(1, 4):
        pos_gap, neg_gap = rules.get_logic_gaps(order)
        print(f"  Order {order}: INTEGER = {pos_gap:.4f}μm (AND/OR), "
              f"HALF = {neg_gap:.4f}μm (NAND/XOR)")

    # Test gap enforcement
    test_gaps = [1.0, 2.5, 4.0, 5.5]
    print("\nGap enforcement (auto-snap to Talbot):")
    for gap in test_gaps:
        lock = rules.enforce_gap(gap)
        print(f"  {gap:.1f}μm → {lock.gap:.4f}μm ({lock.mode}, {lock.polarity.value})")


def demo_moiré_encoding():
    """Demonstrate moiré pattern encoding."""
    print_header("MOIRÉ PATTERN ENCODING")

    # Create encoder
    config = PipelineConfig(
        grid_size=(32, 32),  # Smaller for demo
        lattice_constant=1.0,
        wavelength=550.0,
        cascade_steps=30,
    )
    okm = OpticalKirigamiMoire(config)

    print(f"\nEncoder: {okm}")

    # Create test input (simple gradient)
    print("\nCreating test input (gradient pattern)...")
    x = np.linspace(0, 1, 32)
    y = np.linspace(0, 1, 32)
    X, Y = np.meshgrid(x, y)
    test_input = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)

    print(f"Input shape: {test_input.shape}")
    print(f"Input range: [{test_input.min():.3f}, {test_input.max():.3f}]")

    # Test different computation modes
    modes = [
        ComputationMode.TRANSPARENT,
        ComputationMode.EDGE_DETECT,
        ComputationMode.TEXTURE,
        ComputationMode.COARSE,
    ]

    print("\nEncoding in different modes:")
    for mode in modes:
        okm.set_mode(mode)
        result = okm.encode(test_input)

        print(f"\n  Mode: {mode.value}")
        print(f"    Twist angle: {result.angle_lock.angle:.3f}°")
        print(f"    Gap: {result.gap_lock.gap:.4f}μm ({result.gap_lock.polarity.value})")
        print(f"    Cascade steps: {result.cascade_steps}")
        print(f"    Moiré pattern: {result.moire_pattern.shape}")
        print(f"    Intensity range: [{result.moire_pattern.min():.3f}, {result.moire_pattern.max():.3f}]")

        # Analyze pattern
        analysis = okm.analyze_pattern(result.moire_pattern)
        print(f"    Contrast: {analysis['contrast']:.3f}")
        print(f"    Dominant frequency: {analysis['dominant_frequency']:.1f}")


def demo_feature_extraction():
    """Demonstrate feature extraction for Vision LLM."""
    print_header("FEATURE EXTRACTION")

    config = PipelineConfig(grid_size=(32, 32))
    okm = OpticalKirigamiMoire(config)
    okm.set_mode(ComputationMode.TEXTURE)

    # Create different test patterns
    patterns = []
    labels = []

    print("\nGenerating test patterns...")

    # Pattern 1: Horizontal stripes
    x = np.linspace(0, 4*np.pi, 32)
    y = np.linspace(0, 4*np.pi, 32)
    X, Y = np.meshgrid(x, y)
    patterns.append(np.sin(Y))
    labels.append(0)
    print("  Pattern 0: Horizontal stripes")

    # Pattern 2: Vertical stripes
    patterns.append(np.sin(X))
    labels.append(1)
    print("  Pattern 1: Vertical stripes")

    # Pattern 3: Radial
    cx, cy = 16, 16
    R = np.sqrt((X/4 - cx/16)**2 + (Y/4 - cy/16)**2)
    patterns.append(np.sin(R * 2))
    labels.append(2)
    print("  Pattern 2: Radial pattern")

    # Encode all patterns
    print("\nEncoding and extracting features...")
    encoded_patterns = []
    for i, (pattern, label) in enumerate(zip(patterns, labels)):
        result = okm.encode(pattern)
        encoded_patterns.append(result.moire_pattern)

        print(f"\n  Pattern {label}:")
        print(f"    Feature vector length: {len(result.features)}")
        print(f"    First 5 features: {result.features[:5]}")

    # Show feature similarity
    print("\nFeature similarity (cosine):")
    for i in range(len(encoded_patterns)):
        for j in range(i+1, len(encoded_patterns)):
            f1 = okm.extract_features(encoded_patterns[i])
            f2 = okm.extract_features(encoded_patterns[j])
            similarity = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
            print(f"  Pattern {i} vs {j}: {similarity:.3f}")


def demo_reservoir_classification():
    """Demonstrate reservoir computing classification."""
    print_header("RESERVOIR CLASSIFICATION")

    config = PipelineConfig(grid_size=(32, 32), n_outputs=3)
    okm = OpticalKirigamiMoire(config)
    okm.set_mode(ComputationMode.TEXTURE)

    # Generate training data
    print("\nGenerating training data...")
    n_samples_per_class = 10
    train_patterns = []
    train_labels = []

    for class_idx in range(3):
        for _ in range(n_samples_per_class):
            # Add noise to make it interesting
            noise = np.random.randn(32, 32) * 0.1

            if class_idx == 0:
                # Horizontal stripes
                x = np.linspace(0, 4*np.pi, 32)
                y = np.linspace(0, 4*np.pi, 32)
                X, Y = np.meshgrid(x, y)
                pattern = np.sin(Y + np.random.rand() * 2) + noise

            elif class_idx == 1:
                # Vertical stripes
                x = np.linspace(0, 4*np.pi, 32)
                y = np.linspace(0, 4*np.pi, 32)
                X, Y = np.meshgrid(x, y)
                pattern = np.sin(X + np.random.rand() * 2) + noise

            else:
                # Checkerboard
                pattern = np.zeros((32, 32))
                for i in range(4):
                    for j in range(4):
                        if (i + j) % 2 == 0:
                            pattern[i*8:(i+1)*8, j*8:(j+1)*8] = 1
                pattern = pattern + noise

            result = okm.encode(pattern)
            train_patterns.append(result.moire_pattern)
            train_labels.append(class_idx)

    train_labels = np.array(train_labels)
    print(f"  Training samples: {len(train_patterns)}")
    print(f"  Classes: {np.unique(train_labels)}")

    # Train readout layer
    print("\nTraining readout layer...")
    accuracy = okm.train_readout(train_patterns, train_labels, verbose=True)
    print(f"Training accuracy: {accuracy:.2%}")

    # Test classification
    print("\nTesting classification on new samples...")
    test_patterns = []
    test_labels = []

    for class_idx in range(3):
        noise = np.random.randn(32, 32) * 0.1
        if class_idx == 0:
            x = np.linspace(0, 4*np.pi, 32)
            y = np.linspace(0, 4*np.pi, 32)
            X, Y = np.meshgrid(x, y)
            pattern = np.sin(Y + np.random.rand() * 2) + noise
        elif class_idx == 1:
            x = np.linspace(0, 4*np.pi, 32)
            y = np.linspace(0, 4*np.pi, 32)
            X, Y = np.meshgrid(x, y)
            pattern = np.sin(X + np.random.rand() * 2) + noise
        else:
            pattern = np.zeros((32, 32))
            for i in range(4):
                for j in range(4):
                    if (i + j) % 2 == 0:
                        pattern[i*8:(i+1)*8, j*8:(j+1)*8] = 1
            pattern = pattern + noise

        result = okm.encode(pattern)
        test_patterns.append(result.moire_pattern)
        test_labels.append(class_idx)

    test_labels = np.array(test_labels)

    print("\nTest results:")
    for i, (pattern, true_label) in enumerate(zip(test_patterns, test_labels)):
        pred_class, confidence = okm.classify(pattern)
        status = "✓" if pred_class == true_label else "✗"
        print(f"  Sample {i}: True={true_label}, Pred={pred_class}, "
              f"Conf={confidence:.2%} {status}")


def demo_logic_polarity():
    """Demonstrate logic polarity switching."""
    print_header("LOGIC POLARITY SWITCHING")

    config = PipelineConfig(grid_size=(32, 32))
    okm = OpticalKirigamiMoire(config)

    # Create simple test pattern
    x = np.linspace(0, 2*np.pi, 32)
    y = np.linspace(0, 2*np.pi, 32)
    X, Y = np.meshgrid(x, y)
    test_input = np.sin(X) * np.sin(Y)

    print("\nSwitching between POSITIVE and NEGATIVE logic:")

    for polarity in [LogicPolarity.POSITIVE, LogicPolarity.NEGATIVE]:
        okm.set_logic_polarity(polarity)
        result = okm.encode(test_input)

        print(f"\n  {polarity.value.upper()} Logic:")
        print(f"    Gap: {result.gap_lock.gap:.4f}μm")
        print(f"    Talbot mode: {result.gap_lock.mode}")
        print(f"    Mean intensity: {result.moire_pattern.mean():.3f}")

        # Show pattern statistics
        analysis = okm.analyze_pattern(result.moire_pattern)
        print(f"    Contrast: {analysis['contrast']:.3f}")


def main():
    """Run all demos."""
    print("\n" + "#" * 60)
    print("#  OPTICAL KIRIGAMI MOIRÉ - FULL PIPELINE DEMO")
    print("#  Visual Modal Cognitive Machine")
    print("#" * 60)

    demo_rule_enforcement()
    demo_moiré_encoding()
    demo_feature_extraction()
    demo_reservoir_classification()
    demo_logic_polarity()

    print_header("DEMO COMPLETE")
    print("""
The Optical Kirigami Moiré system demonstrates:

1. RULE ENFORCEMENT: Three rule sets ensure physically valid operation
   - Commensurate angles for periodic superlattices
   - Trilatic tilt symmetry for uniform computation
   - Talbot resonance for high-contrast logic

2. MOIRÉ ENCODING: Input data encoded via:
   - Kirigami reservoir (cascading dynamics)
   - Bichromatic interference (Cyan/Red)
   - Spectral logic gates (AND, OR, XOR, NAND)

3. FEATURE EXTRACTION: Moiré patterns contain:
   - Topological defects (forks)
   - Fringe orientation
   - Spatial frequency content
   - Color channel correlations

4. RESERVOIR CLASSIFICATION: Simple linear readout achieves
   classification by interpreting moiré fringe patterns.

This is computation via light interference, not transistors!
""")


if __name__ == "__main__":
    main()
