"""
HEMOC CLI Entry Point
======================

Usage:
    python -m hemoc verify             Run all Phillips matrix theorem checks
    python -m hemoc verify --fuzz      Run fuzz harness (entry + sign perturbation)
    python -m hemoc registry           List all experiments
    python -m hemoc registry --status  Summary by status/track/renderer
    python -m hemoc render-test        Run renderer contract test suite
    python -m hemoc golden-hadamard    Check Golden Hadamard axioms
"""

import argparse
import json
import sys


def cmd_verify(args):
    """Run Phillips matrix invariant verification."""
    from hemoc.theory.invariant_verifier import PhillipsInvariantVerifier

    verifier = PhillipsInvariantVerifier()
    report = verifier.run_all()

    print(f"\n{'='*60}")
    print(f"Phillips Matrix Invariant Verification")
    print(f"{'='*60}")
    print(f"  Total checks:  {report['n_total']}")
    print(f"  Passed:        {report['n_passed']}")
    print(f"  Failed:        {report['n_failed']}")
    print(f"  Overall:       {'PASS' if report['all_pass'] else 'FAIL'}")
    print()

    for check in report["checks"]:
        status = "PASS" if check["pass"] else "FAIL"
        print(f"  [{status}]  {check.get('theorem', check.get('test', '?'))}")

    if args.fuzz:
        print(f"\n{'='*60}")
        print(f"Fuzz Harness: Entry Perturbation ({args.fuzz_trials} trials)")
        print(f"{'='*60}")
        fuzz_entry = verifier.fuzz_entries(n_trials=args.fuzz_trials, seed=42)
        print(f"  Collision count = 14 rate: {fuzz_entry['collision_14_rate']:.3f}")
        print(f"  Rank = 4 rate:             {fuzz_entry['rank_4_rate']:.3f}")
        print(f"  Collision histogram:       {fuzz_entry['collision_count_histogram']}")

        print(f"\nFuzz Harness: Sign Perturbation ({args.fuzz_trials} trials)")
        print(f"{'='*60}")
        fuzz_sign = verifier.fuzz_signs(n_trials=args.fuzz_trials, seed=42)
        print(f"  Zero-collision rate:       {fuzz_sign['zero_collision_rate']:.3f}")
        print(f"  14-collision rate:         {fuzz_sign['fourteen_collision_rate']:.3f}")
        print(f"  Collision histogram:       {fuzz_sign['collision_histogram']}")

    if report["all_pass"]:
        print("\nAll invariants verified.")
    else:
        print("\nWARNING: Some invariants failed!")
        sys.exit(1)


def cmd_registry(args):
    """List experiments from the registry."""
    from hemoc.experiments.registry import ExperimentRegistry

    reg = ExperimentRegistry()

    if args.status:
        summary = reg.summary()
        print(json.dumps(summary, indent=2))
        return

    experiments = reg.list_all()
    if not experiments:
        print("No experiments found in registry.")
        return

    print(f"\n{'ID':<10} {'Track':<6} {'Status':<14} {'Renderer Dep':<22} {'Name'}")
    print("-" * 90)
    for exp in experiments:
        print(f"{exp.id:<10} {exp.track:<6} {exp.status:<14} "
              f"{exp.renderer_dependence:<22} {exp.name}")

    summary = reg.summary()
    print(f"\nTotal: {summary['total_experiments']} experiments")
    print(f"By status: {summary['by_status']}")


def cmd_render_test(args):
    """Run renderer contract test suite."""
    from hemoc.render.dual_channel_renderer import DualChannelGaloisRenderer
    from hemoc.render.renderer_test_suite import (
        test_contract_compliance,
        test_galois_preservation,
    )

    print("Initializing DualChannelGaloisRenderer...")
    renderer = DualChannelGaloisRenderer(resolution=32)

    print("Running contract compliance tests...")
    compliance = test_contract_compliance(renderer)
    print(f"  Contract: {'PASS' if compliance['all_pass'] else 'FAIL'} "
          f"({compliance['n_passed']}/{compliance['n_total']})")

    for t in compliance["tests"]:
        status = "PASS" if t["pass"] else "FAIL"
        print(f"    [{status}] {t['test']}")

    print("\nRunning Galois preservation test...")
    galois = test_galois_preservation(renderer, n_samples=20)
    print(f"  Galois supported: {galois.get('galois_supported', False)}")
    if galois.get("galois_supported"):
        print(f"  Mean ratio:       {galois['mean_ratio']:.6f} (expected {galois['expected_ratio']:.6f})")
        print(f"  Mean deviation:   {galois['mean_deviation']:.6f}")
        print(f"  Valid rate:       {galois['valid_rate']:.3f}")


def cmd_golden_hadamard(args):
    """Check Golden Hadamard axioms."""
    from hemoc.theory.golden_hadamard import GoldenHadamardChecker
    from hemoc.core.phillips_matrix import PHILLIPS_MATRIX

    checker = GoldenHadamardChecker(PHILLIPS_MATRIX)
    result = checker.check_all()

    print(f"\n{'='*60}")
    print(f"Golden Hadamard Class Check")
    print(f"{'='*60}")
    print(f"  Overall: {'ALL AXIOMS SATISFIED' if result['all_pass'] else 'INCOMPLETE'}")
    print(f"  Passed: {result['n_passed']}/{result['n_total']}")

    for check in result["checks"]:
        status = "PASS" if check["pass"] else "FAIL"
        print(f"  [{status}]  {check['axiom']}")


def main():
    parser = argparse.ArgumentParser(
        prog="hemoc",
        description="HEMOC: Hexagonal Emergent Moire Optical Cognition CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    # verify
    p_verify = subparsers.add_parser("verify", help="Run Phillips matrix verification")
    p_verify.add_argument("--fuzz", action="store_true", help="Run fuzz harness")
    p_verify.add_argument("--fuzz-trials", type=int, default=200, help="Number of fuzz trials")

    # registry
    p_reg = subparsers.add_parser("registry", help="List experiments")
    p_reg.add_argument("--status", action="store_true", help="Show summary only")

    # render-test
    subparsers.add_parser("render-test", help="Run renderer contract tests")

    # golden-hadamard
    subparsers.add_parser("golden-hadamard", help="Check Golden Hadamard axioms")

    args = parser.parse_args()

    if args.command == "verify":
        cmd_verify(args)
    elif args.command == "registry":
        cmd_registry(args)
    elif args.command == "render-test":
        cmd_render_test(args)
    elif args.command == "golden-hadamard":
        cmd_golden_hadamard(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
