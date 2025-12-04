"""
Matching Protocol Comparison Analysis
======================================
Compare random matching vs round-robin matching with statistical validation

Date: 2025-12-04
Author: Statistical Analysis Extension
"""

import numpy as np
from Simulation import SimConfig, CoordinationGame
import json
from scipy import stats
import time


def run_replications(config_params, ref_point_method="bayesian", n_reps=30):
    """Run multiple replications of a simulation condition"""
    results = []

    for rep in range(n_reps):
        # Create config with unique seed
        config = SimConfig(**config_params, seed=1000 + rep)

        # Run simulation
        game = CoordinationGame(config, ref_point_method)
        summary = game.run_simulation(verbose=False)

        results.append(summary)

    return results


def calculate_statistics(results, metrics):
    """Calculate mean, std, CI for specified metrics"""
    stats_dict = {}
    n = len(results)

    for metric in metrics:
        values = [r[metric] for r in results]
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        se = std / np.sqrt(n)

        # 95% confidence interval
        ci_95 = stats.t.interval(0.95, n-1, loc=mean, scale=se)

        stats_dict[metric] = {
            "mean": float(mean),
            "std": float(std),
            "se": float(se),
            "ci_95_lower": float(ci_95[0]),
            "ci_95_upper": float(ci_95[1]),
            "n": n,
            "raw_values": [float(v) for v in values]
        }

    return stats_dict


def compare_conditions_ttest(results1, results2, metric):
    """Perform independent t-test between two conditions"""
    values1 = [r[metric] for r in results1]
    values2 = [r[metric] for r in results2]

    # Independent t-test
    t_stat, p_value = stats.ttest_ind(values1, values2)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(values1, ddof=1) + np.var(values2, ddof=1)) / 2)
    cohens_d = (np.mean(values1) - np.mean(values2)) / pooled_std if pooled_std > 0 else 0

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "significant_at_05": bool(p_value < 0.05),
        "significant_at_01": bool(p_value < 0.01),
        "mean_diff": float(np.mean(values1) - np.mean(values2))
    }


def analyze_matching_protocols(base_config, game_types, n_reps=30):
    """
    Compare random vs round-robin matching across different game types
    """
    print("\n" + "="*70)
    print("MATCHING PROTOCOL COMPARISON")
    print(f"Running {n_reps} replications per condition...")
    print("="*70)

    matching_types = ["random", "round_robin"]
    metrics = ["final_coordination_rate", "final_group_favoritism",
               "overall_coordination_rate", "ingroup_coordination_rate",
               "outgroup_coordination_rate"]

    all_results = {}

    for game_type in game_types:
        print(f"\n{'='*70}")
        print(f"GAME TYPE: {game_type.upper()}")
        print('='*70)

        game_results = {}

        for matching in matching_types:
            print(f"\n  Matching: {matching}...", end=" ")
            start_time = time.time()

            # Create config
            config_params = base_config.copy()
            config_params["game_type"] = game_type
            config_params["matching_type"] = matching

            results = run_replications(config_params, "bayesian", n_reps)
            stats_dict = calculate_statistics(results, metrics)

            # Also track action preferences
            group0_A = [r["group_action_preferences"].get(0, 0) for r in results]
            group1_A = [r["group_action_preferences"].get(1, 0) for r in results]
            stats_dict["group0_prop_A"] = {
                "mean": float(np.mean(group0_A)),
                "std": float(np.std(group0_A, ddof=1)),
                "se": float(np.std(group0_A, ddof=1) / np.sqrt(n_reps))
            }
            stats_dict["group1_prop_A"] = {
                "mean": float(np.mean(group1_A)),
                "std": float(np.std(group1_A, ddof=1)),
                "se": float(np.std(group1_A, ddof=1) / np.sqrt(n_reps))
            }

            game_results[matching] = {
                "statistics": stats_dict,
                "raw_results": results
            }

            elapsed = time.time() - start_time
            print(f"Done ({elapsed:.1f}s)")

            # Print summary
            print(f"    Final coordination: {stats_dict['final_coordination_rate']['mean']:.3f} "
                  f"± {stats_dict['final_coordination_rate']['se']:.3f}")
            print(f"    Group favoritism: {stats_dict['final_group_favoritism']['mean']:.3f} "
                  f"± {stats_dict['final_group_favoritism']['se']:.3f}")
            print(f"    Group 0 prop(A): {stats_dict['group0_prop_A']['mean']:.3f}")
            print(f"    Group 1 prop(A): {stats_dict['group1_prop_A']['mean']:.3f}")

        # Compare random vs round-robin for this game
        print(f"\n  {'='*68}")
        print(f"  Statistical Comparison: Random vs Round-Robin ({game_type})")
        print(f"  {'='*68}")

        comparisons = {}
        for metric in metrics:
            test_result = compare_conditions_ttest(
                game_results["random"]["raw_results"],
                game_results["round_robin"]["raw_results"],
                metric
            )
            comparisons[metric] = test_result

            if test_result["significant_at_05"]:
                sig_marker = "***" if test_result["significant_at_01"] else "**"
                print(f"  {metric:30s}: Δ={test_result['mean_diff']:+.4f}, "
                      f"t={test_result['t_statistic']:6.3f}, "
                      f"p={test_result['p_value']:.4f}, "
                      f"d={test_result['cohens_d']:6.3f} {sig_marker}")

        if not any(c["significant_at_05"] for c in comparisons.values()):
            print(f"  No significant differences (all p > 0.05)")

        all_results[game_type] = {
            "results": game_results,
            "comparisons": comparisons
        }

    return all_results


if __name__ == "__main__":
    print("="*70)
    print("Matching Protocol Analysis with Statistical Validation")
    print("Date: 2025-12-04")
    print("="*70)

    # Base configuration
    base_config = {
        "n_agents": 16,
        "n_rounds": 200,
        "n_groups": 2,
        "lambda_loss": 2.0,
        "initial_group_bias": 0.8,
        "n_initial_rounds": 10,
        "info_treatment": "full"
        # game_type and matching_type will be set per condition
    }

    # Test across all game types
    game_types = ["coordination", "stag_hunt", "chicken"]

    N_REPS = 30

    print(f"\nConfiguration:")
    print(f"  N = {base_config['n_agents']}, T = {base_config['n_rounds']}")
    print(f"  Replications per condition: {N_REPS}")
    print(f"  Game types: {', '.join(game_types)}")
    print(f"  Matching protocols: random, round_robin")
    print(f"  Total simulations: {len(game_types) * 2 * N_REPS}")

    overall_start = time.time()

    # Run analysis
    results = analyze_matching_protocols(base_config, game_types, N_REPS)

    # Prepare summary for JSON
    summary_results = {
        "metadata": {
            "date": "2025-12-04",
            "base_config": base_config,
            "n_replications": N_REPS,
            "game_types": game_types,
            "total_simulations": len(game_types) * 2 * N_REPS
        },
        "results": {}
    }

    for game_type, data in results.items():
        summary_results["results"][game_type] = {
            "random": data["results"]["random"]["statistics"],
            "round_robin": data["results"]["round_robin"]["statistics"],
            "statistical_comparison": data["comparisons"]
        }

    # Save results
    with open("matching_protocol_analysis.json", 'w') as f:
        json.dump(summary_results, f, indent=2)

    overall_elapsed = time.time() - overall_start

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"Total time: {overall_elapsed:.1f} seconds ({overall_elapsed/60:.1f} minutes)")
    print(f"Total simulations run: {len(game_types) * 2 * N_REPS}")
    print(f"Results saved to: matching_protocol_analysis.json")

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY: Key Differences Between Matching Protocols")
    print("="*70)

    for game_type in game_types:
        print(f"\n{game_type.upper()}:")
        random_coord = results[game_type]["results"]["random"]["statistics"]["final_coordination_rate"]["mean"]
        rr_coord = results[game_type]["results"]["round_robin"]["statistics"]["final_coordination_rate"]["mean"]
        diff = rr_coord - random_coord

        comp = results[game_type]["comparisons"]["final_coordination_rate"]
        sig = "***" if comp["significant_at_01"] else ("**" if comp["significant_at_05"] else "ns")

        print(f"  Random:      {random_coord:.3f}")
        print(f"  Round-robin: {rr_coord:.3f}")
        print(f"  Difference:  {diff:+.3f} ({sig})")
        print(f"  p-value:     {comp['p_value']:.4f}")
        print(f"  Cohen's d:   {comp['cohens_d']:.3f}")
