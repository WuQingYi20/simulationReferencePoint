"""
Parameter Sensitivity Analysis: Reference Point Formation Weights
==================================================================
Systematic exploration of how reference point formation weights affect outcomes

Research Question: Which combinations of (weight_recent, weight_group, weight_global)
produce different coordination outcomes, especially in asymmetric games?

Date: 2025-12-04
"""

import numpy as np
from Simulation import SimConfig, CoordinationGame
import json
from scipy import stats
import time
import itertools
from collections import defaultdict


def run_replications(config_params, ref_point_method="bayesian", n_reps=20):
    """Run multiple replications of a simulation condition"""
    results = []

    for rep in range(n_reps):
        config = SimConfig(**config_params, seed=2000 + rep)
        game = CoordinationGame(config, ref_point_method)
        summary = game.run_simulation(verbose=False)
        results.append(summary)

    return results


def calculate_statistics(results, metrics):
    """Calculate mean and std for specified metrics"""
    stats_dict = {}
    n = len(results)

    for metric in metrics:
        values = [r[metric] for r in results]
        mean = np.mean(values)
        std = np.std(values, ddof=1) if n > 1 else 0

        stats_dict[metric] = {
            "mean": float(mean),
            "std": float(std),
            "n": n
        }

    return stats_dict


def generate_weight_grid(n_points=5):
    """
    Generate grid of weight combinations

    Returns:
        List of (weight_recent, weight_group, weight_global) tuples that sum to 1

    Note: Avoiding exact zeros due to division in Bayesian calculation
    Using 0.01 as minimum weight instead of 0.0
    """
    weights = []
    MIN_WEIGHT = 0.01  # Avoid division by zero

    # Near-pure strategies (corners of simplex, but not exact zeros)
    weights.extend([
        (0.98, MIN_WEIGHT, MIN_WEIGHT),  # Dominant recent
        (MIN_WEIGHT, 0.98, MIN_WEIGHT),  # Dominant group
        (MIN_WEIGHT, MIN_WEIGHT, 0.98),  # Dominant global
    ])

    # Edge strategies (two-way mixes with minimal third weight)
    for i in range(1, n_points):
        alpha = i / n_points
        beta = 1 - alpha - MIN_WEIGHT
        if beta > MIN_WEIGHT:
            weights.extend([
                (alpha, beta, MIN_WEIGHT),      # Recent + Group dominant
                (alpha, MIN_WEIGHT, beta),      # Recent + Global dominant
                (MIN_WEIGHT, alpha, beta),      # Group + Global dominant
            ])

    # Interior points (three-way mixes)
    step = 1.0 / n_points
    for w_recent in np.arange(step, 1.0 - 2*MIN_WEIGHT, step):
        for w_group in np.arange(step, 1.0 - w_recent - MIN_WEIGHT, step):
            w_global = 1.0 - w_recent - w_group
            if w_global >= MIN_WEIGHT:  # Ensure all weights positive
                weights.append((w_recent, w_group, w_global))

    # Add some specific interesting points
    weights.extend([
        (0.33, 0.33, 0.34),  # Equal weights
        (0.5, 0.3, 0.2),     # Original default (recent-biased)
        (0.2, 0.6, 0.2),     # Group-biased
        (0.6, 0.2, 0.2),     # Strong recent
        (0.25, 0.5, 0.25),   # Group-centric
    ])

    # Remove duplicates and round
    weights = list(set([(round(w[0], 2), round(w[1], 2), round(w[2], 2)) for w in weights]))

    # Filter to ensure they sum to ~1.0
    weights = [w for w in weights if abs(sum(w) - 1.0) < 0.01]

    return weights


def analyze_weight_sensitivity(base_config, game_types, weight_grid, n_reps=20):
    """
    Test each weight combination across game types
    """
    print("\n" + "="*70)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*70)
    print(f"Weight combinations to test: {len(weight_grid)}")
    print(f"Game types: {len(game_types)}")
    print(f"Replications per condition: {n_reps}")
    print(f"Total simulations: {len(weight_grid) * len(game_types) * n_reps}")
    print("="*70)

    metrics = ["final_coordination_rate", "final_group_favoritism",
               "overall_coordination_rate"]

    all_results = defaultdict(dict)

    total_conditions = len(weight_grid) * len(game_types)
    condition_num = 0

    for game_type in game_types:
        print(f"\n{'='*70}")
        print(f"GAME TYPE: {game_type.upper()}")
        print('='*70)

        for w_recent, w_group, w_global in weight_grid:
            condition_num += 1
            progress = (condition_num / total_conditions) * 100

            # Create config with these weights
            config_params = base_config.copy()
            config_params["game_type"] = game_type
            config_params["weight_recent"] = w_recent
            config_params["weight_group"] = w_group
            config_params["weight_global"] = w_global

            # Run simulations
            start_time = time.time()
            results = run_replications(config_params, "bayesian", n_reps)
            stats_dict = calculate_statistics(results, metrics)

            # Track action preferences
            group0_A = [r["group_action_preferences"].get(0, 0) for r in results]
            group1_A = [r["group_action_preferences"].get(1, 0) for r in results]
            stats_dict["group0_prop_A"] = {
                "mean": float(np.mean(group0_A)),
                "std": float(np.std(group0_A, ddof=1)) if n_reps > 1 else 0
            }
            stats_dict["group1_prop_A"] = {
                "mean": float(np.mean(group1_A)),
                "std": float(np.std(group1_A, ddof=1)) if n_reps > 1 else 0
            }

            # Store results
            weight_key = f"wr{w_recent:.2f}_wg{w_group:.2f}_wG{w_global:.2f}"
            all_results[game_type][weight_key] = {
                "weights": {
                    "recent": w_recent,
                    "group": w_group,
                    "global": w_global
                },
                "statistics": stats_dict
            }

            elapsed = time.time() - start_time

            # Print progress every 10%
            if condition_num % max(1, total_conditions // 10) == 0:
                print(f"  Progress: {progress:5.1f}% | "
                      f"Recent={w_recent:.2f}, Group={w_group:.2f}, Global={w_global:.2f} | "
                      f"Coord={stats_dict['final_coordination_rate']['mean']:.3f} | "
                      f"({elapsed:.1f}s)")

    return all_results


def find_extreme_outcomes(results, game_type, metric):
    """
    Find weight combinations that produce extreme (max/min) outcomes for a metric
    """
    values = []
    for weight_key, data in results[game_type].items():
        value = data["statistics"][metric]["mean"]
        weights = data["weights"]
        values.append((value, weight_key, weights))

    values.sort()

    return {
        "min": {
            "value": values[0][0],
            "weight_key": values[0][1],
            "weights": values[0][2]
        },
        "max": {
            "value": values[-1][0],
            "weight_key": values[-1][1],
            "weights": values[-1][2]
        },
        "range": values[-1][0] - values[0][0],
        "all_values": [v[0] for v in values]
    }


def analyze_variance_explained(results, game_type):
    """
    Calculate how much variance in outcomes is explained by weight choices
    """
    # Extract data
    coord_rates = []
    favoritism = []
    w_recents = []
    w_groups = []
    w_globals = []

    for weight_key, data in results[game_type].items():
        coord_rates.append(data["statistics"]["final_coordination_rate"]["mean"])
        favoritism.append(data["statistics"]["final_group_favoritism"]["mean"])
        w_recents.append(data["weights"]["recent"])
        w_groups.append(data["weights"]["group"])
        w_globals.append(data["weights"]["global"])

    coord_rates = np.array(coord_rates)
    favoritism = np.array(favoritism)
    w_recents = np.array(w_recents)
    w_groups = np.array(w_groups)
    w_globals = np.array(w_globals)

    # Calculate correlations
    correlations = {
        "coordination_rate": {
            "vs_weight_recent": float(np.corrcoef(coord_rates, w_recents)[0, 1]),
            "vs_weight_group": float(np.corrcoef(coord_rates, w_groups)[0, 1]),
            "vs_weight_global": float(np.corrcoef(coord_rates, w_globals)[0, 1])
        },
        "group_favoritism": {
            "vs_weight_recent": float(np.corrcoef(favoritism, w_recents)[0, 1]),
            "vs_weight_group": float(np.corrcoef(favoritism, w_groups)[0, 1]),
            "vs_weight_global": float(np.corrcoef(favoritism, w_globals)[0, 1])
        }
    }

    # Calculate variance
    variance = {
        "coordination_rate": {
            "mean": float(np.mean(coord_rates)),
            "std": float(np.std(coord_rates)),
            "cv": float(np.std(coord_rates) / np.mean(coord_rates) * 100),
            "range": float(np.max(coord_rates) - np.min(coord_rates))
        },
        "group_favoritism": {
            "mean": float(np.mean(favoritism)),
            "std": float(np.std(favoritism)),
            "range": float(np.max(favoritism) - np.min(favoritism))
        }
    }

    return {
        "correlations": correlations,
        "variance": variance
    }


def compare_to_baseline_methods(results, game_type, baseline_results):
    """
    Compare bayesian method with different weights to baseline methods
    """
    comparisons = {}

    # Get baseline performance
    baselines = {}
    for method in ["global", "group", "pairwise", "recent"]:
        if method in baseline_results:
            baselines[method] = baseline_results[method]["statistics"]["final_coordination_rate"]["mean"]

    # Find bayesian configurations that match each baseline
    for method, baseline_value in baselines.items():
        best_match = None
        min_diff = float('inf')

        for weight_key, data in results[game_type].items():
            value = data["statistics"]["final_coordination_rate"]["mean"]
            diff = abs(value - baseline_value)

            if diff < min_diff:
                min_diff = diff
                best_match = {
                    "weight_key": weight_key,
                    "weights": data["weights"],
                    "value": value,
                    "difference": value - baseline_value
                }

        comparisons[method] = best_match

    return comparisons


if __name__ == "__main__":
    print("="*70)
    print("Parameter Sensitivity Analysis")
    print("Reference Point Formation Weights")
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
        "info_treatment": "full",
        "matching_type": "random",
        "recency_decay": 0.9,
        "min_history_for_belief": 1
        # weights will be varied
    }

    # Focus on games where we expect differences
    game_types = ["stag_hunt", "chicken", "coordination"]

    # Generate weight grid
    print("\nGenerating weight combinations...")
    weight_grid = generate_weight_grid(n_points=4)  # 4 points per dimension
    print(f"Generated {len(weight_grid)} unique weight combinations")

    # Show some examples
    print("\nExample weight combinations:")
    for i, (wr, wg, wG) in enumerate(weight_grid[:10]):
        print(f"  {i+1}. Recent={wr:.2f}, Group={wg:.2f}, Global={wG:.2f}")
    if len(weight_grid) > 10:
        print(f"  ... and {len(weight_grid) - 10} more")

    N_REPS = 20  # Balance between statistical power and computation time

    overall_start = time.time()

    # Run parameter sweep
    results = analyze_weight_sensitivity(base_config, game_types, weight_grid, N_REPS)

    print("\n" + "="*70)
    print("ANALYSIS: Finding Extreme Outcomes")
    print("="*70)

    # Analyze each game
    summary = {
        "metadata": {
            "date": "2025-12-04",
            "n_weight_combinations": len(weight_grid),
            "n_replications": N_REPS,
            "total_simulations": len(weight_grid) * len(game_types) * N_REPS,
            "base_config": base_config
        },
        "results": results,
        "analysis": {}
    }

    for game_type in game_types:
        print(f"\n{game_type.upper()}:")
        print("-" * 70)

        # Find extremes
        coord_extremes = find_extreme_outcomes(results, game_type, "final_coordination_rate")
        favor_extremes = find_extreme_outcomes(results, game_type, "final_group_favoritism")

        print(f"\n  Coordination Rate:")
        print(f"    Range: {coord_extremes['range']:.4f} "
              f"({coord_extremes['min']['value']:.3f} to {coord_extremes['max']['value']:.3f})")
        print(f"    Min at: Recent={coord_extremes['min']['weights']['recent']:.2f}, "
              f"Group={coord_extremes['min']['weights']['group']:.2f}, "
              f"Global={coord_extremes['min']['weights']['global']:.2f}")
        print(f"    Max at: Recent={coord_extremes['max']['weights']['recent']:.2f}, "
              f"Group={coord_extremes['max']['weights']['group']:.2f}, "
              f"Global={coord_extremes['max']['weights']['global']:.2f}")

        print(f"\n  Group Favoritism:")
        print(f"    Range: {favor_extremes['range']:.4f} "
              f"({favor_extremes['min']['value']:.3f} to {favor_extremes['max']['value']:.3f})")
        print(f"    Min at: Recent={favor_extremes['min']['weights']['recent']:.2f}, "
              f"Group={favor_extremes['min']['weights']['group']:.2f}, "
              f"Global={favor_extremes['min']['weights']['global']:.2f}")
        print(f"    Max at: Recent={favor_extremes['max']['weights']['recent']:.2f}, "
              f"Group={favor_extremes['max']['weights']['group']:.2f}, "
              f"Global={favor_extremes['max']['weights']['global']:.2f}")

        # Variance analysis
        variance_analysis = analyze_variance_explained(results, game_type)

        print(f"\n  Variance Analysis:")
        print(f"    Coordination rate std: {variance_analysis['variance']['coordination_rate']['std']:.4f} "
              f"(CV: {variance_analysis['variance']['coordination_rate']['cv']:.2f}%)")
        print(f"    Group favoritism std: {variance_analysis['variance']['group_favoritism']['std']:.4f}")

        print(f"\n  Correlations with Coordination Rate:")
        corr_coord = variance_analysis['correlations']['coordination_rate']
        print(f"    vs weight_recent: {corr_coord['vs_weight_recent']:+.3f}")
        print(f"    vs weight_group:  {corr_coord['vs_weight_group']:+.3f}")
        print(f"    vs weight_global: {corr_coord['vs_weight_global']:+.3f}")

        print(f"\n  Correlations with Group Favoritism:")
        corr_favor = variance_analysis['correlations']['group_favoritism']
        print(f"    vs weight_recent: {corr_favor['vs_weight_recent']:+.3f}")
        print(f"    vs weight_group:  {corr_favor['vs_weight_group']:+.3f}")
        print(f"    vs weight_global: {corr_favor['vs_weight_global']:+.3f}")

        # Store in summary
        summary["analysis"][game_type] = {
            "extremes": {
                "coordination_rate": coord_extremes,
                "group_favoritism": favor_extremes
            },
            "variance_analysis": variance_analysis
        }

    # Save results
    with open("parameter_sensitivity_results.json", 'w') as f:
        json.dump(summary, f, indent=2)

    overall_elapsed = time.time() - overall_start

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"Total time: {overall_elapsed:.1f} seconds ({overall_elapsed/60:.1f} minutes)")
    print(f"Total simulations: {len(weight_grid) * len(game_types) * N_REPS}")
    print(f"Results saved to: parameter_sensitivity_results.json")

    # Key findings summary
    print("\n" + "="*70)
    print("KEY FINDINGS SUMMARY")
    print("="*70)

    for game_type in game_types:
        variance = summary["analysis"][game_type]["variance_analysis"]["variance"]
        coord_range = variance["coordination_rate"]["range"]
        favor_range = variance["group_favoritism"]["range"]

        print(f"\n{game_type.upper()}:")
        print(f"  Weight choice explains {coord_range*100:.2f}% range in coordination")
        print(f"  Weight choice explains {favor_range*100:.2f}% range in favoritism")

        if coord_range > 0.05:  # More than 5% range
            print(f"  [YES] WEIGHTS MATTER - substantial variation in outcomes")
        else:
            print(f"  [NO] Weights have minimal effect (< 5% range)")
