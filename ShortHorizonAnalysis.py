"""
Short Time Horizon Analysis (T=50 rounds)
==========================================
Test whether reference point formation weights matter with limited interaction time

Key hypothesis: With only 50 rounds, information sources won't fully converge,
so weight choice should affect outcomes.

Date: 2025-12-04
"""

import numpy as np
from Simulation import SimConfig, CoordinationGame
import json
from scipy import stats
import time


def run_replications(config_params, ref_point_method="bayesian", n_reps=30):
    """Run multiple replications"""
    results = []
    for rep in range(n_reps):
        config = SimConfig(**config_params, seed=4000 + rep)
        game = CoordinationGame(config, ref_point_method)
        summary = game.run_simulation(verbose=False)
        results.append(summary)
    return results


def calculate_statistics(results, metrics):
    """Calculate mean, std, SE, CI for metrics"""
    stats_dict = {}
    n = len(results)

    for metric in metrics:
        values = [r[metric] for r in results]
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        se = std / np.sqrt(n)
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


def compare_ttest(results1, results2, metric):
    """Perform t-test between two conditions"""
    values1 = [r[metric] for r in results1]
    values2 = [r[metric] for r in results2]

    t_stat, p_value = stats.ttest_ind(values1, values2)
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


def analyze_short_horizon(base_config, weight_configs, game_types, n_reps=30):
    """
    Test weight configurations with T=50 rounds
    """
    print("\n" + "="*70)
    print("SHORT TIME HORIZON ANALYSIS (T=50 rounds)")
    print("="*70)

    all_results = {}

    for game_type in game_types:
        print(f"\n{'='*70}")
        print(f"GAME TYPE: {game_type.upper()}")
        print('='*70)

        game_results = {}

        for weight_name, (w_recent, w_group, w_global) in weight_configs.items():
            print(f"\n  Testing: {weight_name} (r={w_recent:.2f}, g={w_group:.2f}, G={w_global:.2f})...", end=" ")

            config_params = base_config.copy()
            config_params["game_type"] = game_type
            config_params["weight_recent"] = w_recent
            config_params["weight_group"] = w_group
            config_params["weight_global"] = w_global

            start_time = time.time()
            results = run_replications(config_params, "bayesian", n_reps)

            metrics = ["final_coordination_rate", "final_group_favoritism",
                      "overall_coordination_rate"]
            stats_dict = calculate_statistics(results, metrics)

            # Track action preferences
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

            game_results[weight_name] = {
                "weights": {"recent": w_recent, "group": w_group, "global": w_global},
                "statistics": stats_dict,
                "raw_results": results
            }

            elapsed = time.time() - start_time
            print(f"Done ({elapsed:.1f}s)")
            print(f"    Final coord: {stats_dict['final_coordination_rate']['mean']:.3f} ± {stats_dict['final_coordination_rate']['se']:.3f}")
            print(f"    Favoritism: {stats_dict['final_group_favoritism']['mean']:.3f} ± {stats_dict['final_group_favoritism']['se']:.3f}")

        # Pairwise comparisons
        print(f"\n  {'='*68}")
        print(f"  Statistical Comparisons ({game_type})")
        print(f"  {'='*68}")

        weight_names = list(weight_configs.keys())
        comparisons = {}

        for i, name1 in enumerate(weight_names):
            for name2 in weight_names[i+1:]:
                key = f"{name1}_vs_{name2}"
                comparisons[key] = {}

                for metric in ["final_coordination_rate", "final_group_favoritism"]:
                    test = compare_ttest(
                        game_results[name1]["raw_results"],
                        game_results[name2]["raw_results"],
                        metric
                    )
                    comparisons[key][metric] = test

                    if test["significant_at_05"]:
                        sig = "***" if test["significant_at_01"] else "**"
                        print(f"  {name1:20s} vs {name2:20s} on {metric:30s}: "
                              f"Δ={test['mean_diff']:+.3f}, p={test['p_value']:.4f}, d={test['cohens_d']:+.3f} {sig}")

        if not any(any(c[m]["significant_at_05"] for m in ["final_coordination_rate", "final_group_favoritism"])
                   for c in comparisons.values()):
            print(f"  No significant differences (all p > 0.05)")

        all_results[game_type] = {
            "results": game_results,
            "comparisons": comparisons
        }

    return all_results


if __name__ == "__main__":
    print("="*70)
    print("Short Time Horizon Analysis")
    print("Testing if weights matter with T=50 rounds")
    print("Date: 2025-12-04")
    print("="*70)

    # Base configuration - SHORT TIME HORIZON
    base_config = {
        "n_agents": 16,
        "n_rounds": 50,  # KEY CHANGE: 50 instead of 200
        "n_groups": 2,
        "lambda_loss": 2.0,
        "initial_group_bias": 0.8,
        "n_initial_rounds": 10,
        "info_treatment": "full",
        "matching_type": "random",
        "recency_decay": 0.9,
        "min_history_for_belief": 1
    }

    # Test key weight configurations
    weight_configs = {
        "Recent-dominant": (0.90, 0.05, 0.05),
        "Group-dominant": (0.05, 0.90, 0.05),
        "Global-dominant": (0.05, 0.05, 0.90),
        "Balanced": (0.33, 0.33, 0.34),
        "Original": (0.50, 0.30, 0.20)
    }

    # Test all three game types
    game_types = ["coordination", "stag_hunt", "chicken"]

    N_REPS = 30

    start_time = time.time()

    # Run analysis
    results = analyze_short_horizon(base_config, weight_configs, game_types, N_REPS)

    # Calculate variance explained by weights
    print("\n" + "="*70)
    print("VARIANCE ANALYSIS")
    print("="*70)

    for game_type in game_types:
        print(f"\n{game_type.upper()}:")

        coord_values = []
        favor_values = []
        weight_labels = []

        for weight_name, data in results[game_type]["results"].items():
            coord_values.append(data["statistics"]["final_coordination_rate"]["mean"])
            favor_values.append(data["statistics"]["final_group_favoritism"]["mean"])
            weight_labels.append(weight_name)

        coord_range = max(coord_values) - min(coord_values)
        favor_range = max(favor_values) - min(favor_values)

        print(f"  Coordination rate range: {coord_range:.4f} ({min(coord_values):.3f} to {max(coord_values):.3f})")
        print(f"  Group favoritism range: {favor_range:.4f} ({min(favor_values):.3f} to {max(favor_values):.3f})")

        if coord_range > 0.02:  # More than 2% range
            print(f"  [YES] WEIGHTS MATTER - {coord_range*100:.1f}% range in coordination!")
            best_idx = np.argmax(coord_values)
            worst_idx = np.argmin(coord_values)
            print(f"    Best: {weight_labels[best_idx]} ({coord_values[best_idx]:.3f})")
            print(f"    Worst: {weight_labels[worst_idx]} ({coord_values[worst_idx]:.3f})")
        else:
            print(f"  [NO] Weights still have minimal effect (< 2% range)")

    # Save results
    summary = {
        "metadata": {
            "date": "2025-12-04",
            "n_rounds": 50,
            "n_replications": N_REPS,
            "base_config": base_config,
            "weight_configurations": {
                name: {"recent": w[0], "group": w[1], "global": w[2]}
                for name, w in weight_configs.items()
            }
        },
        "results": {
            game: {
                weight: {
                    "weights": data["weights"],
                    "statistics": data["statistics"]
                }
                for weight, data in results[game]["results"].items()
            }
            for game in game_types
        },
        "comparisons": {
            game: results[game]["comparisons"]
            for game in game_types
        }
    }

    with open("short_horizon_results.json", 'w') as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - start_time

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Total simulations: {len(weight_configs) * len(game_types) * N_REPS}")
    print(f"Results saved to: short_horizon_results.json")
