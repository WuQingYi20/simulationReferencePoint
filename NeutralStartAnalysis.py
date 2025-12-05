"""
Neutral Start Analysis (T=30 rounds, β=0.5)
============================================
Test with NO initial group bias and very limited time

Key changes:
- initial_group_bias = 0.5 (groups start identical, no convention)
- n_rounds = 30 (extreme time scarcity)
- n_initial_rounds = 5 (shorter convention phase)

Hypothesis: With no initial bias and extreme scarcity, weights should matter
because there's genuine conflict and information sources will diverge.

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
        config = SimConfig(**config_params, seed=5000 + rep)
        game = CoordinationGame(config, ref_point_method)
        summary = game.run_simulation(verbose=False)
        results.append(summary)
    return results


def calculate_statistics(results, metrics):
    """Calculate statistics"""
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
    """T-test between conditions"""
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


def analyze_neutral_start(base_config, weight_configs, game_types, n_reps=30):
    """
    Test with neutral start (no initial bias) and extreme time scarcity
    """
    print("\n" + "="*70)
    print("NEUTRAL START ANALYSIS (T=30 rounds, β=0.5)")
    print("No initial group bias - pure conflict resolution")
    print("="*70)

    all_results = {}

    for game_type in game_types:
        print(f"\n{'='*70}")
        print(f"GAME TYPE: {game_type.upper()}")
        print('='*70)

        game_results = {}

        for weight_name, (w_recent, w_group, w_global) in weight_configs.items():
            print(f"\n  {weight_name:20s} (r={w_recent:.2f}, g={w_group:.2f}, G={w_global:.2f})...", end=" ")

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

            # Action preferences
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
            coord = stats_dict['final_coordination_rate']['mean']
            favor = stats_dict['final_group_favoritism']['mean']
            print(f"Done ({elapsed:.1f}s)")
            print(f"    Coord: {coord:.3f} ± {stats_dict['final_coordination_rate']['se']:.3f}, "
                  f"Favor: {favor:+.3f} ± {stats_dict['final_group_favoritism']['se']:.3f}")

        # Statistical comparisons
        print(f"\n  {'='*68}")
        print(f"  Pairwise Comparisons ({game_type})")
        print(f"  {'='*68}")

        weight_names = list(weight_configs.keys())
        comparisons = {}
        any_significant = False

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
                        any_significant = True
                        sig = "***" if test["significant_at_01"] else "**"
                        print(f"  {name1:18s} vs {name2:18s} | {metric[:15]:15s}: "
                              f"Δ={test['mean_diff']:+.4f}, p={test['p_value']:.4f}, "
                              f"d={test['cohens_d']:+.3f} {sig}")

        if not any_significant:
            print(f"  No significant differences (all p > 0.05)")

        all_results[game_type] = {
            "results": game_results,
            "comparisons": comparisons
        }

    return all_results


if __name__ == "__main__":
    print("="*70)
    print("Neutral Start Analysis")
    print("Testing with no initial bias (β=0.5) and extreme scarcity (T=30)")
    print("Date: 2025-12-04")
    print("="*70)

    # NEUTRAL START configuration
    base_config = {
        "n_agents": 16,
        "n_rounds": 30,           # VERY SHORT
        "n_groups": 2,
        "lambda_loss": 2.25,
        "initial_group_bias": 0.5,  # NO BIAS - groups start identical!
        "n_initial_rounds": 5,      # Shorter initial phase
        "info_treatment": "full",
        "matching_type": "random",
        "recency_decay": 0.9,
        "min_history_for_belief": 1
    }

    # Test key weight configurations
    weight_configs = {
        "Recent-dom": (0.90, 0.05, 0.05),
        "Group-dom": (0.05, 0.90, 0.05),
        "Global-dom": (0.05, 0.05, 0.90),
        "Balanced": (0.33, 0.33, 0.34),
        "Original": (0.50, 0.30, 0.20)
    }

    game_types = ["coordination", "stag_hunt", "chicken"]
    N_REPS = 30

    print(f"\nKey differences from previous tests:")
    print(f"  - initial_group_bias: 0.5 (was 0.8) → NO initial conventions")
    print(f"  - n_rounds: 30 (was 50/200) → EXTREME time scarcity")
    print(f"  - n_initial_rounds: 5 (was 10) → Faster conflict emergence")
    print(f"\nExpectation: With no bias and limited time, weights SHOULD matter!")

    start_time = time.time()

    # Run analysis
    results = analyze_neutral_start(base_config, weight_configs, game_types, N_REPS)

    # Variance analysis
    print("\n" + "="*70)
    print("VARIANCE ANALYSIS - DO WEIGHTS MATTER?")
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
        coord_cv = np.std(coord_values) / np.mean(coord_values) * 100

        print(f"  Coordination rate:")
        print(f"    Range: {coord_range:.4f} ({min(coord_values):.3f} to {max(coord_values):.3f})")
        print(f"    CV: {coord_cv:.2f}%")

        print(f"  Group favoritism:")
        print(f"    Range: {favor_range:.4f} ({min(favor_values):.3f} to {max(favor_values):.3f})")

        # Determine if weights matter
        if coord_range > 0.02:  # More than 2% range
            print(f"  >>> [YES!] WEIGHTS MATTER - {coord_range*100:.1f}% coordination range!")
            best_idx = np.argmax(coord_values)
            worst_idx = np.argmin(coord_values)
            print(f"      Best: {weight_labels[best_idx]} ({coord_values[best_idx]:.3f})")
            print(f"      Worst: {weight_labels[worst_idx]} ({coord_values[worst_idx]:.3f})")
        else:
            print(f"  >>> [NO] Weights still minimal effect ({coord_range*100:.1f}% range)")

    # Save results
    summary = {
        "metadata": {
            "date": "2025-12-04",
            "analysis_type": "neutral_start",
            "n_rounds": 30,
            "initial_group_bias": 0.5,
            "n_initial_rounds": 5,
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

    with open("neutral_start_results.json", 'w') as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - start_time

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Total simulations: {len(weight_configs) * len(game_types) * N_REPS}")
    print(f"Results saved to: neutral_start_results.json")
