"""
Statistical Analysis with Multiple Replications
================================================
Runs each simulation condition multiple times (n=30) with different seeds
for statistical validity and confidence intervals.
"""

import numpy as np
from Simulation import SimConfig, CoordinationGame
import json
from scipy import stats
from dataclasses import asdict
import time


def run_replications(config_params, ref_point_method="bayesian", n_reps=30):
    """
    Run multiple replications of a simulation condition

    Args:
        config_params: Dictionary of SimConfig parameters
        ref_point_method: Reference point method to use
        n_reps: Number of replications

    Returns:
        List of summary dictionaries from each replication
    """
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
    """
    Calculate mean, std, CI for specified metrics across replications

    Args:
        results: List of summary dictionaries
        metrics: List of metric names to analyze

    Returns:
        Dictionary with statistics for each metric
    """
    stats_dict = {}
    n = len(results)

    for metric in metrics:
        values = [r[metric] for r in results]
        mean = np.mean(values)
        std = np.std(values, ddof=1)  # Sample std
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
    """
    Perform independent t-test between two conditions

    Returns:
        Dictionary with t-statistic, p-value, effect size (Cohen's d)
    """
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
        "significant_at_01": bool(p_value < 0.01)
    }


def analyze_methods(base_config, n_reps=30):
    """
    Compare reference point methods with statistical analysis
    """
    print("\n" + "="*70)
    print("ANALYSIS 1: Reference Point Methods (Statistical)")
    print(f"Running {n_reps} replications per method...")
    print("="*70)

    methods = ["global", "group", "pairwise", "recent", "bayesian"]
    metrics = ["final_coordination_rate", "final_group_favoritism",
               "ingroup_coordination_rate", "outgroup_coordination_rate"]

    all_results = {}

    for method in methods:
        print(f"\nRunning method: {method}...", end=" ")
        start_time = time.time()

        results = run_replications(base_config, method, n_reps)
        stats_dict = calculate_statistics(results, metrics)

        all_results[method] = {
            "statistics": stats_dict,
            "raw_results": results
        }

        elapsed = time.time() - start_time
        print(f"Done ({elapsed:.1f}s)")

        # Print summary
        print(f"  Final coordination: {stats_dict['final_coordination_rate']['mean']:.3f} "
              f"± {stats_dict['final_coordination_rate']['se']:.3f} "
              f"(95% CI: [{stats_dict['final_coordination_rate']['ci_95_lower']:.3f}, "
              f"{stats_dict['final_coordination_rate']['ci_95_upper']:.3f}])")
        print(f"  Group favoritism: {stats_dict['final_group_favoritism']['mean']:.3f} "
              f"± {stats_dict['final_group_favoritism']['se']:.3f}")

    # Pairwise comparisons (if there are differences)
    print("\n" + "-"*70)
    print("Pairwise Comparisons (t-tests):")
    print("-"*70)

    comparisons = {}
    for i, method1 in enumerate(methods):
        for method2 in methods[i+1:]:
            key = f"{method1}_vs_{method2}"
            comparisons[key] = {}

            for metric in metrics:
                test_result = compare_conditions_ttest(
                    all_results[method1]["raw_results"],
                    all_results[method2]["raw_results"],
                    metric
                )
                comparisons[key][metric] = test_result

                if test_result["significant_at_05"]:
                    print(f"{method1} vs {method2} on {metric}: "
                          f"t={test_result['t_statistic']:.3f}, "
                          f"p={test_result['p_value']:.4f}, "
                          f"d={test_result['cohens_d']:.3f} *")

    return {
        "results": all_results,
        "comparisons": comparisons
    }


def analyze_games(base_config, n_reps=30):
    """
    Compare game types with statistical analysis
    """
    print("\n" + "="*70)
    print("ANALYSIS 2: Game Types (Statistical)")
    print(f"Running {n_reps} replications per game...")
    print("="*70)

    game_types = ["coordination", "stag_hunt", "chicken"]
    metrics = ["final_coordination_rate", "final_group_favoritism"]

    all_results = {}

    for game_type in game_types:
        print(f"\nRunning game: {game_type}...", end=" ")
        start_time = time.time()

        # Create config with game type
        config_params = base_config.copy()
        config_params["game_type"] = game_type

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

        all_results[game_type] = {
            "statistics": stats_dict,
            "raw_results": results
        }

        elapsed = time.time() - start_time
        print(f"Done ({elapsed:.1f}s)")

        # Print summary
        print(f"  Final coordination: {stats_dict['final_coordination_rate']['mean']:.3f} "
              f"± {stats_dict['final_coordination_rate']['se']:.3f}")
        print(f"  Group favoritism: {stats_dict['final_group_favoritism']['mean']:.3f} "
              f"± {stats_dict['final_group_favoritism']['se']:.3f}")
        print(f"  Group 0 prop(A): {stats_dict['group0_prop_A']['mean']:.3f} "
              f"± {stats_dict['group0_prop_A']['se']:.3f}")
        print(f"  Group 1 prop(A): {stats_dict['group1_prop_A']['mean']:.3f} "
              f"± {stats_dict['group1_prop_A']['se']:.3f}")

    # Pairwise comparisons
    print("\n" + "-"*70)
    print("Pairwise Comparisons (t-tests):")
    print("-"*70)

    comparisons = {}
    for i, game1 in enumerate(game_types):
        for game2 in game_types[i+1:]:
            key = f"{game1}_vs_{game2}"
            comparisons[key] = {}

            for metric in metrics:
                test_result = compare_conditions_ttest(
                    all_results[game1]["raw_results"],
                    all_results[game2]["raw_results"],
                    metric
                )
                comparisons[key][metric] = test_result

                if test_result["significant_at_05"]:
                    print(f"{game1} vs {game2} on {metric}: "
                          f"t={test_result['t_statistic']:.3f}, "
                          f"p={test_result['p_value']:.4f}, "
                          f"d={test_result['cohens_d']:.3f} ***")

    return {
        "results": all_results,
        "comparisons": comparisons
    }


def analyze_info_treatments(base_config, n_reps=30):
    """
    Compare information treatments with statistical analysis
    """
    print("\n" + "="*70)
    print("ANALYSIS 3: Information Treatments (Statistical)")
    print(f"Running {n_reps} replications per treatment...")
    print("="*70)

    treatments = ["full", "group_only", "anonymous"]
    metrics = ["final_coordination_rate", "final_group_favoritism"]

    all_results = {}

    for treatment in treatments:
        print(f"\nRunning treatment: {treatment}...", end=" ")
        start_time = time.time()

        # Create config with info treatment
        config_params = base_config.copy()
        config_params["info_treatment"] = treatment

        results = run_replications(config_params, "bayesian", n_reps)
        stats_dict = calculate_statistics(results, metrics)

        all_results[treatment] = {
            "statistics": stats_dict,
            "raw_results": results
        }

        elapsed = time.time() - start_time
        print(f"Done ({elapsed:.1f}s)")

        # Print summary
        print(f"  Final coordination: {stats_dict['final_coordination_rate']['mean']:.3f} "
              f"± {stats_dict['final_coordination_rate']['se']:.3f}")
        print(f"  Group favoritism: {stats_dict['final_group_favoritism']['mean']:.3f} "
              f"± {stats_dict['final_group_favoritism']['se']:.3f}")

    # Pairwise comparisons
    print("\n" + "-"*70)
    print("Pairwise Comparisons (t-tests):")
    print("-"*70)

    comparisons = {}
    for i, treat1 in enumerate(treatments):
        for treat2 in treatments[i+1:]:
            key = f"{treat1}_vs_{treat2}"
            comparisons[key] = {}

            for metric in metrics:
                test_result = compare_conditions_ttest(
                    all_results[treat1]["raw_results"],
                    all_results[treat2]["raw_results"],
                    metric
                )
                comparisons[key][metric] = test_result

                if test_result["significant_at_05"]:
                    print(f"{treat1} vs {treat2} on {metric}: "
                          f"t={test_result['t_statistic']:.3f}, "
                          f"p={test_result['p_value']:.4f}, "
                          f"d={test_result['cohens_d']:.3f} ***")

    return {
        "results": all_results,
        "comparisons": comparisons
    }


if __name__ == "__main__":
    print("="*70)
    print("Statistical Analysis with Multiple Replications")
    print("="*70)

    # Base configuration (as dictionary for easy modification)
    base_config = {
        "n_agents": 16,
        "n_rounds": 200,
        "n_groups": 2,
        "game_type": "coordination",
        "lambda_loss": 2.0,
        "initial_group_bias": 0.8,
        "n_initial_rounds": 10,
        "info_treatment": "full",
        "matching_type": "random"
        # seed will be set per replication
    }

    N_REPS = 30  # Number of replications per condition

    print(f"\nConfiguration:")
    print(f"  N = {base_config['n_agents']}, T = {base_config['n_rounds']}")
    print(f"  Replications per condition: {N_REPS}")
    print(f"  Statistical power: Two-tailed t-test, α=0.05")

    overall_start = time.time()

    # Run all analyses
    methods_analysis = analyze_methods(base_config, N_REPS)
    games_analysis = analyze_games(base_config, N_REPS)
    info_analysis = analyze_info_treatments(base_config, N_REPS)

    # Prepare summary for JSON (without raw results to keep file manageable)
    summary_results = {
        "configuration": {
            "base_config": base_config,
            "n_replications": N_REPS,
            "total_simulations": N_REPS * (5 + 3 + 3)  # methods + games + treatments
        },
        "methods": {
            method: data["statistics"]
            for method, data in methods_analysis["results"].items()
        },
        "methods_comparisons": methods_analysis["comparisons"],
        "games": {
            game: data["statistics"]
            for game, data in games_analysis["results"].items()
        },
        "games_comparisons": games_analysis["comparisons"],
        "info_treatments": {
            treatment: data["statistics"]
            for treatment, data in info_analysis["results"].items()
        },
        "info_comparisons": info_analysis["comparisons"]
    }

    # Save results
    with open("statistical_analysis_results.json", 'w') as f:
        json.dump(summary_results, f, indent=2)

    overall_elapsed = time.time() - overall_start

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"Total time: {overall_elapsed:.1f} seconds ({overall_elapsed/60:.1f} minutes)")
    print(f"Total simulations run: {N_REPS * 11}")
    print(f"Results saved to: statistical_analysis_results.json")
    print("\nKey findings:")
    print("  - Check p-values < 0.05 for statistical significance")
    print("  - Cohen's d > 0.5 indicates medium effect size")
    print("  - Cohen's d > 0.8 indicates large effect size")
