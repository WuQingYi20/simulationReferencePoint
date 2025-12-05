"""
No Initial Phase Analysis
==========================
Test what happens when we remove the initial convention establishment phase entirely

Changes:
- n_initial_rounds = 0 (no convention phase at all)
- Agents start with KR utility maximization from round 0
- Uninformed priors (reference point = 0.5, no history)

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
        config = SimConfig(**config_params, seed=6000 + rep)
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
            "n": n
        }

    return stats_dict


def analyze_no_initial_phase(weight_configs, game_types, time_horizons, n_reps=30):
    """
    Test across different time horizons with NO initial phase
    """
    print("\n" + "="*70)
    print("NO INITIAL PHASE ANALYSIS")
    print("Pure learning from round 0 with uninformed priors")
    print("="*70)

    all_results = {}

    for T in time_horizons:
        print(f"\n{'='*70}")
        print(f"TIME HORIZON: T={T} rounds")
        print('='*70)

        T_results = {}

        for game_type in game_types:
            print(f"\n  Game: {game_type}")

            game_results = {}

            for weight_name, (w_recent, w_group, w_global) in weight_configs.items():
                base_config = {
                    "n_agents": 16,
                    "n_rounds": T,
                    "n_groups": 2,
                    "lambda_loss": 2.25,
                    "initial_group_bias": 0.8,  # Doesn't matter since n_initial_rounds=0
                    "n_initial_rounds": 0,  # KEY: NO initial phase!
                    "info_treatment": "full",
                    "matching_type": "random",
                    "recency_decay": 0.9,
                    "min_history_for_belief": 1,
                    "game_type": game_type,
                    "weight_recent": w_recent,
                    "weight_group": w_group,
                    "weight_global": w_global
                }

                results = run_replications(base_config, "bayesian", n_reps)

                metrics = ["final_coordination_rate", "final_group_favoritism",
                          "overall_coordination_rate"]
                stats_dict = calculate_statistics(results, metrics)

                # Action preferences
                group0_A = [r["group_action_preferences"].get(0, 0) for r in results]
                group1_A = [r["group_action_preferences"].get(1, 0) for r in results]
                stats_dict["group0_prop_A"] = {
                    "mean": float(np.mean(group0_A)),
                    "std": float(np.std(group0_A, ddof=1))
                }
                stats_dict["group1_prop_A"] = {
                    "mean": float(np.mean(group1_A)),
                    "std": float(np.std(group1_A, ddof=1))
                }

                game_results[weight_name] = stats_dict

            T_results[game_type] = game_results

        all_results[f"T{T}"] = T_results

    return all_results


if __name__ == "__main__":
    print("="*70)
    print("No Initial Phase Analysis")
    print("Testing pure learning without convention establishment")
    print("Date: 2025-12-04")
    print("="*70)

    weight_configs = {
        "Recent-dom": (0.90, 0.05, 0.05),
        "Group-dom": (0.05, 0.90, 0.05),
        "Global-dom": (0.05, 0.05, 0.90),
        "Balanced": (0.33, 0.33, 0.34)
    }

    game_types = ["coordination", "stag_hunt", "chicken"]
    time_horizons = [30, 50, 200]  # Test across multiple T
    N_REPS = 30

    print(f"\nKey change: n_initial_rounds = 0")
    print(f"  - No group-biased convention phase")
    print(f"  - Agents start with uninformed priors (ref_point = 0.5)")
    print(f"  - Pure KR utility maximization from round 0")

    start_time = time.time()

    # Run analysis
    results = analyze_no_initial_phase(weight_configs, game_types, time_horizons, N_REPS)

    # Display results
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    for T_key in [f"T{T}" for T in time_horizons]:
        T = int(T_key[1:])
        print(f"\n{T_key} ({T} rounds):")
        print("-" * 70)

        for game_type in game_types:
            print(f"\n  {game_type.upper()}:")

            coord_values = []
            favor_values = []
            weight_labels = []

            for weight_name in weight_configs.keys():
                coord = results[T_key][game_type][weight_name]["final_coordination_rate"]["mean"]
                favor = results[T_key][game_type][weight_name]["final_group_favoritism"]["mean"]
                coord_values.append(coord)
                favor_values.append(favor)
                weight_labels.append(weight_name)

                print(f"    {weight_name:12s}: Coord={coord:.3f}, Favor={favor:+.3f}")

            coord_range = max(coord_values) - min(coord_values)
            favor_range = max(favor_values) - min(favor_values)

            print(f"    Coordination range: {coord_range:.4f} ({coord_range*100:.2f}%)")
            print(f"    Favoritism range: {favor_range:.4f}")

            if coord_range > 0.02:
                print(f"    >>> WEIGHTS MATTER! ({coord_range*100:.1f}% range)")
            else:
                print(f"    >>> Weights don't matter ({coord_range*100:.1f}% range)")

    # Save results
    summary = {
        "metadata": {
            "date": "2025-12-04",
            "analysis_type": "no_initial_phase",
            "n_initial_rounds": 0,
            "n_replications": N_REPS,
            "time_horizons": time_horizons,
            "game_types": game_types
        },
        "results": results
    }

    with open("no_initial_phase_results.json", 'w') as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - start_time

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Total simulations: {len(weight_configs) * len(game_types) * len(time_horizons) * N_REPS}")
    print(f"Results saved to: no_initial_phase_results.json")

    # Comparison
    print("\n" + "="*70)
    print("COMPARISON: WITH vs WITHOUT Initial Phase")
    print("="*70)
    print("\nCoordination Game, T=200:")
    print("  WITH initial phase (β=0.8, rounds 0-9): 88.7% final coordination")
    print("  WITHOUT initial phase: ", end="")
    no_init_coord = results["T200"]["coordination"]["Balanced"]["final_coordination_rate"]["mean"]
    print(f"{no_init_coord:.1%} final coordination")
    print(f"  Difference: {(no_init_coord - 0.887)*100:+.1f} percentage points")
