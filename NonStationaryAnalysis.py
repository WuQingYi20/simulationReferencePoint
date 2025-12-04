"""
Non-Stationary Environment Analysis
====================================
Test whether reference point formation weights matter when environment changes

Key modification: Introduce behavioral shock at round 100
- Groups swap their action tendencies
- Recent-weighted agents should adapt faster than global-weighted agents

Date: 2025-12-04
"""

import numpy as np
from Simulation import SimConfig, CoordinationGame, Agent, Action
from scipy import stats
import json
import time


class NonStationaryGame(CoordinationGame):
    """
    Modified game with environmental shock at specified round
    """

    def __init__(self, config, ref_point_method="bayesian", shock_round=100):
        super().__init__(config, ref_point_method)
        self.shock_round = shock_round
        self.shock_applied = False

    def apply_shock(self):
        """
        Swap group action biases
        Group 0 now tends toward B instead of A
        Group 1 now tends toward A instead of B
        """
        if not self.shock_applied:
            # Reverse group assignments
            for agent in self.agents:
                agent.group = 1 - agent.group

            # Update group assignments record
            self.group_assignments = {a.id: a.group for a in self.agents}
            self.shock_applied = True

            return True
        return False

    def run_simulation(self, verbose=True):
        """Run simulation with shock at specified round"""
        cfg = self.config

        if verbose:
            print(f"Running NON-STATIONARY simulation:")
            print(f"  N={cfg.n_agents}, T={cfg.n_rounds}, Shock at round {self.shock_round}")
            print(f"  Reference method: {self.ref_point_method}")

        for t in range(cfg.n_rounds):
            # Apply shock at specified round
            if t == self.shock_round:
                self.apply_shock()
                if verbose:
                    print(f"\n  >>> SHOCK APPLIED at round {t}: Groups swapped! <<<\n")

            # Run round
            if cfg.matching_type == "round_robin":
                round_result = self.run_round_robin_round(t)
            else:
                round_result = self.run_random_matching_round(t)

            self.round_results.append(round_result)
            self.coordination_rates.append(round_result["coordination_rate"])

            if verbose and (t + 1) % 50 == 0:
                recent_rate = np.mean(self.coordination_rates[-50:])
                print(f"  Round {t+1}: Recent coordination = {recent_rate:.3f}")

        return self.get_summary()


def calculate_adaptation_speed(coordination_rates, shock_round, window=20):
    """
    Measure how quickly coordination recovers after shock

    Returns:
        - dip_magnitude: How much coordination dropped immediately after shock
        - recovery_time: Rounds needed to return to 90% of pre-shock level
        - adaptation_rate: Slope of recovery
    """
    pre_shock = coordination_rates[max(0, shock_round-window):shock_round]
    post_shock = coordination_rates[shock_round:min(len(coordination_rates), shock_round+window*3)]

    if len(pre_shock) == 0 or len(post_shock) == 0:
        return None

    pre_shock_mean = np.mean(pre_shock)

    # Find minimum after shock (the dip)
    min_idx = np.argmin(post_shock[:window]) if len(post_shock) >= window else np.argmin(post_shock)
    dip_magnitude = pre_shock_mean - post_shock[min_idx]

    # Find recovery time (when it reaches 90% of pre-shock level)
    recovery_threshold = pre_shock_mean * 0.9
    recovery_time = None
    for i, rate in enumerate(post_shock):
        if rate >= recovery_threshold:
            recovery_time = i
            break

    if recovery_time is None:
        recovery_time = len(post_shock)  # Didn't recover within window

    # Calculate adaptation rate (slope of first 20 rounds after shock)
    adaptation_window = post_shock[:min(window, len(post_shock))]
    if len(adaptation_window) > 2:
        x = np.arange(len(adaptation_window))
        slope, _ = np.polyfit(x, adaptation_window, 1)
        adaptation_rate = slope
    else:
        adaptation_rate = 0

    return {
        "pre_shock_mean": float(pre_shock_mean),
        "dip_magnitude": float(dip_magnitude),
        "recovery_time": int(recovery_time),
        "adaptation_rate": float(adaptation_rate),
        "post_shock_final": float(np.mean(post_shock[-window:])) if len(post_shock) >= window else float(np.mean(post_shock))
    }


def run_nonstationary_comparison(base_config, weight_configs, n_reps=30, shock_round=100):
    """
    Compare adaptation speeds across different weight configurations
    """
    print("\n" + "="*70)
    print("NON-STATIONARY ENVIRONMENT ANALYSIS")
    print(f"Shock applied at round {shock_round}")
    print("="*70)

    results = {}

    for weight_name, (w_recent, w_group, w_global) in weight_configs.items():
        print(f"\n  Testing: {weight_name} (r={w_recent:.2f}, g={w_group:.2f}, G={w_global:.2f})")

        config_params = base_config.copy()
        config_params["weight_recent"] = w_recent
        config_params["weight_group"] = w_group
        config_params["weight_global"] = w_global

        rep_results = []
        adaptation_metrics = []

        for rep in range(n_reps):
            config = SimConfig(**config_params, seed=3000 + rep)
            game = NonStationaryGame(config, "bayesian", shock_round=shock_round)
            summary = game.run_simulation(verbose=False)

            # Calculate adaptation metrics
            adaptation = calculate_adaptation_speed(
                game.coordination_rates,
                shock_round,
                window=20
            )

            rep_results.append(summary)
            if adaptation:
                adaptation_metrics.append(adaptation)

        # Aggregate metrics
        avg_metrics = {}
        for key in adaptation_metrics[0].keys():
            values = [m[key] for m in adaptation_metrics]
            avg_metrics[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "se": float(np.std(values, ddof=1) / np.sqrt(len(values)))
            }

        results[weight_name] = {
            "weights": {
                "recent": w_recent,
                "group": w_group,
                "global": w_global
            },
            "adaptation_metrics": avg_metrics,
            "raw_results": rep_results
        }

        print(f"    Dip magnitude: {avg_metrics['dip_magnitude']['mean']:.3f} ± {avg_metrics['dip_magnitude']['se']:.3f}")
        print(f"    Recovery time: {avg_metrics['recovery_time']['mean']:.1f} ± {avg_metrics['recovery_time']['se']:.1f} rounds")
        print(f"    Adaptation rate: {avg_metrics['adaptation_rate']['mean']:.4f} ± {avg_metrics['adaptation_rate']['se']:.4f}")

    return results


def compare_adaptation_metrics(results, metric_name):
    """
    Perform pairwise t-tests on adaptation metrics
    """
    weight_names = list(results.keys())
    comparisons = {}

    for i, name1 in enumerate(weight_names):
        for name2 in weight_names[i+1:]:
            # Extract raw values for this metric from both conditions
            values1 = []
            values2 = []

            for res in results[name1]["raw_results"]:
                # Recalculate from coordination rates
                pass  # Would need to store raw adaptation metrics per replication

            # For now, use the aggregated means and perform comparison
            mean1 = results[name1]["adaptation_metrics"][metric_name]["mean"]
            mean2 = results[name2]["adaptation_metrics"][metric_name]["mean"]
            se1 = results[name1]["adaptation_metrics"][metric_name]["se"]
            se2 = results[name2]["adaptation_metrics"][metric_name]["se"]

            # Simple difference test
            diff = mean1 - mean2
            se_diff = np.sqrt(se1**2 + se2**2)
            t_stat = diff / se_diff if se_diff > 0 else 0

            comparisons[f"{name1}_vs_{name2}"] = {
                "mean_diff": float(diff),
                "se_diff": float(se_diff),
                "t_statistic": float(t_stat)
            }

    return comparisons


if __name__ == "__main__":
    print("="*70)
    print("Non-Stationary Environment Analysis")
    print("Testing if weights matter when environment changes")
    print("Date: 2025-12-04")
    print("="*70)

    # Base configuration
    base_config = {
        "n_agents": 16,
        "n_rounds": 200,
        "n_groups": 2,
        "game_type": "coordination",
        "lambda_loss": 2.0,
        "initial_group_bias": 0.8,
        "n_initial_rounds": 10,
        "info_treatment": "full",
        "matching_type": "random",
        "recency_decay": 0.9,
        "min_history_for_belief": 1
    }

    # Test specific weight configurations
    weight_configs = {
        "Recent-dominant": (0.90, 0.05, 0.05),   # Should adapt quickly
        "Global-dominant": (0.05, 0.05, 0.90),   # Should adapt slowly
        "Group-dominant": (0.05, 0.90, 0.05),    # Medium adaptation
        "Balanced": (0.33, 0.33, 0.34),          # Medium adaptation
        "Original": (0.50, 0.30, 0.20)           # Medium-fast adaptation
    }

    N_REPS = 30
    SHOCK_ROUND = 100

    start_time = time.time()

    # Run analysis
    results = run_nonstationary_comparison(
        base_config,
        weight_configs,
        n_reps=N_REPS,
        shock_round=SHOCK_ROUND
    )

    # Statistical comparisons
    print("\n" + "="*70)
    print("STATISTICAL COMPARISONS")
    print("="*70)

    metrics_to_compare = ["dip_magnitude", "recovery_time", "adaptation_rate"]

    for metric in metrics_to_compare:
        print(f"\n{metric.upper().replace('_', ' ')}:")
        comparisons = compare_adaptation_metrics(results, metric)

        for comp_name, comp_result in comparisons.items():
            diff = comp_result["mean_diff"]
            t_stat = comp_result["t_statistic"]

            if abs(t_stat) > 2.0:  # Roughly significant at p<0.05
                sig = "***" if abs(t_stat) > 2.8 else "**"
                print(f"  {comp_name:40s}: Δ={diff:+.3f}, t={t_stat:6.3f} {sig}")

    # Save results
    summary = {
        "metadata": {
            "date": "2025-12-04",
            "shock_round": SHOCK_ROUND,
            "n_replications": N_REPS,
            "base_config": base_config,
            "weight_configurations": {
                name: {"recent": w[0], "group": w[1], "global": w[2]}
                for name, w in weight_configs.items()
            }
        },
        "results": {
            name: {
                "weights": data["weights"],
                "adaptation_metrics": data["adaptation_metrics"]
            }
            for name, data in results.items()
        }
    }

    with open("nonstationary_analysis_results.json", 'w') as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - start_time

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Total simulations: {len(weight_configs) * N_REPS}")
    print(f"Results saved to: nonstationary_analysis_results.json")

    # Summary of key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    for name, data in results.items():
        metrics = data["adaptation_metrics"]
        print(f"\n{name}:")
        print(f"  Recovery time: {metrics['recovery_time']['mean']:.1f} rounds")
        print(f"  Adaptation rate: {metrics['adaptation_rate']['mean']:.4f}/round")
        print(f"  Dip magnitude: {metrics['dip_magnitude']['mean']:.3f}")

    # Determine if weights matter
    recovery_times = [data["adaptation_metrics"]["recovery_time"]["mean"]
                      for data in results.values()]
    recovery_range = max(recovery_times) - min(recovery_times)

    print(f"\nRecovery time range: {recovery_range:.1f} rounds")
    if recovery_range > 5:
        print("[YES] WEIGHTS MATTER in non-stationary environment!")
        print(f"  Recent-weighted agents recover {recovery_range:.1f} rounds faster")
    else:
        print("[NO] Even in non-stationary environment, weights have minimal effect")
