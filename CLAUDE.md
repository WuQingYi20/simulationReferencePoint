# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a computational social science research project simulating reference point formation in repeated coordination games. The simulation implements Kőszegi-Rabin (2006) reference-dependent preferences to study how agents form expectations and coordinate within/across social groups, with applications to climate change institutional dynamics.

## Running the Simulation

### Basic Execution
```bash
python Simulation.py
```

This runs the full analysis suite with three experiments:
1. Comparison of reference point methods (global, group, pairwise, recent, bayesian)
2. Comparison of game types (coordination, stag_hunt, chicken)
3. Comparison of information treatments (full, group_only, anonymous)

### Output Files
- `full_analysis.json` - Summary statistics from all experiments
- `results_{method}_N{agents}_T{rounds}.json` - Detailed results from individual runs

## Core Architecture

### Configuration System (SimConfig)
All simulation parameters are defined in the `SimConfig` dataclass (lines 26-101). Key parameters:

- **Population**: `n_agents`, `n_groups` (minimal group paradigm)
- **Game type**: "coordination", "stag_hunt", or "chicken" (auto-sets payoff matrices)
- **KR parameters**: `lambda_loss` (loss aversion, λ > 1), `eta` (gain-loss weight)
- **Reference point weights**: `weight_recent`, `weight_group`, `weight_global`
- **Initial conventions**: `initial_group_bias`, `n_initial_rounds`
- **Information treatment**: "full", "group_only", or "anonymous"
- **Matching**: "random" or "round_robin"

### Key Classes and Their Roles

**HistoryTracker** (lines 123-158)
- Maintains multi-level interaction history for each agent
- Provides indexed access: pairwise, group-level, global, recent
- Efficient lookup for reference point calculations

**ReferencePointCalculator** (lines 164-323)
- Core research contribution: different methods of forming expectations
- Methods: global, group, pairwise, recent, bayesian
- Bayesian method uses precision-weighted combination of sources
- Returns probability of coordination (π) as reference point

**KRUtility** (lines 329-419)
- Implements Kőszegi-Rabin (2006) reference-dependent utility
- Formula: U = m(c) + η * n(c|r)
- Loss aversion: μ(x) = x if x ≥ 0, λx if x < 0
- Key method: `calculate_both_utilities()` for decision-making

**Agent** (lines 425-609)
- Autonomous decision-maker with history tracking
- Two decision modes:
  1. Initial rounds (< n_initial_rounds): group-specific convention establishment
  2. Later rounds: KR utility maximization
- Information-contingent partner action estimation
- Detailed decision logging for analysis

**CoordinationGame** (lines 615-958)
- Simulation manager and orchestrator
- Creates agents with minimal group assignment
- Runs matching protocols (random or round-robin)
- Tracks coordination rates (overall, in-group, out-group)
- Generates comprehensive summary statistics

### Game Payoff Structures

Payoffs are automatically set based on `game_type`:

**Coordination** (default): Match=1, Mismatch=0
- Pure coordination problem, two symmetric equilibria

**Stag Hunt**: AA=3, AB=0, BA=2, BB=2
- A is payoff-dominant (risky, high reward)
- B is risk-dominant (safe)
- Loss aversion predicts convergence to B

**Chicken**: AA=2, AB=1, BA=3, BB=0
- Anti-coordination game
- Models escalation/de-escalation dynamics

## Research Design Features

### Minimal Group Paradigm (lines 639-666)
- Agents randomly assigned to arbitrary groups
- No intrinsic meaning to group labels
- Initial rounds establish group-specific conventions via biased action probabilities

### Reference Point Formation (lines 209-323)
The key research question: How do agents weight different information sources?

- **Pairwise history**: Highest specificity, 2x weight
- **Group history**: Medium specificity, weight_group relative weight
- **Recent history**: High relevance, recency_decay exponential weighting
- **Global history**: Low specificity fallback

Bayesian method uses precision-weighting (proportional to sample size).

### Decision Process (lines 450-533)
1. Initial phase: Group convention establishment (stochastic based on `initial_group_bias`)
2. Main phase:
   - Estimate P(partner plays A) using information treatment
   - Calculate U(A) and U(B) using KR utility
   - Choose argmax with trembling hand noise
   - Log decision details (ref point, beliefs, utilities)

### Information Treatments (lines 535-588)
- **Full**: Use pairwise → group → global history (fallback chain)
- **Group only**: Use group → global history
- **Anonymous**: Use only global history

Less information → stronger reliance on group priors → potential for increased group favoritism.

## Analysis Functions

**run_experiment()** (lines 964-990)
- Single simulation run with specified configuration
- Saves detailed JSON results

**compare_methods()** (lines 992-1033)
- Systematic comparison of reference point methods
- Uses same seed for fair comparison
- Reports coordination rates and group favoritism

**compare_games()** (lines 1035-1074)
- Tests different game structures
- Shows how payoff structure affects outcomes

**compare_info_treatments()** (lines 1076-1112)
- Tests information availability effects
- Measures group favoritism sensitivity

## Key Research Questions (from ResearchDesign.md)

1. **Source weighting**: How do agents weight pairwise vs group vs recent vs global history?
2. **Group favoritism**: When does minimal group assignment create in-group bias?
3. **Convention persistence**: How do initial group conventions persist over time?
4. **Loss aversion effects**: How does λ affect convergence speed and group favoritism?

## Customizing Simulations

To modify simulation behavior, adjust parameters in the base_config (lines 1120-1131):

```python
config = SimConfig(
    n_agents=16,           # Population size
    n_rounds=200,          # Duration
    game_type="coordination",  # or "stag_hunt", "chicken"
    lambda_loss=2.0,       # Loss aversion coefficient
    initial_group_bias=0.8,    # Convention strength (0.5=none, 1.0=maximum)
    n_initial_rounds=10,   # Convention establishment period
    info_treatment="full", # Information availability
    matching_type="random",    # Matching protocol
    seed=42                # Reproducibility
)
```

## Important Implementation Details

### Initial Convention Seeding (lines 463-485)
Groups establish different starting norms:
- Group 0: P(A) = initial_group_bias (default 0.8)
- Group 1: P(A) = 1 - initial_group_bias (default 0.2)

This creates divergent group "cultures" that may persist.

### Symmetric Game Assumption (lines 668-692)
All games are symmetric - both players have identical payoff matrices.

### History Recording (lines 590-608)
Every interaction is stored as InteractionRecord with full context:
- Partner ID and group
- Actions taken
- Realized payoffs
- Round number

This enables rich retrospective reference point analysis.

## Dependencies

Required Python packages:
- numpy (random number generation, numerical operations)
- dataclasses (configuration and record structures)
- json (results serialization)
- collections (defaultdict for history indexing)
- enum (Action enumeration)

## Theoretical Foundation

Based on Kőszegi, B., & Rabin, M. (2006). "A model of reference-dependent preferences." Quarterly Journal of Economics.

Key insight: Reference points are rational expectations formed from observable history. This codebase operationalizes that insight in a multi-agent setting with group structure.
