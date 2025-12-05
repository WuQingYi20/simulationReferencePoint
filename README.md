# Reference Point Formation in Repeated Coordination Games

**A Computational Study of Kőszegi-Rabin Preferences with Minimal Group Paradigm**

**Author**: Yifan
**Date**: December 2025

---

## Research Overview

This project investigates how agents form expectations (reference points) in repeated coordination games with group identity, using Kőszegi-Rabin (2006) reference-dependent preferences. The research bridges behavioral economics, social identity theory, and computational modeling to understand coordination dynamics in contexts like climate change negotiations, where institutional actors must coordinate across group boundaries.

### Core Research Question

**How do agents form reference points in repeated coordination games, and does the weighting of different information sources (recent interactions, group-specific history, global population statistics) affect coordination outcomes and group favoritism?**

This question is motivated by the observation that in multi-level coordination problems—such as international climate agreements—actors must balance multiple information sources: their recent bilateral interactions, their history with specific groups or coalitions, and the broader global pattern of behavior.

### Answer: Weights Critically Matter in Conflict Games

**Key Finding** (December 2025 update): After implementing a Bayesian partner belief model where weights affect both reference point formation AND partner behavior prediction, we find that **information weighting has massive effects in strategic conflict situations**:

- **Chicken Game** (anti-coordination): Weight choice explains **32.8% variation in coordination** (T=200) and **25.0% variation** (T=50), with effect sizes d>6 (massive)
- **Coordination Game**: **6.7% variation** at T=50 (significant) but minimal at T=200
- **Stag Hunt**: No effect (risk dominance eliminates strategic conflict)

**Practical implication**: In climate negotiations (analogous to Chicken), group-dominant strategies (prioritizing coalition norms) achieve only 30.8% coordination with 42.5% in-group bias, while global-dominant strategies achieve 55.9% coordination with near-zero bias—a **25 percentage point difference**.

---

## Theoretical Framework

### Kőszegi-Rabin Reference-Dependent Utility

The model implements the KR utility function:
**U = m(c) + η · n(c|r)**

where:
- **m(c)** = consumption utility (standard game payoff)
- **n(c|r)** = gain-loss utility relative to reference point
- **η** = weight on gain-loss component (η = 1.0 in our model)
- **λ** = loss aversion coefficient (**λ = 2.25** in our model, capturing moderate loss aversion)

The gain-loss function exhibits loss aversion: **μ(x) = x if x ≥ 0, λx if x < 0**

### Reference Point as Bayesian Expectation

Agents form reference points by combining multiple information sources:

**π = Σ (w_s · π̂_s · n_s) / Σ (w_s · n_s)**

where sources s ∈ {recent, group-specific, global}, π̂_s is the estimated coordination rate from source s, n_s is the sample size (precision), and w_s are the weights—the focus of our investigation.

### Minimal Group Paradigm

Following Tajfel (1970), agents are assigned to arbitrary groups with no intrinsic meaning. Initial rounds (T₀=10) establish group-specific "conventions" through biased action tendencies, creating the potential for in-group/out-group dynamics to emerge.

### Agent Decision-Making Process (Pseudocode)

```python
# Agent decision process for each interaction
def choose_action(agent, partner_id, partner_group, current_round):

    # PHASE 1: Initial convention establishment (rounds 0-9)
    if current_round < n_initial_rounds:
        if agent.group == 0:
            prob_choose_A = initial_group_bias  # e.g., 0.8
        else:
            prob_choose_A = 1 - initial_group_bias  # e.g., 0.2

        action = random_choice(A with prob_choose_A, B otherwise)
        return action

    # PHASE 2: KR utility maximization (rounds 10+)

    # Step 1: Form reference point (expected coordination probability)
    ref_point = calculate_reference_point(
        agent.history,
        partner_id,
        partner_group,
        method="bayesian"  # or "global", "group", "pairwise", "recent"
    )

    # Step 2: Estimate partner's action probability
    prob_partner_A = estimate_partner_behavior(
        agent.history,
        partner_id,
        partner_group,
        info_treatment  # "full", "group_only", or "anonymous"
    )

    # Step 3: Calculate expected KR utility for both actions
    # For action A:
    payoff_if_partner_A = payoff_AA  # Both choose A
    payoff_if_partner_B = payoff_AB  # I choose A, partner chooses B

    # Consumption utility
    U_consumption_A = prob_partner_A * payoff_AA + (1 - prob_partner_A) * payoff_AB

    # Gain-loss utility (reference-dependent)
    # Compare outcome vs reference in all states
    diff_AB = payoff_if_partner_A - payoff_if_partner_B
    diff_BA = -diff_AB

    U_gain_loss_A = (
        prob_partner_A * (1 - prob_partner_A) * gain_loss(diff_AB) +
        (1 - prob_partner_A) * prob_partner_A * gain_loss(diff_BA)
    )

    # Total KR utility for action A
    U_A = U_consumption_A + eta * U_gain_loss_A

    # Similarly for action B
    U_B = calculate_KR_utility(action=B, prob_partner_A)

    # Step 4: Choose action with trembling hand
    if random() < response_noise:
        action = random_choice(A or B)  # Exploration
    else:
        action = argmax(U_A, U_B)  # Exploitation

    # Step 5: Record outcome after interaction
    agent.history.add(
        partner_id=partner_id,
        partner_group=partner_group,
        my_action=action,
        partner_action=observed_partner_action,
        payoff=realized_payoff
    )

    return action

# Reference point calculation (Bayesian method)
def calculate_reference_point(history, partner_id, partner_group, method="bayesian"):

    if method == "global":
        return coordination_rate(history.all_interactions)

    elif method == "group":
        return coordination_rate(history.by_group[partner_group])

    elif method == "pairwise":
        return coordination_rate(history.by_partner[partner_id])

    elif method == "recent":
        return coordination_rate(history.last_N_interactions, decay=recency_decay)

    elif method == "bayesian":
        # Precision-weighted combination of all sources
        estimates = []
        precisions = []

        # Pairwise history (highest specificity)
        if len(history.by_partner[partner_id]) >= min_history:
            est = coordination_rate(history.by_partner[partner_id])
            n = len(history.by_partner[partner_id])
            estimates.append(est)
            precisions.append(n * 2.0)  # Extra weight for specificity

        # Group history (medium specificity)
        if len(history.by_group[partner_group]) >= min_history:
            est = coordination_rate(history.by_group[partner_group])
            n = len(history.by_group[partner_group])
            estimates.append(est)
            precisions.append(n * weight_group / weight_recent)

        # Recent history (high temporal relevance)
        if len(history.recent) >= min_history:
            est = coordination_rate(history.recent, decay=recency_decay)
            n = len(history.recent)
            estimates.append(est)
            precisions.append(n * weight_recent / weight_group)

        # Global history (fallback)
        if len(history.all_interactions) >= min_history:
            est = coordination_rate(history.all_interactions)
            n = len(history.all_interactions)
            estimates.append(est)
            precisions.append(n * weight_global / weight_recent)

        # Return precision-weighted average
        if estimates:
            return sum(e * p for e, p in zip(estimates, precisions)) / sum(precisions)
        else:
            return 0.5  # Uninformative prior

# Gain-loss function (loss aversion)
def gain_loss(x):
    if x >= 0:
        return x  # Gains valued linearly
    else:
        return lambda_loss * x  # Losses hurt more (lambda > 1)

# Coordination rate calculation
def coordination_rate(interactions, decay=None):
    if not interactions:
        return None

    if decay is None:
        # Simple average
        return sum(1 for i in interactions if i.my_action == i.partner_action) / len(interactions)
    else:
        # Recency-weighted average
        weighted_sum = 0
        weights_sum = 0
        for idx, interaction in enumerate(reversed(interactions)):
            w = decay ** idx  # Exponential decay
            weighted_sum += w * (1 if interaction.coordinated else 0)
            weights_sum += w
        return weighted_sum / weights_sum if weights_sum > 0 else None
```

**Key Decision Points:**
1. **Initial phase** (rounds 0-9): Stochastic group-based convention (creates group identity)
2. **Reference point formation** (rounds 10+): Bayesian combination of pairwise, group, recent, and global history (weighted by w_recent, w_group, w_global)
3. **Partner belief** (Plan C implementation): Bayesian combination of available information sources weighted by the same weights—this ensures weights affect both expectations AND behavior predictions
4. **KR utility**: Combine consumption utility with gain-loss utility (loss aversion **λ=2.25**)
5. **Action selection**: Argmax with trembling hand (ε=0.1 random exploration)
6. **Learning**: Update history after observing outcome

---

## Computational Implementation

### Simulation Design

- **Population**: N=16 agents, 2 groups
- **Time**: T=200 rounds (baseline), also tested T=50 and T=30
- **Games**: Pure coordination, Stag Hunt, Chicken
- **Matching**: Random pairwise or round-robin
- **Initial conditions**: Group-biased conventions (Group 0: 80% choose A, Group 1: 20% choose A)
- **Key parameters**: Loss aversion **λ=2.25**, initial group bias β=0.8
- **Model architecture**: Plan C implementation (weights affect both reference point AND partner belief formation)

### Statistical Validation

All analyses use **30 replications per condition** with different random seeds (1000-1029), providing:
- 95% confidence intervals
- Independent t-tests for condition comparisons
- Effect sizes (Cohen's d)
- Statistical power ≥ 0.80 for medium effects

**Total computational effort**: 3,780 simulations across all analyses (including short time horizon, neutral start, and no-initial-phase tests).

---

## Key Findings

### 1. Reference Point Formation Weights Critically Affect Coordination (Major Positive Result)

**UPDATED WITH PLAN C IMPLEMENTATION** (December 2025): After implementing Bayesian partner belief estimation (where weights affect both reference point formation AND partner behavior prediction), weights now show **game-dependent effects**.

#### Model Architecture Change

**Previous (Plan A)**: Weights only affected reference point formation (π), not partner belief estimation
- Result: Weights irrelevant (0.0% variance)

**Current (Plan C)**: Weights affect BOTH reference point AND partner belief formation
- Partner belief = Bayesian combination of (pairwise, group, recent, global) weighted by (w_recent, w_group, w_global)
- Result: **Weights strongly matter in conflict games**

#### Long Time Horizon Test (T=200 rounds)
**Conditions**: N=16 agents, T=200 rounds, random matching, **λ=2.25**, β=0.8, 20 replications per weight combination

Tested **19 different weight combinations** across **3 game types**:

**Weight configurations tested**:
- Recent-dominant: (0.90, 0.05, 0.05) - Favor recent interactions
- Group-dominant: (0.05, 0.90, 0.05) - Favor group patterns
- Global-dominant: (0.05, 0.05, 0.90) - Favor population statistics
- Balanced: (0.33, 0.33, 0.34) - Equal weighting
- Original: (0.50, 0.30, 0.20) - Default from literature
- ... and 14 additional combinations

**Results - Game-Dependent Effects**:

| Game | Coordination Range | Best Weight | Worst Weight | Variance |
|------|-------------------|-------------|--------------|----------|
| **Chicken** | **32.8%** (23.8% → 56.7%) | Recent-dom (56.7%) | Group-dom (23.8%) | **High** |
| Coordination | 1.3% (90.2% → 91.5%) | Group-dom (91.5%) | Balanced (90.2%) | Low |
| Stag Hunt | 0.0% (91.4% → 91.4%) | All identical | All identical | None |

**Group Favoritism Effects**:

| Game | Favoritism Range | Correlation with weight_group |
|------|-----------------|------------------------------|
| **Chicken** | **58.8%** (-0.9% → +57.8%) | **r = +0.733** (strong) |
| Coordination | 2.2% (-4.5% → -2.2%) | r = +0.473 (moderate) |
| Stag Hunt | 0.0% (-3.3% → -3.3%) | r = 0.000 (none) |

#### Short Time Horizon Test (T=50 rounds) - WEIGHTS MATTER MORE

**Conditions**: N=16 agents, **T=50 rounds** (limited learning time), random matching, **λ=2.25**, β=0.8, 30 replications per weight

**Results - Stronger Weight Effects with Limited Time**:

| Game | Coordination Range | Group Favoritism Range | Effect Size (d) |
|------|-------------------|----------------------|----------------|
| **Chicken** | **25.0%** (30.8% → 55.9%) | **42.6%** (-0.1% → +42.5%) | **d = 6.44** |
| **Coordination** | **6.7%** (82.8% → 89.5%) | **12.0%** (-2.9% → +9.2%) | **d = 1.11** |
| Stag Hunt | 0.0% (90.3% → 90.3%) | 0.0% (-3.8% → -3.8%) | d = 0.00 |

**Key Comparisons (T=50, Chicken game)**:
- Recent-dominant vs Group-dominant: Δ = +23.1%, p < 0.0001, Cohen's d = +6.44 (massive)
- Global-dominant vs Group-dominant: Δ = +25.0%, p < 0.0001, Cohen's d = +5.98 (massive)
- Group-dominant produces 42.5% in-group favoritism vs -0.1% for Recent-dominant

#### Why Weights Matter in Chicken but Not Stag Hunt

**Chicken Game** (anti-coordination incentive):
- Payoffs: AA=2, AB=1, BA=3, BB=0
- **Coordination dilemma**: Agents want OPPOSITE actions
- Group-dominant weights → agents rely on group patterns → lock into group-specific strategies → high favoritism, low coordination
- Recent/Global-dominant weights → agents adapt to individual partners → flexible responses → high coordination, low favoritism

**Stag Hunt** (coordination with risk):
- Payoffs: AA=3, AB=0, BA=2, BB=2
- **Risk dominance**: With λ=2.25 loss aversion, ALL agents converge to safe "Hare" (B) regardless of weights
- Loss aversion eliminates strategic conflict → weights irrelevant

**Coordination Game** (pure coordination):
- Payoffs: AA=1, AB=0, BA=0, BB=1
- **Weak weight effect**: Agents eventually coordinate, but group-dominant weights slow convergence slightly
- Effect increases with time scarcity (6.7% at T=50 vs 1.3% at T=200)

#### Theoretical Implications

**Weight choice reveals behavioral strategy**:

1. **Group-dominant** (high w_group):
   - "Collectivist" strategy: prioritize group norms
   - Chicken: 30.8% coordination, +42.5% favoritism
   - Climate analogy: Coalition-first negotiators (slow global progress, strong regional blocs)

2. **Recent-dominant** (high w_recent):
   - "Adaptive" strategy: respond to immediate feedback
   - Chicken: 54.0% coordination, -0.1% favoritism
   - Climate analogy: Bilateral negotiators (flexible, responsive to partners)

3. **Global-dominant** (high w_global):
   - "Universalist" strategy: follow population statistics
   - Chicken: 55.9% coordination, +1.9% favoritism
   - Climate analogy: Multilateral institutionalists (seek global consensus)

**Scope conditions discovered**:
- **Weights matter when**: Games have strategic conflict (Chicken) + moderate time horizon (T=50-200)
- **Weights don't matter when**: Risk dominance eliminates conflict (Stag Hunt) OR pure coordination (long T)
- **Critical factor**: Weights affect coordination when information sources DIVERGE (group vs global patterns differ)

#### Impact of Initial Convention Phase

**Conditions**: N=16 agents, random matching, **λ=2.25**, full information, 30 replications per condition

Tested WITH vs WITHOUT initial convention establishment phase:

**WITH initial phase** (n_initial_rounds=10, β=0.8):
- Rounds 0-9: Group 0 chooses A with 80% probability, Group 1 chooses B with 80% probability
- Creates artificial group differentiation and conventions

**WITHOUT initial phase** (n_initial_rounds=0):
- Pure KR utility maximization from round 0
- Agents start with uninformed priors (reference point = 0.5)

**Results (T=200 rounds)**:

| Game | WITH Initial Phase | WITHOUT Initial Phase | Difference |
|------|-------------------|----------------------|------------|
| **Coordination** | 88.7% coord, -0.4% favor | **90.4% coord**, -1.0% favor | **+1.7% coord** |
| **Stag Hunt** | 90.2% coord, -2.2% favor | **91.4% coord**, +2.1% favor | **+1.2% coord** |
| **Chicken** | 23.6% coord, +10.9% favor | **29.4% coord**, -4.0% favor | **+5.8% coord** |

**Key findings**:

1. **Initial phase REDUCES coordination efficiency**:
   - Pure learning (no initial phase) achieves +1.7% to +5.8% higher coordination
   - Initial conventions create "coordination debt" that takes 190 rounds to partially resolve

2. **Initial phase CREATES group favoritism**:
   - Chicken with initial phase: +10.9% favoritism (strong in-group bias)
   - Chicken without initial phase: -4.0% favoritism (minimal bias)
   - Difference: 14.9 percentage points of favoritism induced by initial conventions

3. **Chicken game shows largest effect** (+5.8% coordination improvement):
   - With initial phase: Groups develop opposite strategies (aggressive vs passive)
   - Without initial phase: Groups learn balanced strategies together

4. **Weights STILL don't matter** (0.0% variance in both conditions):
   - Tested 4 weight configurations × 3 games × 3 time horizons (T∈{30,50,200})
   - Zero variance across all 36 combinations

**Interpretation for climate negotiations**:

The initial convention phase models **pre-existing institutional norms** (e.g., US vs EU climate policies). Results show:
- **Path dependence**: Initial conventions impose a ~2% "cost of divergence"
- **Persistence**: Group favoritism (+10.9% in Chicken) emerges from different starting norms
- **Adaptive learning**: Despite initial divergence, agents achieve 88.7-90.2% coordination, showing learning can overcome institutional barriers

**Design choice**: The baseline simulations USE the initial phase (β=0.8, 10 rounds) because it more realistically models institutional coordination where actors have pre-existing norms and conventions.

### 2. Game Structure Has Dramatic Effects

**Conditions**: N=16 agents, T=200 rounds, random matching, λ=2.0, β=0.8, full information treatment, Bayesian reference point method, 30 replications per game type

Game payoff structure profoundly affects outcomes (**p < 0.001, Cohen's d > 5** for all pairwise comparisons):

| Game | Payoffs (AA, AB, BA, BB) | Final Coordination | Group Favoritism | Group 0 → A | Group 1 → A |
|------|--------------------------|-------------------|------------------|-------------|-------------|
| **Coordination** | (1, 0, 0, 1) | 88.7% ± 0.5% | -0.4% ± 1.1% | 65.7% | 36.1% |
| **Stag Hunt** | (3, 0, 2, 2) | 90.2% ± 0.3% | -2.2% ± 1.1% | 18.3% | 8.0% |
| **Chicken** | (2, 1, 3, 0) | 23.6% ± 0.7% | +10.9% ± 1.7% | 70.5% | 37.9% |

**Statistical comparisons**:
- Coordination vs Stag Hunt: t=-2.337, p=0.023, d=-0.603 (significant)
- Coordination vs Chicken: t=74.602, p<0.001, d=19.262 (highly significant)
- Stag Hunt vs Chicken: t=86.881, p<0.001, d=22.433 (highly significant)

**Critical insight on Stag Hunt**: With λ=2.0 loss aversion, both groups converge to risk-dominant "Hare" (B) equilibrium despite lower payoff—only 18% and 8% choose risky "Stag" (A). This validates loss aversion effects in KR preferences.

### 3. Information Availability Matters

**Conditions**: N=16 agents, T=200 rounds, random matching, coordination game, λ=2.0, β=0.8, Bayesian reference point method, 30 replications per treatment

Information treatment effects (**p < 0.05, Cohen's d ≈ 0.7**):

| Treatment | Agent Knows | Final Coordination | 95% CI | Group Favoritism |
|-----------|-------------|-------------------|--------|------------------|
| **Full** | Partner ID + Group | 88.7% ± 0.5% | [87.6%, 89.8%] | -0.4% ± 1.1% |
| **Group-only** | Partner Group only | 90.6% ± 0.4% | [89.8%, 91.4%] | -1.6% ± 1.1% |
| **Anonymous** | No information | 90.5% ± 0.4% | [89.7%, 91.4%] | -1.8% ± 1.2% |

**Statistical comparisons**:
- Full vs Group-only: t=-2.825, p=0.007, d=-0.729 (significant)
- Full vs Anonymous: t=-2.597, p=0.012, d=-0.671 (significant)
- Group-only vs Anonymous: t=0.127, p=0.900, d=0.033 (not significant)

**Counterintuitive finding**: Less information → +1.9% better coordination. Agents without partner tracking rely on global statistics and converge faster to single convention.

### 4. Matching Protocol Creates Path Dependence

**Conditions**: N=16 agents, T=200 rounds, λ=2.0, β=0.8, full information, Bayesian method, 30 replications per protocol × game combination

Round-robin (all pairs interact each round) vs random matching (pairwise random sampling):

| Game | Random Matching | Round-Robin Matching | Difference | Statistical Test |
|------|----------------|---------------------|------------|------------------|
| **Coordination** | 88.7% ± 0.5% | 90.5% ± 0.1% | +1.8% | t=-3.337, p=0.001, d=-0.862 |
| **Stag Hunt** | 90.2% ± 0.3% | 90.4% ± 0.1% | +0.2% | t=-0.711, p=0.478, d=-0.185 (ns) |
| **Chicken** | 23.6% ± 0.7% | 9.5% ± 0.1% | -14.2% | t=20.345, p<0.001, d=5.253 |

Round-robin produces:
- **More stable coordination** (lower variance: ±0.1% vs ±0.5-0.7%)
- **Faster learning** in coordination games (+1.8%)
- **More conflict** in anti-coordination games (-14.2%)

### 5. Non-Stationary Environments Still Show No Weight Effects

**Conditions**: N=16 agents, T=200 rounds, random matching, coordination game, λ=2.0, β=0.8, environmental shock at round 100 (group tendencies swapped), 30 replications per weight configuration

Five weight configurations tested:
- Recent-dominant (0.90, 0.05, 0.05)
- Global-dominant (0.05, 0.05, 0.90)
- Group-dominant (0.05, 0.90, 0.05)
- Balanced (0.33, 0.33, 0.34)
- Original (0.50, 0.30, 0.20)

**Results** (all weight configurations identical):
- Dip magnitude after shock: 0.256 ± 0.019 (all configurations)
- Recovery time: 0.3 ± 0.1 rounds (all configurations)
- Adaptation rate: 0.0012 ± 0.0008 per round (all configurations)
- Recovery time range: 0.0 rounds (no difference between configurations)

**Conclusion**: Even with environmental shocks disrupting information sources, agents adapt nearly instantly regardless of reference point formation weights. The modeling framework provides sufficient learning capacity that weight differences are negligible.

---

## Future Research Directions

To make reference point weights meaningful, future work should:

1. **Structured matching networks** (80% in-group, 20% out-group interactions)
2. **Heterogeneous agent populations** (different strategy types)
3. **Shorter time horizons** (T=50 rounds to prevent full convergence)
4. **Explicit belief elicitation** (measure predictions, not just actions)
5. **Empirical validation** through laboratory experiments with minimal group paradigm

The codebase is fully documented and ready for these extensions.

---

## Repository Structure

**Code**:
- `Simulation.py` - Core simulation engine (1,182 lines)
- `StatisticalAnalysis.py` - Multi-replication validation (T=200, β=0.8, initial phase)
- `ParameterSensitivityAnalysis.py` - Weight sweep analysis (19 combinations, T=200)
- `MatchingAnalysis.py` - Random vs round-robin comparison
- `NonStationaryAnalysis.py` - Environmental shock tests (T=200)
- `ShortHorizonAnalysis.py` - Short time horizon tests (T=50, β=0.8, initial phase)
- `NeutralStartAnalysis.py` - Neutral start tests (T=30, β=0.5, initial phase)
- `NoInitialPhaseAnalysis.py` - **No initial phase tests (pure learning from round 0)**

**Documentation**:
- `README.md` - This file (comprehensive overview for supervisors)
- `CLAUDE.md` - Technical architecture documentation
- `ResearchDesign.md` - Detailed theoretical framework
- `STATISTICAL_REPORT.md` - Complete findings (10 pages)
- `PARAMETER_SENSITIVITY_REPORT.md` - Weight analysis findings

**Data**:
- `statistical_analysis_results.json` - Main statistical results (T=200, β=0.8, with initial phase)
- `parameter_sensitivity_results.json` - 19 weight combinations (T=200, β=0.8)
- `matching_protocol_analysis.json` - Matching comparison
- `nonstationary_analysis_results.json` - Environmental shocks (T=200)
- `short_horizon_results.json` - Short time horizon (T=50, β=0.8, with initial phase)
- `neutral_start_results.json` - Neutral start (T=30, β=0.5, with initial phase)
- `no_initial_phase_results.json` - **No initial phase (pure learning from round 0)**

All analyses use reproducible random seeds and are fully replicable.
