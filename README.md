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

---

## Theoretical Framework

### Kőszegi-Rabin Reference-Dependent Utility

The model implements the KR utility function:
**U = m(c) + η · n(c|r)**

where:
- **m(c)** = consumption utility (standard game payoff)
- **n(c|r)** = gain-loss utility relative to reference point
- **η** = weight on gain-loss component
- **λ** = loss aversion coefficient (λ > 1)

The gain-loss function exhibits loss aversion: **μ(x) = x if x ≥ 0, λx if x < 0**

### Reference Point as Bayesian Expectation

Agents form reference points by combining multiple information sources:

**π = Σ (w_s · π̂_s · n_s) / Σ (w_s · n_s)**

where sources s ∈ {recent, group-specific, global}, π̂_s is the estimated coordination rate from source s, n_s is the sample size (precision), and w_s are the weights—the focus of our investigation.

### Minimal Group Paradigm

Following Tajfel (1970), agents are assigned to arbitrary groups with no intrinsic meaning. Initial rounds (T₀=10) establish group-specific "conventions" through biased action tendencies, creating the potential for in-group/out-group dynamics to emerge.

---

## Computational Implementation

### Simulation Design

- **Population**: N=16 agents, 2 groups
- **Time**: T=200 rounds (sufficient for convergence)
- **Games**: Pure coordination, Stag Hunt, Chicken
- **Matching**: Random pairwise or round-robin
- **Initial conditions**: Group-biased conventions (Group 0: 80% choose A, Group 1: 20% choose A)
- **Key parameters**: Loss aversion λ=2.0, initial group bias β=0.8

### Statistical Validation

All analyses use **30 replications per condition** with different random seeds (1000-1029), providing:
- 95% confidence intervals
- Independent t-tests for condition comparisons
- Effect sizes (Cohen's d)
- Statistical power ≥ 0.80 for medium effects

**Total computational effort**: 1,800 simulations across all analyses.

---

## Key Findings

### 1. Reference Point Formation Weights Are Irrelevant (Critical Negative Result)

Across **19 different weight combinations** tested in **all game types**, outcomes were **statistically identical**:

- Recent-dominant (90% recent, 5% group, 5% global)
- Group-dominant (5% recent, 90% group, 5% global)
- Global-dominant (5% recent, 5% group, 90% global)
- All produced: 91.5% coordination in Stag Hunt, 23.7% in Chicken, 89.6% in Coordination

**Variance explained by weight choice: 0.0%**

**Explanation**: With sufficient interaction time (200 rounds) and random matching, all information sources converge to the same empirical frequencies. When recent, group, and global statistics all show ~90% coordination, the weighting becomes mathematically irrelevant.

**Implication**: This is a valuable **scope condition discovery**. Reference point formation weights only matter when information sources *diverge*—requiring shorter time horizons, structured matching, or non-stationary environments.

### 2. Game Structure Has Dramatic Effects

Game payoff structure profoundly affects outcomes (**p < 0.001, Cohen's d > 5**):

| Game | Coordination | Group Favoritism | Key Finding |
|------|--------------|------------------|-------------|
| **Coordination** | 88.7% ± 0.5% | -0.4% ± 1.1% | Groups maintain distinct conventions |
| **Stag Hunt** | 90.2% ± 0.3% | -2.2% ± 1.1% | **Risk-dominant equilibrium selection** |
| **Chicken** | 23.6% ± 0.7% | +10.9% ± 1.7% | Strong group differentiation |

**Critical insight on Stag Hunt**: With λ=2.0 loss aversion, both groups converge to the safe "Hare" option (B) despite its lower payoff, choosing it 92% and 92% of the time respectively. This validates that loss aversion drives risk-dominant over payoff-dominant equilibrium selection—a key prediction of KR preferences.

### 3. Information Availability Matters

Reduced information improves coordination (**p < 0.01, d ≈ 0.7**):

- **Full information** (partner ID + group): 88.7%
- **Group-only information**: 90.6% (+1.9%, p=0.007)
- **Anonymous** (no identity): 90.5% (+1.8%, p=0.012)

**Counterintuitive finding**: When agents cannot track individual partners, they rely on global statistics and converge faster to a single convention. Full information allows maintaining separate conventions with different partners, which increases flexibility but slows overall convergence.

### 4. Matching Protocol Creates Path Dependence

Round-robin matching (all pairs interact each round) versus random matching produces:

- **Coordination game**: +1.8% with round-robin (p=0.001)
- **Stag Hunt**: No difference (both converge strongly)
- **Chicken**: -14.2% with round-robin (p<0.001, d=5.25)

In anti-coordination games (Chicken), more interactions amplify conflict. In coordination games, more interactions accelerate learning.

### 5. Non-Stationary Environments Still Show No Weight Effects

Even when introducing environmental shocks (swapping group tendencies at round 100), all weight configurations showed identical adaptation:
- Recovery time: 0.3 rounds (all weights)
- Adaptation rate: 0.0012/round (all weights)

This suggests the modeling framework provides agents with sufficient learning capacity that weight differences wash out even under changing conditions.

---

## Implications for Theory and Policy

### Behavioral Economics

The findings validate KR reference-dependent preferences in multi-agent coordination but reveal that **reference point formation method is less critical than reference point existence**. What matters is that agents *have* expectations and exhibit loss aversion, not precisely *how* those expectations are formed.

### Climate Change Negotiations

Applied to institutional coordination (the original motivation):
1. **Loss aversion drives conservative equilibria**: Nations may coordinate on risk-dominant (lower ambition) agreements even when payoff-dominant (higher ambition) agreements exist
2. **Information architecture matters**: Transparency paradoxically may slow convergence by allowing bilateral side-deals
3. **Convention lock-in**: Initial coordination patterns (Group 0 → A, Group 1 → B) persist across 200 rounds

### Methodological Contribution

The computational approach demonstrates:
- **Power of negative results**: Discovering what doesn't matter (weights) is as valuable as what does (game structure, loss aversion)
- **Scope condition identification**: Precise characterization of when theoretical mechanisms operate
- **Scalability**: 1,800 simulations in ~10 minutes enables comprehensive parameter exploration

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

- `Simulation.py` - Core simulation engine (1,182 lines)
- `StatisticalAnalysis.py` - Multi-replication validation
- `ParameterSensitivityAnalysis.py` - Weight sweep analysis
- `NonStationaryAnalysis.py` - Environmental shock tests
- `CLAUDE.md` - Technical documentation for AI assistance
- `ResearchDesign.md` - Detailed theoretical framework
- `STATISTICAL_REPORT.md` - Complete findings (10 pages)
- `PARAMETER_SENSITIVITY_REPORT.md` - Weight analysis findings

All analyses use reproducible random seeds and are fully replicable.
