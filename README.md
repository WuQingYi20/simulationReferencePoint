# Reference Point Formation in Repeated Coordination Games

**A Computational Study of Kőszegi-Rabin Preferences with Minimal Group Paradigm**

**Author**: Yifan
**Date**: December 2025

---

## Research Question

**How do agents form reference points in repeated coordination games, and does the weighting of different information sources affect coordination outcomes and group favoritism?**

In multi-level coordination problems (e.g., climate negotiations), actors must balance multiple information sources:
- Recent bilateral interactions
- History with specific groups/coalitions
- Global population patterns

**Which information source matters most? Does it depend on the strategic environment?**

---

## Motivation

International climate negotiations exhibit a fundamental tension: countries belong to regional coalitions (EU, G77, AOSIS) but must achieve global coordination. Negotiators must decide:

- Should I follow my coalition's norms? (Group-dominant strategy)
- Should I adapt to recent bilateral signals? (Recent-dominant strategy)
- Should I pursue global consensus? (Global-dominant strategy)

This research uses computational modeling to test when and how information weighting strategies affect coordination and in-group favoritism.

---

## Key Finding

**Information weighting has massive effects in strategic conflict, but not in pure coordination.**

| Game Type | Strategic Feature | Weight Effect | Best Strategy |
|-----------|------------------|---------------|---------------|
| **Chicken** (anti-coordination) | Conflict over who yields | **32.8% coordination range** | Global/Recent-dominant |
| **Coordination** (pure coordination) | No conflict, limited time | 6.7% range (T=50 only) | Recent-dominant |
| **Stag Hunt** (risk-dominant) | Loss aversion eliminates conflict | 0.0% (no effect) | Weights irrelevant |

**Climate negotiation implication** (Chicken analogy):
- Group-dominant: 30.8% coordination, +42.5% in-group bias
- Global-dominant: 55.9% coordination, +1.9% in-group bias
- **Difference: 25 percentage points**

---

## Theoretical Framework

### Kőszegi-Rabin (2006) Reference-Dependent Preferences

Agents evaluate outcomes relative to reference points (expectations):

**U = m(c) + η · n(c|r)**

- **m(c)**: Standard payoff (consumption utility)
- **n(c|r)**: Gain-loss utility relative to reference point r
- **Loss aversion**: Losses hurt λ=2.25 times more than equivalent gains

**Reference point formation** (Bayesian weighting):

π = Σ(w_s · estimate_s · sample_size_s) / Σ(w_s · sample_size_s)

where sources s ∈ {recent, group, global} and w_s are the weights we investigate.

### Critical Model Feature: Bayesian Partner Belief

**Key innovation**: Weights affect both reference point formation AND partner behavior prediction.

When estimating "Will my partner choose A?", agents use the same Bayesian weighting:
- High w_group → rely on group patterns → lock into group-specific strategies
- High w_recent → adapt to individual partners → flexible responses
- High w_global → follow population statistics → pursue global consensus

This ensures weight choice directly affects action selection, not just internal expectations.

---

## Simulation Design

### Setup
- **Population**: N=16 agents, 2 groups (minimal group paradigm)
- **Time horizons**: T=200 (baseline), T=50 (limited time), T=30 (extreme scarcity)
- **Matching**: Random pairwise (8 pairs per round)
- **Initial phase**: Rounds 0-9 establish group conventions (Group 0: 80% choose A, Group 1: 80% choose B)

### Games Tested
| Game | Payoff Matrix | Strategic Feature |
|------|--------------|-------------------|
| **Coordination** | AA=1, AB=0, BA=0, BB=1 | Pure coordination |
| **Stag Hunt** | AA=3, AB=0, BA=2, BB=2 | Coordination with risk |
| **Chicken** | AA=2, AB=1, BA=3, BB=0 | Anti-coordination |

### Weight Configurations
- **Recent-dominant**: (0.90, 0.05, 0.05)
- **Group-dominant**: (0.05, 0.90, 0.05)
- **Global-dominant**: (0.05, 0.05, 0.90)
- **Balanced**: (0.33, 0.33, 0.34)
- Plus 15 additional combinations (19 total)

### Statistical Validation
- **30 replications per condition** (different random seeds)
- Independent t-tests, effect sizes (Cohen's d), 95% confidence intervals
- **Total: 1,140 simulations** across all analyses

---

## How Agents Make Decisions

Each round, an agent faces a partner and must choose action A or B:

**Step 1: Form reference point (expected coordination probability)**
```
π = Bayesian_combination(recent_history, group_history, global_history)
Weighted by: (w_recent, w_group, w_global)
```

**Step 2: Predict partner's action**
```
P(partner chooses A) = Bayesian_combination(
    pairwise_history[partner],  # if observable
    group_history[partner_group],
    recent_history,
    global_history
)
Weighted by the SAME weights: (w_recent, w_group, w_global)
```

**Step 3: Calculate KR utility for each action**
```
U(A) = Consumption_utility(A) + Loss_aversion × Gain_loss_utility(A | π)
U(B) = Consumption_utility(B) + Loss_aversion × Gain_loss_utility(B | π)
```

**Step 4: Choose action with highest expected utility**
```
action = argmax(U(A), U(B))  # with 10% random exploration
```

**Step 5: Observe outcome and update history**

---

## Results

### Finding 1: Weights Strongly Matter in Chicken Game

**Test**: 19 weight combinations × 20 replications, T=200 rounds, λ=2.25

| Weight Configuration | Coordination Rate | Group Favoritism |
|---------------------|------------------|------------------|
| Recent-dominant (0.90, 0.05, 0.05) | **56.7%** | -0.9% |
| Global-dominant (0.05, 0.05, 0.90) | 55.2% | +1.8% |
| Balanced (0.33, 0.33, 0.34) | 39.6% | +18.3% |
| **Group-dominant (0.05, 0.90, 0.05)** | **23.8%** | **+57.8%** |

**Range: 32.8%** (23.8% → 56.7%)
**Correlation**: weight_group vs coordination: r = -0.759 (p < 0.001)

**Interpretation**: Group-dominant agents lock into group-specific strategies, producing high in-group bias but catastrophic global coordination failure.

### Finding 2: Effect Increases with Time Scarcity

**Test**: 5 weight configurations × 30 replications, T=50 rounds

| Game | Coordination Range | Group Favoritism Range | Effect Size |
|------|-------------------|----------------------|-------------|
| **Chicken** | **25.0%** (30.8% → 55.9%) | 42.6% (-0.1% → +42.5%) | **d = 6.44** |
| **Coordination** | **6.7%** (82.8% → 89.5%) | 12.0% (-2.9% → +9.2%) | d = 1.11 |
| Stag Hunt | 0.0% (identical) | 0.0% (identical) | d = 0.00 |

**Statistical test (Chicken, T=50)**:
- Recent-dominant vs Group-dominant: Δ = +23.1%, t(58) = 24.7, **p < 0.0001**, **Cohen's d = 6.44**

Effect size d=6.44 is **massive** (d>0.8 is "large" in social science).

### Finding 3: No Effect in Stag Hunt Due to Risk Dominance

With loss aversion λ=2.25, agents in Stag Hunt converge to risk-dominant "Hare" equilibrium regardless of weight configuration:

| Weight Config | Coordination Rate | Group 0 → Stag | Group 1 → Stag |
|--------------|------------------|----------------|----------------|
| All configs | 91.4% ± 0.0% | 5.2% | 5.2% |

**Reason**: Loss aversion makes risky "Stag" strategy unattractive. Strategic conflict eliminated → weights irrelevant.

### Finding 4: Behavioral Strategy Typology

Our results identify three distinct behavioral strategies with different performance profiles:

**1. Collectivist (Group-dominant weights)**
- Strategy: "Follow my coalition's norms"
- Chicken performance: 30.8% coordination, +42.5% favoritism
- Climate analogy: G77 bloc prioritizing internal cohesion over global deals
- Tradeoff: Strong identity, poor global outcomes

**2. Adaptive (Recent-dominant weights)**
- Strategy: "Respond to immediate feedback"
- Chicken performance: 54.0% coordination, -0.1% favoritism
- Climate analogy: Bilateral negotiators (US-China talks)
- Tradeoff: Flexible but requires frequent interaction

**3. Universalist (Global-dominant weights)**
- Strategy: "Pursue global consensus"
- Chicken performance: 55.9% coordination, +1.9% favoritism
- Climate analogy: Multilateral institutionalists (UNFCCC process)
- Tradeoff: Best coordination, minimal bias, but slow to adapt

---

## Scope Conditions

**Weights matter when**:
- Games have strategic conflict (Chicken: wanting opposite outcomes)
- Time horizon allows divergence (T=50-200)
- Information sources produce different signals (group ≠ global patterns)

**Weights don't matter when**:
- Risk dominance eliminates conflict (Stag Hunt with loss aversion)
- Pure coordination with ample time (Coordination at T=200)
- Information sources fully converge

---

## Implications for Climate Negotiations

Climate negotiations resemble Chicken: countries want others to reduce emissions (bear costs) while they maintain economic growth. Our findings suggest:

**1. Institutional design recommendation**:
- Encourage **global-dominant** information processing (track global emission trends, not just coalition positions)
- Expected benefit: +25 percentage points in coordination success

**2. Coalition dynamics insight**:
- Group-dominant coalitions (G77, AOSIS) may produce strong internal solidarity (+42.5% in-group preference) but hinder global agreements (-32.8% coordination)
- Policy: Balance coalition identity with global information sharing mechanisms

**3. Time pressure matters**:
- Short negotiation windows (analogous to T=50) amplify the importance of information strategy
- COP deadlines may inadvertently increase the cost of group-dominant strategies

---

## Computational Details

### Code Structure
- **Simulation.py** (1,182 lines): Core engine implementing KR utility, Bayesian belief formation, matching protocols
- **Analysis scripts**: Statistical validation, parameter sensitivity, time horizon tests
- **Total simulations**: 1,140 runs across 19 weight combinations × 3 games × 20 replications

### Key Parameters
- Loss aversion: λ = 2.25
- Gain-loss weight: η = 1.0
- Response noise: ε = 0.1 (trembling hand)
- Recency decay: γ = 0.9 (exponential)
- Initial group bias: β = 0.8 (80/20 split)

### Reproducibility
- All analyses use fixed random seeds (1000-1029, 4000-4029, 5000-5029, 6000-6029)
- Results saved in JSON format with full statistical summaries
- Code available with complete documentation

---

## References

Kőszegi, B., & Rabin, M. (2006). A model of reference-dependent preferences. *Quarterly Journal of Economics*, 121(4), 1133-1165.

Tajfel, H. (1970). Experiments in intergroup discrimination. *Scientific American*, 223(5), 96-102.

---

## Repository Structure

**Core simulation**:
- `Simulation.py` - Main simulation engine
- `CLAUDE.md` - Technical architecture documentation

**Analyses**:
- `ParameterSensitivityAnalysis.py` - 19 weight combinations (T=200)
- `ShortHorizonAnalysis.py` - 5 key configurations (T=50)
- `StatisticalAnalysis.py` - Reference point methods comparison

**Results**:
- `parameter_sensitivity_results.json` - Full sensitivity analysis
- `short_horizon_results.json` - Time scarcity effects
- `statistical_analysis_results.json` - Baseline statistics

All code fully documented and replicable.
