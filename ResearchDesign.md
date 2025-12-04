# Reference Point Formation in Repeated Coordination Games: A Simulation Study

**Author**: Yifan  

---

## 1. Research Context and Motivation

### 1.1 Background: Institutional Dynamics in Climate Change

Climate change presents a complex multi-level coordination problem:

| Level | Actors | Game Structure |
|-------|--------|----------------|
| Macro | Nations, Institutions | Public goods → Chicken → Stag Hunt (as conditions change) |
| Meso | Organizations, Communities | Coordination with group identity |
| Micro | Individuals | Reference-dependent decision making |

Existing research (e.g., Paris Agreement studies) focuses on macro-level institutional interactions. However, **how individuals form expectations and coordinate within/across groups** remains under-explored.

### 1.2 From I-We Frame to Reference Points

Previous approach (I-We frame):
- Conceptually rich but computationally vague
- Difficult to operationalize in simulations

New approach (Reference Points):
- Grounded in Kőszegi-Rabin (2006) theory
- Computable from observable interaction history
- Connects psychological mechanisms to behavioral outcomes

**Core insight**: In coordination games, agents form **expectations as reference points** based on:
1. Historical interactions (pairwise, group-level, global)
2. Recency-weighted experiences
3. Group membership signals (minimal group paradigm)

---

## 2. Research Questions

### Primary Question
**How do agents form reference points in repeated coordination games, and how does reference point formation affect coordination outcomes and group favoritism?**

### Sub-questions

1. **Source weighting**: When forming expectations, how do agents weight:
   - Pairwise history (specific partner)
   - Group-specific history (in-group vs out-group)
   - Recent interactions vs distant past
   - Global population statistics

2. **Group favoritism emergence**: Under what conditions does minimal group assignment lead to differential coordination rates (in-group > out-group)?

3. **Convention persistence**: How do initial group-specific conventions persist or dissolve over time?

4. **Loss aversion effects**: How does the loss aversion parameter (λ) affect:
   - Speed of coordination convergence
   - Strength of group favoritism
   - Rigidity of established conventions

---

## 3. Theoretical Framework

### 3.1 Kőszegi-Rabin Reference-Dependent Utility

The utility function:

$$U = m(c) + \eta \cdot n(c|r)$$

Where:
- $m(c)$ = consumption utility (standard payoff)
- $n(c|r)$ = gain-loss utility relative to reference point
- $\eta$ = weight on gain-loss utility
- $r$ = reference point (a probability distribution over outcomes)

The gain-loss function:

$$\mu(x) = \begin{cases} x & \text{if } x \geq 0 \\ \lambda x & \text{if } x < 0 \end{cases}$$

Where $\lambda > 1$ captures loss aversion.

### 3.2 Reference Point as Expectation

Following KR (2006), the reference point is the agent's **rational expectation** about outcomes. In repeated games, this expectation is formed from:

$$\pi_{expected} = f(\text{History}, \text{Context})$$

**Key innovation of this research**: Specifying how $f(\cdot)$ combines different information sources.

### 3.3 Bayesian Reference Point Formation

Proposed model:

$$\pi = \frac{\sum_{s \in \text{Sources}} w_s \cdot \hat{\pi}_s \cdot n_s}{\sum_{s \in \text{Sources}} w_s \cdot n_s}$$

Where:
- $s \in \{\text{pairwise}, \text{group}, \text{recent}, \text{global}\}$
- $\hat{\pi}_s$ = coordination rate estimate from source $s$
- $n_s$ = number of observations in source $s$ (precision)
- $w_s$ = weight on source $s$ (to be estimated)

The weights $w_s$ are the **research focus** — they represent how agents cognitively weight different information.

---

## 4. Simulation Design

### 4.1 Basic Setup

| Parameter | Default Value | Notes |
|-----------|---------------|-------|
| N (agents) | 16 | Can vary: 8, 16, 32, 64 |
| T (rounds) | 200 | Sufficient for convergence |
| Groups | 2 | Minimal group paradigm |
| λ (loss aversion) | 2.0 | Standard value from literature |
| η (gain-loss weight) | 1.0 | |
| Noise | 0.1 | Trembling hand |

### 4.2 Game Structure

**Pure Coordination Game**:
|  | A | B |
|--|---|---|
| A | (1, 1) | (0, 0) |
| B | (0, 0) | (1, 1) |

**Extensions** (see Section 6):
- Stag Hunt (payoff-dominant vs risk-dominant)
- Chicken (conflict game)
- Asymmetric coordination

### 4.3 Matching Protocol

**Option 1: Random Matching**
- Each round: agents randomly paired
- Pro: Realistic, forces use of group priors
- Con: Variable interaction frequency per pair

**Option 2: Round Robin**
- Each agent plays every other agent each "super-round"
- Pro: Balanced data
- Con: Less realistic

**Recommended**: Random matching (more interesting dynamics)

### 4.4 Initial Conditions

**Minimal Group Paradigm Implementation**:
- Arbitrary group assignment (e.g., "overestimators" vs "underestimators")
- No intrinsic meaning to group labels

**Initial Convention Seeding**:
- First $T_0$ rounds: biased action choice
- Group 0: P(A) = $\beta$
- Group 1: P(A) = $1 - \beta$
- Where $\beta \in [0.5, 1.0]$ controls initial convention strength

This models that groups may have different "cultures" or starting norms.

### 4.5 Reference Point Methods (Treatments)

| Method | Description | Research Question |
|--------|-------------|-------------------|
| `global` | Use all interaction history | Baseline |
| `group` | Use group-specific history only | Does group info matter? |
| `pairwise` | Use partner-specific history | Importance of individual reputation |
| `recent` | Exponentially decay old interactions | Recency effects |
| `bayesian` | Weighted combination (Section 3.3) | Full model |

### 4.6 Agent Decision Rule

```
For each interaction:
1. Observe: partner_id, partner_group
2. Calculate: reference_point π based on method
3. Estimate: P(partner plays A) from history
4. Compute: U(A), U(B) using KR utility
5. Choose: argmax with noise ε
6. Record: outcome to history
```

---

## 5. Outcome Measures and Hypotheses

### 5.1 Primary Outcomes

1. **Coordination Rate**: Proportion of successful coordination
   - Overall
   - In-group pairs
   - Out-group pairs

2. **Group Favoritism**: 
   $$\text{Favoritism} = \text{InGroupRate} - \text{OutGroupRate}$$

3. **Convention Strength**:
   - Group 0: P(A)
   - Group 1: P(A)
   - Divergence = |P(A|Group0) - P(A|Group1)|

4. **Convergence Speed**: Rounds to reach 90% coordination

### 5.2 Hypotheses

**H1 (Group Favoritism)**: With initial group conventions and group-based reference points, in-group coordination > out-group coordination.

**H2 (Method Effects)**: Different reference point methods produce different outcomes:
- `group` method → strongest group favoritism
- `pairwise` method → fastest convergence within pairs
- `global` method → fastest overall convergence but less group favoritism

**H3 (Loss Aversion)**: Higher λ → 
- Stronger adherence to established conventions
- Slower adaptation to new partners
- Greater group favoritism persistence

**H4 (Initial Bias)**: Stronger initial bias ($\beta$ closer to 1) →
- Stronger group favoritism
- But lower overall coordination (harder to adapt across groups)

---

## 6. Extensions and Robustness

### 6.1 Game Structure Variations

**Stag Hunt** (Risk vs Payoff Dominance):
|  | A (Stag) | B (Hare) |
|--|----------|----------|
| A | (3, 3) | (0, 2) |
| B | (2, 0) | (2, 2) |

Why interesting: 
- A is payoff-dominant
- B is risk-dominant
- Reference point affects risk perception

**Prediction**: Higher λ → more risk-averse → more B choices

**Chicken** (Anti-coordination):
|  | Swerve | Straight |
|--|--------|----------|
| Swerve | (2, 2) | (1, 3) |
| Straight | (3, 1) | (0, 0) |

Why interesting:
- Models escalation/de-escalation
- Relevant to climate chicken game

### 6.2 Information Variations

| Treatment | Agent Knows |
|-----------|-------------|
| Full info | Partner ID + Group |
| Group only | Partner Group only |
| Anonymous | Nothing (only global history) |

**Prediction**: Less information → group priors more important → more group favoritism

### 6.3 Population Structure

- **Homogeneous mixing**: Current setup
- **Network structure**: Clustered interactions
- **Migration**: Agents can change groups (endogenous identity)

### 6.4 Sensitivity Analysis

| Parameter | Range | Purpose |
|-----------|-------|---------|
| N | 8, 16, 32, 64 | Scale effects |
| λ | 1.0 - 4.0 | Loss aversion |
| β (initial bias) | 0.5 - 1.0 | Initial convention strength |
| T₀ (initial rounds) | 5, 10, 20 | Convention establishment |
| ε (noise) | 0.01 - 0.2 | Decision stochasticity |
| Recency decay | 0.8 - 0.99 | Memory structure |

---

## 7. Validation Strategy

### 7.1 Internal Validation

1. **Boundary cases**: 
   - λ = 1 should reduce to expected utility
   - β = 0.5 should produce no group favoritism (initially)
   - ε = 1 should produce random behavior

2. **Convergence checks**:
   - Run until stability
   - Multiple seeds for variance estimation

3. **Analytical benchmarks**:
   - Compare to known equilibria in simple cases

### 7.2 External Validation (Empirical Testing)

**Proposed Experimental Design**:

| Phase | Description |
|-------|-------------|
| 1. Group assignment | Random "overestimator/underestimator" (minimal group) |
| 2. Training | Initial rounds with group-mates only (establish convention) |
| 3. Main game | Mixed matching with group info visible |
| 4. Elicitation | Measure expectations before each round |

**Key measurements**:
- Action choices
- Stated expectations (to directly test reference point formation)
- Coordination outcomes

**Comparison to simulation**:
- Estimate weights $w_s$ from experimental data
- Test if simulated behavior matches actual behavior
- Identify systematic deviations

### 7.3 Connecting to Climate Context

| Simulation Feature | Climate Analogue |
|--------------------|------------------|
| Groups | Countries / Blocs (e.g., EU, US, China) |
| Initial conventions | Existing norms and policies |
| Reference points | Expectations about others' commitments |
| Coordination failure | Failed climate agreements |
| Game transitions | Public goods → Chicken → Stag Hunt |

---

## 8. Implementation Plan

### Phase 1: Basic Simulation (Current)
- [x] Pure coordination game
- [x] Multiple reference point methods
- [x] Initial group conventions
- [x] Basic analysis

### Phase 2: Extended Games
- [ ] Stag Hunt implementation
- [ ] Chicken game implementation  
- [ ] Asymmetric payoffs

### Phase 3: Information Treatments
- [ ] Anonymous matching
- [ ] Group-only information
- [ ] Belief elicitation (explicit reference point tracking)

### Phase 4: Analysis and Writing
- [ ] Systematic parameter sweeps
- [ ] Statistical analysis
- [ ] Draft paper
- [ ] Connect to experimental design

---

## 9. Questions for Discussion (12.8 Meeting)

1. **Scope**: Should we focus on pure coordination first, or move directly to Stag Hunt?

2. **Empirical connection**: What experimental paradigm would best test these predictions?

3. **Climate framing**: How explicit should the climate change connection be?

4. **Theoretical contribution**: Is the Bayesian reference point formation model novel enough?

5. **Ontology fit**: How does this connect to the broader research program?

---

## 10. References

- Akerlof, G. A., & Kranton, R. E. (2000). Economics and identity. *Quarterly Journal of Economics*.
- Kőszegi, B., & Rabin, M. (2006). A model of reference-dependent preferences. *Quarterly Journal of Economics*.
- Tajfel, H. (1970). Experiments in intergroup discrimination. *Scientific American*. (Minimal group paradigm)
- [Climate game theory literature - to be added]

---

## Appendix: Code Structure

```
kr_coordination_sim/
├── simulation.py      # Main simulation engine
│   ├── SimConfig      # All parameters
│   ├── Agent          # Agent with history and decision
│   ├── HistoryTracker # Multi-level history management
│   ├── ReferencePointCalculator  # Different methods
│   ├── KRUtility      # Utility calculations
│   └── CoordinationGame  # Game manager
├── analysis.py        # Analysis and visualization
└── figures/           # Generated plots
```

### Key Configuration Parameters

```python
SimConfig(
    n_agents=16,
    n_rounds=200,
    n_groups=2,
    game_type="coordination",     # or "stag_hunt", "chicken"
    lambda_loss=2.0,              # Loss aversion
    eta=1.0,                      # Gain-loss weight
    weight_recent=0.5,            # Bayesian weights
    weight_group=0.3,
    weight_global=0.2,
    recency_decay=0.9,
    response_noise=0.1,
    initial_group_bias=0.8,       # Convention strength
    n_initial_rounds=10,
    info_treatment="full",        # or "group_only", "anonymous"
    matching_type="random",
    seed=42
)
```

---

## Appendix B: Preliminary Simulation Results

### B.1 Game Type Comparison (N=16, T=200, λ=2.0)

| Game | Payoff Matrix | Final Coord | Favoritism | Group 0 (A) | Group 1 (A) |
|------|---------------|-------------|------------|-------------|-------------|
| **Coordination** | AA=1, AB=0, BA=0, BB=1 | **91.3%** | +1.0% | 64% | 32% |
| **Stag Hunt** | AA=3, AB=0, BA=2, BB=2 | **90.6%** | -3.3% | 16% | 9% |
| **Chicken** | AA=2, AB=1, BA=3, BB=0 | **17.5%** | +7.9% | 64% | 43% |

**Key Findings:**

1. **Stag Hunt**: Both groups converge to **Hare (B)** — the risk-dominant but not payoff-dominant strategy. Loss aversion (λ=2) makes agents avoid the risky Stag choice. This is a key prediction!

2. **Chicken**: Low "coordination" rate is expected (it's an anti-coordination game). Strong group favoritism (+7.9%) emerges because groups maintain different aggression levels.

3. **Coordination**: Groups maintain distinct conventions (Group 0→A, Group 1→B) with slight in-group advantage.

### B.2 Information Treatment Comparison (Coordination Game)

| Treatment | Agent Knows | Final Coord | Favoritism |
|-----------|-------------|-------------|------------|
| **Full** | Partner ID + Group | 91.3% | +1.0% |
| **Group Only** | Partner Group | 93.8% | -2.9% |
| **Anonymous** | Nothing | 94.4% | -2.5% |

**Key Finding**: Less information → better overall coordination but less group favoritism. This is because:
- With full info, agents adapt to specific partners → can maintain separate conventions
- With less info, everyone uses global statistics → faster convergence to single convention

### B.3 Reference Point Method Comparison

All methods produce **identical results** in the pure coordination game. This confirms that in symmetric coordination, the reference point formation method doesn't affect the optimal response — agents simply best-respond to their belief about partner's action.

**Implication**: Need asymmetric games (Stag Hunt, Chicken) to see method effects.