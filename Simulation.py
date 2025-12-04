"""
Reference Point Formation in Repeated Coordination Games
=========================================================
Research Question: How do agents form reference points?
- From recent interactions?
- From group-specific history?
- Bayesian combination?

Framework: Kőszegi-Rabin (2006) reference-dependent preferences
Game: Pure coordination game with minimal group paradigm
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import json
from collections import defaultdict


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SimConfig:
    """Simulation parameters - easily adjustable"""
    # Population
    n_agents: int = 16
    n_groups: int = 2  # Minimal group paradigm: arbitrary assignment
    
    # Time
    n_rounds: int = 200
    
    # Game type: "coordination", "stag_hunt", "chicken"
    game_type: str = "coordination"
    
    # Game payoffs - will be overridden based on game_type if not set
    # Pure coordination: match=1, mismatch=0
    # Stag Hunt: AA=3, AB=0, BA=2, BB=2
    # Chicken: AA=2, AB=1, BA=3, BB=0
    payoff_AA: float = None  # Both choose A
    payoff_AB: float = None  # I choose A, partner chooses B
    payoff_BA: float = None  # I choose B, partner chooses A
    payoff_BB: float = None  # Both choose B
    
    # KR utility parameters
    lambda_loss: float = 2.0       # Loss aversion coefficient (λ > 1)
    eta: float = 1.0               # Weight on gain-loss utility
    
    # Reference point formation weights (research variables)
    weight_recent: float = 0.5     # Weight on recent interactions
    weight_group: float = 0.3      # Weight on group-specific history
    weight_global: float = 0.2     # Weight on global history
    
    # Memory parameters
    recency_decay: float = 0.9     # Exponential decay for recency weighting
    min_history_for_belief: int = 1  # Minimum interactions before using history
    
    # Decision noise
    response_noise: float = 0.1    # Probability of random choice (trembling hand)
    
    # Initial group convention (KEY FOR GROUP FAVORITISM)
    initial_group_bias: float = 0.8  # 0.5 = no bias, 0.8 = strong initial convention
    n_initial_rounds: int = 10       # Rounds with biased initial choices
    
    # Information treatment: "full", "group_only", "anonymous"
    info_treatment: str = "full"
    
    # Matching type
    matching_type: str = "random"  # "round_robin" or "random"
    
    # Random seed
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Set default payoffs based on game type"""
        if self.game_type == "coordination":
            if self.payoff_AA is None: self.payoff_AA = 1.0
            if self.payoff_AB is None: self.payoff_AB = 0.0
            if self.payoff_BA is None: self.payoff_BA = 0.0
            if self.payoff_BB is None: self.payoff_BB = 1.0
        elif self.game_type == "stag_hunt":
            # A = Stag (risky, high reward), B = Hare (safe)
            if self.payoff_AA is None: self.payoff_AA = 3.0
            if self.payoff_AB is None: self.payoff_AB = 0.0
            if self.payoff_BA is None: self.payoff_BA = 2.0
            if self.payoff_BB is None: self.payoff_BB = 2.0
        elif self.game_type == "chicken":
            # A = Swerve (safe), B = Straight (aggressive)
            if self.payoff_AA is None: self.payoff_AA = 2.0
            if self.payoff_AB is None: self.payoff_AB = 1.0
            if self.payoff_BA is None: self.payoff_BA = 3.0
            if self.payoff_BB is None: self.payoff_BB = 0.0
        else:
            # Default to coordination
            if self.payoff_AA is None: self.payoff_AA = 1.0
            if self.payoff_AB is None: self.payoff_AB = 0.0
            if self.payoff_BA is None: self.payoff_BA = 0.0
            if self.payoff_BB is None: self.payoff_BB = 1.0


class Action(Enum):
    A = 0
    B = 1


# =============================================================================
# History Tracking
# =============================================================================

@dataclass
class InteractionRecord:
    """Single interaction record"""
    round: int
    partner_id: int
    partner_group: int
    my_action: Action
    partner_action: Action
    payoff: float


class HistoryTracker:
    """
    Tracks interaction history at multiple levels:
    - Global: all interactions
    - Group-level: interactions with each group
    - Pairwise: interactions with each specific partner
    """
    
    def __init__(self, agent_id: int, agent_group: int):
        self.agent_id = agent_id
        self.agent_group = agent_group
        
        # Store all interactions
        self.all_interactions: List[InteractionRecord] = []
        
        # Indexed views for efficient lookup
        self.by_partner: Dict[int, List[InteractionRecord]] = defaultdict(list)
        self.by_group: Dict[int, List[InteractionRecord]] = defaultdict(list)
    
    def add(self, record: InteractionRecord):
        self.all_interactions.append(record)
        self.by_partner[record.partner_id].append(record)
        self.by_group[record.partner_group].append(record)
    
    def get_partner_history(self, partner_id: int) -> List[InteractionRecord]:
        return self.by_partner[partner_id]
    
    def get_group_history(self, group_id: int) -> List[InteractionRecord]:
        return self.by_group[group_id]
    
    def get_all_history(self) -> List[InteractionRecord]:
        return self.all_interactions
    
    def get_recent_history(self, n: int = 10) -> List[InteractionRecord]:
        return self.all_interactions[-n:] if self.all_interactions else []


# =============================================================================
# Reference Point Formation
# =============================================================================

class ReferencePointCalculator:
    """
    Calculates reference point (expected probability of coordination)
    using different information sources
    """
    
    def __init__(self, config: SimConfig):
        self.config = config
    
    def calculate_coordination_rate(
        self, 
        history: List[InteractionRecord],
        decay: Optional[float] = None
    ) -> Optional[float]:
        """
        Calculate coordination rate from history
        Optionally apply recency weighting with exponential decay
        """
        if not history:
            return None
        
        if decay is None:
            # Simple average
            coordinated = sum(
                1 for r in history 
                if r.my_action == r.partner_action
            )
            return coordinated / len(history)
        else:
            # Recency-weighted average
            weights = []
            coordinated_weighted = 0
            for i, record in enumerate(reversed(history)):
                w = decay ** i
                weights.append(w)
                if record.my_action == record.partner_action:
                    coordinated_weighted += w
            return coordinated_weighted / sum(weights) if weights else None
    
    def get_reference_point(
        self,
        history_tracker: HistoryTracker,
        partner_id: int,
        partner_group: int,
        method: str = "bayesian"
    ) -> float:
        """
        Calculate reference point π (expected prob of coordination)
        
        Methods:
        - "global": use all history
        - "group": use group-specific history only
        - "pairwise": use partner-specific history only  
        - "recent": use recent history only
        - "bayesian": weighted combination (research focus)
        """
        cfg = self.config
        
        # Get histories
        pairwise_hist = history_tracker.get_partner_history(partner_id)
        group_hist = history_tracker.get_group_history(partner_group)
        global_hist = history_tracker.get_all_history()
        recent_hist = history_tracker.get_recent_history(n=10)
        
        if method == "global":
            rate = self.calculate_coordination_rate(global_hist)
            return rate if rate is not None else 0.5
        
        elif method == "group":
            rate = self.calculate_coordination_rate(group_hist)
            return rate if rate is not None else 0.5
        
        elif method == "pairwise":
            rate = self.calculate_coordination_rate(pairwise_hist)
            if rate is not None:
                return rate
            # Fallback to group if no pairwise history
            rate = self.calculate_coordination_rate(group_hist)
            return rate if rate is not None else 0.5
        
        elif method == "recent":
            rate = self.calculate_coordination_rate(
                recent_hist, 
                decay=cfg.recency_decay
            )
            return rate if rate is not None else 0.5
        
        elif method == "bayesian":
            # Weighted combination based on information precision
            # This is the key research contribution
            return self._bayesian_reference_point(
                pairwise_hist, group_hist, global_hist, recent_hist
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _bayesian_reference_point(
        self,
        pairwise_hist: List[InteractionRecord],
        group_hist: List[InteractionRecord],
        global_hist: List[InteractionRecord],
        recent_hist: List[InteractionRecord]
    ) -> float:
        """
        Bayesian combination of different information sources
        
        Key insight: Weight by precision (inverse variance)
        More observations → higher precision → higher weight
        """
        cfg = self.config
        
        estimates = []
        precisions = []
        
        # Pairwise estimate (highest specificity)
        if len(pairwise_hist) >= cfg.min_history_for_belief:
            rate = self.calculate_coordination_rate(pairwise_hist)
            if rate is not None:
                # Precision proportional to sample size
                # Could be more sophisticated (beta posterior)
                precision = len(pairwise_hist)
                estimates.append(rate)
                precisions.append(precision * 2.0)  # Extra weight for specificity
        
        # Group estimate (medium specificity)
        if len(group_hist) >= cfg.min_history_for_belief:
            rate = self.calculate_coordination_rate(group_hist)
            if rate is not None:
                precision = len(group_hist)
                estimates.append(rate)
                precisions.append(precision * cfg.weight_group / cfg.weight_recent)
        
        # Recent estimate (high relevance)
        if len(recent_hist) >= cfg.min_history_for_belief:
            rate = self.calculate_coordination_rate(
                recent_hist, 
                decay=cfg.recency_decay
            )
            if rate is not None:
                estimates.append(rate)
                precisions.append(len(recent_hist) * cfg.weight_recent / cfg.weight_group)
        
        # Global estimate (low specificity, fallback)
        if len(global_hist) >= cfg.min_history_for_belief:
            rate = self.calculate_coordination_rate(global_hist)
            if rate is not None:
                precision = len(global_hist)
                estimates.append(rate)
                precisions.append(precision * cfg.weight_global / cfg.weight_recent)
        
        # Precision-weighted average
        if estimates:
            total_precision = sum(precisions)
            weighted_sum = sum(e * p for e, p in zip(estimates, precisions))
            return weighted_sum / total_precision
        else:
            # No history: uninformative prior
            return 0.5


# =============================================================================
# KR Utility Calculation  
# =============================================================================

class KRUtility:
    """
    Kőszegi-Rabin (2006) reference-dependent utility
    
    U = m(c) + η * n(c|r)
    
    where:
    - m(c) = consumption utility (standard payoff)
    - n(c|r) = gain-loss utility relative to reference point
    - η = weight on gain-loss utility
    
    Gain-loss function μ(x):
    - μ(x) = x      if x ≥ 0 (gains)
    - μ(x) = λx     if x < 0 (losses, λ > 1)
    """
    
    def __init__(self, config: SimConfig):
        self.config = config
    
    def gain_loss_function(self, x: float) -> float:
        """μ(x) with loss aversion"""
        if x >= 0:
            return x
        else:
            return self.config.lambda_loss * x
    
    def calculate_expected_utility(
        self,
        my_action: Action,
        prob_partner_A: float
    ) -> float:
        """
        Calculate expected KR utility for an action
        
        Args:
            my_action: The action I'm considering
            prob_partner_A: My belief about P(partner plays A)
        
        Returns:
            Expected utility under KR preferences
        """
        cfg = self.config
        π = prob_partner_A
        
        # My possible payoffs given my action
        if my_action == Action.A:
            payoff_if_partner_A = cfg.payoff_AA  # I play A, partner plays A
            payoff_if_partner_B = cfg.payoff_AB  # I play A, partner plays B
        else:  # Action.B
            payoff_if_partner_A = cfg.payoff_BA  # I play B, partner plays A
            payoff_if_partner_B = cfg.payoff_BB  # I play B, partner plays B
        
        # Expected consumption utility
        m_expected = π * payoff_if_partner_A + (1 - π) * payoff_if_partner_B
        
        # Gain-loss utility using KR framework
        # Reference point is the same distribution as outcome (rational expectations)
        # We compare all outcome-reference pairs
        
        # Four cases (outcome × reference):
        # 1. Outcome=partner_A, Ref=partner_A: μ(payoff_A - payoff_A) = 0
        # 2. Outcome=partner_A, Ref=partner_B: μ(payoff_A - payoff_B)
        # 3. Outcome=partner_B, Ref=partner_A: μ(payoff_B - payoff_A)
        # 4. Outcome=partner_B, Ref=partner_B: μ(payoff_B - payoff_B) = 0
        
        diff_AB = payoff_if_partner_A - payoff_if_partner_B  # Could be + or -
        diff_BA = payoff_if_partner_B - payoff_if_partner_A  # = -diff_AB
        
        n_expected = (
            π * π * self.gain_loss_function(0) +                    # Case 1
            π * (1-π) * self.gain_loss_function(diff_AB) +          # Case 2
            (1-π) * π * self.gain_loss_function(diff_BA) +          # Case 3
            (1-π) * (1-π) * self.gain_loss_function(0)              # Case 4
        )
        
        return m_expected + cfg.eta * n_expected
    
    def calculate_both_utilities(
        self,
        prob_partner_A: float
    ) -> Tuple[float, float]:
        """
        Calculate expected utility for both actions
        
        Returns:
            (U(A), U(B)) tuple
        """
        u_A = self.calculate_expected_utility(Action.A, prob_partner_A)
        u_B = self.calculate_expected_utility(Action.B, prob_partner_A)
        return u_A, u_B


# =============================================================================
# Agent
# =============================================================================

class Agent:
    """
    Agent in coordination game
    Makes decisions based on KR utility with reference-dependent preferences
    """
    
    def __init__(
        self, 
        agent_id: int, 
        group_id: int, 
        config: SimConfig,
        ref_point_method: str = "bayesian"
    ):
        self.id = agent_id
        self.group = group_id
        self.config = config
        self.ref_point_method = ref_point_method
        
        self.history = HistoryTracker(agent_id, group_id)
        self.ref_calculator = ReferencePointCalculator(config)
        self.kr_utility = KRUtility(config)
        
        # Track decisions for analysis
        self.decision_log: List[Dict] = []
    
    def choose_action(
        self, 
        partner_id: int, 
        partner_group: int,
        current_round: int,
        rng: np.random.Generator
    ) -> Action:
        """
        Choose action based on KR utility maximization
        """
        cfg = self.config
        
        # INITIAL ROUNDS: Use group-specific convention
        # This models that groups may have different initial "cultures"
        if current_round < cfg.n_initial_rounds:
            if self.group == 0:
                prob_A = cfg.initial_group_bias  # Group 0 tends to choose A
            else:
                prob_A = 1 - cfg.initial_group_bias  # Group 1 tends to choose B
            
            action = Action.A if rng.random() < prob_A else Action.B
            decision_type = "initial_convention"
            
            # Log decision
            self.decision_log.append({
                "round": current_round,
                "partner_id": partner_id,
                "partner_group": partner_group,
                "ref_point": 0.5,
                "prob_partner_A": prob_A,
                "u_A": 0,
                "u_B": 0,
                "action": action.name,
                "decision_type": decision_type
            })
            return action
        
        # AFTER INITIAL ROUNDS: KR utility maximization
        # Estimate P(partner plays A) based on information treatment
        prob_partner_A = self._estimate_partner_action(
            partner_id, partner_group, cfg.info_treatment
        )
        
        # Calculate expected utilities for both actions
        u_A, u_B = self.kr_utility.calculate_both_utilities(prob_partner_A)
        
        # Decision with noise
        if rng.random() < cfg.response_noise:
            # Trembling hand: random choice
            action = Action.A if rng.random() < 0.5 else Action.B
            decision_type = "noise"
        else:
            # Utility maximization
            if u_A > u_B:
                action = Action.A
            elif u_B > u_A:
                action = Action.B
            else:
                # Tie: random
                action = Action.A if rng.random() < 0.5 else Action.B
            decision_type = "utility"
        
        # Get reference point for logging
        ref_point = self.ref_calculator.get_reference_point(
            self.history,
            partner_id,
            partner_group,
            method=self.ref_point_method
        )
        
        # Log decision
        self.decision_log.append({
            "round": current_round,
            "partner_id": partner_id,
            "partner_group": partner_group,
            "ref_point": ref_point,
            "prob_partner_A": prob_partner_A,
            "u_A": u_A,
            "u_B": u_B,
            "action": action.name,
            "decision_type": decision_type
        })
        
        return action
    
    def _estimate_partner_action(
        self,
        partner_id: int,
        partner_group: int,
        info_treatment: str
    ) -> float:
        """
        Estimate P(partner plays A) based on available information
        
        info_treatment:
        - "full": know partner_id and partner_group
        - "group_only": know only partner_group
        - "anonymous": know nothing specific
        """
        if info_treatment == "full":
            # Use pairwise history if available, fallback to group, then global
            partner_hist = self.history.get_partner_history(partner_id)
            if partner_hist:
                return sum(1 for r in partner_hist if r.partner_action == Action.A) / len(partner_hist)
            
            group_hist = self.history.get_group_history(partner_group)
            if group_hist:
                return sum(1 for r in group_hist if r.partner_action == Action.A) / len(group_hist)
            
            global_hist = self.history.get_all_history()
            if global_hist:
                return sum(1 for r in global_hist if r.partner_action == Action.A) / len(global_hist)
            
            return 0.5
        
        elif info_treatment == "group_only":
            # Only use group-level history
            group_hist = self.history.get_group_history(partner_group)
            if group_hist:
                return sum(1 for r in group_hist if r.partner_action == Action.A) / len(group_hist)
            
            # Fallback to global
            global_hist = self.history.get_all_history()
            if global_hist:
                return sum(1 for r in global_hist if r.partner_action == Action.A) / len(global_hist)
            
            return 0.5
        
        elif info_treatment == "anonymous":
            # Only use global history
            global_hist = self.history.get_all_history()
            if global_hist:
                return sum(1 for r in global_hist if r.partner_action == Action.A) / len(global_hist)
            
            return 0.5
        
        else:
            # Default to full
            return self._estimate_partner_action(partner_id, partner_group, "full")
    
    def record_interaction(
        self,
        round_num: int,
        partner_id: int,
        partner_group: int,
        my_action: Action,
        partner_action: Action,
        payoff: float
    ):
        """Record interaction outcome"""
        record = InteractionRecord(
            round=round_num,
            partner_id=partner_id,
            partner_group=partner_group,
            my_action=my_action,
            partner_action=partner_action,
            payoff=payoff
        )
        self.history.add(record)


# =============================================================================
# Game / Simulation Manager
# =============================================================================

class CoordinationGame:
    """
    Manages the repeated coordination game simulation
    """
    
    def __init__(
        self, 
        config: SimConfig,
        ref_point_method: str = "bayesian"
    ):
        self.config = config
        self.ref_point_method = ref_point_method
        
        # Set random seed
        self.rng = np.random.default_rng(config.seed)
        
        # Create agents with group assignments (minimal group paradigm)
        self.agents: List[Agent] = []
        self._create_agents()
        
        # Results storage
        self.round_results: List[Dict] = []
        self.coordination_rates: List[float] = []
    
    def _create_agents(self):
        """Create agents with arbitrary group assignment"""
        cfg = self.config
        
        # Shuffle agent IDs and assign to groups
        agent_ids = list(range(cfg.n_agents))
        self.rng.shuffle(agent_ids)
        
        agents_per_group = cfg.n_agents // cfg.n_groups
        
        for i, agent_id in enumerate(agent_ids):
            group_id = i // agents_per_group
            if group_id >= cfg.n_groups:
                group_id = cfg.n_groups - 1  # Handle remainder
            
            agent = Agent(
                agent_id=agent_id,
                group_id=group_id,
                config=cfg,
                ref_point_method=self.ref_point_method
            )
            self.agents.append(agent)
        
        # Sort by ID for consistent indexing
        self.agents.sort(key=lambda a: a.id)
        
        # Store group assignments
        self.group_assignments = {a.id: a.group for a in self.agents}
    
    def get_payoff(self, action1: Action, action2: Action) -> Tuple[float, float]:
        """Get payoffs for both players based on game type"""
        cfg = self.config
        
        # Player 1's payoff
        if action1 == Action.A and action2 == Action.A:
            p1 = cfg.payoff_AA
        elif action1 == Action.A and action2 == Action.B:
            p1 = cfg.payoff_AB
        elif action1 == Action.B and action2 == Action.A:
            p1 = cfg.payoff_BA
        else:  # B, B
            p1 = cfg.payoff_BB
        
        # Player 2's payoff (symmetric game)
        if action2 == Action.A and action1 == Action.A:
            p2 = cfg.payoff_AA
        elif action2 == Action.A and action1 == Action.B:
            p2 = cfg.payoff_AB
        elif action2 == Action.B and action1 == Action.A:
            p2 = cfg.payoff_BA
        else:  # B, B
            p2 = cfg.payoff_BB
        
        return p1, p2
    
    def run_random_matching_round(self, round_num: int) -> Dict:
        """
        Run one round with random pair matching
        Each agent is matched with one random partner
        """
        cfg = self.config
        n = cfg.n_agents
        
        # Shuffle and pair up agents
        indices = list(range(n))
        self.rng.shuffle(indices)
        
        interactions = []
        total_coordination = 0
        total_pairs = 0
        
        # Pair consecutive agents in shuffled list
        for k in range(0, n - 1, 2):
            i = indices[k]
            j = indices[k + 1]
            
            agent_i = self.agents[i]
            agent_j = self.agents[j]
            
            # Both agents choose simultaneously
            action_i = agent_i.choose_action(
                partner_id=j,
                partner_group=agent_j.group,
                current_round=round_num,
                rng=self.rng
            )
            action_j = agent_j.choose_action(
                partner_id=i,
                partner_group=agent_i.group,
                current_round=round_num,
                rng=self.rng
            )
            
            # Get payoffs
            payoff_i, payoff_j = self.get_payoff(action_i, action_j)
            
            # Record interactions
            agent_i.record_interaction(
                round_num, j, agent_j.group,
                action_i, action_j, payoff_i
            )
            agent_j.record_interaction(
                round_num, i, agent_i.group,
                action_j, action_i, payoff_j
            )
            
            # Track coordination
            coordinated = (action_i == action_j)
            total_coordination += int(coordinated)
            total_pairs += 1
            
            # Track in-group vs out-group
            same_group = (agent_i.group == agent_j.group)
            
            interactions.append({
                "round": round_num,
                "agent_i": i,
                "agent_j": j,
                "group_i": agent_i.group,
                "group_j": agent_j.group,
                "same_group": same_group,
                "action_i": action_i.name,
                "action_j": action_j.name,
                "coordinated": coordinated,
                "payoff_i": payoff_i,
                "payoff_j": payoff_j
            })
        
        coordination_rate = total_coordination / total_pairs if total_pairs > 0 else 0
        
        return {
            "round": round_num,
            "coordination_rate": coordination_rate,
            "interactions": interactions
        }
    
    def run_round_robin_round(self, round_num: int) -> Dict:
        """
        Run one round of round-robin matching
        Each agent plays with every other agent once
        """
        cfg = self.config
        n = cfg.n_agents
        
        interactions = []
        total_coordination = 0
        total_pairs = 0
        
        # Round-robin: each pair plays once
        for i in range(n):
            for j in range(i + 1, n):
                agent_i = self.agents[i]
                agent_j = self.agents[j]
                
                # Both agents choose simultaneously
                action_i = agent_i.choose_action(
                    partner_id=j,
                    partner_group=agent_j.group,
                    current_round=round_num,
                    rng=self.rng
                )
                action_j = agent_j.choose_action(
                    partner_id=i,
                    partner_group=agent_i.group,
                    current_round=round_num,
                    rng=self.rng
                )
                
                # Get payoffs
                payoff_i, payoff_j = self.get_payoff(action_i, action_j)
                
                # Record interactions
                agent_i.record_interaction(
                    round_num, j, agent_j.group,
                    action_i, action_j, payoff_i
                )
                agent_j.record_interaction(
                    round_num, i, agent_i.group,
                    action_j, action_i, payoff_j
                )
                
                # Track coordination
                coordinated = (action_i == action_j)
                total_coordination += int(coordinated)
                total_pairs += 1
                
                # Track in-group vs out-group
                same_group = (agent_i.group == agent_j.group)
                
                interactions.append({
                    "round": round_num,
                    "agent_i": i,
                    "agent_j": j,
                    "group_i": agent_i.group,
                    "group_j": agent_j.group,
                    "same_group": same_group,
                    "action_i": action_i.name,
                    "action_j": action_j.name,
                    "coordinated": coordinated,
                    "payoff_i": payoff_i,
                    "payoff_j": payoff_j
                })
        
        coordination_rate = total_coordination / total_pairs if total_pairs > 0 else 0
        
        return {
            "round": round_num,
            "coordination_rate": coordination_rate,
            "interactions": interactions
        }
    
    def run_simulation(self, verbose: bool = True) -> Dict:
        """Run full simulation"""
        cfg = self.config
        
        if verbose:
            print(f"Running simulation: N={cfg.n_agents}, T={cfg.n_rounds}, λ={cfg.lambda_loss}")
            print(f"Reference point method: {self.ref_point_method}")
            print(f"Matching type: {cfg.matching_type}")
            print(f"Groups: {cfg.n_groups}")
        
        for t in range(cfg.n_rounds):
            if cfg.matching_type == "round_robin":
                round_result = self.run_round_robin_round(t)
            else:  # random matching
                round_result = self.run_random_matching_round(t)
            
            self.round_results.append(round_result)
            self.coordination_rates.append(round_result["coordination_rate"])
            
            if verbose and (t + 1) % 50 == 0:
                recent_rate = np.mean(self.coordination_rates[-50:])
                print(f"  Round {t+1}: Recent coordination rate = {recent_rate:.3f}")
        
        return self.get_summary()
    
    def get_summary(self) -> Dict:
        """Generate summary statistics"""
        cfg = self.config
        
        # Overall coordination
        overall_rate = np.mean(self.coordination_rates)
        final_rate = np.mean(self.coordination_rates[-20:])
        
        # In-group vs out-group coordination BY ROUND
        ingroup_by_round = []
        outgroup_by_round = []
        
        for rr in self.round_results:
            ingroup_coords = []
            outgroup_coords = []
            
            for interaction in rr["interactions"]:
                if interaction["same_group"]:
                    ingroup_coords.append(int(interaction["coordinated"]))
                else:
                    outgroup_coords.append(int(interaction["coordinated"]))
            
            ingroup_by_round.append(np.mean(ingroup_coords) if ingroup_coords else 0)
            outgroup_by_round.append(np.mean(outgroup_coords) if outgroup_coords else 0)
        
        # Overall in-group vs out-group
        ingroup_rate = np.mean(ingroup_by_round)
        outgroup_rate = np.mean(outgroup_by_round)
        
        # Final rates (last 20 rounds)
        final_ingroup = np.mean(ingroup_by_round[-20:])
        final_outgroup = np.mean(outgroup_by_round[-20:])
        
        # Action distribution
        all_actions_A = 0
        all_actions_B = 0
        for rr in self.round_results:
            for interaction in rr["interactions"]:
                all_actions_A += int(interaction["action_i"] == "A")
                all_actions_A += int(interaction["action_j"] == "A")
                all_actions_B += int(interaction["action_i"] == "B")
                all_actions_B += int(interaction["action_j"] == "B")
        
        total_actions = all_actions_A + all_actions_B
        prop_A = all_actions_A / total_actions if total_actions > 0 else 0.5
        
        # Group-specific action preferences (to see if groups maintain different conventions)
        group_action_prefs = {g: {"A": 0, "B": 0} for g in range(cfg.n_groups)}
        for rr in self.round_results[-50:]:  # Last 50 rounds
            for interaction in rr["interactions"]:
                group_action_prefs[interaction["group_i"]][interaction["action_i"]] += 1
                group_action_prefs[interaction["group_j"]][interaction["action_j"]] += 1
        
        # Convert to proportions
        group_prop_A = {}
        for g, counts in group_action_prefs.items():
            total = counts["A"] + counts["B"]
            group_prop_A[g] = counts["A"] / total if total > 0 else 0.5
        
        return {
            "config": {
                "n_agents": cfg.n_agents,
                "n_rounds": cfg.n_rounds,
                "n_groups": cfg.n_groups,
                "lambda_loss": cfg.lambda_loss,
                "initial_group_bias": cfg.initial_group_bias,
                "ref_point_method": self.ref_point_method
            },
            "overall_coordination_rate": overall_rate,
            "final_coordination_rate": final_rate,
            "ingroup_coordination_rate": ingroup_rate,
            "outgroup_coordination_rate": outgroup_rate,
            "final_ingroup_rate": final_ingroup,
            "final_outgroup_rate": final_outgroup,
            "group_favoritism": ingroup_rate - outgroup_rate,
            "final_group_favoritism": final_ingroup - final_outgroup,
            "proportion_action_A": prop_A,
            "group_action_preferences": group_prop_A,
            "coordination_rates_by_round": self.coordination_rates,
            "ingroup_rates_by_round": ingroup_by_round,
            "outgroup_rates_by_round": outgroup_by_round,
            "group_assignments": self.group_assignments
        }


# =============================================================================
# Main Entry Point
# =============================================================================

def run_experiment(
    config: SimConfig = None,
    ref_point_method: str = "bayesian",
    save_results: bool = True
) -> Dict:
    """
    Run a single experiment
    """
    if config is None:
        config = SimConfig()
    
    game = CoordinationGame(config, ref_point_method)
    summary = game.run_simulation(verbose=True)
    
    if save_results:
        filename = f"results_{ref_point_method}_N{config.n_agents}_T{config.n_rounds}.json"
        with open(filename, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            summary_json = summary.copy()
            summary_json["coordination_rates_by_round"] = [
                float(x) for x in summary["coordination_rates_by_round"]
            ]
            json.dump(summary_json, f, indent=2)
        print(f"\nResults saved to {filename}")
    
    return summary


def compare_methods(config: SimConfig = None) -> Dict:
    """
    Compare different reference point formation methods
    """
    if config is None:
        config = SimConfig()
    
    methods = ["global", "group", "pairwise", "recent", "bayesian"]
    results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print('='*60)
        
        # Use same seed for comparison
        cfg = SimConfig(
            n_agents=config.n_agents,
            n_rounds=config.n_rounds,
            n_groups=config.n_groups,
            game_type=config.game_type,
            lambda_loss=config.lambda_loss,
            initial_group_bias=config.initial_group_bias,
            n_initial_rounds=config.n_initial_rounds,
            info_treatment=config.info_treatment,
            matching_type=config.matching_type,
            seed=config.seed
        )
        
        game = CoordinationGame(cfg, method)
        summary = game.run_simulation(verbose=False)
        results[method] = summary
        
        print(f"  Final coordination rate: {summary['final_coordination_rate']:.3f}")
        print(f"  In-group (final): {summary['final_ingroup_rate']:.3f}")
        print(f"  Out-group (final): {summary['final_outgroup_rate']:.3f}")
        print(f"  Group favoritism (final): {summary['final_group_favoritism']:.3f}")
        print(f"  Group 0 prop(A): {summary['group_action_preferences'].get(0, 0):.3f}")
        print(f"  Group 1 prop(A): {summary['group_action_preferences'].get(1, 0):.3f}")
    
    return results


def compare_games(base_config: SimConfig = None, method: str = "bayesian") -> Dict:
    """
    Compare different game types
    """
    if base_config is None:
        base_config = SimConfig()
    
    game_types = ["coordination", "stag_hunt", "chicken"]
    results = {}
    
    for game_type in game_types:
        print(f"\n{'='*60}")
        print(f"Game: {game_type}")
        print('='*60)
        
        cfg = SimConfig(
            n_agents=base_config.n_agents,
            n_rounds=base_config.n_rounds,
            n_groups=base_config.n_groups,
            game_type=game_type,
            lambda_loss=base_config.lambda_loss,
            initial_group_bias=base_config.initial_group_bias,
            n_initial_rounds=base_config.n_initial_rounds,
            info_treatment=base_config.info_treatment,
            matching_type=base_config.matching_type,
            seed=base_config.seed
        )
        
        game = CoordinationGame(cfg, method)
        summary = game.run_simulation(verbose=False)
        results[game_type] = summary
        
        print(f"  Payoffs: AA={cfg.payoff_AA}, AB={cfg.payoff_AB}, BA={cfg.payoff_BA}, BB={cfg.payoff_BB}")
        print(f"  Final coordination rate: {summary['final_coordination_rate']:.3f}")
        print(f"  Group favoritism (final): {summary['final_group_favoritism']:.3f}")
        print(f"  Group 0 prop(A): {summary['group_action_preferences'].get(0, 0):.3f}")
        print(f"  Group 1 prop(A): {summary['group_action_preferences'].get(1, 0):.3f}")
    
    return results


def compare_info_treatments(base_config: SimConfig = None, method: str = "bayesian") -> Dict:
    """
    Compare different information treatments
    """
    if base_config is None:
        base_config = SimConfig()
    
    treatments = ["full", "group_only", "anonymous"]
    results = {}
    
    for treatment in treatments:
        print(f"\n{'='*60}")
        print(f"Info treatment: {treatment}")
        print('='*60)
        
        cfg = SimConfig(
            n_agents=base_config.n_agents,
            n_rounds=base_config.n_rounds,
            n_groups=base_config.n_groups,
            game_type=base_config.game_type,
            lambda_loss=base_config.lambda_loss,
            initial_group_bias=base_config.initial_group_bias,
            n_initial_rounds=base_config.n_initial_rounds,
            info_treatment=treatment,
            matching_type=base_config.matching_type,
            seed=base_config.seed
        )
        
        game = CoordinationGame(cfg, method)
        summary = game.run_simulation(verbose=False)
        results[treatment] = summary
        
        print(f"  Final coordination rate: {summary['final_coordination_rate']:.3f}")
        print(f"  Group favoritism (final): {summary['final_group_favoritism']:.3f}")
    
    return results


if __name__ == "__main__":
    print("="*70)
    print("Reference Point Formation in Coordination Games - Full Analysis")
    print("="*70)
    
    # Base configuration
    base_config = SimConfig(
        n_agents=16,
        n_rounds=200,
        n_groups=2,
        game_type="coordination",
        lambda_loss=2.0,
        initial_group_bias=0.8,
        n_initial_rounds=10,
        info_treatment="full",
        matching_type="random",
        seed=42
    )
    
    print("\nBase Configuration:")
    print(f"  N = {base_config.n_agents}, T = {base_config.n_rounds}")
    print(f"  Groups = {base_config.n_groups}")
    print(f"  Game type = {base_config.game_type}")
    print(f"  λ (loss aversion) = {base_config.lambda_loss}")
    print(f"  Initial group bias = {base_config.initial_group_bias}")
    print(f"  Info treatment = {base_config.info_treatment}")
    
    # 1. Compare reference point methods
    print("\n\n" + "="*70)
    print("ANALYSIS 1: Reference Point Methods")
    print("="*70)
    results_methods = compare_methods(base_config)
    
    # 2. Compare game types
    print("\n\n" + "="*70)
    print("ANALYSIS 2: Game Types")
    print("="*70)
    results_games = compare_games(base_config)
    
    # 3. Compare information treatments
    print("\n\n" + "="*70)
    print("ANALYSIS 3: Information Treatments")
    print("="*70)
    results_info = compare_info_treatments(base_config)
    
    # Save all results
    all_results = {
        "methods": {m: {
            "final_coord": results_methods[m]["final_coordination_rate"],
            "favoritism": results_methods[m]["final_group_favoritism"]
        } for m in results_methods},
        "games": {g: {
            "final_coord": results_games[g]["final_coordination_rate"],
            "favoritism": results_games[g]["final_group_favoritism"],
            "group0_A": results_games[g]["group_action_preferences"].get(0, 0),
            "group1_A": results_games[g]["group_action_preferences"].get(1, 0)
        } for g in results_games},
        "info_treatments": {t: {
            "final_coord": results_info[t]["final_coordination_rate"],
            "favoritism": results_info[t]["final_group_favoritism"]
        } for t in results_info}
    }
    
    with open("full_analysis.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n\n" + "="*70)
    print("Analysis complete! Results saved to full_analysis.json")
    print("="*70)