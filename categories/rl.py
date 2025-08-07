# Reinforcement Learning Category

from typing import Any, Callable, Dict, Tuple
from .base import Category

class RLObject:
    def __init__(self, states: Any, actions: Any, transition_fn: Callable, reward_fn: Callable):
        self.states = states  # e.g., list of states
        self.actions = actions  # e.g., list of actions
        self.P = transition_fn  # function: (s, a) -> s'
        self.R = reward_fn      # function: (s, a) -> reward

    def __repr__(self):
        return f"RLObject(states={len(self.states)}, actions={len(self.actions)})"


class ReinforcementLearningCategory(Category):
    def __init__(self):
        super().__init__()

    def add_rl_object(self, rl_obj: RLObject):
        self.add_object(rl_obj)

    def add_policy_morphism(self, source: RLObject, target: RLObject, policy_transform: Callable):
        """A morphism represents a policy update or transformation (π_t → π_{t+1})"""
        if not isinstance(source, RLObject) or not isinstance(target, RLObject):
            raise ValueError("Source and target must be RLObject instances.")
        self.add_morphism(source, target, policy_transform)

    def apply_policy(self, rl_obj: RLObject, policy: Callable) -> Dict:
        """Apply a policy π(s) → a to each state"""
        return {s: policy(s) for s in rl_obj.states}
