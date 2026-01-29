from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class ForbidWhenRule:
    """
    Forbid an action when some predicate condition holds.
    Example: forbid pick-inside-box when (not (opened ?box)).
    """
    action_name: str
    param_pattern: Tuple[str, ...]   # e.g. ("?obj",)
    condition_pred: Tuple[str, Tuple[str, ...]]  
    # e.g. ("inside-and-closed", ("?obj", "?box")) 
    # This implies we might need to inject ?box into the action parameters if it's not there.

@dataclass
class OrderingRule:
    """
    Enforce partial ordering between abstract actions.
    Example: clear-lid must happen before open-lid.
    """
    before_action: str
    after_action: str

@dataclass
class PlanningRuleSet:
    """
    Container for all rules we want to apply to the domain.
    """
    forbid_when_rules: List[ForbidWhenRule] = field(default_factory=list)
    ordering_rules: List[OrderingRule] = field(default_factory=list)

    def forbid_action_when(
        self,
        action_name: str,
        param_pattern: Tuple[str, ...],
        condition_pred: Tuple[str, Tuple[str, ...]],
    ) -> None:
        self.forbid_when_rules.append(
            ForbidWhenRule(
                action_name=action_name,
                param_pattern=param_pattern,
                condition_pred=condition_pred,
            )
        )

    def enforce_order(self, before_action: str, after_action: str) -> None:
        self.ordering_rules.append(
            OrderingRule(before_action=before_action, after_action=after_action)
        )
