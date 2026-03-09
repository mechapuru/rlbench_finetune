from planning_rules import PlanningRuleSet

def add_ground_truth_rules(rules: PlanningRuleSet) -> None:
    """
    Add commonsense rules for the kitchen domain.
    These are your manual 'ground truth' constraints.
    """

    # 1) Symbolic version of:
    # "Don't pick an object inside a box if the lid is not opened."
    #
    # We assume the domain patcher will handle the injection of ?box into parameters
    # and the construction of the precondition.
    #
    # The semantic we want is:
    #   For pick(?o), if inside(?o, ?box) is true, then opened(?box) must be true.
    #   In PDDL (ADL), this is: (forall (?box) (imply (inside ?o ?box) (opened ?box)))
    #   
    #   However, for our simple patcher, we might just inject:
    #   (opened ?box)
    #   and ensure ?box is added to parameters and bound correctly (or rely on ADL).
    #
    #   Let's define the rule as:
    #   Forbid 'pick' when 'inside-and-closed' is true.
    #   The patcher will interpret 'inside-and-closed' as needing to check (inside ?o ?box) and (not (opened ?box)).
    #   Or simpler: Require (opened ?box) if (inside ?o ?box).
    
    rules.forbid_action_when(
        action_name="pick",
        param_pattern=("?o",), 
        condition_pred=("inside-and-closed", ("?o", "?box")),
    )

    # 2) Rule: Don't open lid if something is on top (in box-top region)
    # This prevents sliding the lid while the mug is still on it.
    rules.forbid_action_when(
        action_name="open-lid",
        param_pattern=("?o",),
        condition_pred=("something-on-top", ("?o",))
    )

    # 3) Optional: an ordering rule at the abstract level
    # "Clear lid"/"open-lid" must occur before high-level goal
    # that involves retrieving from box interior.
    # rules.enforce_order(before_action="open-lid", after_action="pick-from-box")
