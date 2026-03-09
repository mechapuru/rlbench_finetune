from pathlib import Path
from typing import Tuple
from planning_rules import PlanningRuleSet, ForbidWhenRule

def apply_rules_to_domain(
    domain_path: str,
    rules: PlanningRuleSet,
    output_path: str = None,
) -> str:
    """
    Read the base PDDL domain, modify it according to rules, and write
    out a new PDDL domain file. Returns the new path.
    """
    domain_text = Path(domain_path).read_text()

    # Ensure 'inside' predicate is declared
    domain_text = _ensure_inside_predicate(domain_text)
    
    # Ensure 'box-top' constant is declared
    domain_text = _ensure_constants(domain_text, ["box-top"])

    # Ensure 'pick' clears 'in-region'
    domain_text = _ensure_pick_clears_region(domain_text)

    # Apply forbid rules
    domain_text = _apply_forbid_when_rules(domain_text, rules.forbid_when_rules)

    if output_path is None:
        output_path = domain_path.replace(".pddl", "_constrained.pddl")

    Path(output_path).write_text(domain_text)
    return output_path


def _apply_forbid_when_rules(
    domain_text: str,
    forbid_rules: list[ForbidWhenRule],
) -> str:
    """
    Simple string-based patcher.
    """
    for rule in forbid_rules:
        if rule.action_name == "pick":
            domain_text = _inject_pick_inside_lid_open(domain_text, rule)
        elif rule.action_name == "open-lid":
            domain_text = _inject_open_lid_clear_top(domain_text, rule)

    return domain_text


def _inject_open_lid_clear_top(domain_text: str, rule: ForbidWhenRule) -> str:
    """
    Injects precondition: (forall (?x) (not (in-region ?x box-top)))
    This ensures nothing is on top of the box before opening the lid.
    """
    start = domain_text.find("(:action open-lid")
    if start == -1:
        return domain_text

    precond_marker = ":precondition"
    idx_precond = domain_text.find(precond_marker, start)
    if idx_precond == -1:
        return domain_text
    
    idx_and_start = domain_text.find("(and", idx_precond)
    if idx_and_start == -1:
        return domain_text

    insertion_point = idx_and_start + 4 
    
    # We use the constant 'box-top' which we ensure is defined in :constants
    new_condition = "\n      (forall (?x) (not (in-region ?x box-top)))"
    
    domain_text_modified = domain_text[:insertion_point] + new_condition + domain_text[insertion_point:]
    
    return domain_text_modified


def _inject_pick_inside_lid_open(domain_text: str, rule: ForbidWhenRule) -> str:
    """
    Specific helper:
    Ensure 'pick' has a precondition that enforces (opened ?box) if (inside ?o ?box).
    
    We will use ADL's quantified precondition:
    (forall (?box) (imply (inside ?o ?box) (opened ?box)))
    
    This avoids needing to add ?box to the action parameters.
    """

    # 1) Find the 'pick' action block.
    start = domain_text.find("(:action pick")
    if start == -1:
        return domain_text

    # Find the end of the action block (heuristic: next (:action or end of file)
    # Better heuristic: count parentheses, but let's assume standard formatting for now
    # or just find the :precondition block.
    
    precond_marker = ":precondition"
    idx_precond = domain_text.find(precond_marker, start)
    if idx_precond == -1:
        return domain_text
    
    # We assume the precondition starts with (and ...
    # We want to insert our new condition inside the (and ... )
    
    # Find the opening '(' of the (and
    idx_and_start = domain_text.find("(and", idx_precond)
    if idx_and_start == -1:
        # Maybe it's not an (and ...), just a single predicate?
        # If so, we wrap it in (and ...). But usually it is (and.
        return domain_text

    # We will insert right after "(and"
    insertion_point = idx_and_start + 4 
    
    # The condition we want to add:
    # (forall (?box) (imply (inside ?o ?box) (opened ?box)))
    # Note: ?o must match the parameter name in the domain.
    # In rlbench_kitchen_domain.pddl, pick params are (?o ?p ?g ?q1 ?q2 ?t)
    # So ?o is correct.
    
    new_condition = "\n      (forall (?box) (imply (inside ?o ?box) (opened ?box)))"
    
    domain_text_modified = domain_text[:insertion_point] + new_condition + domain_text[insertion_point:]
    
    return domain_text_modified

def _ensure_inside_predicate(domain_text: str) -> str:
    """
    Checks if (inside ?o ?box) is defined in predicates. If not, adds it.
    """
    if "(inside ?o ?box)" in domain_text or "(inside ?x ?y)" in domain_text:
        return domain_text
        
    # Find :predicates section
    pred_marker = "(:predicates"
    idx_pred = domain_text.find(pred_marker)
    if idx_pred == -1:
        return domain_text
        
    # Insert after (:predicates
    insertion_point = idx_pred + len(pred_marker)
    new_pred = "\n      (inside ?o ?box)"
    
    return domain_text[:insertion_point] + new_pred + domain_text[insertion_point:]

def _ensure_constants(domain_text: str, constants: list[str]) -> str:
    """
    Ensures the given constants are declared in the domain.
    """
    if not constants:
        return domain_text
        
    # Check if :constants block exists
    const_marker = "(:constants"
    idx_const = domain_text.find(const_marker)
    
    # In untyped PDDL (which this domain seems to be), we don't use "- type".
    # We just list the constants. The type is defined by a unary predicate in init.
    constants_str = "\n      " + " ".join(constants)
    
    if idx_const != -1:
        # Insert into existing block
        insertion_point = idx_const + len(const_marker)
        return domain_text[:insertion_point] + constants_str + domain_text[insertion_point:]
    else:
        # Create new block after :requirements or :types
        # Let's put it after :predicates for safety, or before it.
        # Standard PDDL: requirements, types, constants, predicates...
        
        # Find :types
        types_marker = "(:types"
        idx_types = domain_text.find(types_marker)
        if idx_types != -1:
            # Find end of types block ')'
            # This is tricky without parsing.
            # Let's assume it ends before :predicates
            pred_marker = "(:predicates"
            idx_pred = domain_text.find(pred_marker)
            if idx_pred != -1:
                insertion_point = idx_pred
                new_block = f"(:constants {constants_str})\n\n  "
                return domain_text[:insertion_point] + new_block + domain_text[insertion_point:]
        
        # Fallback: Insert before :predicates
        pred_marker = "(:predicates"
        idx_pred = domain_text.find(pred_marker)
        if idx_pred != -1:
            new_block = f"(:constants {constants_str})\n\n  "
            return domain_text[:idx_pred] + new_block + domain_text[idx_pred:]
            
    return domain_text


def _ensure_pick_clears_region(domain_text: str) -> str:
    """
    Injects effect: (forall (?r) (when (in-region ?o ?r) (not (in-region ?o ?r))))
    into the 'pick' action.
    """
    start = domain_text.find("(:action pick")
    if start == -1:
        return domain_text

    effect_marker = ":effect"
    idx_effect = domain_text.find(effect_marker, start)
    if idx_effect == -1:
        return domain_text
    
    idx_and_start = domain_text.find("(and", idx_effect)
    if idx_and_start == -1:
        return domain_text

    insertion_point = idx_and_start + 4 
    
    new_effect = "\n      (forall (?r) (when (in-region ?o ?r) (not (in-region ?o ?r))))"
    
    domain_text_modified = domain_text[:insertion_point] + new_effect + domain_text[insertion_point:]
    
    return domain_text_modified

