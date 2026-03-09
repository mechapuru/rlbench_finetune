from planning_rules import PlanningRuleSet
from kitchen_rules_manual import add_ground_truth_rules
from domain_patcher import apply_rules_to_domain
from pddlstream.algorithms.meta import solve as pddlstream_solve
from pddlstream.language.constants import PDDLProblem
# from rlbench_kitchen_streams import get_stream_map # REMOVED to avoid double initialization
from pddlstream.utils import read
import os

def solve_with_rules(domain_pddl_path, stream_pddl_path, init_atoms, goal_atoms, stream_map=None):
    """
    Wrap PDDLStream solving with a rules->domain-patch step.
    """

    # 1) Build rule set and fill with 'ground truth' rules.
    rules = PlanningRuleSet()
    add_ground_truth_rules(rules)

    # 2) Apply rules to domain, get constrained domain path.
    constrained_domain_path = apply_rules_to_domain(domain_pddl_path, rules)
    
    print(f"  [Info] Using constrained domain: {constrained_domain_path}")

    # 3) Build PDDLStream problem using constrained domain.
    # We need to read the text of the constrained domain
    domain_pddl = read(constrained_domain_path)
    stream_pddl = read(stream_pddl_path)

    if stream_map is None:
        from rlbench_kitchen_streams import get_stream_map
        stream_map = get_stream_map()

    problem = PDDLProblem(
        domain_pddl=domain_pddl,
        constant_map={'box-top': 'box-top'},
        stream_pddl=stream_pddl,
        stream_map=stream_map,
        init=init_atoms,
        goal=goal_atoms,
    )

    # 4) Call the usual PDDLStream solver.
    # We use the same parameters as the baseline
    solution = pddlstream_solve(problem, algorithm='adaptive', verbose=True, max_time=120)
    return solution
