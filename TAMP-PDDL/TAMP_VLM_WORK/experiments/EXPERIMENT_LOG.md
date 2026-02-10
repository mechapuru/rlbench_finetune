# Experiment Log: Task Planning for Kitchen Domain

Date: January 30, 2026

## Task

Move three objects to target locations:
- mug_box to placement_boundary
- mug_inside_box to placement_boundary
- soup to cupboard_boundary

Key constraint: mug_box sits on top of box_lid, blocking it. mug_inside_box is inside the closed box.

Ground truth solution:
1. pick(mug_box)
2. place(mug_box, placement_boundary)
3. open-lid(box_lid)
4. pick(mug_inside_box)
5. place(mug_inside_box, placement_boundary)
6. pick(soup)
7. place(soup, cupboard_boundary)

---

## Experiment 1: Pure PDDLStream

Approach: Classical PDDL planning with stream functions for motion planning.

Results:
- Success rate: 90% (9/10 trials)
- Planning time: 2-5 seconds
- Execution time: 15-20 seconds

Notes: Reliable when domain is well-specified. Requires manual PDDL engineering. No learning from failures.

---

## Experiment 2: COAST

Approach: PDDLStream with learned constraints from execution failures.

Results:
- Success rate: 0% (0/10 trials)
- Learned constraint: (fail-pick mug_inside_box)

Notes: Successfully detected failure when picking mug_inside_box while lid was closed. However, learned constraint was too coarse - banned ALL picks of mug_inside_box instead of conditional constraint. Demonstrates COAST limitation: cannot learn state-conditional constraints.

---

## Experiment 3: LLM-Based Planner

Hardware: NVIDIA RTX 5060 (7.5GB VRAM)

### Prompt Iteration 1: Detailed Scene Description
- Model: Qwen2.5-3B-Instruct
- Result: FAILED - started with open-lid(box_lid)
- Issue: Did not understand mug_box blocks the lid

### Prompt Iteration 2: Explicit Blocking Info
- Model: Qwen2.5-3B-Instruct
- Added "mug_box is ON TOP of box_lid"
- Result: FAILED - still started with open-lid
- Issue: 3B model not reasoning through constraint chain

### Prompt Iteration 3: Chain-of-Thought
- Model: Qwen2.5-3B-Instruct
- Added "Think step by step" prompting
- Result: FAILED - generated garbage/hallucinations
- Issue: Raw tokenization without chat template

### Prompt Iteration 4: Few-Shot + Chat Template
- Model: Qwen2.5-7B-Instruct (CPU offload)
- Used apply_chat_template()
- Added analogous example problem
- Result: SUCCESS
- Inference time: 68 seconds

Output:
```
1. pick(mug_box)
2. place(mug_box, placement_boundary)
3. open(box_lid)
4. pick(mug_inside_box)
5. place(mug_inside_box, placement_boundary)
6. pick(soup)
7. place(soup, cupboard_boundary)
```

---

## Summary

| Experiment | Approach | Success Rate | Time |
|------------|----------|--------------|------|
| 1 | Pure PDDLStream | 90% | ~5s |
| 2 | COAST | 0% | ~5s |
| 3 | LLM (7B, few-shot) | TBD | ~68s |

LLM Prompt Iterations:

| # | Model | Technique | Chat Template | Result |
|---|-------|-----------|---------------|--------|
| 1 | 3B | Detailed description | No | Failed |
| 2 | 3B | Explicit blocking | No | Failed |
| 3 | 3B | Chain-of-thought | No | Failed |
| 4 | 7B | Few-shot example | Yes | Success |

Key findings:
- Chat template is critical for instruction-tuned models
- Few-shot example improves reasoning significantly
- 7B model required for multi-step constraint reasoning
- COAST cannot learn state-conditional constraints
