# Variation 2

This directory contains the dedicated ground-truth runner for:

- Scene: `task1_variation2.ttt`
- Script: `ground_truth_orchestrator_variation2.py`

Ground-truth flow:

1. Cupboard mug -> placement boundary
2. Table grocery -> cupboard
3. Box-top mug -> placement boundary
4. Open box lid
5. Box grocery -> cupboard
6. Both table mugs -> box (with non-overlapping slot finalization + fallback)

## Run

From project root:

```bash
python variation_2/ground_truth_orchestrator_variation2.py
```

## Run With Live Segmentation

```bash
python variation_2/run_live_segmentation_variation2.py
```

Optional backend selection:

```bash
LIVE_SEG_VIEWER_BACKEND=tkinter python variation_2/run_live_segmentation_variation2.py
```
