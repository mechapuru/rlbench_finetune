# Variation 3 Hard

This directory contains the dedicated ground-truth runner for:

- Scene: `task1_variation3.ttt`
- Script: `ground_truth_orchestrator_variation3_hard.py`

Ground-truth flow:

1. Cupboard mug -> placement boundary
2. Table grocery -> cupboard
3. Box-top mug -> placement boundary
4. Open box lid
5. Box grocery -> cupboard
6. All table mugs (3) -> box (with non-overlapping slot finalization + fallback)

## Run

From project root:

```bash
python variation_3_hard/ground_truth_orchestrator_variation3_hard.py
```

## Run With Live Segmentation

```bash
python variation_3_hard/run_live_segmentation_variation3_hard.py
```

Optional backend selection:

```bash
LIVE_SEG_VIEWER_BACKEND=tkinter python variation_3_hard/run_live_segmentation_variation3_hard.py
```
