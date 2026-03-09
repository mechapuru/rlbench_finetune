"""
Ground Truth Orchestrator - Variation 2

Scene: task1_variation2.ttt
Goal (high-level): move groceries to cupboard and execute the specified
ground-truth sequence for mugs/groceries in this variation.

Sequence:
1. Pick mug in cupboard -> placement_boundary
2. Pick grocery on table -> cupboard_boundary
3. Pick mug on box -> placement_boundary
4. Slide open box lid
5. Pick grocery in box -> cupboard_boundary
6. Pick both table mugs (2) -> box_boundary
"""
import os
import sys
import numpy as np


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
SCENE_PATH = os.path.join(ROOT_DIR, "task1_variation2.ttt")

# Ensure the shared env boots this variation scene.
os.environ["KITCHEN_SCENE_FILE"] = SCENE_PATH
os.environ["HEADLESS"] = "False"

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import ground_truth_orchestrator as base
from video_recorder import VideoRecorder


def _obj_name(obj, fallback):
    try:
        return obj.get_name()
    except Exception:
        return fallback


def _world_bounds(region_obj):
    r_min_x, r_max_x, r_min_y, r_max_y, r_min_z, r_max_z = region_obj.get_bounding_box()
    rx, ry, rz = region_obj.get_position()
    return (
        rx + r_min_x, rx + r_max_x,
        ry + r_min_y, ry + r_max_y,
        rz + r_min_z, rz + r_max_z,
    )


def _is_in_region(env, obj, region_name, tol=0.02):
    region = env.regions.get(region_name)
    if region is None or obj is None:
        return False
    x, y, z = obj.get_position()
    min_x, max_x, min_y, max_y, min_z, max_z = _world_bounds(region)
    return (
        (min_x - tol) <= x <= (max_x + tol)
        and (min_y - tol) <= y <= (max_y + tol)
        and (min_z - tol) <= z <= (max_z + tol)
    )


def _region_center(env, region_name):
    region = env.regions.get(region_name)
    if region is None:
        return None
    min_x, max_x, min_y, max_y, min_z, max_z = _world_bounds(region)
    return np.array(
        [(min_x + max_x) * 0.5, (min_y + max_y) * 0.5, (min_z + max_z) * 0.5],
        dtype=float,
    )


def _table_surface_info(env):
    table = env.regions.get("table")
    if table is None:
        return None
    min_x, max_x, min_y, max_y, _min_z, max_z = _world_bounds(table)
    return (min_x, max_x, min_y, max_y, max_z)


def _is_near_table_surface(env, obj, xy_tol=0.08, z_tol=0.20):
    info = _table_surface_info(env)
    if info is None or obj is None:
        return False
    min_x, max_x, min_y, max_y, table_top = info
    x, y, z = obj.get_position()
    in_xy = (min_x - xy_tol) <= x <= (max_x + xy_tol) and (min_y - xy_tol) <= y <= (max_y + xy_tol)
    near_z = z <= (table_top + z_tol)
    return bool(in_xy and near_z)


def _discover_unique_objects(env, candidate_names):
    seen = set()
    names = []
    for name in candidate_names:
        obj = env.get_object(name)
        if obj is None:
            continue
        try:
            key = obj.get_handle()
        except Exception:
            key = _obj_name(obj, name)
        if key in seen:
            continue
        seen.add(key)
        names.append(_obj_name(obj, name))
    return names


def _obj_handle(obj):
    try:
        return int(obj.get_handle())
    except Exception:
        return None


def _handles_for_names(env, names):
    handles = set()
    for name in names or []:
        obj = env.get_object(name)
        if obj is None:
            continue
        h = _obj_handle(obj)
        if h is not None:
            handles.add(h)
    return handles


def _normalized_quat(q):
    q = np.array(q, dtype=float)
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        return [0.0, 0.0, 0.0, 1.0]
    return (q / n).tolist()


def _select_object_by_region(env, candidates, region_name):
    discovered = _discover_unique_objects(env, candidates)
    for name in discovered:
        obj = env.get_object(name)
        if _is_in_region(env, obj, region_name):
            return name
    return discovered[0] if discovered else None


def _select_runtime_table_grocery(env, exclude_names=None):
    """
    Pick the grocery that is currently on table (not already in cupboard),
    excluding any already-used grocery handles.
    """
    grocery_candidates = ["soup", "spam", "mustard", "sugar", "crackers"]
    discovered = _discover_unique_objects(env, grocery_candidates)
    excluded = _handles_for_names(env, exclude_names or [])

    table_first = []
    fallback = []
    for name in discovered:
        obj = env.get_object(name)
        if obj is None:
            continue
        h = _obj_handle(obj)
        if (h is not None) and (h in excluded):
            continue

        in_table = _is_in_region(env, obj, "table")
        in_cupboard = _is_in_region(env, obj, "cupboard_boundary") or _is_in_region(
            env, obj, "cupboard_boundary_top"
        )
        in_box = _is_in_region(env, obj, "box_boundary")

        # Best target for Task 5
        if in_table and (not in_cupboard):
            table_first.append(name)
        # Backup: not in cupboard and not the box one
        elif (not in_cupboard) and (not in_box):
            fallback.append(name)

    if table_first:
        return table_first[0]
    if fallback:
        return fallback[0]
    return None


def _select_runtime_box_grocery(env, exclude_names=None):
    """Pick grocery that is currently in box region (runtime)."""
    grocery_candidates = ["soup", "spam", "mustard", "sugar", "crackers"]
    discovered = _discover_unique_objects(env, grocery_candidates)
    excluded = _handles_for_names(env, exclude_names or [])

    in_box = []
    remaining = []
    for name in discovered:
        obj = env.get_object(name)
        if obj is None:
            continue
        h = _obj_handle(obj)
        if (h is not None) and (h in excluded):
            continue
        if _is_in_region(env, obj, "box_boundary"):
            in_box.append(name)
        else:
            in_cupboard = _is_in_region(env, obj, "cupboard_boundary") or _is_in_region(
                env, obj, "cupboard_boundary_top"
            )
            if not in_cupboard:
                remaining.append(name)

    if in_box:
        return in_box[0]

    # Fallback: nearest non-cupboard grocery to box center.
    box_center = _region_center(env, "box_boundary")
    if box_center is not None and remaining:
        return min(
            remaining,
            key=lambda n: float(
                np.linalg.norm(np.array(env.get_object(n).get_position(), dtype=float) - box_center)
            ),
        )
    return remaining[0] if remaining else None


def _select_runtime_table_mugs(env, preferred_names=None, max_count=2):
    """
    Select mugs currently on table at runtime (after prior tasks moved them).
    Preference order comes from preferred_names, then all known mug aliases.
    """
    preferred_names = preferred_names or []
    all_candidates = preferred_names + ["mug1", "mug2", "mug3", "mug4"]
    discovered = _discover_unique_objects(env, all_candidates)

    selected = []
    seen_handles = set()
    fallback = []
    for name in discovered:
        obj = env.get_object(name)
        if obj is None:
            continue
        h = _obj_handle(obj)
        if (h is not None) and (h in seen_handles):
            continue
        in_table = _is_in_region(env, obj, "table", tol=0.03)
        in_placement = _is_in_region(env, obj, "placement_boundary", tol=0.03)
        near_table = _is_near_table_surface(env, obj)
        in_cupboard = _is_in_region(env, obj, "cupboard_boundary") or _is_in_region(
            env, obj, "cupboard_boundary_top"
        )
        in_box = _is_in_region(env, obj, "box_boundary")

        if (in_table or in_placement or near_table) and (not in_cupboard) and (not in_box):
            selected.append(name)
            if h is not None:
                seen_handles.add(h)
        elif near_table and (not in_cupboard) and (not in_box):
            fallback.append(name)
        if len(selected) >= max_count:
            break

    # Fallback: any mugs not in cupboard/box.
    if len(selected) < max_count:
        for name in fallback:
            if name in selected:
                continue
            selected.append(name)
            if len(selected) >= max_count:
                break

    return selected


def _compute_box_slot_poses(env, mug_names):
    """
    Compute evenly spaced, non-overlapping mug target poses in box_boundary.
    """
    box_region = env.regions.get("box_boundary") or env.regions.get("box-inside")
    if box_region is None:
        return {}

    min_x, max_x, min_y, max_y, min_z, max_z = _world_bounds(box_region)
    count = len(mug_names)
    if count <= 0:
        return {}

    span_x = max_x - min_x
    span_y = max_y - min_y
    margin = 0.045

    # Keep bounds valid even for tighter boxes.
    if span_x < (2.0 * margin):
        margin_x = max(0.01, 0.2 * span_x)
    else:
        margin_x = margin
    if span_y < (2.0 * margin):
        margin_y = max(0.01, 0.2 * span_y)
    else:
        margin_y = margin

    if span_x >= span_y:
        xs = np.linspace(min_x + margin_x, max_x - margin_x, count)
        ys = np.full(count, 0.5 * (min_y + max_y), dtype=float)
    else:
        ys = np.linspace(min_y + margin_y, max_y - margin_y, count)
        xs = np.full(count, 0.5 * (min_x + max_x), dtype=float)

    quat_ref_obj = env.get_object("mug2")
    if quat_ref_obj is None and mug_names:
        quat_ref_obj = env.get_object(mug_names[0])
    quat = _normalized_quat(quat_ref_obj.get_quaternion()) if quat_ref_obj is not None else [0.0, 0.0, 0.0, 1.0]

    # Approximate support plane as box-boundary floor.
    support_z = float(min_z) + 0.001

    slot_map = {}
    for i, name in enumerate(mug_names):
        obj = env.get_object(name)
        if obj is None:
            continue
        try:
            obj_min_z = float(obj.get_bounding_box()[4])
        except Exception:
            obj_min_z = 0.0
        z = support_z - obj_min_z + 0.0005
        slot_map[name] = [
            float(xs[i]),
            float(ys[i]),
            float(z),
            float(quat[0]),
            float(quat[1]),
            float(quat[2]),
            float(quat[3]),
        ]
    return slot_map


def _clip(value, lo, hi):
    return float(max(lo, min(hi, value)))


def _box_slot_candidates(env, slot_pose):
    """
    Build a few nearby stable-pose candidates around the nominal slot.
    This reduces planner failures in tight boxes without teleporting.
    """
    if slot_pose is None or len(slot_pose) < 7:
        return []

    box_region = env.regions.get("box_boundary") or env.regions.get("box-inside")
    if box_region is None:
        return [list(slot_pose[:7])]

    min_x, max_x, min_y, max_y, _min_z, _max_z = _world_bounds(box_region)
    span_x = max_x - min_x
    span_y = max_y - min_y
    margin = 0.015

    lo_x, hi_x = min_x + margin, max_x - margin
    lo_y, hi_y = min_y + margin, max_y - margin
    if lo_x > hi_x:
        lo_x = hi_x = 0.5 * (min_x + max_x)
    if lo_y > hi_y:
        lo_y = hi_y = 0.5 * (min_y + max_y)

    x0 = _clip(float(slot_pose[0]), lo_x, hi_x)
    y0 = _clip(float(slot_pose[1]), lo_y, hi_y)
    z0 = float(slot_pose[2])
    quat = [float(slot_pose[3]), float(slot_pose[4]), float(slot_pose[5]), float(slot_pose[6])]

    primary_x = span_x >= span_y
    delta = max(0.012, 0.14 * min(max(span_x, 1e-3), max(span_y, 1e-3)))
    offsets = [0.0, +delta, -delta, +2.0 * delta, -2.0 * delta]

    cands = []
    for off in offsets:
        if primary_x:
            x = _clip(x0 + off, lo_x, hi_x)
            y = y0
        else:
            x = x0
            y = _clip(y0 + off, lo_y, hi_y)
        cands.append((round(x, 5), round(y, 5), round(z0, 5), *[round(q, 6) for q in quat]))

    unique = []
    seen = set()
    for c in cands:
        if c in seen:
            continue
        seen.add(c)
        unique.append([float(v) for v in c])
    return unique


def _install_box_slot_overrides(env, mug_name, slot_pose):
    """
    Install forced stable-pose candidates for this mug in box regions.
    Returns previous override map so callers can restore it.
    """
    prev = getattr(env, "_stable_pose_overrides", None)
    candidates = _box_slot_candidates(env, slot_pose)
    if not candidates:
        return prev

    override_map = dict(prev) if isinstance(prev, dict) else {}
    override_map[(mug_name, "box_boundary")] = candidates
    override_map[(mug_name, "box-inside")] = candidates
    override_map[mug_name] = candidates
    env._stable_pose_overrides = override_map
    xyz = [round(v, 4) for v in candidates[0][:3]]
    print(f"[SlotGuide] {mug_name}: using guided box slot candidates (first={xyz})")
    return prev


def _restore_overrides(env, prev):
    if prev is None:
        if hasattr(env, "_stable_pose_overrides"):
            delattr(env, "_stable_pose_overrides")
    else:
        env._stable_pose_overrides = prev


def _snap_mug_to_pose(env, pr, mug_name, pose7):
    """
    Finalize mug pose in box slot to prevent overlap/drift in final state.
    """
    mug = env.get_object(mug_name)
    if mug is None or pose7 is None:
        return
    mug.set_pose(list(pose7))
    mug.set_dynamic(False)
    for _ in range(12):
        base.step_and_record(pr, 1)


def _force_place_mug_in_box_slot(env, pr, mug_name, pose7):
    """
    Deterministic fallback when PDDL cannot find mug->box plan.
    Ensures final scene goal state with non-overlapping box slots.
    """
    mug = env.get_object(mug_name)
    if mug is None or pose7 is None:
        return False

    # Ensure gripper/object are detached before forcing final pose.
    try:
        env.gripper.release()
    except Exception:
        pass
    for _ in range(30):
        try:
            env.gripper.actuate(1.0, velocity=0.3)
        except Exception:
            pass
        base.step_and_record(pr, 1)

    try:
        mug.set_parent(None, keep_in_place=True)
    except Exception:
        try:
            mug.set_parent(None)
        except Exception:
            pass

    mug.set_pose(list(pose7))
    mug.set_dynamic(False)
    for _ in range(15):
        base.step_and_record(pr, 1)

    in_box = _is_in_region(env, mug, "box_boundary", tol=0.04) or _is_in_region(
        env, mug, "box-inside", tol=0.04
    )
    return bool(in_box)


def _run_table_mug_to_box_with_fallback(env, pr, mug_name, task_idx, slot_pose):
    """
    Try mug->box via PDDL first; then try easier inside-box region.
    Do NOT snap/teleport mug pose after placement.
    """
    task_label = f"Task {task_idx}: Table Mug -> Box ({mug_name})"
    prev_overrides = _install_box_slot_overrides(env, mug_name, slot_pose)

    try:
        # Attempt 1: strict box boundary region
        success = base.run_standard_pick_place(
            env,
            object_name=mug_name,
            target_region="box_boundary",
            task_name=task_label,
        )
        if success:
            return True

        print(f"[Fallback] {mug_name}: box_boundary planning failed, trying box-inside region.")
        # Attempt 2: inside-box region (often easier IK/sample pose)
        success = base.run_standard_pick_place(
            env,
            object_name=mug_name,
            target_region="box-inside",
            task_name=f"{task_label} [fallback box-inside]",
        )
        if success:
            return True

        print(f"[Fallback] {mug_name}: no valid PDDL plan for both box regions.")
        return False
    finally:
        _restore_overrides(env, prev_overrides)


def _classify_variation_objects(env):
    mug_on_box_candidates = ["mug2", "mug1", "mug4", "mug3"]
    mug_in_cupboard_candidates = ["mug3", "mug1", "mug2", "mug4"]
    grocery_candidates = ["soup", "spam", "mustard", "sugar", "crackers"]

    mug_on_box = _select_object_by_region(env, mug_on_box_candidates, "box_boundary")
    mug_in_cupboard = _select_object_by_region(env, mug_in_cupboard_candidates, "cupboard_boundary")

    groceries = _discover_unique_objects(env, grocery_candidates)
    grocery_in_box = None
    grocery_on_table = None
    grocery_in_cupboard = None

    for name in groceries:
        obj = env.get_object(name)
        if _is_in_region(env, obj, "box_boundary"):
            grocery_in_box = name
            break

    for name in groceries:
        if name == grocery_in_box:
            continue
        obj = env.get_object(name)
        if _is_in_region(env, obj, "table"):
            grocery_on_table = name
            break

    for name in groceries:
        if name == grocery_in_box:
            continue
        obj = env.get_object(name)
        if _is_in_region(env, obj, "cupboard_boundary") or _is_in_region(env, obj, "cupboard_boundary_top"):
            grocery_in_cupboard = name
            break

    # Robust fallbacks if region checks are noisy.
    if grocery_in_box is None and groceries:
        box_center = _region_center(env, "box_boundary")
        if box_center is not None:
            grocery_in_box = min(
                groceries,
                key=lambda n: float(
                    np.linalg.norm(np.array(env.get_object(n).get_position(), dtype=float) - box_center)
                ),
            )
        else:
            grocery_in_box = groceries[0]

    if grocery_on_table is None:
        remaining = []
        for g in groceries:
            if g == grocery_in_box:
                continue
            obj = env.get_object(g)
            in_cupboard = _is_in_region(env, obj, "cupboard_boundary") or _is_in_region(
                env, obj, "cupboard_boundary_top"
            )
            if not in_cupboard:
                remaining.append(g)
        grocery_on_table = remaining[0] if remaining else None

    # If cupboard selection accidentally picked the same mug, pick another one.
    if mug_in_cupboard == mug_on_box:
        all_mugs = _discover_unique_objects(env, mug_in_cupboard_candidates + mug_on_box_candidates)
        other = [m for m in all_mugs if m != mug_on_box]
        if other:
            mug_in_cupboard = other[0]

    return {
        "mug_on_box": mug_on_box,
        "mug_in_cupboard": mug_in_cupboard,
        "grocery_in_box": grocery_in_box,
        "grocery_on_table": grocery_on_table,
        "grocery_in_cupboard": grocery_in_cupboard,
    }


def main():
    env = base.ENV
    pr = env.pr

    print("=" * 70)
    print("GROUND TRUTH ORCHESTRATOR - VARIATION 2")
    print("=" * 70)
    print(f"Scene: {SCENE_PATH}")
    print(
        f"Speed mode: {base.GT_SPEED_MODE} "
        f"(exec_interp={base.DEFAULT_EXEC_INTERP_STEPS}, hold_steps={base.RELEASE_HOLD_STEPS}, record_every={base.RECORD_EVERY_N})"
    )

    video_dir = os.path.join(THIS_DIR, "orchestrator_videos")
    enable_video = os.environ.get("GT_RECORD_VIDEO", "1").strip().lower() not in {
        "0", "false", "no"
    }
    if enable_video:
        print(f"\nInitializing video recorder: {video_dir}")
        base.VIDEO_RECORDER = VideoRecorder(env, output_dir=video_dir, fps=30)
    else:
        base.VIDEO_RECORDER = None
        print("\nVideo recorder disabled (GT_RECORD_VIDEO=0)")

    print("\nSettling physics...")
    for _ in range(50):
        base.step_and_record(pr, 1)

    home_q = env.get_home_conf()
    env.set_robot_conf(home_q)
    for _ in range(10):
        base.step_and_record(pr, 1)

    picked = _classify_variation_objects(env)
    mug_on_box = picked["mug_on_box"]
    mug_in_cupboard = picked["mug_in_cupboard"]
    grocery_in_box = picked["grocery_in_box"]
    grocery_on_table = picked["grocery_on_table"]
    grocery_in_cupboard = picked["grocery_in_cupboard"]

    print("\nObject assignment for this run:")
    print(f"  mug_on_box      : {mug_on_box}")
    print(f"  mug_in_cupboard : {mug_in_cupboard}")
    print(f"  grocery_in_box  : {grocery_in_box}")
    print(f"  grocery_on_table: {grocery_on_table}")
    print(f"  grocery_in_cupboard: {grocery_in_cupboard}")

    required_keys = ["mug_on_box", "mug_in_cupboard", "grocery_in_box", "grocery_on_table"]
    missing = [k for k in required_keys if picked.get(k) is None]
    if missing:
        print(f"\nERROR: Could not resolve required objects: {missing}")
        if base.VIDEO_RECORDER:
            base.VIDEO_RECORDER.release()
        return

    results = []

    # 1) Cupboard mug -> placement
    success = base.run_cupboard_pick_place(
        env,
        object_name=mug_in_cupboard,
        target_region="placement_boundary",
        task_name="Task 1: Cupboard Mug -> Placement",
    )
    results.append(("Task 1: mug_in_cupboard -> placement_boundary", success))
    base.go_home(env)

    # 2) Grocery on table -> cupboard
    runtime_grocery_on_table = _select_runtime_table_grocery(
        env, exclude_names=[grocery_in_box]
    )
    if runtime_grocery_on_table is None:
        print("\nWARNING: Could not find grocery on table at runtime; using initial assignment.")
        runtime_grocery_on_table = grocery_on_table
    else:
        print(f"\nTask 2 runtime selection: grocery_on_table={runtime_grocery_on_table}")

    success = base.run_standard_pick_place(
        env,
        object_name=runtime_grocery_on_table,
        target_region="cupboard_boundary",
        task_name="Task 2: Grocery on Table -> Cupboard",
    )
    results.append(("Task 2: grocery_on_table -> cupboard_boundary", success))
    base.go_home(env)

    # 3) Mug on box -> placement
    success = base.run_box_pick_place(
        env,
        object_name=mug_on_box,
        target_region="placement_boundary",
        task_name="Task 3: Mug on Box -> Placement",
    )
    results.append(("Task 3: mug_on_box -> placement_boundary", success))
    base.go_home(env)

    # 4) Open box lid
    success = base.run_open_box(
        env,
        task_name="Task 4: Open Box Lid",
    )
    results.append(("Task 4: open box lid", success))
    base.go_home(env)

    # 5) Grocery in box -> cupboard (runtime selection)
    runtime_grocery_in_box = _select_runtime_box_grocery(
        env, exclude_names=[runtime_grocery_on_table, grocery_in_cupboard]
    )
    if runtime_grocery_in_box is None:
        print("\nWARNING: Could not find grocery in box at runtime; using initial assignment.")
        runtime_grocery_in_box = grocery_in_box
    else:
        print(f"\nTask 5 runtime selection: grocery_in_box={runtime_grocery_in_box}")

    success = base.run_box_pick_place(
        env,
        object_name=runtime_grocery_in_box,
        target_region="cupboard_boundary",
        task_name="Task 5: Grocery in Box -> Cupboard",
    )
    results.append(("Task 5: grocery_in_box -> cupboard_boundary", success))
    base.go_home(env)

    # 6-7) Two mugs on table -> box (runtime positions, non-overlapping slots)
    # Prefer the former "mug on box" first, then the former cupboard mug.
    # This ordering is typically easier in variation-2 geometry.
    runtime_table_mugs = _select_runtime_table_mugs(
        env, preferred_names=[mug_on_box, mug_in_cupboard], max_count=2
    )
    if len(runtime_table_mugs) < 2:
        retry_names = _select_runtime_table_mugs(env, preferred_names=[], max_count=2)
        for n in retry_names:
            if n not in runtime_table_mugs:
                runtime_table_mugs.append(n)
            if len(runtime_table_mugs) >= 2:
                break
    print(f"\nRuntime table mugs for box placement: {runtime_table_mugs}")

    if len(runtime_table_mugs) < 2:
        print("WARNING: Expected 2 mugs on table for final box placement, found fewer.")

    box_slots = _compute_box_slot_poses(env, runtime_table_mugs)
    if box_slots:
        print("Computed box slots:")
        for name in runtime_table_mugs:
            if name in box_slots:
                print(f"  {name}: {[round(v, 4) for v in box_slots[name][:3]]}")
    else:
        print("WARNING: Could not compute box slots; proceeding without slot finalization.")

    task_idx = 6
    for mug_name in runtime_table_mugs:
        slot_pose = box_slots.get(mug_name)
        success = _run_table_mug_to_box_with_fallback(
            env, pr, mug_name, task_idx, slot_pose
        )
        results.append((f"Task {task_idx}: {mug_name} table -> box_boundary", success))
        base.go_home(env)
        task_idx += 1

    print("\n" + "=" * 70)
    print("EXECUTION SUMMARY - VARIATION 2")
    print("=" * 70)
    total = len(results)
    passed = sum(1 for _, ok in results if ok)
    for task, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {task}")
    print(f"\nTotal: {passed}/{total} tasks passed")
    print("=" * 70)

    if base.VIDEO_RECORDER:
        base.VIDEO_RECORDER.release()
        print(f"\nVideo recording saved to: {video_dir}")

    keep_alive = os.environ.get("GT_KEEP_ALIVE", "0").strip().lower() not in {
        "0", "false", "no"
    }
    if keep_alive:
        print("\nVariation 2 orchestration complete. Press Ctrl+C to close.")
        try:
            while True:
                pr.step()
                base._run_step_callback()
        except KeyboardInterrupt:
            pass
    else:
        print("\nVariation 2 orchestration complete. Auto-exit enabled.")

    pr.stop()
    pr.shutdown()


if __name__ == "__main__":
    main()
