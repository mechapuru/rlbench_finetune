(define (stream rlbench_kitchen_streams)
  ;; (:requirements :typing) removed

  ;; sample a stable placement pose of object o in region r
  (:stream sample-stable-pose
    :inputs  (?o ?r)
    :domain  (and (movable ?o) (region ?r))
    :outputs (?p)
    :certified (and (pose ?p) (stable ?o ?p ?r)))

  ;; sample a feasible pick (IK + collision-free traj) for object o at pose p
  (:stream sample-pick-kin
    :inputs  (?o ?p)
    :domain  (and (movable ?o) (pose ?p))
    :outputs (?g ?q1 ?q2 ?t)
    :certified (and (grasp ?g) (conf ?q1) (conf ?q2) (traj ?t) (kin ?o ?p ?g ?q1 ?q2 ?t)))

  ;; sample a feasible place trajectory to pose p in region r
  (:stream sample-place-kin
    :inputs  (?o ?p ?r)
    :domain  (and (movable ?o) (pose ?p) (region ?r) (stable ?o ?p ?r))
    :outputs (?g ?q1 ?q2 ?t)
    :certified (and (grasp ?g) (conf ?q1) (conf ?q2) (traj ?t) (place-kin ?o ?p ?g ?q1 ?q2 ?t)))

  ;; sample motion plan between two configurations
  (:stream sample-motion
    :inputs  (?q1 ?q2)
    :domain  (and (conf ?q1) (conf ?q2))
    :outputs (?t)
    :certified (and (traj ?t) (motion ?q1 ?q2 ?t)))

  ;; sample open lid trajectory (combined grasp, slide, return)
  (:stream sample-open-lid
    :inputs  (?o)
    :domain  (lid ?o)
    :outputs (?g ?q1 ?q2 ?t)
    :certified (and (grasp ?g) (conf ?q1) (conf ?q2) (traj ?t) (can-open-lid ?o ?g ?q1 ?q2 ?t)))
)
