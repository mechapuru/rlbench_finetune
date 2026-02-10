(define (domain grill_task)
  (:requirements :strips :equality)

  (:predicates
      ;; Types (as unary predicates)
      (robot ?x)
      (movable ?x)
      (region ?x)
      (conf ?x)
      (pose ?x)
      (grasp ?x)
      (traj ?x)

      ;; Robot & object state
      (at-conf ?q)
      (hand-empty)
      (holding ?o)

      ;; Geometry / placements
      (at-pose ?o ?p)
      (in-region ?o ?r)
      (is-home ?q)

      ;; Stream-certified symbols
      (stable ?o ?p ?r)                       ; pose places o stably in region r
      (kin ?o ?p ?g ?q-start ?q-end ?t)       ; feasible pick trajectory
      (place-kin ?o ?p ?g ?q-start ?q-end ?t) ; feasible place trajectory
      (motion ?q1 ?q2 ?t)                     ; feasible motion trajectory

      ;; Grill lid specific (hinged rotation)
      (lid ?o)
      (grill-open ?o)
      (grill-closed ?o)
      (can-close-grill ?o ?g ?q1 ?q2 ?t)     ; feasible close-grill trajectory
      (can-open-grill ?o ?g ?q1 ?q2 ?t)      ; feasible open-grill trajectory

      ;; Grill surface
      (grill-surface ?r)
      (on-grill ?o)
  )

  ;;--- ACTIONS ------------------------------------------------------------

  (:action move
    :parameters (?q1 ?q2 ?t)
    :precondition (and (conf ?q1) (conf ?q2) (traj ?t)
                       (at-conf ?q1)
                       (motion ?q1 ?q2 ?t))
    :effect (and (not (at-conf ?q1))
                 (at-conf ?q2)))

  (:action pick
    :parameters (?o ?p ?g ?q1 ?q2 ?t)
    :precondition (and
        (movable ?o) (pose ?p) (grasp ?g) (conf ?q1) (conf ?q2) (traj ?t)
        (not (lid ?o))
        (hand-empty)
        (at-conf ?q1)
        (at-pose ?o ?p))
        ;; Note: kin predicate is certified by sample-pick-kin stream
        ; (kin ?o ?p ?g ?q1 ?q2 ?t)
    :effect (and
        (not (hand-empty))
        (holding ?o)
        (not (at-pose ?o ?p))
        (at-conf ?q2)))

  (:action place
    :parameters (?o ?p ?g ?r ?q1 ?q2 ?t)
    :precondition (and
        (movable ?o) (pose ?p) (grasp ?g) (region ?r) (conf ?q1) (conf ?q2) (traj ?t)
        (holding ?o)
        (at-conf ?q1)
        (stable ?o ?p ?r)
        (place-kin ?o ?p ?g ?q1 ?q2 ?t))
    :effect (and
        (hand-empty)
        (not (holding ?o))
        (at-pose ?o ?p)
        (in-region ?o ?r)
        (at-conf ?q2)
        ;; If placing on grill surface, mark as on-grill
        (when (grill-surface ?r) (on-grill ?o))))

  (:action close-grill
    :parameters (?o ?g ?q1 ?q2 ?t)
    :precondition (and
        (lid ?o) (grill-open ?o) (grasp ?g) (conf ?q1) (conf ?q2) (traj ?t)
        (hand-empty)
        (at-conf ?q1)
        (can-close-grill ?o ?g ?q1 ?q2 ?t))
    :effect (and
        (hand-empty)
        (grill-closed ?o)
        (not (grill-open ?o))
        (at-conf ?q2)
        (not (at-conf ?q1))))

  (:action open-grill
    :parameters (?o ?g ?q1 ?q2 ?t)
    :precondition (and
        (lid ?o) (grill-closed ?o) (grasp ?g) (conf ?q1) (conf ?q2) (traj ?t)
        (hand-empty)
        (at-conf ?q1)
        (can-open-grill ?o ?g ?q1 ?q2 ?t))
    :effect (and
        (hand-empty)
        (grill-open ?o)
        (not (grill-closed ?o))
        (at-conf ?q2)
        (not (at-conf ?q1))))
)
