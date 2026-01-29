(define (domain rlbench_kitchen)
  (:requirements :strips :equality)

  (:predicates
      ; Types
      (robot ?x)
      (movable ?x)
      (region ?x)
      (conf ?x)
      (pose ?x)
      (grasp ?x)
      (traj ?x)

      ;; robot & object state
      (at-conf ?q)
      (hand-empty)
      (holding ?o)
      (retreated ?o)

      ;; geometry / placements
      (at-pose ?o ?p)
      (in-region ?o ?r)
      (is-home ?q)

      ;; stream-certified symbols
      (stable ?o ?p ?r)      ; pose places o stably in region r
      (kin ?o ?p ?g ?q-start ?q-end ?t)   ; feasible pick trajectory
      (place-kin ?o ?p ?g ?q-start ?q-end ?t) ; feasible place trajectory
      (motion ?q1 ?q2 ?t) ; feasible motion trajectory
      
      ;; Lid specific
      (lid ?o)
      (closed ?o)
      (opened ?o)
      (can-open-lid ?o ?g ?q1 ?q2 ?t)
      
      ;; Obstruction logic
      (obstructs ?blocker ?blocked)
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
        (at-pose ?o ?p)
        (kin ?o ?p ?g ?q1 ?q2 ?t)
        ;; Obstruction check: Nothing must obstruct ?o
        (forall (?x) (not (obstructs ?x ?o))))
    :effect (and
        (not (hand-empty))
        (holding ?o)
        (not (at-pose ?o ?p))
        (at-conf ?q2)
        ;; Clear obstructions caused by ?o
        (forall (?y) (when (obstructs ?o ?y) (not (obstructs ?o ?y))))))

  (:action open-lid
    :parameters (?o ?g ?q1 ?q2 ?t)
    :precondition (and
        (lid ?o) (closed ?o) (grasp ?g) (conf ?q1) (conf ?q2) (traj ?t)
        (hand-empty)
        (at-conf ?q1)
        (can-open-lid ?o ?g ?q1 ?q2 ?t)
        ;; Obstruction check: Nothing must obstruct the lid
        (forall (?x) (not (obstructs ?x ?o))))
    :effect (and
        (hand-empty)
        (opened ?o)
        (not (closed ?o))
        (at-conf ?q2)
        (not (at-conf ?q1))
        ;; Opening the lid clears obstructions it causes
        (forall (?y) (when (obstructs ?o ?y) (not (obstructs ?o ?y))))))

  (:action retreat
    :parameters (?o ?q1 ?q2 ?t)
    :precondition (and
        (movable ?o) (conf ?q1) (conf ?q2) (traj ?t)
        (holding ?o)
        (at-conf ?q1)
        (is-home ?q2)
        (motion ?q1 ?q2 ?t))
    :effect (and
        (not (at-conf ?q1))
        (at-conf ?q2)
        (retreated ?o)))

  (:action place
    :parameters (?o ?p ?g ?r ?q1 ?q2 ?t)
    :precondition (and
        (movable ?o) (pose ?p) (grasp ?g) (region ?r) (conf ?q1) (conf ?q2) (traj ?t)
        (holding ?o)
        ; (retreated ?o) ; Removed to allow direct placement
        (at-conf ?q1)
        (stable ?o ?p ?r)
        (place-kin ?o ?p ?g ?q1 ?q2 ?t))
    :effect (and
        (hand-empty)
        (not (holding ?o))
        ; (not (retreated ?o))
        (at-pose ?o ?p)
        (in-region ?o ?r)
        (at-conf ?q2)))
)
