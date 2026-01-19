;; Streams PDDL for COAST integration
;; Defines stream-related predicates and placeholder actions for constraints

(define (domain long-horizon-grill)
    (:requirements :strips :typing :equality :negative-preconditions :conditional-effects)
    
    (:types
        object - abstract
        location - abstract
        timestep - abstract
        grill - location
        plate - location
        rack - location
        ;; Stream types
        conf - abstract
        pose - abstract
        grasp - abstract
        traj - abstract
    )
    
    (:predicates
        ;; Object state (same as domain.pddl)
        (On ?o - object ?l - location)
        (Holding ?o - object)
        (HandEmpty)
        (Cooked ?o - object)
        (OnGrill ?o - object)
        (OnPlate ?o - object)
        (GrillOpen)
        (GrillClosed)
        (AtTimestep ?t - timestep)
        (Next ?t1 - timestep ?t2 - timestep)
        
        ;; Failure predicates
        (FailPick ?o - object ?l - location ?t1 - timestep ?t2 - timestep)
        (FailPlace ?o - object ?l - location ?t1 - timestep ?t2 - timestep)
        (FailOpenGrill ?t1 - timestep ?t2 - timestep)
        (FailCloseGrill ?t1 - timestep ?t2 - timestep)
        
        ;; Log predicates
        (LogPick ?o - object ?l - location ?t1 - timestep ?t2 - timestep)
        (LogPlace ?o - object ?l - location ?t1 - timestep ?t2 - timestep)
        
        ;; Stream certified facts
        (AtConf ?q - conf)
        (AtPose ?o - object ?p - pose)
        (GraspSampled ?o - object ?g - grasp)
        (PoseSampled ?o - object ?l - location ?p - pose)
        (IKSolved ?o - object ?p - pose ?g - grasp ?q - conf)
        (MotionPlanned ?q1 - conf ?q2 - conf ?t - traj)
        (CollisionFree ?t - traj)
    )
    
    ;; Placeholder actions for handling stream failures
    ;; These are used by COAST to reset failure predicates
    
    (:action FailPick_reset
        :parameters (?o - object ?l - location ?t1 - timestep ?t2 - timestep)
        :precondition (and)
        :effect (not (FailPick ?o ?l ?t1 ?t2))
    )
    
    (:action FailPlace_reset
        :parameters (?o - object ?l - location ?t1 - timestep ?t2 - timestep)
        :precondition (and)
        :effect (not (FailPlace ?o ?l ?t1 ?t2))
    )
    
    (:action FailOpenGrill_reset
        :parameters (?t1 - timestep ?t2 - timestep)
        :precondition (and)
        :effect (not (FailOpenGrill ?t1 ?t2))
    )
    
    (:action FailCloseGrill_reset
        :parameters (?t1 - timestep ?t2 - timestep)
        :precondition (and)
        :effect (not (FailCloseGrill ?t1 ?t2))
    )
)
