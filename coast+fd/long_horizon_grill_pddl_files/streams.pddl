;; Streams PDDL for COAST integration
;; Defines stream-related predicates for LongHorizonGrillTask

(define (domain long-horizon-grill-streams)
    (:requirements :strips :typing :equality :negative-preconditions)
    
    (:types
        object - abstract
        location - abstract
        timestep - abstract
        
        ;; Stream types for motion planning
        conf - abstract      ; Robot joint configuration
        pose - abstract      ; Object pose (position + orientation)
        grasp - abstract     ; Grasp pose for object
        traj - abstract      ; Robot trajectory
    )
    
    (:predicates
        ;; Object state (same as domain.pddl)
        (On ?o - object ?l - location)
        (Holding ?o - object)
        (HandEmpty)
        (GrillOpen)
        (GrillClosed)
        (ChickenOnGrill)
        (ChickenCooked)
        (PlateAtTarget)
        (ChickenOnPlate)
        
        ;; Timestep tracking
        (AtTimestep ?t - timestep)
        (Next ?t1 - timestep ?t2 - timestep)
        
        ;; Failure predicates (for COAST constraint learning)
        (FailPick ?o - object ?l - location ?t1 - timestep ?t2 - timestep)
        (FailPlace ?o - object ?l - location ?t1 - timestep ?t2 - timestep)
        (FailCloseLid ?t1 - timestep ?t2 - timestep)
        (FailOpenLid ?t1 - timestep ?t2 - timestep)
        
        ;; Stream certified facts (geometric state)
        (AtConf ?q - conf)
        (AtPose ?o - object ?p - pose)
        (GraspSampled ?o - object ?g - grasp)
        (PoseSampled ?o - object ?l - location ?p - pose)
        (IKSolved ?o - object ?p - pose ?g - grasp ?q - conf)
        (MotionPlanned ?q1 - conf ?q2 - conf ?t - traj)
        (CollisionFree ?t - traj)
        
        ;; Lid-specific predicates
        (LidTrajectoryPlanned ?t - traj)
    )
    
    ;; Placeholder actions for handling stream failures
    ;; Used by COAST to reset failure predicates during replanning
    
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
    
    (:action FailCloseLid_reset
        :parameters (?t1 - timestep ?t2 - timestep)
        :precondition (and)
        :effect (not (FailCloseLid ?t1 ?t2))
    )
    
    (:action FailOpenLid_reset
        :parameters (?t1 - timestep ?t2 - timestep)
        :precondition (and)
        :effect (not (FailOpenLid ?t1 ?t2))
    )
)
