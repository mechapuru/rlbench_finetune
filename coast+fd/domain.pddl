;; PDDL Domain for LongHorizonGrillTask
;; Defines actions for the grill cooking task with COAST integration

(define (domain long-horizon-grill)
    (:requirements :strips :typing :equality :negative-preconditions :conditional-effects)
    
    (:types
        object - abstract
        location - abstract
        timestep - abstract
        grill - location
        plate - location
        rack - location
    )
    
    (:constants
        ;; Objects
        steak - object
        chicken - object
        the_plate - object
        
        ;; Locations
        grill_surface - grill
        plate_source - rack
        plate_target - location
        
        ;; Timesteps (for sequence constraints)
        t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 - timestep
        t11 t12 t13 t14 t15 t16 t17 t18 t19 t20 - timestep
    )
    
    (:predicates
        ;; Object state
        (On ?o - object ?l - location)
        (Holding ?o - object)
        (HandEmpty)
        (Cooked ?o - object)
        (OnGrill ?o - object)
        (OnPlate ?o - object)
        
        ;; Grill state
        (GrillOpen)
        (GrillClosed)
        
        ;; Timestep tracking (for COAST constraints)
        (AtTimestep ?t - timestep)
        (Next ?t1 - timestep ?t2 - timestep)
        
        ;; Failure predicates (for COAST constraints)
        (FailPick ?o - object ?l - location ?t1 - timestep ?t2 - timestep)
        (FailPlace ?o - object ?l - location ?t1 - timestep ?t2 - timestep)
        (FailOpenGrill ?t1 - timestep ?t2 - timestep)
        (FailCloseGrill ?t1 - timestep ?t2 - timestep)
        
        ;; Logging predicates (for sequence tracking)
        (LogPick ?o - object ?l - location ?t1 - timestep ?t2 - timestep)
        (LogPlace ?o - object ?l - location ?t1 - timestep ?t2 - timestep)
    )
    
    ;; ==================== Actions ====================
    
    (:action pick
        :parameters (?o - object ?l - location ?t1 - timestep ?t2 - timestep)
        :precondition (and
            (On ?o ?l)
            (HandEmpty)
            (not (FailPick ?o ?l ?t1 ?t2))
            (not (= ?t1 ?t2))
            (AtTimestep ?t1)
            (Next ?t1 ?t2)
        )
        :effect (and
            (Holding ?o)
            (not (On ?o ?l))
            (not (HandEmpty))
            (not (AtTimestep ?t1))
            (AtTimestep ?t2)
            (LogPick ?o ?l ?t1 ?t2)
        )
    )
    
    (:action place
        :parameters (?o - object ?l - location ?t1 - timestep ?t2 - timestep)
        :precondition (and
            (Holding ?o)
            (not (FailPlace ?o ?l ?t1 ?t2))
            (not (= ?t1 ?t2))
            (AtTimestep ?t1)
            (Next ?t1 ?t2)
        )
        :effect (and
            (On ?o ?l)
            (not (Holding ?o))
            (HandEmpty)
            (not (AtTimestep ?t1))
            (AtTimestep ?t2)
            (LogPlace ?o ?l ?t1 ?t2)
            ;; Track if placed on grill or plate
            (when (= ?l grill_surface) (OnGrill ?o))
            (when (= ?l plate_target) (OnPlate ?o))
        )
    )
    
    (:action place_on_grill
        :parameters (?o - object ?t1 - timestep ?t2 - timestep)
        :precondition (and
            (Holding ?o)
            (GrillOpen)
            (not (FailPlace ?o grill_surface ?t1 ?t2))
            (not (= ?t1 ?t2))
            (AtTimestep ?t1)
            (Next ?t1 ?t2)
        )
        :effect (and
            (On ?o grill_surface)
            (OnGrill ?o)
            (not (Holding ?o))
            (HandEmpty)
            (not (AtTimestep ?t1))
            (AtTimestep ?t2)
            (LogPlace ?o grill_surface ?t1 ?t2)
        )
    )
    
    (:action open_grill
        :parameters (?t1 - timestep ?t2 - timestep)
        :precondition (and
            (GrillClosed)
            (HandEmpty)
            (not (FailOpenGrill ?t1 ?t2))
            (not (= ?t1 ?t2))
            (AtTimestep ?t1)
            (Next ?t1 ?t2)
        )
        :effect (and
            (GrillOpen)
            (not (GrillClosed))
            (not (AtTimestep ?t1))
            (AtTimestep ?t2)
        )
    )
    
    (:action close_grill
        :parameters (?t1 - timestep ?t2 - timestep)
        :precondition (and
            (GrillOpen)
            (HandEmpty)
            (not (FailCloseGrill ?t1 ?t2))
            (not (= ?t1 ?t2))
            (AtTimestep ?t1)
            (Next ?t1 ?t2)
        )
        :effect (and
            (GrillClosed)
            (not (GrillOpen))
            (not (AtTimestep ?t1))
            (AtTimestep ?t2)
        )
    )
    
    (:action cook
        :parameters (?o - object ?t1 - timestep ?t2 - timestep)
        :precondition (and
            (OnGrill ?o)
            (GrillClosed)
            (HandEmpty)
            (not (= ?t1 ?t2))
            (AtTimestep ?t1)
            (Next ?t1 ?t2)
        )
        :effect (and
            (Cooked ?o)
            (not (AtTimestep ?t1))
            (AtTimestep ?t2)
        )
    )
)
