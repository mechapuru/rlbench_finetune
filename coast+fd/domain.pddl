;; PDDL Domain for LongHorizonGrillTask
;; Task: Cook chicken on grill, move plate, serve chicken on plate
;;
;; Sequence:
;; 1. Pick chicken → Place on grill
;; 2. Close grill lid (constrained trajectory)
;; 3. Pick plate from dish_rack → Place on plate_target
;; 4. Open grill lid
;; 5. Pick chicken from grill → Place on plate

(define (domain long-horizon-grill)
    (:requirements :strips :typing :equality :negative-preconditions)
    
    (:types
        object - abstract
        location - abstract
        timestep - abstract
    )
    
    (:constants
        ;; Manipulable objects
        chicken - object
        plate - object
        lid - object
        
        ;; Distractor (not used in planning but present in scene)
        ;; steak - object  
        
        ;; Locations
        grill_surface - location    ; Where meat is placed on grill
        dish_rack - location        ; Where plate starts
        plate_target - location     ; Where plate should end up
        lid_open_position - location    ; Lid resting position when open
        lid_closed_position - location  ; Lid position when closed
        
        ;; Timesteps (for COAST constraint tracking)
        t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 - timestep
        t11 t12 t13 t14 t15 t16 t17 t18 t19 t20 - timestep
    )
    
    (:predicates
        ;; Object location state
        (On ?o - object ?l - location)
        (Holding ?o - object)
        (HandEmpty)
        
        ;; Grill state
        (GrillOpen)
        (GrillClosed)
        
        ;; Task progress tracking
        (ChickenOnGrill)
        (ChickenCooked)
        (PlateAtTarget)
        (ChickenOnPlate)
        
        ;; Timestep tracking (for COAST constraints)
        (AtTimestep ?t - timestep)
        (Next ?t1 - timestep ?t2 - timestep)
        
        ;; Failure predicates (for COAST constraint learning)
        (FailPick ?o - object ?l - location ?t1 - timestep ?t2 - timestep)
        (FailPlace ?o - object ?l - location ?t1 - timestep ?t2 - timestep)
        (FailCloseLid ?t1 - timestep ?t2 - timestep)
        (FailOpenLid ?t1 - timestep ?t2 - timestep)
    )
    
    ;; ==================== Actions ====================
    
    ;; Pick up an object from a location
    (:action pick
        :parameters (?o - object ?l - location ?t1 - timestep ?t2 - timestep)
        :precondition (and
            (On ?o ?l)
            (HandEmpty)
            (not (FailPick ?o ?l ?t1 ?t2))
            (AtTimestep ?t1)
            (Next ?t1 ?t2)
        )
        :effect (and
            (Holding ?o)
            (not (On ?o ?l))
            (not (HandEmpty))
            (not (AtTimestep ?t1))
            (AtTimestep ?t2)
        )
    )
    
    ;; Place an object at a location
    (:action place
        :parameters (?o - object ?l - location ?t1 - timestep ?t2 - timestep)
        :precondition (and
            (Holding ?o)
            (not (FailPlace ?o ?l ?t1 ?t2))
            (AtTimestep ?t1)
            (Next ?t1 ?t2)
        )
        :effect (and
            (On ?o ?l)
            (not (Holding ?o))
            (HandEmpty)
            (not (AtTimestep ?t1))
            (AtTimestep ?t2)
        )
    )
    
    ;; Place chicken on grill (specialized action)
    (:action place_chicken_on_grill
        :parameters (?t1 - timestep ?t2 - timestep)
        :precondition (and
            (Holding chicken)
            (GrillOpen)
            (not (FailPlace chicken grill_surface ?t1 ?t2))
            (AtTimestep ?t1)
            (Next ?t1 ?t2)
        )
        :effect (and
            (On chicken grill_surface)
            (ChickenOnGrill)
            (not (Holding chicken))
            (HandEmpty)
            (not (AtTimestep ?t1))
            (AtTimestep ?t2)
        )
    )
    
    ;; Place plate at target
    (:action place_plate_at_target
        :parameters (?t1 - timestep ?t2 - timestep)
        :precondition (and
            (Holding plate)
            (not (FailPlace plate plate_target ?t1 ?t2))
            (AtTimestep ?t1)
            (Next ?t1 ?t2)
        )
        :effect (and
            (On plate plate_target)
            (PlateAtTarget)
            (not (Holding plate))
            (HandEmpty)
            (not (AtTimestep ?t1))
            (AtTimestep ?t2)
        )
    )
    
    ;; Place chicken on plate (final serving)
    (:action place_chicken_on_plate
        :parameters (?t1 - timestep ?t2 - timestep)
        :precondition (and
            (Holding chicken)
            (PlateAtTarget)
            (ChickenCooked)
            (AtTimestep ?t1)
            (Next ?t1 ?t2)
        )
        :effect (and
            (On chicken plate_target)
            (ChickenOnPlate)
            (not (Holding chicken))
            (HandEmpty)
            (not (AtTimestep ?t1))
            (AtTimestep ?t2)
        )
    )
    
    ;; Close grill lid (requires constrained trajectory)
    ;; Robot must grasp lid and move it along semi-circular path
    (:action close_lid
        :parameters (?t1 - timestep ?t2 - timestep)
        :precondition (and
            (GrillOpen)
            (HandEmpty)
            (ChickenOnGrill)
            (not (FailCloseLid ?t1 ?t2))
            (AtTimestep ?t1)
            (Next ?t1 ?t2)
        )
        :effect (and
            (GrillClosed)
            (not (GrillOpen))
            (ChickenCooked)  ; Chicken is cooked once lid is closed
            (not (AtTimestep ?t1))
            (AtTimestep ?t2)
        )
    )
    
    ;; Open grill lid (requires constrained trajectory)
    (:action open_lid
        :parameters (?t1 - timestep ?t2 - timestep)
        :precondition (and
            (GrillClosed)
            (HandEmpty)
            (not (FailOpenLid ?t1 ?t2))
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
    
    ;; Pick chicken from grill (after cooking)
    (:action pick_chicken_from_grill
        :parameters (?t1 - timestep ?t2 - timestep)
        :precondition (and
            (On chicken grill_surface)
            (GrillOpen)
            (ChickenCooked)
            (HandEmpty)
            (not (FailPick chicken grill_surface ?t1 ?t2))
            (AtTimestep ?t1)
            (Next ?t1 ?t2)
        )
        :effect (and
            (Holding chicken)
            (not (On chicken grill_surface))
            (not (ChickenOnGrill))
            (not (HandEmpty))
            (not (AtTimestep ?t1))
            (AtTimestep ?t2)
        )
    )
)
