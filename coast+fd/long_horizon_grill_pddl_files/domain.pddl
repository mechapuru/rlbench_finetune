;; PDDL Domain for LongHorizonGrillTask
;; Task: Cook chicken on grill, move plate, serve chicken on plate
;;
;; Sequence:
;; 1. pick(chicken, dish_rack) → place(chicken, grill_surface)
;; 2. close_lid
;; 3. pick(plate, dish_rack) → place(plate, plate_target)
;; 4. open_lid
;; 5. pick(chicken, grill_surface) → place(chicken, plate_target)

(define (domain long-horizon-grill)
    (:requirements :strips :typing :equality :negative-preconditions :conditional-effects)
    
    (:types
        object - abstract
        location - abstract
        timestep - abstract
    )
    
    (:constants
        ;; Manipulable objects
        chicken - object
        plate - object
        
        ;; Locations
        grill_side - location           ; Where chicken starts (on grill but not cooking area)
        grill_cooking_area - location   ; Where meat cooks (under lid)
        dish_rack - location            ; Where plate starts
        plate_target - location         ; Where plate should end up
        
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
    
    ;; Generic pick action - pick any object from any location
    (:action pick
        :parameters (?o - object ?l - location ?t1 - timestep ?t2 - timestep)
        :precondition (and
            (On ?o ?l)
            (HandEmpty)
            ;; Special case: picking from grill requires grill open and chicken cooked
            (or
                (not (= ?l grill_cooking_area))
                (and (= ?l grill_cooking_area) (GrillOpen) (ChickenCooked))
            )
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
            ;; Clear ChickenOnGrill if picking chicken from grill
            (when (and (= ?o chicken) (= ?l grill_cooking_area))
                (not (ChickenOnGrill)))
        )
    )
    
    ;; Generic place action - place any object at any location
    (:action place
        :parameters (?o - object ?l - location ?t1 - timestep ?t2 - timestep)
        :precondition (and
            (Holding ?o)
            ;; Special case: placing on grill requires grill open
            (or
                (not (= ?l grill_cooking_area))
                (and (= ?l grill_cooking_area) (GrillOpen))
            )
            ;; Special case: placing chicken on plate requires plate at target and chicken cooked
            (or
                (not (and (= ?o chicken) (= ?l plate_target)))
                (and (= ?o chicken) (= ?l plate_target) (PlateAtTarget) (ChickenCooked))
            )
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
            ;; Track ChickenOnGrill
            (when (and (= ?o chicken) (= ?l grill_cooking_area))
                (ChickenOnGrill))
            ;; Track PlateAtTarget
            (when (and (= ?o plate) (= ?l plate_target))
                (PlateAtTarget))
            ;; Track ChickenOnPlate
            (when (and (= ?o chicken) (= ?l plate_target) (ChickenCooked))
                (ChickenOnPlate))
        )
    )
    
    ;; Close grill lid (requires constrained trajectory)
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
)
