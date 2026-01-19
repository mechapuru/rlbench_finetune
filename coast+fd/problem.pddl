;; PDDL Problem for LongHorizonGrillTask
;;
;; Initial State: Grill open, chicken on dish_rack, plate on dish_rack
;; Goal: Chicken cooked and served on plate at target location
;;
;; Required Sequence:
;; 1. pick(chicken) → place_chicken_on_grill
;; 2. close_lid
;; 3. pick(plate) → place_plate_at_target
;; 4. open_lid
;; 5. pick_chicken_from_grill → place_chicken_on_plate

(define (problem grill-task)
    (:domain long-horizon-grill)
    
    (:init
        ;; Initial object positions
        (On chicken dish_rack)      ; Chicken starts on dish rack
        (On plate dish_rack)        ; Plate starts on dish rack
        
        ;; Initial gripper state
        (HandEmpty)
        
        ;; Initial grill state - LID IS OPEN
        (GrillOpen)
        
        ;; Timestep initialization (for COAST constraint tracking)
        (AtTimestep t1)
        (Next t1 t2)
        (Next t2 t3)
        (Next t3 t4)
        (Next t4 t5)
        (Next t5 t6)
        (Next t6 t7)
        (Next t7 t8)
        (Next t8 t9)
        (Next t9 t10)
        (Next t10 t11)
        (Next t11 t12)
        (Next t12 t13)
        (Next t13 t14)
        (Next t14 t15)
        (Next t15 t16)
        (Next t16 t17)
        (Next t17 t18)
        (Next t18 t19)
        (Next t19 t20)
    )
    
    (:goal
        (and
            ;; Chicken must be cooked and on plate
            (ChickenOnPlate)
            (ChickenCooked)
            
            ;; Plate must be at target
            (PlateAtTarget)
            
            ;; Grill must be open (from task success conditions)
            (GrillOpen)
            
            ;; Nothing in hand
            (HandEmpty)
        )
    )
)
