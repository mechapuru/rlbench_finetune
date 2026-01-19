;; PDDL Problem for LongHorizonGrillTask
;;
;; Initial State: Grill open, chicken on grill_side, plate on dish_rack
;; Goal: Chicken cooked and served on plate at target location
;;
;; Required Sequence:
;; 1. pick(chicken, grill_side) → place(chicken, grill_cooking_area)
;; 2. close_lid
;; 3. pick(plate, dish_rack) → place(plate, plate_target)
;; 4. open_lid
;; 5. pick(chicken, grill_cooking_area) → place(chicken, plate_target)

(define (problem grill-task)
    (:domain long-horizon-grill)
    
    (:init
        ;; Initial object positions
        (On chicken grill_side)     ; Chicken starts on grill side (not cooking area)
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
