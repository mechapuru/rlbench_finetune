;; PDDL Problem for LongHorizonGrillTask
;; Goal: Put chicken on grill, move plate from rack to target, close grill

(define (problem grill-task)
    (:domain long-horizon-grill)
    
    (:objects
        ;; No additional objects needed - using constants from domain
    )
    
    (:init
        ;; Initial object positions
        (On chicken plate_source)
        (On steak plate_source)
        (On the_plate plate_source)
        
        ;; Initial gripper state
        (HandEmpty)
        
        ;; Initial grill state
        (GrillClosed)
        
        ;; Timestep initialization
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
            ;; Chicken must be cooked (on grill with lid closed)
            (OnGrill chicken)
            
            ;; Plate moved from source rack to target
            (On the_plate plate_target)
            
            ;; Chicken on the plate
            (OnPlate chicken)
            
            ;; Grill closed after cooking
            (GrillOpen)
            
            ;; Nothing in hand
            (HandEmpty)
        )
    )
)
