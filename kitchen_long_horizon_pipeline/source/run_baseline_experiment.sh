#!/bin/bash

# Create logs directory
mkdir -p baseline_logs

# Log file for the aggregate summary
SUMMARY_LOG="baseline_logs/aggregate_summary_$(date +%Y%m%d_%H%M%S).log"

echo "Starting Baseline 0 (Just PDDL) - 100 Runs" | tee -a "$SUMMARY_LOG"

SUCCESS_COUNT=0
TOTAL_RUNS=2

for i in $(seq 1 $TOTAL_RUNS); do
    echo "----------------------------------------------------------------" | tee -a "$SUMMARY_LOG"
    echo "Run #$i / $TOTAL_RUNS" | tee -a "$SUMMARY_LOG"
    
    # Run the python script for exactly 1 run
    # We capture stdout/stderr to a specific log file for this run
    RUN_LOG="baseline_logs/run_${i}.log"
    
    # Run with timeout to prevent hanging
    timeout 350s python3 baseline_0_just_PDDL.py --runs 1 --object soup > "$RUN_LOG" 2>&1
    EXIT_CODE=$?
    
    # Check if "Result Episode 1: SUCCESS" is in the log
    if grep -q "Result Episode 1: SUCCESS" "$RUN_LOG"; then
        echo "Result: SUCCESS" | tee -a "$SUMMARY_LOG"
        SUCCESS_COUNT=$((SUCCESS_COUNT+1))
    else
        echo "Result: FAILURE" | tee -a "$SUMMARY_LOG"
        # Extract reason if possible
        REASON=$(grep "Result Episode 1: FAILURE" "$RUN_LOG" | sed 's/.*FAILURE //')
        if [ -z "$REASON" ]; then
             # Check for timeout or crash
             if [ $EXIT_CODE -eq 124 ]; then
                 REASON="(Timeout)"
             else
                 REASON="(Crash/Unknown - Check $RUN_LOG)"
             fi
        fi
        echo "Reason: $REASON" | tee -a "$SUMMARY_LOG"
    fi
    
    # Optional: Kill any lingering CoppeliaSim processes if they get stuck
    # pkill -f coppelia
    
    # Small sleep to ensure cleanup
    sleep 2
done

echo "================================================================" | tee -a "$SUMMARY_LOG"
echo "Final Results:" | tee -a "$SUMMARY_LOG"
echo "Total Runs: $TOTAL_RUNS" | tee -a "$SUMMARY_LOG"
echo "Successes: $SUCCESS_COUNT" | tee -a "$SUMMARY_LOG"
echo "Success Rate: ${SUCCESS_COUNT}%" | tee -a "$SUMMARY_LOG"
