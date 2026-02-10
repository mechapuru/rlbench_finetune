#!/bin/bash
# =============================================================================
# Run experiments with multiple seeds for benchmarking
# =============================================================================
#
# Usage:
#   ./run_benchmark.sh                  # Run both experiments, seeds 0-9
#   ./run_benchmark.sh --exp1           # Run only Experiment 1
#   ./run_benchmark.sh --exp2           # Run only Experiment 2
#   ./run_benchmark.sh --seeds 5        # Run seeds 0-4 only
#
# Results saved to:
#   experiments/results/exp1_pure_pddl/seed_XX_*.json
#   experiments/results/exp2_coast/seed_XX_*.json
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults
NUM_SEEDS=10
RUN_EXP1=true
RUN_EXP2=true
OBJECT="soup"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp1)
            RUN_EXP2=false
            shift
            ;;
        --exp2)
            RUN_EXP1=false
            shift
            ;;
        --seeds)
            NUM_SEEDS="$2"
            shift 2
            ;;
        --object)
            OBJECT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "TAMP Benchmark: PDDLStream vs COAST"
echo "=============================================="
echo "Seeds: 0 to $((NUM_SEEDS-1))"
echo "Object: $OBJECT"
echo "Run Exp1 (Pure): $RUN_EXP1"
echo "Run Exp2 (COAST): $RUN_EXP2"
echo "=============================================="

# Create results directories
mkdir -p results/exp1_pure_pddl
mkdir -p results/exp2_coast

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Run experiments for each seed
for seed in $(seq 0 $((NUM_SEEDS-1))); do
    echo ""
    echo "========================================"
    echo "SEED $seed / $((NUM_SEEDS-1))"
    echo "========================================"
    
    if [ "$RUN_EXP1" = true ]; then
        echo "--- Running Exp1 (Pure PDDLStream) with seed $seed ---"
        python exp1_pure_pddl.py --object $OBJECT --seed $seed 2>&1 | tee -a results/exp1_run_$TIMESTAMP.log
    fi
    
    if [ "$RUN_EXP2" = true ]; then
        echo "--- Running Exp2 (COAST) with seed $seed ---"
        python exp2_coast.py --object $OBJECT --seed $seed 2>&1 | tee -a results/exp2_run_$TIMESTAMP.log
    fi
done

echo ""
echo "=============================================="
echo "Benchmark Complete!"
echo "=============================================="

# Generate comparison report
python compare_results.py 2>/dev/null || echo "Run compare_results.py manually"
