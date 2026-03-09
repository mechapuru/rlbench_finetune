#!/bin/bash
mkdir -p baseline_plans
echo "Starting Baseline Experiment (100 Runs)..."
for i in {1..100}
do
   echo "----------------------------------------"
   echo "Running Episode $i..."
   python3 baseline_0_just_PDDL.py --episode_num $i --object soup
   echo "Episode $i Complete."
done
echo "Experiment Complete. Check baseline_logs.txt and baseline_plans/ folder."
