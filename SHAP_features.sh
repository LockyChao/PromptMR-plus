#!/bin/bash

# Loop through numbers 1 to 10
for i in {0..40}; do
    echo "Running register.sh with parameter: $i"
    sbatch job-calcWM.sh "$i" &
done

wait
echo "All jobs finished"