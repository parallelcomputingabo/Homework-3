#!/bin/bash

# Make results directory if it doesn't exist
mkdir -p results

# Output file
OUTPUT_FILE="./results/results.txt"

# Clear previous results
echo "Test Results" > "$OUTPUT_FILE"
echo "=========================" >> "$OUTPUT_FILE"

# Loop through modes 0 to 9
for M in {0..9}
do
    # Run 5 times for each mode
    for R in {1..5}
    do
        echo "Run $R" >> "$OUTPUT_FILE"
        ./build/app "$M" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
    done

    echo "" >> "$OUTPUT_FILE"
done

echo "All runs complete."