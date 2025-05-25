#!/bin/bash

# Script to run all test cases and collect results

# Check for CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo "CUDA compiler (nvcc) not found. Please ensure CUDA toolkit is installed and in your PATH."
    exit 1
fi

# Build the application
echo "Building application with CMake..."
cmake -DCMAKE_CUDA_COMPILER=nvcc .
make

# Run tests for all 10 cases
for case in {0..9}; do
    echo "Running case $case..."
    
    # Create directory for results if it doesn't exist
    mkdir -p "data/$case"
    
    # Run the CUDA implementation
    ./app $case | tee "data/$case/output_log.txt"
    
    # Extract timing information
    naive_cuda_time=$(grep "Naive CUDA time" "data/$case/output_log.txt" | awk '{print $4}')
    tiled_cuda_time=$(grep "Tiled CUDA time" "data/$case/output_log.txt" | awk '{print $4}')
    
    # Store timing results
    echo "$naive_cuda_time $tiled_cuda_time" > "data/$case/cuda_results.txt"
    
    echo "Case $case completed."
    echo "------------------------------"
done

# Build and run the comparison tool
make compare_results
./compare_results > performance_table.txt

echo "All tests completed. Performance table generated in 'performance_table.txt'."
echo "You can copy this table to your README.md for submission."
