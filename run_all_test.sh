#!/bin/bash

# Run all test cases and collect results
echo "Running CUDA Matrix Multiplication Tests"
echo "========================================"

# Create results file with header
results_file="performance_results.csv"
echo "Test Case,Dimensions,Naive CUDA (s),Tiled CUDA (s),Tiled vs Naive Speedup,Status" > $results_file

# Check if executable exists
if [ ! -f "./matrix_mult" ]; then
    echo "Error: matrix_mult executable not found!"
    echo "Please build first with: cmake -DCMAKE_CUDA_COMPILER=nvcc . && make"
    exit 1
fi

# Check GPU availability
if ! nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not available. GPU may not be accessible."
fi

successful_tests=0
failed_tests=0

# Run each test case
for i in {0..9}; do
    echo ""
    echo "Running Test Case $i..."
    echo "========================"
    
    # Check if data files exist
    data_dir="data/$i"
    if [ ! -d "$data_dir" ]; then
        echo "Warning: Data directory $data_dir not found, skipping..."
        echo "$i,N/A,N/A,N/A,N/A,MISSING_DATA" >> $results_file
        ((failed_tests++))
        continue
    fi
    
    if [ ! -f "$data_dir/input0.raw" ] || [ ! -f "$data_dir/input1.raw" ]; then
        echo "Warning: Input files missing for test case $i, skipping..."
        echo "$i,N/A,N/A,N/A,N/A,MISSING_INPUT" >> $results_file
        ((failed_tests++))
        continue
    fi
    
    # Run the test with timeout
    timeout 300 ./matrix_mult $i > test_case_${i}.log 2>&1
    exit_code=$?
    
    if [ $exit_code -eq 124 ]; then
        echo "Test case $i timed out (>300s)"
        echo "$i,N/A,N/A,N/A,N/A,TIMEOUT" >> $results_file
        ((failed_tests++))
    elif [ $exit_code -ne 0 ]; then
        echo "Test case $i failed with exit code $exit_code"
        echo "Error output:"
        cat test_case_${i}.log | tail -5
        echo "$i,N/A,N/A,N/A,N/A,FAILED" >> $results_file
        ((failed_tests++))
    else
        echo "Test case $i completed successfully"
        # Extract CSV line and append to results
        csv_line=$(grep "^CSV:" test_case_${i}.log | cut -d' ' -f2-)
        if [ ! -z "$csv_line" ]; then
            echo "${csv_line},SUCCESS" >> $results_file
            ((successful_tests++))
        else
            echo "$i,N/A,N/A,N/A,N/A,NO_OUTPUT" >> $results_file
            ((failed_tests++))
        fi
    fi
done

echo ""
echo "============================================"
echo "All tests completed. Results saved to $results_file"
echo "Successful tests: $successful_tests"
echo "Failed tests: $failed_tests"
echo "============================================"
echo ""
echo "Performance Summary:"
echo "===================="
cat $results_file

echo ""
echo "Detailed logs available in test_case_*.log files"