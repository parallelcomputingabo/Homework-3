#!/bin/sh

# Usage: ./test.sh
# This script runs all CUDA test cases (0..9) using main.cu's executable.
# It expects the following directory structure:
#   data/<case>/input0.raw, input1.raw, output.raw

# You must provide the correct dimensions for each test case below.
# Fill in the m, n, p values for each case as needed.
# Example: CASE_DIMS[0]="1024 1024 1024"
declare -a CASE_DIMS
CASE_DIMS[0]="1024 1024 1024"
CASE_DIMS[1]="512 512 512"
CASE_DIMS[2]="256 256 256"
CASE_DIMS[3]="128 128 128"
CASE_DIMS[4]="64 64 64"
CASE_DIMS[5]="2048 2048 2048"
CASE_DIMS[6]="1024 2048 512"
CASE_DIMS[7]="512 1024 2048"
CASE_DIMS[8]="2048 512 1024"
CASE_DIMS[9]="4096 4096 4096"

for i in $(seq 0 9); do
    dims=${CASE_DIMS[$i]}
    echo "Running CUDA case $i with dims $dims"
    ./main $i $dims > data/$i/run_cuda.log 2>&1
    if grep -q "INVALID" data/$i/run_cuda.log; then
        echo "Case $i: FAIL (see data/$i/run_cuda.log)"
    else
        echo "Case $i: PASS"
    fi
done