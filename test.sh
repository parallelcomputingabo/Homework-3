#!/bin/sh

# Remember to make file executable with chmod +x test.sh
# Usage: ./test.sh
# This script runs all CUDA test cases (0..9) using main.cu's executable.

declare -a CASE_DIMENSIONS
CASE_DIMENSIONS[0]="1024 1024 1024"
CASE_DIMENSIONS[1]="512 512 512"
CASE_DIMENSIONS[2]="256 256 256"
CASE_DIMENSIONS[3]="128 128 128"
CASE_DIMENSIONS[4]="64 64 64"
CASE_DIMENSIONS[5]="2048 2048 2048"
CASE_DIMENSIONS[6]="1024 2048 512"
CASE_DIMENSIONS[7]="512 1024 2048"
CASE_DIMENSIONS[8]="2048 512 1024"
CASE_DIMENSIONS[9]="4096 4096 4096"

for i in $(seq 0 9); do
    dims=${CASE_DIMENSIONS[$i]}
    echo "Running CUDA case $i with dims $dims"
    ./main $i $dims > data/$i/run_cuda.log 2>&1
    if grep -q "INVALID" data/$i/run_cuda.log; then
        echo "Case $i: FAIL (see data/$i/run_cuda.log)"
    else
        echo "Case $i: PASS"
    fi
done