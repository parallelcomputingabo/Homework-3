#!/bin/bash

echo "Checking test data files..."
echo "=========================="

for i in {0..9}; do
    dir="data/$i"
    if [ ! -d "$dir" ]; then
        echo "Case $i: Missing directory"
        continue
    fi
    
    input0="$dir/input0.raw"
    input1="$dir/input1.raw"
    output="$dir/output.raw"
    
    if [ ! -f "$input0" ] || [ ! -f "$input1" ] || [ ! -f "$output" ]; then
        echo "Case $i: Missing files"
        continue
    fi
    
    # Get dimensions
    dim0=$(head -c 8 "$input0" | od -tu4 -N4 -An | xargs)
    dim1=$(head -c 8 "$input1" | od -tu4 -N4 -An | xargs)
    dim_out=$(head -c 8 "$output" | od -tu4 -N4 -An | xargs)
    
    echo "Case $i:"
    echo "  A: $dim0"
    echo "  B: $dim1"
    echo "  Output: $dim_out"
    
    # Check compatibility
    a_cols=$(head -c 8 "$input0" | od -tu4 -N4 -An -j4 | xargs)
    b_rows=$(head -c 8 "$input1" | od -tu4 -N4 -An | xargs)
    
    if [ "$a_cols" -eq "$b_rows" ]; then
        echo "  Compatible: Yes"
    else
        echo "  Compatible: No (A cols $a_cols != B rows $b_rows)"
    fi
done