#!/bin/bash

# Setup script for CUDA Matrix Multiplication Assignment
echo "Setting up CUDA Matrix Multiplication Assignment..."

# Create directory structure if it doesn't exist
echo "Creating directory structure..."
mkdir -p data

# Make scripts executable
echo "Setting up permissions..."
chmod +x run_all_tests.sh
chmod +x submit_job.sh

# Check if we're on CSC Mahti
if [[ $(hostname) == *"mahti"* ]]; then
    echo "Detected CSC Mahti environment"
    
    # Load modules
    echo "Loading required modules..."
    module purge
    module load gcc/11.3.0
    module load cuda/11.7.0
    module load cmake/3.24.2
    
    # Display module list
    echo "Loaded modules:"
    module list
    
    echo ""
    echo "Environment setup complete for CSC Mahti!"
    echo ""
    echo "Next steps:"
    echo "1. Ensure your data directory contains test cases 0-9"
    echo "2. Edit submit_job.sh to use your project account"
    echo "3. Submit job: sbatch submit_job.sh"
    echo "   OR build locally: cmake -DCMAKE_CUDA_COMPILER=nvcc . && make"
    
else
    echo "Not on CSC Mahti - manual module loading may be required"
    echo ""
    echo "To build manually:"
    echo "1. Load CUDA and CMake modules"
    echo "2. Run: cmake -DCMAKE_CUDA_COMPILER=nvcc ."
    echo "3. Run: make"
    echo "4. Test: ./matrix_mult 0"
fi

echo ""
echo "Directory structure:"
ls -la

echo ""
echo "Setup complete!"