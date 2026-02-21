#!/bin/bash

# Unido-Africa Rice Model Export - Installation Script

if [ -z "$BASH_VERSION" ]; then
    exec bash "$0" "$@"
fi

set -e  # Exit on error

echo "=========================================="
echo "Unido-Africa Rice Model Export Requirements Installation Script"
echo "=========================================="
echo ""

# Step 1: Check if conda is installed
echo "Step 1: Checking if conda is installed..."
if command -v conda &> /dev/null; then
    echo "✓ Conda is already installed"
    conda_version=$(conda --version)
    echo "  Version: $conda_version"
else
    echo "✗ Conda is not installed. Installing Miniconda..."
    
    # Determine OS and download appropriate installer
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "  Detected Linux system"
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  Detected macOS system"
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
    else
        echo "  Unsupported OS. Please install conda manually: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    
    # Install miniconda
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
    
    # Initialize conda
    $HOME/miniconda3/bin/conda init
    
    echo "✓ Miniconda installed successfully!"
    echo "  Please run 'source ~/.bashrc' to activate conda in your current shell"
    source $HOME/miniconda3/bin/activate
fi

echo ""

# Step 2: Ask if user wants to create a new environment
echo "Step 2: Setting up Python environment..."
read -p "Do you want to create a new conda environment? (y/n) [default: y]: " create_env
create_env=${create_env:-y}

if [[ "$create_env" == "y" || "$create_env" == "Y" ]]; then
    read -p "Enter environment name [default: unido-app-model-export]: " env_name
    env_name=${env_name:-unido-app-model-export}
    
    echo "Creating environment '$env_name'..."
    conda create -y -n "$env_name" python=3.11
    echo "✓ Environment '$env_name' created successfully"
else
    echo "Skipping environment creation"
    read -p "Enter existing environment name to activate [default: base]: " env_name
    env_name=${env_name:-base}
fi

echo ""

# Step 3: Activate environment and install requirements
echo "Step 3: Installing requirements..."
echo "Activating conda environment '$env_name'..."

# Get the conda shell activation script
eval "$(conda shell.bash hook)"
conda activate "$env_name"

# Check if requirements.txt exists
if [ -f "RiceChallenge_final/requirements.txt" ]; then
    echo "Installing packages from requirements.txt..."
    pip install -r RiceChallenge_final/requirements.txt
    echo "✓ All packages installed successfully"
else
    echo "✗ requirements.txt not found in current directory"
    exit 1
fi

echo ""

# Step 4: Summary
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Environment name: $env_name"
echo ""
echo "To activate your environment in future sessions, run:"
echo "  conda activate $env_name"
echo ""
echo "To deactivate the environment, run:"
echo "  conda deactivate"
echo ""
echo "To start Jupyter Notebook, run:"
echo "  jupyter notebook"
echo ""
echo "Happy exploring! 🚀"
