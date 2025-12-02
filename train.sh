#!/bin/bash

# Simple one-command training script
# Usage: ./train.sh

set -e

cd "$(dirname "$0")"

echo "Road Accident Detection - Quick Training"
echo ""

# Check datasets
if [ ! -d "datasets/1/data" ]; then
    echo "Dataset not found at datasets/1/data"
    echo "   Please download datasets first"
    exit 1
fi

echo "Datasets found"
echo ""

# Check if model exists
if [ -f "dict_models/vgg16_baseline.pth" ]; then
    echo "Model already exists: dict_models/vgg16_baseline.pth"
    read -p "Do you want to retrain? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Using existing model"
        exit 0
    fi
fi

# Setup venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies (this may take a minute)..."
pip install -q --upgrade pip
pip install -q torch torchvision opencv-python numpy Pillow 2>/dev/null || {
    echo "Installing with verbose output..."
    pip install torch torchvision opencv-python numpy Pillow
}

echo ""
echo "Dependencies installed"
echo ""
echo "Starting model training..."
echo "   • Device: CPU"
echo "   • Epochs: 10"
echo "   • Time: ~15-30 minutes"
echo "   • You can monitor progress below"
echo ""
echo "Press Ctrl+C to cancel"
echo ""
sleep 2

# Run training
python -m src.models.model_train.baseline_vgg

# Check if model was created
if [ -f "models/vgg16_baseline.pth" ]; then
    echo ""
    echo "Training completed!"
    echo ""
    
    # Move model to dict_models
    mkdir -p dict_models
    cp models/vgg16_baseline.pth dict_models/
    echo "Model copied to dict_models/vgg16_baseline.pth"
    
    # Restart Docker if running
    if docker ps | grep -q accident-detection-server; then
        echo ""
        echo "Restarting Docker service..."
        docker-compose restart accident-detection-server
        echo "Server restarted with new model"
    fi
    
    echo ""
    echo "   python src/cloud/client.py --camera 0"
    echo ""
else
    echo ""
    echo "Model file not found after training"
    echo "Check the output above for errors"
    exit 1
fi

deactivate

