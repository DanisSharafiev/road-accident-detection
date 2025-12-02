#!/bin/bash

# Simple Training Script - Just train the model and start services
# Assumes datasets are already downloaded

set -e

cd "$(dirname "$0")/.."

echo "Road Accident Detection - Quick Train & Start"
echo "=================================================="
echo ""

# Check if datasets exist
if [ ! -d "datasets/1/data" ]; then
    echo "Dataset not found at datasets/1/data"
    echo "   Please download datasets first using the Jupyter notebook:"
    echo "   src/data/data_download.ipynb"
    exit 1
fi

echo "Datasets found"
echo ""

# Check if model already exists
if [ -f "dict_models/vgg16_baseline.pth" ]; then
    echo "Model already exists!"
    echo ""
    read -p "Do you want to retrain? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping training, starting services..."
        docker-compose restart accident-detection-server
        echo ""
        echo "Server restarted with existing model!"
        echo "   API: http://localhost:8000"
        echo "   Test: python src/cloud/client.py --camera 0"
        exit 0
    fi
fi

echo "Setting up virtual environment..."
echo ""

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing/updating dependencies..."
pip install -q --upgrade pip
pip install -q torch torchvision opencv-python numpy Pillow

echo "Dependencies ready"

echo ""
echo "Starting model training..."
echo "   This will take 10-30 minutes depending on your hardware"
echo "   Press Ctrl+C to cancel"
echo ""

# Run training
python -m src.models.model_train.baseline_vgg

# Check if model was created
if [ -f "models/vgg16_baseline.pth" ]; then
    # Move to dict_models
    mkdir -p dict_models
    cp models/vgg16_baseline.pth dict_models/
    echo ""
    echo "Model trained and copied to dict_models/"
    
    # Restart Docker service
    echo ""
    echo "Restarting Docker service with new model..."
    docker-compose restart accident-detection-server
    
    echo ""
    echo "="*80
    echo "Success!"
    echo "="*80
    echo ""
    echo "Model trained and server restarted!"
    echo ""
    echo "Access points:"
    echo "   - API Server: http://localhost:8000"
    echo "   - API Docs: http://localhost:8000/docs"
    echo "   - Grafana: http://localhost:3000 (admin/admin)"
    echo ""
    echo "Test with:"
    echo "   python src/cloud/client.py --camera 0"
    echo ""
else
    echo ""
    echo "Model file not found after training"
    echo "   Please check the errors above"
    exit 1
fi

# Deactivate venv
deactivate

