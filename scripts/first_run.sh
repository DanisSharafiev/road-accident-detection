#!/bin/bash

# First Run Script - Automatic Setup and Launch
# This script will download datasets, train model, and start all services

set -e

echo "Road Accident Detection - First Run Setup"
echo "=============================================="
echo ""

# Check if model exists
if [ -f "dict_models/vgg16_baseline.pth" ]; then
    echo "Model already exists!"
    echo ""
    read -p "Do you want to retrain? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping training, starting services..."
        docker-compose up -d
        exit 0
    fi
fi

echo "This script will:"
echo "   1. Install Python dependencies"
echo "   2. Download datasets from Kaggle and Google Drive"
echo "   3. Train the VGG16 model (~10-30 minutes)"
echo "   4. Start all Docker services"
echo ""

# Check for Kaggle credentials
if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
    echo "Kaggle credentials not found in environment"
    echo ""
    echo "To download datasets automatically, you need Kaggle API credentials:"
    echo "   1. Go to https://www.kaggle.com/settings"
    echo "   2. Click 'Create New API Token'"
    echo "   3. Extract kaggle.json and set environment variables:"
    echo "      export KAGGLE_USERNAME=your_username"
    echo "      export KAGGLE_KEY=your_key"
    echo ""
    read -p "Continue without Kaggle? (manual dataset download required) (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
read -p "Ready to start? (Y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Starting setup..."
echo ""

# Setup virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q torch torchvision opencv-python numpy Pillow

echo ""
echo "Starting model training..."
echo "   This will take 10-30 minutes on CPU"
echo ""

# Run training
python -m src.models.model_train.baseline_vgg

# Deactivate venv
deactivate

# Check if model was created
if [ -f "dict_models/vgg16_baseline.pth" ]; then
    echo ""
    echo "="*80
    echo "Setup Complete!"
    echo "="*80
    echo ""
    echo "Starting Docker services..."
    docker-compose up -d
    
    echo ""
    echo "All services are running!"
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
    echo "Model file not found. Please check the errors above."
    echo "   You can try running the setup script manually:"
    echo "   python3 scripts/setup_and_train.py"
    echo ""
    exit 1
fi

