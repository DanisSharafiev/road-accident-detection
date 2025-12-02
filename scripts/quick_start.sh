#!/bin/bash

# Quick Start Script for Road Accident Detection Cloud Service
# This script helps you quickly deploy and test the cloud service

set -e

echo "Road Accident Detection - Cloud Service Quick Start"
echo "========================================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "ERROR: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "Docker and Docker Compose are installed"
echo ""

# Check if model file exists
if [ ! -f "dict_models/vgg16_baseline.pth" ]; then
    echo "WARNING: Model file not found at dict_models/vgg16_baseline.pth"
    echo "   Please ensure you have the trained model before starting the server."
    read -p "   Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p logs
mkdir -p config/dashboards
echo "Directories created"
echo ""

# Start services
echo "Starting cloud services..."
echo "   This will start:"
echo "   - FastAPI Server (port 8000)"
echo "   - Loki (port 3100)"
echo "   - Promtail"
echo "   - Grafana (port 3000)"
echo ""

docker-compose up -d

echo ""
echo "Waiting for services to start..."
sleep 10

# Check if services are running
echo ""
echo "Checking service status..."
docker-compose ps

echo ""
echo "Services started successfully!"
echo ""
echo "Access points:"
echo "   - API Server: http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo "   - Grafana Dashboard: http://localhost:3000 (admin/admin)"
echo "   - Loki: http://localhost:3100"
echo ""
echo "📡 To connect a client:"
echo "   Camera: python src/cloud/client.py --camera 0"
echo "   Video:  python src/cloud/client.py --video your_video.mp4"
echo "   Logs:   python src/cloud/client.py --logs-only"
echo ""
echo "🛑 To stop services:"
echo "   docker-compose down"
echo ""
echo "📝 View logs:"
echo "   docker-compose logs -f"
echo ""
echo "Setup complete! Happy monitoring!"

