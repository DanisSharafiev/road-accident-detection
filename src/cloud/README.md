# Cloud Service for Road Accident Detection
Cloud-based solution for real-time road accident detection with WebSocket and REST API support. The service provides scalable infrastructure for video stream processing, real-time logging, and alert management with monitoring capabilities through Grafana dashboards.

# Architecture Overview
The cloud service implements a microservices architecture with the following components:
- **FastAPI Server**: Handles video processing and inference with dual WebSocket connections
- **Loki**: Log aggregation system for centralized logging
- **Promtail**: Log shipper that collects and forwards logs to Loki
- **Grafana**: Visualization platform with custom dashboards for monitoring

# Key Features
- **Dual WebSocket Connections**: Separate channels for video streaming (`/ws/video`) and real-time log transmission (`/ws/logs`)
- **REST API**: Complete alert management system with endpoints for creating, retrieving, and managing alerts
- **Real-time Processing**: Asynchronous video processing with low latency (<100ms)
- **Smart Detection**: Queue-based filtering to reduce false positives with noise detection
- **Scalability**: Docker-based deployment supporting multiple concurrent clients
- **Monitoring Stack**: Integrated Loki + Grafana for comprehensive observability

# How to Launch

## Option 1: Docker Compose (Recommended)
For production deployment with full monitoring stack:
```bash
# Start all services (API server, Loki, Promtail, Grafana)
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

Services will be available at:
- **API Server**: http://localhost:8000
- **Grafana Dashboard**: http://localhost:3000 (credentials: admin/admin)
- **Loki API**: http://localhost:3100

## Option 2: Local Development
For development and testing:
```bash
# Install dependencies
pip install -r requirements.txt

# Run the server with hot reload
python -m uvicorn src.cloud.server:app --host 0.0.0.0 --port 8000 --reload
```

## Client Usage

### Video Stream Connection
Connect camera or video file to the cloud service:
```bash
# Real-time camera stream
python src/cloud/client.py --camera 0

# Video file processing
python src/cloud/client.py --video path/to/video.mp4

# Connect to remote server
python src/cloud/client.py --server ws://your-server:8000 --camera 0
```

### Log Stream Only
Monitor logs without video processing:
```bash
python src/cloud/client.py --logs-only
```

# API Endpoints

## WebSocket Endpoints

### `/ws/video` - Video Stream
Handles real-time video frame processing and inference.

**Send frame** (Client → Server):
```json
{
  "type": "frame",
  "frame": "base64_encoded_jpeg_image",
  "timestamp": "2024-01-01T12:00:00"
}
```

**Receive prediction** (Server → Client):
```json
{
  "timestamp": "2024-01-01T12:00:00",
  "prediction": "Accident",
  "confidence": 0.95,
  "accident_detected": true,
  "noise_detected": false
}
```

### `/ws/logs` - Log Stream
Real-time log transmission for monitoring and debugging.

**Receive logs** (Server → Client):
```json
{
  "timestamp": "2024-01-01T12:00:00",
  "level": "CRITICAL",
  "message": "ACCIDENT DETECTED!",
  "service": "accident-detection",
  "confidence": 0.95
}
```

## REST API Endpoints

### `GET /health`
Health check endpoint for service monitoring.
```bash
curl http://localhost:8000/health
```

### `POST /api/alerts`
Create a new alert manually or from external systems.
```bash
curl -X POST http://localhost:8000/api/alerts \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2024-01-01T12:00:00",
    "severity": "critical",
    "message": "Accident detected",
    "confidence": 0.95
  }'
```

### `GET /api/alerts`
Retrieve alerts with optional filtering.
```bash
# Get all recent alerts
curl http://localhost:8000/api/alerts

# Filter by severity
curl http://localhost:8000/api/alerts?severity=critical&limit=50
```

### `GET /api/stats`
Get service statistics and metrics.
```bash
curl http://localhost:8000/api/stats
```

# Monitoring with Grafana

## Dashboard Access
1. Open http://localhost:3000 in your browser
2. Login with credentials: `admin` / `admin`
3. Navigate to Dashboards → "Road Accident Detection Dashboard"

## Dashboard Components
- **Accident Detection Rate**: Time-series graph showing critical and noisy accident detections
- **Event Counters**: Real-time gauges for critical accidents, total events, and service status
- **Events by Level**: Bar chart showing log distribution by severity level
- **Real-time Logs**: Live log stream with filtering capabilities
- **Critical Events**: Dedicated panel for accident alerts

# Configuration Files

## Loki Configuration
File: `config/loki-config.yml`
- Log retention period: 31 days
- Ingestion rate limit: 10 MB/s
- Storage: Local filesystem with BoltDB shipper

## Promtail Configuration
File: `config/promtail-config.yml`
- Automatically parses JSON-formatted logs
- Extracts timestamp, level, and message fields
- Forwards to Loki with proper labels

## Grafana Configuration
- Datasources: `config/grafana-datasources.yml` (auto-provisions Loki)
- Dashboard provisioning: `config/grafana-dashboards.yml`
- Custom dashboard: `config/dashboards/accident-detection-dashboard.json`

# System Architecture

```
┌─────────────┐
│   Client    │
│  (Camera/   │
│   Video)    │
└──────┬──────┘
       │ WebSocket /ws/video
       │ (base64 frames)
       ▼
┌─────────────────────────────┐
│   FastAPI Server            │
│   - Video processing        │
│   - Inference engine        │
│   - Alert management        │
└──────┬──────────────┬───────┘
       │              │
       │ Logs         │ REST API
       │              │
       ▼              ▼
┌─────────────┐  ┌─────────────┐
│   Promtail  │  │   Clients   │
│             │  │  (Alerts)   │
└──────┬──────┘  └─────────────┘
       │
       │ Push logs
       ▼
┌─────────────┐
│    Loki     │
│  (Storage)  │
└──────┬──────┘
       │
       │ Query logs
       ▼
┌─────────────┐
│   Grafana   │
│ (Dashboard) │
└─────────────┘
       ▲
       │ WebSocket /ws/logs
       │
┌──────┴──────┐
│   Client    │
│   (Logs)    │
└─────────────┘
```

# Technical Implementation

## Python Client API
Example usage of the cloud client:
```python
import asyncio
from src.cloud.client import CloudClient

async def main():
    # Initialize client
    client = CloudClient(server_url="ws://localhost:8000")
    
    # Connect to both WebSocket endpoints
    await client.connect_video()
    await client.connect_logs()
    
    # Stream video (camera or file)
    await client.stream_video(source=0)
    
    # Cleanup
    await client.close()

asyncio.run(main())
```

## Log Format
All logs are written in JSON format for Loki compatibility:
```json
{
  "time": "2024-01-01T12:00:00",
  "level": "INFO",
  "message": "Prediction: Accident (conf: 0.95)",
  "service": "accident-detection"
}
```

**Log Levels**:
- `INFO`: Normal operational events
- `WARNING`: Warning conditions
- `ERROR`: Error events
- `CRITICAL`: Critical events (accidents detected)

# Production Deployment

## Environment Variables
Create a `.env` file for production configuration:
```bash
MODEL_PATH=dict_models/vgg16_baseline.pth
LOG_LEVEL=INFO
MAX_CONNECTIONS=100
```

## Nginx Reverse Proxy
Example configuration for production deployment:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

# Development

## Running in Development Mode
```bash
# Start server with hot reload
uvicorn src.cloud.server:app --reload --host 0.0.0.0 --port 8000

# Start only monitoring stack
docker-compose up loki promtail grafana
```

## Testing WebSocket Connections
```python
import asyncio
import websockets
import json

async def test_logs():
    async with websockets.connect("ws://localhost:8000/ws/logs") as ws:
        async for message in ws:
            print(json.loads(message))

asyncio.run(test_logs())
```

# Contributors
Name Surname|Mail|Position
-|-|-
Danis Sharafiev|d.sharafiev@innopolis.university|ML Engineer
Anton Korotkov|a.korotkov@innopolis.university|Data Scientist
Alex Kachmazov|a.kachmazov@innopolis.university|ML Engineer
Nikita Shiyanov|n.shiyanov@innopolis.university|ML Engineer


