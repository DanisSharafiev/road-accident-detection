"""
Cloud Server for Road Accident Detection
Supports:
- WebSocket 1: Video/Camera stream input
- WebSocket 2: Real-time logs output
- REST API: Alerts endpoint
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import cv2
import numpy as np
import torch
import base64

from src.realtime.video_inference import VideoInference
from src.realtime.q import Queue


# Logging configuration for Loki
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s", "level":"%(levelname)s", "message":"%(message)s", "service":"accident-detection"}',
    handlers=[
        logging.FileHandler("logs/cloud_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Road Accident Detection Cloud Service")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models
class Alert(BaseModel):
    timestamp: str
    severity: str  # "low", "medium", "high", "critical"
    message: str
    confidence: float
    metadata: Optional[Dict] = None


class AlertResponse(BaseModel):
    alert_id: str
    status: str
    received_at: str


# Global state
class ConnectionManager:
    def __init__(self):
        self.video_connections: List[WebSocket] = []
        self.log_connections: List[WebSocket] = []
        self.alerts: List[Alert] = []
        self.inference_engine: Optional[VideoInference] = None
        
    async def connect_video(self, websocket: WebSocket):
        await websocket.accept()
        self.video_connections.append(websocket)
        logger.info(f"Video client connected. Total: {len(self.video_connections)}")
        
    async def connect_logs(self, websocket: WebSocket):
        await websocket.accept()
        self.log_connections.append(websocket)
        logger.info(f"Log client connected. Total: {len(self.log_connections)}")
        
    def disconnect_video(self, websocket: WebSocket):
        if websocket in self.video_connections:
            self.video_connections.remove(websocket)
        logger.info(f"Video client disconnected. Total: {len(self.video_connections)}")
        
    def disconnect_logs(self, websocket: WebSocket):
        if websocket in self.log_connections:
            self.log_connections.remove(websocket)
        logger.info(f"Log client disconnected. Total: {len(self.log_connections)}")
    
    async def broadcast_log(self, log_data: dict):
        """Send logs to all connected log clients"""
        disconnected = []
        for connection in self.log_connections:
            try:
                await connection.send_json(log_data)
            except Exception as e:
                logger.error(f"Error broadcasting log: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect_logs(conn)
    
    def add_alert(self, alert: Alert):
        self.alerts.append(alert)
        # Keep only last 1000 alerts
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]


manager = ConnectionManager()


@app.on_event("startup")
async def startup_event():
    """Initialize the inference engine on startup"""
    try:
        manager.inference_engine = VideoInference(model_path="dict_models/vgg16_baseline.pth")
        logger.info("Inference engine initialized successfully")
        await manager.broadcast_log({
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "message": "Server started and inference engine initialized",
            "service": "accident-detection"
        })
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")


@app.get("/")
async def root():
    return {
        "service": "Road Accident Detection Cloud Service",
        "version": "1.0.0",
        "endpoints": {
            "video_ws": "/ws/video",
            "logs_ws": "/ws/logs",
            "alerts_rest": "/api/alerts",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "video_connections": len(manager.video_connections),
        "log_connections": len(manager.log_connections),
        "total_alerts": len(manager.alerts),
        "inference_ready": manager.inference_engine is not None
    }


@app.websocket("/ws/video")
async def websocket_video_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for video/camera stream
    Expects base64 encoded frames
    """
    await manager.connect_video(websocket)
    
    try:
        await manager.broadcast_log({
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "message": "New video stream connected",
            "service": "accident-detection"
        })
        
        while True:
            # Receive frame data
            data = await websocket.receive_json()
            
            if data.get("type") == "frame":
                # Decode base64 frame
                frame_data = data.get("frame")
                if not frame_data:
                    continue
                
                try:
                    # Decode base64 to numpy array
                    img_bytes = base64.b64decode(frame_data)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        await manager.broadcast_log({
                            "timestamp": datetime.now().isoformat(),
                            "level": "WARNING",
                            "message": "Failed to decode frame",
                            "service": "accident-detection"
                        })
                        continue
                    
                    # Run inference
                    if manager.inference_engine:
                        label, confidence = manager.inference_engine.predict(frame)
                        class_name = manager.inference_engine.class_names[label]
                        
                        # Check for accident with queue
                        accident, noise = manager.inference_engine.queue.check(
                            confidence if label == 1 else 1 - confidence
                        )
                        
                        # Prepare response
                        response = {
                            "timestamp": datetime.now().isoformat(),
                            "prediction": class_name,
                            "confidence": float(confidence),
                            "accident_detected": accident,
                            "noise_detected": noise
                        }
                        
                        # Send prediction back
                        await websocket.send_json(response)
                        
                        # Log prediction
                        await manager.broadcast_log({
                            "timestamp": datetime.now().isoformat(),
                            "level": "INFO" if not accident else "WARNING",
                            "message": f"Prediction: {class_name} (conf: {confidence:.2f})",
                            "service": "accident-detection",
                            "data": response
                        })
                        
                        # If accident detected, create alert
                        if accident and not noise:
                            alert = Alert(
                                timestamp=datetime.now().isoformat(),
                                severity="critical",
                                message="ACCIDENT DETECTED!",
                                confidence=float(confidence),
                                metadata={"noise": False}
                            )
                            manager.add_alert(alert)
                            
                            await manager.broadcast_log({
                                "timestamp": datetime.now().isoformat(),
                                "level": "CRITICAL",
                                "message": "ACCIDENT DETECTED!",
                                "service": "accident-detection",
                                "confidence": float(confidence)
                            })
                        
                        elif accident and noise:
                            alert = Alert(
                                timestamp=datetime.now().isoformat(),
                                severity="high",
                                message="ACCIDENT DETECTED WITH NOISE",
                                confidence=float(confidence),
                                metadata={"noise": True}
                            )
                            manager.add_alert(alert)
                            
                            await manager.broadcast_log({
                                "timestamp": datetime.now().isoformat(),
                                "level": "ERROR",
                                "message": "ACCIDENT DETECTED (with noise)",
                                "service": "accident-detection",
                                "confidence": float(confidence)
                            })
                    
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    await manager.broadcast_log({
                        "timestamp": datetime.now().isoformat(),
                        "level": "ERROR",
                        "message": f"Error processing frame: {str(e)}",
                        "service": "accident-detection"
                    })
            
            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        manager.disconnect_video(websocket)
        await manager.broadcast_log({
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "message": "Video stream disconnected",
            "service": "accident-detection"
        })
    except Exception as e:
        logger.error(f"WebSocket video error: {e}")
        manager.disconnect_video(websocket)


@app.websocket("/ws/logs")
async def websocket_logs_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time log streaming
    Clients connect here to receive logs
    """
    await manager.connect_logs(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "message": "Connected to log stream",
            "service": "accident-detection"
        })
        
        # Keep connection alive
        while True:
            # Wait for client messages (ping/pong)
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({
                    "type": "keepalive",
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        manager.disconnect_logs(websocket)
    except Exception as e:
        logger.error(f"WebSocket logs error: {e}")
        manager.disconnect_logs(websocket)


# REST API for Alerts
@app.post("/api/alerts", response_model=AlertResponse)
async def create_alert(alert: Alert):
    """
    REST API endpoint to manually create alerts
    """
    manager.add_alert(alert)
    
    alert_id = f"alert_{len(manager.alerts)}_{datetime.now().timestamp()}"
    
    # Broadcast alert as log
    await manager.broadcast_log({
        "timestamp": alert.timestamp,
        "level": alert.severity.upper(),
        "message": alert.message,
        "service": "accident-detection",
        "confidence": alert.confidence,
        "metadata": alert.metadata
    })
    
    logger.info(f"Alert created: {alert.message}")
    
    return AlertResponse(
        alert_id=alert_id,
        status="received",
        received_at=datetime.now().isoformat()
    )


@app.get("/api/alerts")
async def get_alerts(limit: int = 100, severity: Optional[str] = None):
    """
    Get recent alerts
    """
    alerts = manager.alerts[-limit:]
    
    if severity:
        alerts = [a for a in alerts if a.severity == severity]
    
    return {
        "total": len(alerts),
        "alerts": alerts
    }


@app.delete("/api/alerts")
async def clear_alerts():
    """
    Clear all alerts
    """
    count = len(manager.alerts)
    manager.alerts.clear()
    
    await manager.broadcast_log({
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "message": f"Cleared {count} alerts",
        "service": "accident-detection"
    })
    
    return {"status": "cleared", "count": count}


@app.get("/api/stats")
async def get_stats():
    """
    Get service statistics
    """
    return {
        "video_connections": len(manager.video_connections),
        "log_connections": len(manager.log_connections),
        "total_alerts": len(manager.alerts),
        "alerts_by_severity": {
            "critical": len([a for a in manager.alerts if a.severity == "critical"]),
            "high": len([a for a in manager.alerts if a.severity == "high"]),
            "medium": len([a for a in manager.alerts if a.severity == "medium"]),
            "low": len([a for a in manager.alerts if a.severity == "low"]),
        },
        "inference_ready": manager.inference_engine is not None
    }


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

