"""
Client example for connecting to the Cloud Server
Demonstrates both WebSocket connections (video + logs)
"""

import asyncio
import base64
import json
import cv2
import websockets
import argparse
from datetime import datetime


class CloudClient:
    def __init__(self, server_url="ws://localhost:8000"):
        self.server_url = server_url
        self.video_ws = None
        self.logs_ws = None
        
    async def connect_video(self):
        """Connect to video WebSocket"""
        try:
            self.video_ws = await websockets.connect(f"{self.server_url}/ws/video")
            print(f"Connected to video stream: {self.server_url}/ws/video")
            return True
        except Exception as e:
            print(f"Failed to connect to video stream: {e}")
            return False
    
    async def connect_logs(self):
        """Connect to logs WebSocket"""
        try:
            self.logs_ws = await websockets.connect(f"{self.server_url}/ws/logs")
            print(f"Connected to log stream: {self.server_url}/ws/logs")
            return True
        except Exception as e:
            print(f"Failed to connect to log stream: {e}")
            return False
    
    async def send_frame(self, frame):
        """Send a video frame to the server"""
        if not self.video_ws:
            return None
        
        try:
            # Resize frame for faster processing (optional, adjust as needed)
            # Uncomment next line to resize to 640x480 for better performance
            # frame = cv2.resize(frame, (640, 480))
            
            # Encode frame to JPEG with lower quality for faster transfer
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]  # 70% quality
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            # Convert to base64
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send frame
            await self.video_ws.send(json.dumps({
                "type": "frame",
                "frame": frame_base64,
                "timestamp": datetime.now().isoformat()
            }))
            
            # Wait for response
            response = await asyncio.wait_for(self.video_ws.recv(), timeout=5.0)
            return json.loads(response)
            
        except asyncio.TimeoutError:
            print("Timeout waiting for prediction")
            return None
        except Exception as e:
            print(f"Error sending frame: {e}")
            return None
    
    async def listen_logs(self):
        """Listen to log stream"""
        if not self.logs_ws:
            return
        
        try:
            async for message in self.logs_ws:
                log_data = json.loads(message)
                
                # Format log output
                timestamp = log_data.get("timestamp", "")
                level = log_data.get("level", "INFO")
                msg = log_data.get("message", "")
                
                # Color coding
                colors = {
                    "INFO": "\033[94m",      # Blue
                    "WARNING": "\033[93m",   # Yellow
                    "ERROR": "\033[91m",     # Red
                    "CRITICAL": "\033[95m",  # Magenta
                }
                reset = "\033[0m"
                
                color = colors.get(level, "")
                print(f"{color}[{level}] {timestamp} - {msg}{reset}")
                
                # Show additional data if present
                if "data" in log_data:
                    print(f"  └─ Data: {log_data['data']}")
                    
        except websockets.exceptions.ConnectionClosed:
            print("Log stream connection closed")
        except Exception as e:
            print(f"Error in log stream: {e}")
    
    async def stream_video(self, source=0):
        """Stream video from camera or file"""
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Cannot open video source: {source}")
            return
        
        # Set camera FPS (if it's a camera)
        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_FPS, 15)  # Limit to 15 FPS
        
        print(f"Starting video stream from: {source}")
        print("Press 'q' to quit, 'p' to pause")
        print("Processing every frame")
        
        paused = False
        frame_count = 0
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("Video stream ended")
                        break
                    
                    frame_count += 1
                    
                    # Send frame and get prediction (every frame)
                    result = await self.send_frame(frame)
                    
                    if result:
                        # Display prediction on frame
                        prediction = result.get("prediction", "Unknown")
                        confidence = result.get("confidence", 0.0)
                        accident = result.get("accident_detected", False)
                        
                        # Log every frame prediction
                        print(f"Frame {frame_count}: {prediction} (conf: {confidence:.2f})", end="")
                        if accident:
                            print(" - ACCIDENT DETECTED!")
                        else:
                            print()
                        
                        color = (0, 255, 0) if prediction == "Non-Accident" else (0, 0, 255)
                        text = f"{prediction} ({confidence:.2f})"
                        cv2.putText(frame, text, (20, 40),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        
                        if accident:
                            cv2.putText(frame, "ACCIDENT DETECTED!", (20, 80),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    
                    cv2.putText(frame, f"Frame: {frame_count}", (20, frame.shape[0] - 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow("Cloud Client - Video Stream", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print(f"{'Paused' if paused else 'Resumed'}")
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    async def close(self):
        """Close all connections"""
        if self.video_ws:
            await self.video_ws.close()
        if self.logs_ws:
            await self.logs_ws.close()
        print("Disconnected from server")


async def main():
    parser = argparse.ArgumentParser(description="Cloud Client for Road Accident Detection")
    parser.add_argument("--server", type=str, default="ws://localhost:8000",
                       help="Server URL (default: ws://localhost:8000)")
    parser.add_argument("--video", type=str, default=None,
                       help="Path to video file")
    parser.add_argument("--camera", type=int, default=None,
                       help="Camera ID (default: 0)")
    parser.add_argument("--logs-only", action="store_true",
                       help="Only connect to logs stream (no video)")
    
    args = parser.parse_args()
    
    client = CloudClient(server_url=args.server)
    
    # Connect to logs stream
    await client.connect_logs()
    
    if args.logs_only:
        # Only listen to logs
        print("Listening to logs only...")
        await client.listen_logs()
    else:
        # Connect to video stream
        await client.connect_video()
        
        # Determine video source
        source = args.camera if args.camera is not None else (args.video if args.video else 0)
        
        # Start both tasks
        video_task = asyncio.create_task(client.stream_video(source))
        logs_task = asyncio.create_task(client.listen_logs())
        
        # Wait for video to finish
        await video_task
        
        # Cancel logs task
        logs_task.cancel()
    
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())

