"""
Test script for WebSocket connections
"""

import asyncio
import websockets
import json
from datetime import datetime


async def test_logs_websocket():
    """Test logs WebSocket endpoint"""
    print("🔌 Connecting to logs WebSocket...")
    
    try:
        async with websockets.connect("ws://localhost:8000/ws/logs") as websocket:
            print("Connected to logs stream!")
            
            # Receive messages
            for i in range(10):
                message = await websocket.recv()
                data = json.loads(message)
                print(f"📨 [{data.get('level', 'INFO')}] {data.get('message', '')}")
                
                if i % 3 == 0:
                    # Send ping
                    await websocket.send(json.dumps({"type": "ping"}))
                    
    except Exception as e:
        print(f"ERROR: {e}")


async def test_video_websocket():
    """Test video WebSocket endpoint (without actual video)"""
    print("🔌 Connecting to video WebSocket...")
    
    try:
        async with websockets.connect("ws://localhost:8000/ws/video") as websocket:
            print("Connected to video stream!")
            
            # Send ping
            await websocket.send(json.dumps({"type": "ping"}))
            response = await websocket.recv()
            print(f"📨 Response: {response}")
            
    except Exception as e:
        print(f"ERROR: {e}")


async def main():
    print("🧪 Testing WebSocket connections...\n")
    
    # Test logs
    print("=" * 50)
    print("Testing Logs WebSocket")
    print("=" * 50)
    await test_logs_websocket()
    
    print("\n")
    
    # Test video
    print("=" * 50)
    print("Testing Video WebSocket")
    print("=" * 50)
    await test_video_websocket()
    
    print("\nTests completed!")


if __name__ == "__main__":
    asyncio.run(main())

