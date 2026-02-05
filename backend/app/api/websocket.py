"""
WebSocket endpoint for real-time sign language recognition.
"""

import json
from collections import deque
from typing import Dict

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from ..ml.inference import SignLanguagePredictor


# Frame buffers for each connection (connection_id -> deque of frames)
frame_buffers: Dict[int, deque] = {}

# Number of frames to accumulate before prediction
BUFFER_SIZE = 60


async def websocket_endpoint(websocket: WebSocket, predictor: SignLanguagePredictor):
    """
    WebSocket endpoint for real-time gesture recognition.

    Protocol:
        Client sends: {"keypoints": [N float values]}
        Server responds:
            - Buffering: {"status": "buffering", "progress": 0.5}
            - Prediction: {"status": "prediction", "predicted_sign": "...", ...}
    """
    await websocket.accept()
    connection_id = id(websocket)
    frame_buffers[connection_id] = deque(maxlen=BUFFER_SIZE)

    print(f"Client {connection_id} connected")

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if "keypoints" not in message:
                await websocket.send_json({"error": "Missing keypoints field"})
                continue

            keypoints = message["keypoints"]
            expected_features = predictor.input_size

            if len(keypoints) != expected_features:
                await websocket.send_json(
                    {"error": f"Expected {expected_features} keypoints, got {len(keypoints)}"}
                )
                continue

            frame_buffers[connection_id].append(keypoints)

            if len(frame_buffers[connection_id]) < BUFFER_SIZE:
                await websocket.send_json(
                    {
                        "status": "buffering",
                        "progress": len(frame_buffers[connection_id]) / BUFFER_SIZE,
                    }
                )
                continue

            try:
                sequence = np.array(list(frame_buffers[connection_id]), dtype=np.float32)
                result = predictor.predict(sequence, top_k=5)

                await websocket.send_json(
                    {
                        "status": "prediction",
                        "predicted_sign": result["predicted_sign"],
                        "confidence": float(result["confidence"]),
                        "top_k": result["top_k"],
                    }
                )

                # Start fresh for next gesture.
                frame_buffers[connection_id].clear()

            except Exception as e:
                print(f"Prediction error: {e}")
                await websocket.send_json({"error": f"Prediction failed: {str(e)}"})

    except WebSocketDisconnect:
        if connection_id in frame_buffers:
            del frame_buffers[connection_id]
        print(f"Client {connection_id} disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        if connection_id in frame_buffers:
            del frame_buffers[connection_id]
        raise


def get_active_connections() -> int:
    """Get number of active WebSocket connections."""
    return len(frame_buffers)


def clear_connection(connection_id: int) -> None:
    """Clear buffer for a specific connection."""
    if connection_id in frame_buffers:
        frame_buffers[connection_id].clear()
