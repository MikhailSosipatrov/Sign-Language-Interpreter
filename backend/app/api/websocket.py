"""
WebSocket endpoint for real-time sign language recognition
"""

from fastapi import WebSocket, WebSocketDisconnect
import json
import numpy as np
from collections import deque
from typing import Dict

from ..ml.inference import SignLanguagePredictor


# Frame buffers for each connection (connection_id -> deque of frames)
frame_buffers: Dict[int, deque] = {}

# Configuration
BUFFER_SIZE = 60  # Number of frames to accumulate before prediction
SEQUENCE_LENGTH = 60  # Model input sequence length


async def websocket_endpoint(websocket: WebSocket, predictor: SignLanguagePredictor):
    """
    WebSocket endpoint for real-time gesture recognition

    Protocol:
        Client sends: {"keypoints": [225 float values]}
        Server responds:
            - Buffering: {"status": "buffering", "progress": 0.5}
            - Prediction: {"predicted_sign": "Привет", "confidence": 0.95,
                          "top_k": [{"sign": "Привет", "confidence": 0.95}, ...]}

    Args:
        websocket: WebSocket connection
        predictor: ML model predictor instance
    """
    await websocket.accept()
    connection_id = id(websocket)
    frame_buffers[connection_id] = deque(maxlen=BUFFER_SIZE)

    print(f"Client {connection_id} connected")

    try:
        while True:
            # Receive keypoints from client
            data = await websocket.receive_text()
            message = json.loads(data)

            if 'keypoints' not in message:
                await websocket.send_json({
                    'error': 'Missing keypoints field'
                })
                continue

            keypoints = message['keypoints']

            # Validate keypoints
            if len(keypoints) != 225:
                await websocket.send_json({
                    'error': f'Expected 225 keypoints, got {len(keypoints)}'
                })
                continue

            # Add to buffer
            frame_buffers[connection_id].append(keypoints)

            # Check if buffer is full
            if len(frame_buffers[connection_id]) < BUFFER_SIZE:
                # Still buffering
                await websocket.send_json({
                    'status': 'buffering',
                    'progress': len(frame_buffers[connection_id]) / BUFFER_SIZE
                })
            else:
                # Buffer full - make prediction
                try:
                    # Convert buffer to numpy array
                    sequence = np.array(list(frame_buffers[connection_id]), dtype=np.float32)

                    # Make prediction
                    result = predictor.predict(sequence, top_k=5)

                    # Send result
                    await websocket.send_json({
                        'status': 'prediction',
                        'predicted_sign': result['predicted_class'],
                        'confidence': float(result['confidence']),
                        'top_k': [
                            {
                                'sign': pred['class_name'],
                                'confidence': float(pred['confidence'])
                            }
                            for pred in result['top_k_predictions']
                        ]
                    })

                    # Clear buffer (or keep sliding window)
                    # For now, clear to start fresh gesture
                    frame_buffers[connection_id].clear()

                except Exception as e:
                    print(f"Prediction error: {e}")
                    await websocket.send_json({
                        'error': f'Prediction failed: {str(e)}'
                    })

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
    """Get number of active WebSocket connections"""
    return len(frame_buffers)


def clear_connection(connection_id: int) -> None:
    """Clear buffer for a specific connection"""
    if connection_id in frame_buffers:
        frame_buffers[connection_id].clear()
