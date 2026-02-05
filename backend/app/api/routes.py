"""
API routes for sign language recognition
"""

from fastapi import APIRouter, HTTPException, Request
import numpy as np

from ..schemas.prediction import PredictionResponse, KeypointsRequest, HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """
    Health check endpoint

    Returns service status and model information
    """
    predictor = getattr(request.app.state, 'predictor', None)

    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        model_type=predictor.model_type if predictor else None
    )


@router.post("/predict/keypoints", response_model=PredictionResponse)
async def predict_from_keypoints(
    data: KeypointsRequest,
    request: Request
):
    """
    Predict sign from keypoints

    Accepts keypoints array and returns predicted sign with confidence scores.

    Args:
        data: KeypointsRequest with keypoints array of shape (num_frames, input_size)

    Returns:
        PredictionResponse with predicted sign and top-K results

    Example:
        ```json
        {
            "keypoints": [[0.1, 0.2, ...], [0.15, 0.22, ...], ...]
        }
        ```
    """
    try:
        # Get predictor from app state
        predictor = request.app.state.predictor

        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Convert to numpy array
        keypoints_array = np.array(data.keypoints, dtype=np.float32)

        # Validate shape
        if keypoints_array.ndim != 2:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid keypoints shape. Expected 2D array, got {keypoints_array.ndim}D"
            )

        expected_features = predictor.input_size
        if keypoints_array.shape[1] != expected_features:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid number of features. "
                    f"Expected {expected_features}, got {keypoints_array.shape[1]}"
                )
            )

        # Make prediction
        result = predictor.predict(keypoints_array)

        return PredictionResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.get("/stats")
async def get_stats(request: Request):
    """
    Get model statistics

    Returns information about the loaded model
    """
    predictor = getattr(request.app.state, 'predictor', None)

    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": predictor.model_type,
        "num_classes": len(predictor.idx_to_class),
        "sequence_length": predictor.sequence_length,
        "device": str(predictor.device),
        "input_size": predictor.input_size
    }
