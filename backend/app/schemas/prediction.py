"""
Pydantic schemas for API requests and responses
"""

from pydantic import BaseModel, Field
from typing import List


class TopKPrediction(BaseModel):
    """Single prediction with confidence"""
    sign: str = Field(..., description="Sign name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class PredictionResponse(BaseModel):
    """Response model for prediction endpoints"""
    predicted_sign: str = Field(..., description="Most likely sign")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for top prediction")
    top_k: List[TopKPrediction] = Field(..., description="Top-K predictions with confidence scores")

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_sign": "привет",
                "confidence": 0.92,
                "top_k": [
                    {"sign": "привет", "confidence": 0.92},
                    {"sign": "здравствуйте", "confidence": 0.05},
                    {"sign": "добрый день", "confidence": 0.02},
                    {"sign": "до свидания", "confidence": 0.01},
                    {"sign": "пока", "confidence": 0.005}
                ]
            }
        }


class KeypointsRequest(BaseModel):
    """Request model for keypoints prediction"""
    keypoints: List[List[float]] = Field(..., description="Keypoints array of shape (num_frames, 225)")

    class Config:
        json_schema_extra = {
            "example": {
                "keypoints": [[0.1] * 225 for _ in range(60)]
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_type: str = Field(None, description="Type of loaded model")
