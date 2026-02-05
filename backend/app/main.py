"""
FastAPI application for sign language recognition
"""

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys
from pathlib import Path

from .config import settings
from .api import routes
from .api.websocket import websocket_endpoint
from .ml.inference import SignLanguagePredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup: Load model
    print("\n" + "="*60)
    print("Starting Sign Language Recognition API")
    print("="*60)

    try:
        app.state.predictor = SignLanguagePredictor(settings.model_path)
        print("\n✓ Model loaded successfully")
    except Exception as e:
        print(f"\n✗ Failed to load model: {e}")
        print(f"  Model path: {settings.model_path}")
        print(f"  Make sure the model file exists and is valid")
        app.state.predictor = None

    print("\n" + "="*60)
    print(f"API running at http://{settings.host}:{settings.port}")
    print(f"Documentation: http://{settings.host}:{settings.port}/docs")
    print("="*60 + "\n")

    yield

    # Shutdown
    print("\nShutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(routes.router, prefix="/api/v1", tags=["prediction"])


@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "Sign Language Recognition API",
        "version": settings.api_version,
        "docs": "/docs",
        "health": "/api/v1/health",
        "endpoints": {
            "predict": "/api/v1/predict/keypoints",
            "stats": "/api/v1/stats",
            "websocket": "/ws/predict"
        }
    }


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    """
    WebSocket endpoint for real-time sign language recognition

    Client sends: {"keypoints": [N float values]} where N matches model input_size (189 now)
    Server responds with predictions
    """
    if app.state.predictor is None:
        await websocket.close(code=1003, reason="Model not loaded")
        return

    await websocket_endpoint(websocket, app.state.predictor)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
