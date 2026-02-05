"""
Configuration settings for the FastAPI application
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings"""

    # API metadata
    api_title: str = "Sign Language Recognition API"
    api_description: str = "Real-time Russian Sign Language recognition using deep learning"
    api_version: str = "1.0.0"

    # Model settings
    model_path: str = "../training/models/lstm_test/best_model.pth"

    # CORS settings
    allowed_origins: List[str] = ["*"]  # In production, specify exact origins

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Create settings instance
settings = Settings()
