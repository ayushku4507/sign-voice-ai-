import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="SignVoice API",
    version="0.1.0",
    description="Backend scaffold. No trained ISL recognition model is connected."
)

origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in origins if origin.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

class ModelStatus(BaseModel):
    available: bool
    model_name: str | None
    message: str

@app.get("/health")
def health():
    return {"status": "ok", "service": "signvoice-api"}

@app.get("/v1/model/status", response_model=ModelStatus)
def model_status():
    # Intentionally unavailable until a real, evaluated model is configured.
    return ModelStatus(
        available=False,
        model_name=None,
        message="No trained ISL model is configured. Predictions are disabled."
    )
