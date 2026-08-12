import os
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from .db import engine, Base
from .routers import recordings, predictions, severity, explanations
from .config import settings

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Tables are managed by Alembic
# Base.metadata.create_all(bind=engine)

app = FastAPI(title="ApneaGuard AI API")

# Add CORS middleware to allow the frontend to call the API
# IMPORTANT: This API serves physiological predictions. An explicit allow-list is critical.
# Never use wildcard ["*"] in production to prevent unauthorized cross-origin requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recordings.router, prefix="/recordings", tags=["Recordings"])
app.include_router(predictions.router, prefix="/predictions", tags=["Predictions"])
app.include_router(severity.router, prefix="/recordings", tags=["Severity"])
app.include_router(explanations.router, prefix="/predictions", tags=["Explanations"])

# Serve frontend static files
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    logger.warning("Frontend directory not found. Serving API only.")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up ApneaGuard AI API")
    from api.services.model_loader import load_all_models
    load_all_models()
