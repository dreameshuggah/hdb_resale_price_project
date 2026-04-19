"""
server.py — FastAPI inference server for HDB Resale Price Prediction.

Endpoints
---------
GET  /              → health check / welcome message
GET  /metadata      → returns valid categorical options from rank maps
POST /predict       → predicts resale price from input features

Usage
-----
    Development:
        uvicorn server:app --reload --host 0.0.0.0 --port 8000

    Production (High Concurrency):
        gunicorn -w 4 -k uvicorn.workers.UvicornWorker server:app --bind 0.0.0.0:8000
     (Change the -w 4 to match however many CPU cores you have available on your production host machine!)

Test with curl:
http://127.0.0.1:8000/docs#/Prediction/predict_predict_post
http://0.0.0.0:8000/docs#/Prediction/predict_predict_post

copy the curl code and run it on terminal to generate a prediction.

Model files expected (relative to this script or set via env vars):
    MODEL_PATH   : path to model.json or model.xgb  (default: ../streamlit_app/model.json)
    DATA_DIR     : directory containing rank-map JSONs (default: ../streamlit_app/)
"""

import os
import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (override with env vars if needed)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "../streamlit_app"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", DATA_DIR / "model.json"))

FEATURE_COLS = [
    "flat_type_rank",
    "region_ura_rank",
    "town_rank",
    "storey_range_rank",
    "flat_model_rank",
    "distance_to_cbd",
    "floor_area_sqm",
    "remaining_lease_years",
    "resale_price_index",
]

# ---------------------------------------------------------------------------
# Load artefacts at startup (module-level so they are cached for all requests)
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


logger.info("Loading rank maps …")
flat_type_map: dict   = _load_json(DATA_DIR / "flat_type_rank_map.json")
region_map: dict      = _load_json(DATA_DIR / "region_rank_map.json")
town_map: dict        = _load_json(DATA_DIR / "town_rank_map.json")
storey_map: dict      = _load_json(DATA_DIR / "storey_range_rank_map.json")
flat_model_map: dict  = _load_json(DATA_DIR / "flat_model_rank_map.json")

logger.info("Loading XGBoost model from %s …", MODEL_PATH)
_model = xgb.XGBRegressor()
_model.load_model(str(MODEL_PATH))
logger.info("Model loaded successfully.")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="HDB Resale Price Prediction API",
    description=(
        "Algorithmic HDB resale price estimator powered by an XGBoost model "
        "trained on historical Singapore HDB resale transactions."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Input features for one prediction."""

    flat_type: str = Field(
        ...,
        example="4 ROOM",
        description="HDB flat type (e.g. '3 ROOM', '4 ROOM', 'EXECUTIVE').",
    )
    region_ura: str = Field(
        ...,
        example="CENTRAL REGION",
        description="URA planning region (e.g. 'CENTRAL REGION', 'EAST REGION').",
    )
    town: str = Field(
        ...,
        example="TAMPINES",
        description="HDB planning area / town (e.g. 'TAMPINES', 'BISHAN').",
    )
    storey_range: str = Field(
        ...,
        example="10 TO 12",
        description="Storey range band (e.g. '01 TO 03', '10 TO 12').",
    )
    flat_model: str = Field(
        ...,
        example="Model A",
        description="Flat model name (e.g. 'Standard', 'Improved', 'Model A').",
    )
    distance_to_cbd: float = Field(
        ...,
        gt=0,
        example=14500.0,
        description="Straight-line distance from the unit to the CBD, in metres.",
    )
    floor_area_sqm: float = Field(
        ...,
        gt=0,
        example=93.0,
        description="Floor area of the flat in square metres.",
    )
    remaining_lease_years: float = Field(
        ...,
        gt=0,
        le=99,
        example=65.0,
        description="Remaining lease in years at point of transaction.",
    )
    resale_price_index: float = Field(
        ...,
        gt=0,
        example=182.3,
        description=(
            "HDB Resale Price Index value for the target quarter "
            "(1Q2009 = 100, source: SingStat)."
        ),
    )

    # Validate categorical fields against known maps
    @field_validator("flat_type")
    @classmethod
    def validate_flat_type(cls, v: str) -> str:
        if v not in flat_type_map:
            raise ValueError(
                f"Unknown flat_type '{v}'. Valid values: {sorted(flat_type_map.keys())}"
            )
        return v

    @field_validator("region_ura")
    @classmethod
    def validate_region(cls, v: str) -> str:
        if v not in region_map:
            raise ValueError(
                f"Unknown region_ura '{v}'. Valid values: {sorted(region_map.keys())}"
            )
        return v

    @field_validator("town")
    @classmethod
    def validate_town(cls, v: str) -> str:
        if v not in town_map:
            raise ValueError(
                f"Unknown town '{v}'. Valid values: {sorted(town_map.keys())}"
            )
        return v

    @field_validator("storey_range")
    @classmethod
    def validate_storey(cls, v: str) -> str:
        if v not in storey_map:
            raise ValueError(
                f"Unknown storey_range '{v}'. Valid values: {sorted(storey_map.keys())}"
            )
        return v

    @field_validator("flat_model")
    @classmethod
    def validate_flat_model(cls, v: str) -> str:
        if v not in flat_model_map:
            raise ValueError(
                f"Unknown flat_model '{v}'. Valid values: {sorted(flat_model_map.keys())}"
            )
        return v


class PredictResponse(BaseModel):
    """Prediction result."""

    predicted_resale_price: float = Field(
        ..., description="Predicted resale price in SGD."
    )
    input_features: dict = Field(
        ..., description="Encoded feature vector sent to the model."
    )
    model_file: str = Field(..., description="Model file used for inference.")


class MetadataResponse(BaseModel):
    """Valid categorical options."""

    flat_types: list[str]
    regions: list[str]
    towns: list[str]
    storey_ranges: list[str]
    flat_models: list[str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    """Health-check / welcome endpoint."""
    return {
        "service": "HDB Resale Price Prediction API",
        "status": "ok",
        "model": str(MODEL_PATH.name),
        "docs": "/docs",
    }


@app.get("/metadata", response_model=MetadataResponse, tags=["Metadata"])
def get_metadata():
    """Return all valid categorical values for each input field."""
    return MetadataResponse(
        flat_types=sorted(flat_type_map.keys()),
        regions=sorted(region_map.keys()),
        towns=sorted(town_map.keys()),
        storey_ranges=sorted(storey_map.keys()),
        flat_models=sorted(flat_model_map.keys()),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Predict the HDB resale price for the given property features.

    All categorical fields are validated against the training-time rank maps.
    The numeric fields are passed through directly.
    
    Note: Uses `async def` and NumPy directly to avoid the overhead of thread-pool
    dispatching and Pandas DataFrame construction, maximizing concurrency throughput.
    """
    try:
        # Construct feature array directly to avoid pandas DataFrame overhead 
        # which heavily bottlenecks high-concurrency requests.
        features_array = np.array([[
            flat_type_map[request.flat_type],
            region_map[request.region_ura],
            town_map[request.town],
            storey_map[request.storey_range],
            flat_model_map[request.flat_model],
            request.distance_to_cbd,
            request.floor_area_sqm,
            request.remaining_lease_years,
            request.resale_price_index,
        ]])

        prediction: float = float(_model.predict(features_array)[0])
        
        # Assemble variables for logging and response
        encoded = dict(zip(FEATURE_COLS, features_array[0].tolist()))

        logger.info(
            "Prediction: SGD %.2f | town=%s flat_type=%s storey=%s",
            prediction,
            request.town,
            request.flat_type,
            request.storey_range,
        )

        return PredictResponse(
            predicted_resale_price=round(prediction, 2),
            input_features=encoded,
            model_file=MODEL_PATH.name,
        )

    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
