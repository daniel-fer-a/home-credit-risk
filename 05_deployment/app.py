from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts"

MODEL_PATH = ARTIFACTS / "champion_model.joblib"
COLS_PATH = ARTIFACTS / "champion_numeric_cols.joblib"


app = FastAPI(title="Home Credit Risk API", version="1.0")


class PredictRequest(BaseModel):
    """
    Recibe un diccionario de features (por ejemplo desde model_X).
    Ejemplo:
    {
      "features": {
        "EXT_SOURCE_1": 0.7,
        "DAYS_BIRTH": -12000,
        ...
      }
    }
    """
    features: Dict[str, Any]


@app.on_event("startup")
def load_artifacts():
    global model, numeric_cols
    model = joblib.load(MODEL_PATH)
    numeric_cols = joblib.load(COLS_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    
    row = pd.DataFrame([req.features])

    
    row = row.reindex(columns=numeric_cols)

    
    row = row.replace([np.inf, -np.inf], np.nan)

    proba = float(model.predict_proba(row)[:, 1][0])
    pred = int(proba >= 0.5)

    return {
        "prediction": pred,
        "probability_default": proba,
        "threshold": 0.5
    }
