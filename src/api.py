from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Literal
from fastapi.responses import FileResponse
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


ROOT = Path(__file__).resolve().parents[1]


class PredictionInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        json_schema_extra={
            "example": {
                "type": "M",
                "air_temperature": 298.1,
                "process_temperature": 308.6,
                "rotational_speed": 1551,
                "torque": 42.8,
                "tool_wear": 0,
            }
        },
    )

    type: Literal["L", "M", "H"]
    air_temperature: float = Field(gt=0, description="Kelvin")
    process_temperature: float = Field(gt=0, description="Kelvin")
    rotational_speed: float = Field(gt=0, description="rpm")
    torque: float = Field(ge=0, description="Nm")
    tool_wear: float = Field(ge=0, description="Dakika")


class PredictionOutput(BaseModel):
    prediction: int
    status: Literal["Normal", "Fault"]
    probability: float
    decision_threshold: float
    model_name: str
    model_version: str
    timestamp_utc: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    metadata_path = ROOT / "models" / "metadata.json"
    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )
    model_path = ROOT / "models" / metadata["artifact"]

    app.state.model = joblib.load(model_path)
    app.state.metadata = metadata

    yield


app = FastAPI(
    title="Predictive Maintenance API",
    description="AI4I verisiyle egitilmis ariza tespit prototipi.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": app.state.metadata["model_version"],
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput):
    row = {
        "Type": payload.type,
        "Air temperature [K]": payload.air_temperature,
        "Process temperature [K]": payload.process_temperature,
        "Rotational speed [rpm]": payload.rotational_speed,
        "Torque [Nm]": payload.torque,
        "Tool wear [min]": payload.tool_wear,
    }

    metadata = app.state.metadata
    frame = pd.DataFrame(
        [row],
        columns=metadata["features"],
    )

    probability = float(
        app.state.model.predict_proba(frame)[0, 1]
    )
    threshold = float(metadata["decision_threshold"])
    prediction = int(probability >= threshold)

    return PredictionOutput(
        prediction=prediction,
        status="Fault" if prediction else "Normal",
        probability=probability,
        decision_threshold=threshold,
        model_name=metadata["model_name"],
        model_version=metadata["model_version"],
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
@app.get("/", include_in_schema=False)
def home():
    return FileResponse(ROOT / "web" / "index.html")