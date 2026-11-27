import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List
import joblib
import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, field_validator

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts: load both models into memory."""
    try:
        app.state.models = {
            "category": joblib.load(MODEL_DIR / "baseline_category.joblib"),
            "priority": joblib.load(MODEL_DIR / "baseline_priority.joblib"),
        }
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Model files not found in {MODEL_DIR}. Run task2_baseline.py first. ({e})"
        )
    yield
    app.state.models = {}


app = FastAPI(
    title="Support Ticket Classifier",
    description="Predicts the category and priority of a customer support ticket.",
    version="1.0.0",
    lifespan=lifespan,
)

class Ticket(BaseModel):
    text: str = Field(..., max_length=2000, examples=["I was charged twice for my subscription"])

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty")   # -> HTTP 422
        return v


class Prediction(BaseModel):
    category: str
    priority: str
    confidence: float


class BatchRequest(BaseModel):
    tickets: List[Ticket] = Field(..., min_length=1, max_length=100)


class BatchResponse(BaseModel):
    predictions: List[Prediction]

def classify(models: dict, texts: List[str]) -> List[Prediction]:
    """
    confidence = the LOWER of the category and priority probabilities
    (conservative: the API is only as sure as its weakest answer).
    """
    cat_model, pri_model = models["category"], models["priority"]
    cat_proba = cat_model.predict_proba(texts)
    pri_proba = pri_model.predict_proba(texts)

    results = []
    for cp, pp in zip(cat_proba, pri_proba):
        ci, pi = int(np.argmax(cp)), int(np.argmax(pp))
        results.append(
            Prediction(
                category=str(cat_model.classes_[ci]),
                priority=str(pri_model.classes_[pi]),
                confidence=round(float(min(cp[ci], pp[pi])), 4),
            )
        )
    return results

@app.get("/health")
def health(request: Request):
    loaded = bool(getattr(request.app.state, "models", None))
    return {"status": "ok" if loaded else "models_not_loaded", "models_loaded": loaded}


@app.post("/predict", response_model=Prediction)
def predict(ticket: Ticket, request: Request):
    return classify(request.app.state.models, [ticket.text])[0]
@app.post("/predict-batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest, request: Request):
    texts = [t.text for t in batch.tickets]
    return BatchResponse(predictions=classify(request.app.state.models, texts))
