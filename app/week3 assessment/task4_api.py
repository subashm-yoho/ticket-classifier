import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "models/distilbert-feedback"   

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

app = FastAPI(title="Customer Feedback Classifier")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)  


class PredictResponse(BaseModel):
    label: str
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text.strip():               
        raise HTTPException(status_code=422, detail="text must not be empty")

    inputs = tokenizer(req.text, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    idx = int(torch.argmax(probs))
    return PredictResponse(label=model.config.id2label[idx], confidence=round(float(probs[idx]), 4))
