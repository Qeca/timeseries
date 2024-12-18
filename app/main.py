from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
import os
import torch

from informer_model import load_model, run_inference

app = FastAPI()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Словарь с путями к разным горизонтам предсказания
weights_map = {
    1: "app/weights/1_best_informer_model.pth",
    5: "app/weights/5_best_informer_model.pth",
    10: "app/weights/10_best_informer_model.pth",
    20: "app/weights/20_best_informer_model.pth",
    30: "app/weights/30_best_informer_model.pth"
}

models_cache = {}

def get_model_for_horizon(horizon):
    if horizon not in weights_map:
        raise HTTPException(status_code=400, detail="Invalid horizon")

    if horizon not in models_cache:
        path = weights_map[horizon]
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Model weights not found")
        models_cache[horizon] = load_model(path, out_len=horizon, device=device)
    return models_cache[horizon]

class PredictionRequest(BaseModel):
    # input_data - список списков: [[val], [val], ...] для seq_len точек
    # Например: [[0.5],[0.6],[0.7],...]
    input_data: list

@app.post("/predict/{horizon}")
def predict(horizon: int, request: PredictionRequest):
    model = get_model_for_horizon(horizon)
    input_data = np.array(request.input_data)  # shape: (seq_len, 1)
    preds = run_inference(model, input_data, out_len=horizon, device=device)
    # preds: (1, horizon, 1)
    return {"prediction": preds.squeeze().tolist()}
