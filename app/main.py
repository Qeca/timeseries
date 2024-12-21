from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
import os
import torch
import joblib
import pandas as pd
from informer_model import load_model, run_inference

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE_DIR, "weights", "scaler.pkl")
CSV_PATH = os.path.join(BASE_DIR, "weights", "gazprom_historical_data.csv")
weights_map = {
    1:  os.path.join(BASE_DIR, "weights", "1_best_informer_model.pth"),
    5:  os.path.join(BASE_DIR, "weights", "5_best_informer_model.pth"),
    10: os.path.join(BASE_DIR, "weights", "10_best_informer_model.pth"),
    20: os.path.join(BASE_DIR, "weights", "20_best_informer_model.pth"),
    30: os.path.join(BASE_DIR, "weights", "30_best_informer_model.pth"),
}
models_cache = {}
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_last_n_from_csv(csv_path, seq_len: int) -> np.ndarray:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df['BOARDID'] == 'TQBR']

    if 'LEGALCLOSEPRICE' not in df.columns:
        raise ValueError("Column 'LEGALCLOSEPRICE' not found in CSV file.")

    df['LEGALCLOSEPRICE'] = pd.to_numeric(df['LEGALCLOSEPRICE'], errors='coerce')
    df = df.dropna(subset=['LEGALCLOSEPRICE'])

    if 'TRADEDATE' in df.columns:
        df = df.sort_values('TRADEDATE').reset_index(drop=True)

    if len(df) < seq_len:
        raise ValueError(f"Not enough rows in CSV. Need >= {seq_len}, got {len(df)}.")

    last_values = df['LEGALCLOSEPRICE'].values[-seq_len:]
    last_values = last_values.reshape(-1, 1)
    return last_values

def get_model_for_horizon(horizon: int):
    if horizon not in weights_map:
        raise HTTPException(status_code=400, detail=f"Invalid horizon {horizon}")

    model_path = weights_map[horizon]
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model weights not found: {model_path}")

    if horizon in models_cache:
        return models_cache[horizon]

    seq_len = 60
    label_len = 30
    out_len = horizon
    try:
        model = load_model(
            weights_path=model_path,
            seq_len=seq_len,
            label_len=label_len,
            out_len=out_len,
            d_model=512,
            n_heads=4,
            e_layers=2,
            d_layers=1,
            d_ff=2048,
            dropout=0.2,
            device=device
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

    models_cache[horizon] = model
    return model

class PredictionRequest(BaseModel):
    input_data: list = []

@app.post("/predict/{horizon}")
def predict(horizon: int, request: PredictionRequest):
    model = get_model_for_horizon(horizon)

    if not os.path.exists(SCALER_PATH):
        raise HTTPException(status_code=500, detail="Scaler file not found.")

    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading scaler: {str(e)}")

    seq_len = 60
    if not request.input_data:
        try:
            input_data = get_last_n_from_csv(CSV_PATH, seq_len)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading CSV: {str(e)}")
    else:
        input_data = np.array(request.input_data, dtype=np.float32)
        if len(input_data.shape) == 1:
            input_data = np.expand_dims(input_data, axis=-1)
        if len(input_data.shape) != 2 or input_data.shape[-1] != 1:
            raise HTTPException(
                status_code=400,
                detail="`input_data` must have shape (60, 1). Example: [[100.1],[101.2],...]"
            )
        if input_data.shape[0] != seq_len:
            raise HTTPException(
                status_code=400,
                detail=f"`input_data` must have length {seq_len}, got {input_data.shape[0]}"
            )

    try:
        input_scaled = scaler.transform(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scaler error: {str(e)}")

    try:
        result = run_inference(
            model=model,
            input_data=input_scaled,
            out_len=horizon,
            scaler=scaler,
            label_len=30,
            device=device
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
