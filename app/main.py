from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
import os
import torch
from datetime import datetime
import joblib

from informer_model import load_model, run_inference
app = FastAPI()

# Определяем устройство для выполнения (CPU или CUDA)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Словарь с путями к разным горизонтам предсказания
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
weights_map = {
    1: os.path.join(BASE_DIR, "weights/1_best_informer_model.pth"),
    5: os.path.join(BASE_DIR, "weights/5_best_informer_model.pth"),
    10: os.path.join(BASE_DIR, "weights/10_best_informer_model.pth"),
    20: os.path.join(BASE_DIR, "weights/20_best_informer_model.pth"),
    30: os.path.join(BASE_DIR, "weights/30_best_informer_model.pth"),
}
# Кэш моделей, чтобы не загружать их заново
models_cache = {}

def get_model_for_horizon(horizon):
    if horizon not in weights_map:
        raise HTTPException(status_code=400, detail="Invalid horizon")

    if horizon not in models_cache:
        path = weights_map[horizon]
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Model weights not found at {path}")
        models_cache[horizon] = load_model(path, out_len=horizon, device=device)
    return models_cache[horizon]

class PredictionRequest(BaseModel):
    input_data: list = []  # Данные по умолчанию пусты

@app.post("/predict/{horizon}")
def predict(horizon: int, request: PredictionRequest):
    model = get_model_for_horizon(horizon)

    # Обработка пустых данных - заглушка из случайных чисел
    if not request.input_data:
        seq_len = 60  # Длина входа по умолчанию
        input_data = np.random.rand(seq_len, 1)  # Заполняем случайными значениями
    else:
        input_data = np.array(request.input_data)

    # Проверка на корректный формат
    if len(input_data.shape) == 1:
        input_data = np.expand_dims(input_data, axis=-1)
    if len(input_data.shape) != 2 or input_data.shape[-1] != 1:
        raise HTTPException(status_code=400, detail="Input data must be a 2D list with shape (seq_len, 1)")
    scaler = joblib.load('weights/scaler.pkl')

    # Выполняем инференс
    result = run_inference(model, input_data, out_len=horizon, device=device, scaler=scaler)
    return result


# Запуск сервера
if __name__ == "__main__":
    print("🚀 Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
