import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, data, dates, seq_length, label_length, pred_length):
        self.data = data
        self.dates = dates
        self.seq_length = seq_length      # Длина входа для энкодера
        self.label_length = label_length  # Длина метки для декодера
        self.pred_length = pred_length    # Длина предсказания

    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length

    def __getitem__(self, idx):
        # Входы для энкодера
        x_enc = self.data[idx : idx + self.seq_length]

        # Инициализируем x_dec нулями и заполняем известными данными
        x_dec = np.zeros((self.label_length + self.pred_length, self.data.shape[1]))
        x_dec[:self.label_length] = self.data[idx + self.seq_length - self.label_length : idx + self.seq_length]

        # Целевые значения (будущие данные)
        y = self.data[idx + self.seq_length : idx + self.seq_length + self.pred_length]
        y_dates = self.dates[idx + self.seq_length : idx + self.seq_length + self.pred_length]

        # Преобразуем даты в список строк
        y_dates = [str(date) for date in y_dates]
        
        return torch.FloatTensor(x_enc), torch.FloatTensor(x_dec), torch.FloatTensor(y), y_dates
class TimeSeriesWithNewsDataset(Dataset):
    def __init__(self, data, dates, seq_length, label_length, pred_length):
        self.data = data
        self.dates = dates
        self.seq_length = seq_length
        self.label_length = label_length
        self.pred_length = pred_length

    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length

    def __getitem__(self, idx):
        x_enc = self.data[idx: idx + self.seq_length]  # (seq_length, 769)
        
        x_dec = np.zeros((self.label_length + self.pred_length, self.data.shape[1]))
        x_dec[:self.label_length] = self.data[idx + self.seq_length - self.label_length : idx + self.seq_length]

        y = self.data[idx + self.seq_length : idx + self.seq_length + self.pred_length, 0:1]  # Предполагаем, что целевой признак - цена (первый канал)
        y_dates = self.dates[idx + self.seq_length : idx + self.seq_length + self.pred_length]
        y_dates = [str(d) for d in y_dates]

        return torch.FloatTensor(x_enc), torch.FloatTensor(x_dec), torch.FloatTensor(y), y_dates

def custom_collate_fn(batch):
    x_enc_batch = []
    x_dec_batch = []
    y_batch = []
    y_dates_batch = []

    for x_enc, x_dec, y, y_dates in batch:
        x_enc_batch.append(x_enc)
        x_dec_batch.append(x_dec)
        y_batch.append(y)
        y_dates_batch.extend(y_dates)  # Собираем даты в один список

    x_enc_batch = torch.stack(x_enc_batch)
    x_dec_batch = torch.stack(x_dec_batch)
    y_batch = torch.stack(y_batch)

    return x_enc_batch, x_dec_batch, y_batch, y_dates_batch