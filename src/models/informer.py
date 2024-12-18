import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.dates as mdates

def plot_predictions_with_dates(prediction_dates, actuals, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(prediction_dates, actuals, label='Actual Prices', color='blue')
    plt.plot(prediction_dates, predictions, label='Predicted Prices', color='orange')
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    # Форматируем даты на оси X
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()  # Автоматический поворот дат для удобства
    plt.grid(True)
    plt.show()
    
# Эмбеддинги данных и позиций
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        pe[:, 0::2] = torch.sin(position * div_term)      # Четные индексы
        pe[:, 1::2] = torch.cos(position * div_term)      # Нечетные индексы
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

class DataEmbedding(nn.Module):
    def __init__(self, input_dim, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = nn.Linear(input_dim, d_model)
        self.position_embedding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.value_embedding(x)
        x = self.position_embedding(x)
        return self.dropout(x)

# Механизмы внимания (Attention)
class FullAttention(nn.Module):
    def __init__(self, mask_flag=False, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.mask_flag = mask_flag
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
    
    def forward(self, queries, keys, values, attn_mask=None):
        # queries: [B, H, L_Q, D_k]
        # keys: [B, H, L_K, D_k]
        # values: [B, H, L_V, D_v]
        scale = self.scale or 1. / math.sqrt(queries.size(-1))
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale  # [B, H, L_Q, L_K]
        
        if self.mask_flag and attn_mask is not None:
            scores = scores.masked_fill(attn_mask.bool(), -np.inf)
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, values)  # [B, H, L_Q, D_v]
        return output, attn
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
    
    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, _ = queries.shape  # Длина запросов
        B, L_K, _ = keys.shape     # Длина ключей
        B, L_V, _ = values.shape   # Длина значений
        H = self.n_heads

        # Проекция
        queries = self.query_projection(queries).view(B, L_Q, H, -1)
        keys = self.key_projection(keys).view(B, L_K, H, -1)
        values = self.value_projection(values).view(B, L_V, H, -1)

        # Перестановка
        queries = queries.permute(0, 2, 1, 3)  # [B, H, L_Q, D_k]
        keys = keys.permute(0, 2, 1, 3)        # [B, H, L_K, D_k]
        values = values.permute(0, 2, 1, 3)    # [B, H, L_V, D_v]

        # Механизм внимания
        out, attn = self.inner_attention(queries, keys, values, attn_mask=attn_mask)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L_Q, -1)
        return self.out_projection(out)

# Слои энкодера и декодера
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = AttentionLayer(attention, d_model, n_heads=8)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    
    def forward(self, x, attn_mask=None):
        new_x = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        
        y = x.permute(0, 2, 1)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        y = y.permute(0, 2, 1)
        
        x = x + y
        x = self.norm2(x)
        return x
class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
    
    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = AttentionLayer(self_attention, d_model, n_heads=8)
        self.cross_attention = AttentionLayer(cross_attention, d_model, n_heads=8)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    
    def forward(self, x, enc_output, self_attn_mask=None, cross_attn_mask=None):
        new_x = self.self_attention(x, x, x, attn_mask=self_attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        
        new_x = self.cross_attention(x, enc_output, enc_output, attn_mask=cross_attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm2(x)
        
        y = x.permute(0, 2, 1)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        y = y.permute(0, 2, 1)
        
        x = x + y
        x = self.norm3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection or nn.Linear(layers[0].norm3.normalized_shape[0], 1)
    
    def forward(self, x, enc_output, self_attn_mask=None, cross_attn_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, self_attn_mask=self_attn_mask, cross_attn_mask=cross_attn_mask)
        if self.norm is not None:
            x = self.norm(x)
        x = self.projection(x)
        return x
class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=512, dropout=0.1):
        super(Informer, self).__init__()
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)
        self.encoder = Encoder(
            [EncoderLayer(FullAttention(), d_model, d_ff, dropout) for _ in range(e_layers)],
            norm_layer=nn.LayerNorm(d_model)
        )
        self.decoder = Decoder(
            [DecoderLayer(FullAttention(mask_flag=True), FullAttention(), d_model, d_ff, dropout) for _ in range(d_layers)],
            norm_layer=nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out)
        )
        self.seq_len = seq_len
        self.label_len = label_len
        self.out_len = out_len

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, x_enc, x_dec):
        enc_out = self.enc_embedding(x_enc)
        enc_out = self.encoder(enc_out)
        
        dec_out = self.dec_embedding(x_dec)
        device = x_enc.device
        seq_len = x_dec.size(1)
        self_attn_mask = self.generate_square_subsequent_mask(seq_len).to(device)
        
        dec_out = self.decoder(dec_out, enc_out, self_attn_mask=self_attn_mask)
        return dec_out

def train(num_epochs, model, criterion, optimizer, scheduler, train_loader, val_loader, label_length, pred_length, device, save_path='best_model.pth'):
    train_losses = []
    val_losses = []
    learning_rates = []
    best_val_loss = float('inf')  # Инициализация лучшего val_loss как бесконечности

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for x_enc, x_dec, y, _ in train_loader:
            x_enc = x_enc.to(device)
            x_dec = x_dec.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x_enc, x_dec)
            output = output[:, -pred_length:, :]  # Берём только последние pred_length шагов

            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_enc.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Валидация
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x_enc, _, y, _ in val_loader:
                x_enc = x_enc.to(device)
                y = y.to(device)

                batch_size = x_enc.size(0)
                x_dec_input = torch.zeros(batch_size, label_length + pred_length, x_enc.size(2)).to(device)
                x_dec_input[:, :label_length, :] = x_enc[:, -label_length:, :]

                output = model(x_enc, x_dec_input)
                output = output[:, -pred_length:, :]  # Берём только последние pred_length шагов

                loss = criterion(output, y)
                val_loss += loss.item() * x_enc.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # Обновление планировщика
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)  # Для ReduceLROnPlateau передаём val_loss
        else:
            scheduler.step()  # Для остальных планировщиков просто вызываем step()

        # Получаем текущий learning rate
        lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        learning_rates.append(lrs)

        # Проверяем, улучшился ли val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }, save_path)
            print(f'--> Сохранена лучшая модель на эпохе {epoch+1} с Val Loss: {val_loss:.6f}')

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.6f}, '
              f'Val Loss: {val_loss:.6f}, '
              f'Learning Rate: {lrs}')

    print(f'Обучение завершено. Лучшая Val Loss: {best_val_loss:.6f}')
    return train_losses, val_losses, learning_rates

# ===================== Функция тестирования =====================

def test(model, val_loader, scaler, device, label_length, pred_length):
    model.eval()
    predictions = []
    actuals = []
    prediction_dates = []

    with torch.no_grad():
        for x_enc, _, y, y_dates in val_loader:
            x_enc = x_enc.to(device)
            y = y.to(device)

            batch_size = x_enc.size(0)
            x_dec_input = torch.zeros(batch_size, label_length + pred_length, x_enc.size(2)).to(device)
            x_dec_input[:, :label_length, :] = x_enc[:, -label_length:, :]

            output = model(x_enc, x_dec_input)
            output = output[:, -pred_length:, :]  # Берём только последние pred_length шагов

            # Приводим тензоры к одномерному виду
            output = output.reshape(-1)
            y = y.reshape(-1)

            predictions.extend(output.cpu().numpy())
            actuals.extend(y.cpu().numpy())
            prediction_dates.extend(y_dates)

    # Преобразуем даты
    prediction_dates = [pd.to_datetime(date) for date in prediction_dates]

    # Преобразуем в numpy массивы
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Обратное масштабирование
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actuals = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

    # Расчёт метрик
    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    mape = mean_absolute_percentage_error(actuals, predictions)
    rmse = np.sqrt(mse)
    plot_predictions_with_dates(prediction_dates, actuals, predictions)

    return mae, mse, mape, rmse