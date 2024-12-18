import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ===================== Определение функции визуализации =====================

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
    
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Энкодерная часть LSTM
        self.lstm_enc = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Декодерная часть LSTM
        self.lstm_dec = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Полносвязный слой для предсказания
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x_enc, x_dec):
        # Энкодер
        h0 = torch.zeros(self.num_layers, x_enc.size(0), self.hidden_size).to(x_enc.device)
        c0 = torch.zeros(self.num_layers, x_enc.size(0), self.hidden_size).to(x_enc.device)
        enc_out, (hn, cn) = self.lstm_enc(x_enc, (h0, c0))
        
        # Декодер
        dec_out, _ = self.lstm_dec(x_dec, (hn, cn))
        
        # Предсказание
        out = self.fc(dec_out)
        return out
def train(num_epochs, model, criterion, optimizer, scheduler, train_loader, val_loader, label_length, pred_length, device, save_path='best_lstm_model.pth'):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')  # Инициализация лучшего val_loss как бесконечности

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for x_enc, x_dec, y, _ in train_loader:
            x_enc = x_enc.to(device)
            x_dec = x_dec.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x_enc, x_dec)  # [batch_size, label_length + pred_length, 1]
            # Берём только предсказанные шаги
            output = output[:, -pred_length:, :]  # [batch_size, pred_length, 1]
            loss = criterion(output, y)
            
            if torch.isnan(loss):
                print("NaN detected in loss")
                continue  # Пропустить этот шаг, чтобы избежать остановки обучения
            
            loss.backward()
            # Добавление градиентного клиппинга
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * x_enc.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Валидация
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x_enc, x_dec, y, _ in val_loader:
                x_enc = x_enc.to(device)
                x_dec = x_dec.to(device)
                y = y.to(device)

                output = model(x_enc, x_dec)  # [batch_size, pred_length, 1]
                output = output[:, -pred_length:, :]  # [batch_size, pred_length, 1]
                loss = criterion(output, y)
                
                if torch.isnan(loss):
                    print("NaN detected in val loss")
                    continue  # Пропустить этот шаг
                
                val_loss += loss.item() * x_enc.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # Обновление планировщика
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)  # Для ReduceLROnPlateau передаём val_loss
        else:
            scheduler.step()  # Для остальных планировщиков просто вызываем step()

        # Сохранение лучшей модели
        if val_loss < best_val_loss and not np.isnan(val_loss):
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
              f'Val Loss: {val_loss:.6f}')

    print(f'Обучение завершено. Лучшая Val Loss: {best_val_loss:.6f}')
    return train_losses, val_losses
def test(model, val_loader, scaler, device, label_length, pred_length):
    model.eval()
    predictions = []
    actuals = []
    prediction_dates = []

    with torch.no_grad():
        for x_enc, x_dec, y, y_dates in val_loader:
            x_enc = x_enc.to(device)
            x_dec = x_dec.to(device)
            y = y.to(device)

            # Получаем предсказания модели
            output = model(x_enc, x_dec)  # [batch_size, seq_length, 1]
            output = output[:, -pred_length:, :]  # Берем только последние pred_length шагов

            # Преобразуем к numpy
            predictions.append(output.cpu().numpy())
            actuals.append(y[:, -pred_length:, :].cpu().numpy())  # Сравниваем с последними pred_length целевыми значениями
            prediction_dates.extend(y_dates)

    # Преобразуем списки в numpy массивы
    predictions = np.concatenate(predictions, axis=0)  # [N, pred_length, 1]
    actuals = np.concatenate(actuals, axis=0)          # [N, pred_length, 1]

    # Преобразуем dates в массив [N, pred_length]
    prediction_dates = np.array(prediction_dates).reshape(-1, pred_length)

    # Обратное масштабирование
    inv_predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    inv_actuals = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

    # Преобразуем обратно к массивам [N, pred_length]
    inv_predictions = inv_predictions.reshape(-1, pred_length)
    inv_actuals = inv_actuals.reshape(-1, pred_length)

    # Расчёт метрик
    mae = mean_absolute_error(inv_actuals.flatten(), inv_predictions.flatten())
    mse = mean_squared_error(inv_actuals.flatten(), inv_predictions.flatten())
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((inv_actuals.flatten() - inv_predictions.flatten()) / inv_actuals.flatten())) * 100

    print(f'\nМетрики на тестовой выборке:')
    print(f'MAE: {mae:.2f}')
    print(f'MSE: {mse:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print(f'MAPE: {mape:.2f}%')

    # Визуализация
    flat_dates = prediction_dates.flatten()
    plot_predictions_with_dates(flat_dates, inv_actuals.flatten(), inv_predictions.flatten())

    return mae, mse, mape, rmse