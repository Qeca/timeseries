FROM python:3.9-slim

# Рабочая директория
WORKDIR /app

# Копируем файлы проекта
COPY ./app /app

# Устанавливаем зависимости
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем AWS CLI для скачивания из S3
RUN pip install boto3

# Создаём директорию для весов
RUN mkdir -p /app/weights

# Порт для FastAPI
EXPOSE 8000

# Запуск бота и API
CMD ["sh", "-c", "python download_weights.py && uvicorn main:app --host 0.0.0.0 --port 8000 & python telegram_bot.py"]
