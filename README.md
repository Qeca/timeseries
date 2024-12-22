
# Описание проекта

Этот проект представляет собой веб-приложение, разработанное с использованием FastAPI и интегрированное с Telegram-ботом. Приложение контейнеризовано с помощью Docker и включает следующие компоненты:

- **FastAPI**: фреймворк для создания API.
- **Uvicorn**: ASGI-сервер для работы с FastAPI.
- **Docker**: контейнеризация приложения.
- **Telegram Bot**: бот для взаимодействия с пользователями.

## Структура проекта

```plaintext
project/
├── app/
│   ├── main.py              # Основной файл FastAPI приложения
│   ├── download_weights.py  # Скрипт для загрузки весов модели
│   ├── telegram_bot.py      # Telegram-бот
│   └── requirements.txt     # Зависимости Python
├── weights/                 # Директория для хранения весов модели
├── Dockerfile               # Конфигурация для сборки Docker-образа
└── README.md                # Инструкции по запуску
```

## Предварительные требования

Перед началом работы убедитесь, что на вашем компьютере установлены следующие инструменты:

- **Docker**: [Установка Docker](https://docs.docker.com/get-docker/)
- **Docker Compose**: [Установка Docker Compose](https://docs.docker.com/compose/install/)

## Установка и запуск

1. **Клонирование репозитория**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Создайте файл `.env`**

   В корневой директории создайте файл `.env` и добавьте токен Telegram-бота:

   ```env
   TELEGRAM_API_TOKEN=ваш_токен_бота
   ```

3. **Сборка Docker-образа**

   Выполните команду для сборки Docker-образа:

   ```bash
   docker build -t my-fastapi-app .
   ```

4. **Запуск контейнера**

   Запустите контейнер:

   ```bash
   docker run -d -p 8000:8000 --name my-container my-fastapi-app
   ```

5. **Проверка работы приложения**

   - Приложение FastAPI будет доступно по адресу: [http://localhost:8000](http://localhost:8000)
   - Документация API:
     - Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
     - ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

6. **Взаимодействие с Telegram-ботом**

   Найдите бота в Telegram по имени, указанному при создании. Начните диалог, отправив команду `/start`.

## Структура Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY ./app /app

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install boto3

RUN mkdir -p /app/weights

EXPOSE 8000

CMD ["sh", "-c", "python download_weights.py && uvicorn main:app --host 0.0.0.0 --port 8000 & python telegram_bot.py"]
```

## Полезные команды Docker

- **Просмотр активных контейнеров**

  ```bash
  docker ps
  ```

- **Просмотр логов контейнера**

  ```bash
  docker logs my-container
  ```

- **Остановка контейнера**

  ```bash
  docker stop my-container
  ```

- **Удаление контейнера**

  ```bash
  docker rm my-container
  ```

- **Запуск контейнера заново**

  ```bash
  docker start my-container
  ```

## Полезные ссылки

- [Документация FastAPI](https://fastapi.tiangolo.com/)
- [Документация Docker](https://docs.docker.com/)
- [Документация Telegram Bot API](https://core.telegram.org/bots)

Следуя этим инструкциям, вы сможете успешно запустить и использовать приложение.
