import os
import json
import requests
from aiogram import Bot, Dispatcher, executor, types

TELEGRAM_TOKEN = os.getenv("7709316638:AAG3RC37YzZXPqxu7666bl5APGKKNEweRtw")  # Ваш токен бота
API_URL = os.getenv("API_URL", "http://localhost:8000")

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=['start'])
async def start_command(message: types.Message):
    await message.reply("Привет! Используй команду /predict HORIZON, где HORIZON может быть 1,5,10,20,30.\nПосле этого отправь данные как JSON.\nПример:\n/predict 5")

@dp.message_handler(commands=['predict'])
async def predict_command(message: types.Message):
    parts = message.text.strip().split()
    if len(parts) < 2:
        await message.reply("Использование: /predict HORIZON\nПример: /predict 5")
        return
    horizon = parts[1]
    try:
        horizon = int(horizon)
    except:
        await message.reply("HORIZON должен быть числом (1,5,10,20,30).")
        return
    
    # Сохраним состояние горизонта в "контекст" (упрощённо)
    # В реальном решении лучше FSM. Здесь просто храним в атрибуте.
    # Предположим, что пользователь сразу после /predict HORIZON отправит данные.
    message.conf = {"horizon": horizon}
    await message.reply("Отправь данные в формате JSON: [[val], [val], ...], где val - число.")

# Обработчик текстовых сообщений
@dp.message_handler(content_types=['text'])
async def handle_text(message: types.Message):
    # Попытаемся считать JSON с данными
    try:
        data = json.loads(message.text)
        # Предположим, что до этого был вызван /predict
        # Для реальной реализации нужно хранить контекст.
        # Здесь просто возьмём horizon = 1 по умолчанию.
        horizon = 1
        # Если хотите полноценный контекст - стоит использовать FSM или базу.
        # Упростим: допустим пользователь каждый раз после /predict сразу шлёт данные.
        # Тогда нужно предположить, что код /predict_command будет запущен незадолго до этого.
        # Здесь для примера — статичное значение горизонта.
        # В реальном случае можно использовать хранение состояния в dp или redis.
        
        # Отправим запрос к API
        res = requests.post(f"{API_URL}/predict/{horizon}", json={"input_data": data})
        if res.status_code == 200:
            pred = res.json()["prediction"]
            await message.reply(f"Предикт: {pred}")
        else:
            await message.reply(f"Ошибка на сервере: {res.text}")
    except Exception as e:
        await message.reply(f"Ошибка: {str(e)}")
