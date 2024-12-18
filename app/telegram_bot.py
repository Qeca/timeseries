import os
import requests
from aiogram import Bot, Dispatcher, types
from aiogram.types import ParseMode
from aiogram.utils import executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage

# Настройки
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7709316638:AAG3RC37YzZXPqxu7666bl5APGKKNEweRtw")  # Токен бота
API_URL = os.getenv("API_URL", "http://localhost:8000")

bot = Bot(token=TELEGRAM_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

@dp.message_handler(commands=['start'])
async def start_command(message: types.Message):
    await message.reply(
        "👋 Привет!\nЯ бот для получения предсказаний. \n\n"
        "Используй команду:\n"
        "`/predict HORIZON`\n\n"
        "Где HORIZON может быть: 1, 5, 10, 20, 30.\n\n"
        "После этого я автоматически покажу предсказания на выбранное количество дней.",
        parse_mode=ParseMode.MARKDOWN
    )

@dp.message_handler(commands=['predict'])
async def predict_command(message: types.Message):
    parts = message.text.strip().split()
    if len(parts) != 2:
        await message.reply("⚠️ Использование: `/predict HORIZON`\nПример: `/predict 5`", parse_mode=ParseMode.MARKDOWN)
        return

    try:
        horizon = int(parts[1])
        if horizon not in [1, 5, 10, 20, 30]:
            raise ValueError("Недопустимый горизонт.")
    except ValueError:
        await message.reply("⚠️ HORIZON должен быть числом из списка: `1, 5, 10, 20, 30`.", parse_mode=ParseMode.MARKDOWN)
        return

    await message.reply("⏳ Выполняю предсказание, пожалуйста подождите...")

    try:
        # Отправляем запрос на сервер с пустыми данными
        response = requests.post(f"{API_URL}/predict/{horizon}", json={"input_data": []})

        if response.status_code == 200:
            result = response.json()
            dates = result.get("dates", [])
            predictions = result.get("predictions", [])
            formatted_result = "\n".join([f"📅 {date}: {val:.4f}" for date, val in zip(dates, predictions)])
            await message.reply(
                f"📊 Предсказание на **{horizon} дней**:\n\n{formatted_result}",
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await message.reply(f"❌ Ошибка на сервере: {response.text}")

    except Exception as e:
        await message.reply(f"❌ Произошла ошибка: {str(e)}")

if __name__ == "__main__":
    print("🚀 Бот запущен!")
    executor.start_polling(dp, skip_updates=True)