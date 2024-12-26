import os
import requests
from aiogram import Bot, Dispatcher, types
from aiogram.types import ParseMode, InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils import executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7709316638:AAG3RC37YzZXPqxu7666bl5APGKKNEweRtw")
API_URL = os.getenv("API_URL", "http://localhost:8000")

bot = Bot(token=TELEGRAM_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

def get_horizon_keyboard() -> InlineKeyboardMarkup:
    keyboard = InlineKeyboardMarkup()
    for horizon in [1, 5, 10, 20, 30]:
        button = InlineKeyboardButton(
            text=f"{horizon} день{'(дней)' if horizon != 1 else ''}",
            callback_data=f"predict_{horizon}"
        )
        keyboard.add(button)
    return keyboard

@dp.message_handler(commands=['start'])
async def start_command(message: types.Message):
    keyboard = InlineKeyboardMarkup()
    keyboard.add(
        InlineKeyboardButton(
            text="Получить предсказание",
            callback_data="start_predict"
        )
    )

    await message.reply(
        "👋 Привет!\nЯ бот для получения предсказаний.\n\n"
        "Нажми на кнопку ниже, чтобы выбрать горизонт предсказания:",
        parse_mode=ParseMode.MARKDOWN,
        reply=False,
        reply_markup=keyboard
    )

@dp.callback_query_handler(text="start_predict")
async def start_predict_callback(call: types.CallbackQuery):
    keyboard = get_horizon_keyboard()
    await call.message.edit_text(
        text="Выберите, на сколько дней сделать предсказание:",
        reply_markup=keyboard
    )

@dp.callback_query_handler(lambda c: c.data.startswith("predict_"))
async def handle_predict_callback(call: types.CallbackQuery):

    horizon_str = call.data.split("_")[1]
    try:
        horizon = int(horizon_str)
    except ValueError:
        await call.message.edit_text("Не удалось определить горизонт предсказания.")
        return

    await call.answer("⏳ Выполняю предсказание, пожалуйста подождите...", show_alert=False)

    try:
        response = requests.post(f"{API_URL}/predict/{horizon}", json={"input_data": []})

        if response.status_code == 200:
            result = response.json()
            dates = result.get("dates", [])
            predictions = result.get("predictions", [])
            formatted_result = "\n".join(
                [f"📅 {date}: {val:.4f}" for date, val in zip(dates, predictions)]
            )

            back_keyboard = InlineKeyboardMarkup()
            back_keyboard.add(
                InlineKeyboardButton(
                    text="Вернуться",
                    callback_data="start_predict"
                )
            )

            await call.message.edit_text(
                text=(
                    f"📊 Предсказание на **{horizon} дней**:\n\n"
                    f"{formatted_result}"
                ),
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=back_keyboard
            )
        else:
            await call.message.edit_text(f"❌ Ошибка на сервере: {response.text}")

    except Exception as e:
        await call.message.edit_text(f"❌ Произошла ошибка: {str(e)}")


if __name__ == "__main__":
    print("🚀 Бот запущен!")
    executor.start_polling(dp, skip_updates=True)
