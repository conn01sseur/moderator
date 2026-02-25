import telebot
import time
from datetime import datetime
from config import TOKEN


# Логирование
def log_message(msg):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {msg}")


# Токен бота


# Создание бота
log_message("Инициализация бота...")
bot = telebot.TeleBot(TOKEN)


# Обработчик команды /start
@bot.message_handler(commands=['start'])
def start(message):
    log_message(f"Пользователь {message.chat.id} (@{message.chat.username}) запустил бота")
    bot.reply_to(message, "привет! 👋")


# Обработчик команды /help
@bot.message_handler(commands=['help'])
def help_command(message):
    log_message(f"Пользователь {message.chat.id} запросил помощь")
    bot.reply_to(message, "Доступные команды:\n/start - приветствие\n/help - помощь\n/hello - сказать привет")


# Обработчик команды /hello
@bot.message_handler(commands=['hello'])
def hello(message):
    log_message(f"Пользователь {message.chat.id} сказал привет")
    bot.reply_to(message, "привет!")


# Обработчик текстовых сообщений
@bot.message_handler(func=lambda message: True)
def echo_all(message):
    log_message(f"Получено сообщение от {message.chat.id}: '{message.text}'")

    if message.text.lower() == "привет":
        bot.reply_to(message, "привет!")
    else:
        bot.reply_to(message, "Напиши 'привет' или используй /hello")


# Запуск бота
log_message("Бот запущен и готов к работе...")

while True:
    try:
        bot.polling(none_stop=True)
    except Exception as e:
        log_message(f"Ошибка: {e}")
        log_message("Перезапуск через 3 секунды...")
        time.sleep(3)