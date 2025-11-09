# src/notifier.py

import os
import asyncio
import smtplib
from email.mime.text import MIMEText
import logging
from telegram import Bot
from telegram.error import TelegramError
from src.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

def send_email_notification(subject, message):
    try:
        host = os.getenv("EMAIL_HOST")
        port = int(os.getenv("EMAIL_PORT", 587))
        user = os.getenv("EMAIL_USER")
        password = os.getenv("EMAIL_PASS")
        to_addr = os.getenv("EMAIL_TO")
        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = to_addr
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, [to_addr], msg.as_string())
    except Exception as e:
        logging.error(f"Failed to send email notification: {e}")

class TelegramNotifier:
    def __init__(self):
        self.enabled = True
        try:
            if TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_TOKEN != 'dummy_token_for_testing':
                self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
                self.chat_id = TELEGRAM_CHAT_ID
                logging.info("Telegram Notifier initialized.")
            else:
                self.enabled = False
                self.bot = None
                self.chat_id = None
                logging.info("Telegram notifications disabled (no valid token provided).")
        except Exception as e:
            self.enabled = False
            self.bot = None
            self.chat_id = None
            logging.warning(f"Failed to initialize Telegram bot: {e}. Notifications disabled.")

    def send_message(self, message):
        if not self.enabled or not self.bot:
            logging.debug("Telegram notifications disabled. Message not sent.")
            return

        try:
            asyncio.run(self.bot.send_message(chat_id=self.chat_id, text=message))
            logging.info(f"Telegram message sent: '{message}'")
        except TelegramError as e:
            logging.error(f"Failed to send Telegram message: {e}", exc_info=True)
        except Exception as e:
            logging.error(f"An unexpected error occurred while sending Telegram message: {e}", exc_info=True)