"""ค่าตั้งค่าทั้งหมดของแอป อ่านจากไฟล์ .env (ดูตัวอย่างที่ .env.example)"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATA_FILE = BASE_DIR / "data" / "Thai Emotion.xlsx"
MODEL_FILE = BASE_DIR / "models" / "sentiment_model.joblib"

PORT = int(os.getenv("PORT", "5000"))
DEBUG = os.getenv("FLASK_DEBUG", "1") == "1"
