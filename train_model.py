"""เทรนโมเดลใหม่จาก data/Thai Emotion.xlsx แล้วเซฟลง models/

รันสคริปต์นี้เมื่อแก้ไขชุดข้อมูลใน Excel:  python train_model.py
"""
from config import DATA_FILE, MODEL_FILE
from sentiment_classifier import train_and_save

if __name__ == "__main__":
    print(f"กำลังเทรนโมเดลจาก {DATA_FILE.name} ...")
    train_and_save()
    print(f"เสร็จแล้ว เซฟไว้ที่ {MODEL_FILE}")
