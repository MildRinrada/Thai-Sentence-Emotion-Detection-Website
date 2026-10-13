"""โมเดลวิเคราะห์ความรู้สึกของประโยคภาษาไทย (Naive Bayes + attacut)

โมเดลถูกเทรนครั้งเดียวแล้วเซฟเป็นไฟล์ .joblib เวลาใช้งานจริงจะโหลดไฟล์นั้นมาใช้เลย
ไม่ต้องอ่าน Excel และตัดคำทั้งชุดข้อมูลใหม่ทุกครั้ง
"""
import re

import joblib
import pandas as pd
from attacut import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

from config import DATA_FILE, MODEL_FILE

# label ในไฟล์ข้อมูล -> (ชื่ออารมณ์ภาษาไทย, อีโมจิที่ใช้แสดงผล)
EMOTIONS = {
    0: ("ความเศร้า", "😢"),
    1: ("ความสุข", "😄"),
    2: ("ความรัก", "🥰"),
    3: ("ความโกรธ", "😡"),
}

LABEL_NAMES = {label: name for label, (name, _) in EMOTIONS.items()}
EMOJI_BY_NAME = {name: emoji for name, emoji in EMOTIONS.values()}


def normalize(text: str) -> str:
    """ตัวอักษรไทยที่พิมพ์ซ้ำ 3 ตัวขึ้นไปให้เหลือตัวเดียว เช่น "มากกกก" -> "มาก" """
    return re.sub(r"([฀-๿])\1{2,}", r"\1", text.lower())


def train_and_save():
    """เทรนโมเดลจากไฟล์ Excel แล้วเซฟลง MODEL_FILE คืนค่า pipeline ที่เทรนแล้ว"""
    df = pd.read_excel(DATA_FILE)

    # ใช้ทั้งคำที่ตัดด้วย attacut และชิ้นตัวอักษร 2-4 ตัว (char n-gram)
    # ส่วน char n-gram ช่วยให้ยังจับความหมายได้แม้คำนั้นตัดผิดหรือไม่เคยเจอในชุดข้อมูล
    features = FeatureUnion([
        ("word", TfidfVectorizer(tokenizer=tokenize, token_pattern=None,
                                 preprocessor=normalize, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(2, 4),
                                 preprocessor=normalize, sublinear_tf=True)),
    ])
    pipeline = Pipeline([
        ("features", features),
        ("clf", MultinomialNB(alpha=0.1)),
    ])
    pipeline.fit(df["text"], df["label"])

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)
    return pipeline


def load_model():
    """โหลดโมเดลที่เซฟไว้ ถ้ายังไม่มีไฟล์จะเทรนให้อัตโนมัติหนึ่งครั้ง"""
    if MODEL_FILE.exists():
        return joblib.load(MODEL_FILE)
    return train_and_save()


class ThaiSentimentClassifier:
    """หุ้มโมเดลไว้ให้เรียก predict() ได้ โหลดโมเดลแบบ lazy และโหลดใหม่เมื่อไฟล์โมเดลเปลี่ยน"""

    def __init__(self):
        self._pipeline = None
        self._model_mtime = None

    def predict(self, text: str) -> str:
        """คืนชื่ออารมณ์ภาษาไทยของประโยคที่รับเข้ามา"""
        # ถ้ารัน python train_model.py ขณะเว็บเปิดอยู่ จะได้ใช้โมเดลใหม่ทันทีโดยไม่ต้องรีสตาร์ทเว็บ
        mtime = MODEL_FILE.stat().st_mtime if MODEL_FILE.exists() else None
        if self._pipeline is None or mtime != self._model_mtime:
            self._pipeline = load_model()
            self._model_mtime = MODEL_FILE.stat().st_mtime
        return LABEL_NAMES[self._pipeline.predict([text])[0]]
