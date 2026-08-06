"""โมเดลวิเคราะห์ความรู้สึกของประโยคภาษาไทย (Naive Bayes + attacut)

โมเดลถูกเทรนครั้งเดียวแล้วเซฟเป็นไฟล์ .joblib เวลาใช้งานจริงจะโหลดไฟล์นั้นมาใช้เลย
ไม่ต้องอ่าน Excel และตัดคำทั้งชุดข้อมูลใหม่ทุกครั้ง
"""
import joblib
import pandas as pd
from attacut import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

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


def train_and_save():
    """เทรนโมเดลจากไฟล์ Excel แล้วเซฟลง MODEL_FILE คืนค่า pipeline ที่เทรนแล้ว"""
    df = pd.read_excel(DATA_FILE)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(tokenizer=tokenize, token_pattern=None, use_idf=False)),
        ("clf", MultinomialNB()),
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
    """หุ้มโมเดลไว้ให้เรียก predict() ได้ โดยโหลดโมเดลแบบ lazy ครั้งเดียว"""

    def __init__(self):
        self._pipeline = None

    def predict(self, text: str) -> str:
        """คืนชื่ออารมณ์ภาษาไทยของประโยคที่รับเข้ามา"""
        if self._pipeline is None:
            self._pipeline = load_model()
        return LABEL_NAMES[self._pipeline.predict([text])[0]]
