"""เว็บตรวจจับความรู้สึกของประโยคภาษาไทย"""
from flask import Flask, redirect, render_template, request, url_for

from config import DEBUG, PORT
from sentiment_classifier import EMOJI_BY_NAME, ThaiSentimentClassifier

app = Flask(__name__)
classifier = ThaiSentimentClassifier()


@app.route("/")
def home():
    result = request.args.get("result", "")
    return render_template(
        "home.html",
        text=request.args.get("text", ""),
        result=result,
        emoji=EMOJI_BY_NAME.get(result, ""),
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    input_text = request.form.get("input", "").strip()
    if not input_text:
        return redirect(url_for("home"))

    result = classifier.predict(input_text)
    return redirect(url_for("home", text=input_text, result=result))


if __name__ == "__main__":
    app.run(debug=DEBUG, port=PORT)
