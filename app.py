from flask import Flask, request, render_template
import numpy as np
import pickle
import sqlite3
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)


model = load_model("next_word_lstm.h5")

with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)


MAX_SEQUENCE_LEN = 14  


def init_db():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            seed_text TEXT,
            predicted_word TEXT,
            timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    seed_text = request.form["text"]

    
    sequence = tokenizer.texts_to_sequences([seed_text])[0]
    sequence = pad_sequences(
        [sequence],
        maxlen=MAX_SEQUENCE_LEN,
        padding="pre"
    )

    
    prediction = model.predict(sequence, verbose=0)
    predicted_index = np.argmax(prediction)

    
    predicted_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            predicted_word = word
            break

    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (seed_text, predicted_word, timestamp) VALUES (?, ?, ?)",
        (seed_text, predicted_word, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

    return render_template(
        "index.html",
        seed_text=seed_text,
        predicted_word=predicted_word
    )


if __name__ == "__main__":
    app.run(debug=True)
