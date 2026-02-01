##-------------------------Next Word Prediction using LSTM (Flask Web App)-------------------------------##

1. Project Overview
-This project implements a Next Word Prediction system using a Long Short-Term Memory (LSTM) neural network trained on Shakespeare's Hamlet text.
-Given a Short seed text, the model predicts the most likely next word based on learned language patterns.

-The trained deep learning model is deployed as a flask web application, allowing users to interact with the      model through a simple web interface. User inputs and model predictions are optionally stored in a SQLITE database for analysis and improvement.

2. Key Concepts Used
-Long Short-Term Memory (LSTM) RNN
-Tokenization & sequence padding
-Language modeling (next-word prediction)
-Model serialization(.h5 , pickle)
-Flask Web application
-SQLite database integrations

3. Technology Stack
-Python
-Tensorflow/Keras
-Flask 
-NumPy
-SQLite
-HTML & CSS

4. Project Structure

NEXT_WORD_PREDICTION/
│── app.py
│── next_word_lstm.h5
│── tokenizer.pickle
│── predictions.db
│── requirements.txt
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css

5. How it Works:
   |
   |-1.User enters a short seed sentence.
   |-2.Text is tokenized and padded to a fixed sequence length.
   |-3.The LSTM model predicts the most probable next word.
   |-4.The prediction is displayed on the web page.
   |-5.Input and output are logged in a SQLite database.


6. Limitation
   |
   |-Model predictions are domain-specific (trained only on Hamlet).
   |-Performance degrades for long input text due to fixed context window.
   |-LSTM handles limited long-range dependencies compared to Transformer models.



7. Outcome 

This project demonstrates how a deep learning NLP model can be trained, serialized, and deployed as a full-stack application while highlighting practical limitations of LSTM-based language models.