from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import sqlite3
import os

# -----------------------------
# 1. Flask setup
# -----------------------------
app = Flask(__name__)

# Load your trained CNN model
model = load_model("face_emotionModel.h5")

# Emotion class labels (same order as training)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Ensure upload folder exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# -----------------------------
# 2. Database setup (SQLite)
# -----------------------------
def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    emotion TEXT
                )''')
    conn.commit()
    conn.close()

init_db()


# -----------------------------
# 3. Define routes
# -----------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get uploaded image
        file = request.files["image"]
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Preprocess the image
            img = image.load_img(filepath, target_size=(48, 48), color_mode="grayscale")
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Predict emotion
            predictions = model.predict(img_array)
            emotion_index = np.argmax(predictions)
            predicted_emotion = emotion_labels[emotion_index]

            # Save to database
            conn = sqlite3.connect("database.db")
            c = conn.cursor()
            c.execute("INSERT INTO predictions (filename, emotion) VALUES (?, ?)",
                      (file.filename, predicted_emotion))
            conn.commit()
            conn.close()

            return render_template("index.html", prediction=predicted_emotion, image_path=filepath)

    return render_template("index.html", prediction=None)


# -----------------------------
# 4. Run app
# -----------------------------
if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


