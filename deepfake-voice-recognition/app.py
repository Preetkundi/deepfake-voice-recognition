import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─── Constants (copy these exactly from your training run) ───────────────────
MODEL_PATH = 'models/deepfake_final_model.h5'
MAX_LEN    = 56239    # the 95th-percentile padding length you used
THR_LOW    = 0.426    # perfect-recall threshold
THR_HIGH   = 0.566    # max-F1 threshold

# ─── Flask setup ────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)
os.makedirs('uploads', exist_ok=True)

# ─── Load the trained model once ─────────────────────────────────────────────
model = tf.keras.models.load_model(MODEL_PATH)

# ─── Feature extraction (must match your notebook) ──────────────────────────
def extract_features(audio, sr, n_mfcc=13):
    # Compute MFCC + delta + delta-delta
    mfcc    = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    delta1  = librosa.feature.delta(mfcc)
    delta2  = librosa.feature.delta(mfcc, order=2)
    stacked = np.vstack([mfcc, delta1, delta2])
    # Per-feature mean-variance normalization
    normed  = (stacked - stacked.mean(axis=1, keepdims=True)) / (
                stacked.std(axis=1, keepdims=True) + 1e-6
              )
    return normed.T   # shape: (frames, features)

def preprocess_audio(path):
    # 1) Load WAV at 16 kHz mono
    audio, sr = librosa.load(path, sr=16000, mono=True)
    # 2) Extract features
    feats = extract_features(audio, sr)
    # 3) Pad or truncate to MAX_LEN
    padded = pad_sequences(
        [feats],
        maxlen=MAX_LEN,
        dtype='float32',
        padding='post',
        truncating='post'
    )
    return padded

# ─── Routes ─────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1) Check upload
    if 'audio' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    f = request.files['audio']
    if f.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 2) Save temporarily
    tmp_path = os.path.join('uploads', f.filename)
    f.save(tmp_path)

    try:
        # 3) Preprocess & predict
        x       = preprocess_audio(tmp_path)        # shape: (1, MAX_LEN, features)
        probs   = model.predict(x)[0]               # [P(real), P(fake)]
        prob_f  = float(probs[1])

        # 4) Three-way decision
        if prob_f >= THR_HIGH:
            label = 'Fake'
        elif prob_f < THR_LOW:
            label = 'Real'
        else:
            label = 'Review'

        # 5) Build response
        res = {
            'prediction': label,
            'prob_real':  round(float(probs[0]), 4),
            'prob_fake':  round(prob_f,        4),
            'thr_low':    THR_LOW,
            'thr_high':   THR_HIGH
        }
    except Exception as e:
        res = {'error': f'Processing error: {str(e)}'}
    finally:
        # 6) Clean up
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return jsonify(res)

# ─── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
