# Deepfake Voice Detector

An end‐to‐end pipeline for detecting deepfake voice recordings. From data preprocessing and model training in a Jupyter notebook, to a live Flask web app with an interactive, dark‐themed frontend.

---

## 🚀 Features

- **Robust Audio Preprocessing**  
  - MFCC (13 coefficients) + delta + delta–delta features  
  - Data augmentation: time-stretch (0.8×, 1.2×) and pitch-shift (±2 semitones)  
  - Padding/truncation to the 95th-percentile sequence length  

- **High‐Performance BiLSTM Model**  
  - Bidirectional LSTM layers (128 → 64 units) with masking  
  - Dropout & Dense layers for regularization  
  - Adam optimizer & categorical cross-entropy loss  
  - Class-weight balancing, EarlyStopping & ReduceLROnPlateau  
  - Threshold-based three-way decision: **Real**, **Review**, **Fake**  

- **Evaluation & Thresholding**  
  - 80/20 train/test split with stratification  
  - Precision/Recall/F₁ analysis  
  - Automatic branch: 100 % accuracy on “clear” cases  
  - Human-in-the-loop “Review” zone (~5 % of samples)  

- **Interactive Web App**  
  - Flask backend exposing a `/predict` JSON API  
  - Drag-and-drop upload + click selector  
  - Live audio waveform visualization & playback  
  - Theme toggle (dark/light), wobble animation, custom cursor & ripple effects  
  - Session history of past predictions  

---



