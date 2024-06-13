import os
import numpy as np
import pickle
from flask import Flask, request, render_template
import librosa

app = Flask(__name__)

# Load the trained models, scaler, and genre dictionary
models = {}
models['svm'] = pickle.load(open('pickle/svm_model.pkl', 'rb'))
models['knn'] = pickle.load(open('pickle/knn_model.pkl', 'rb'))
models['logreg'] = pickle.load(open('pickle/logreg_model.pkl', 'rb'))
models['xgbc'] = pickle.load(open('pickle/xgbc_model.pkl', 'rb'))
scaler = pickle.load(open('pickle/scaler.pkl', 'rb'))
genre_dict = pickle.load(open('pickle/label_encoder.pkl', 'rb'))

# Function to extract metadata from a WAV file
def getmetadata(filename):
    y, sr = librosa.load(filename)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zero_crossing = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    metadata_dict = {'tempo': tempo, 'chroma_stft': np.mean(chroma_stft), 'rmse': np.mean(rmse),
                     'spectral_centroid': np.mean(spec_centroid), 'spectral_bandwidth': np.mean(spec_bw),
                     'rolloff': np.mean(spec_rolloff), 'zero_crossing_rates': np.mean(zero_crossing)}
    for i in range(1, 21):
        metadata_dict.update({'mfcc' + str(i): np.mean(mfcc[i - 1])})

    return list(metadata_dict.values())

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling file upload and making predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)

        # Extract features from the uploaded WAV file
        features = getmetadata(filepath)
        features = np.array(features, dtype=object).reshape(1, -1)
        features_scaled = scaler.transform(features)

        # Make predictions using the loaded models
        predictions = {name: model.predict(features_scaled)[0] for name, model in models.items()}
        decoded_predictions = {name: genre_dict[pred] for name, pred in predictions.items()}

        return render_template('result.html', predictions=decoded_predictions)

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
