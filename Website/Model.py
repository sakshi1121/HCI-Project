from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import glob
import json
import openpyxl
from openpyxl import Workbook
from pathlib import Path
import numpy as np
import pickle
import joblib
import pandas as pd
import operator
import librosa
import soundfile
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pickle
import tensorflow as tf
from tensorflow import keras


ALLOWED_EXTENSIONS = {'wav'}
UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + r'/uploads/'
folder_path = os.path.dirname(os.path.abspath(__file__)) + r'\uploads'
app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # if 'file' not in request.files:
        #     print('No file attached in request')
        #     return redirect(request.url)
        file1 = request.files['q1']
        file2 = request.files['q2']
        file3 = request.files['q3']
        file4 = request.files['q4']
        file5 = request.files['q5']

        if file1.filename == '' or file2.filename == '' or file3.filename == '' or file4.filename == '' or file5.filename == '':
            print('No file selected')
            return redirect(request.url)

        if file1 and file2 and file3 and file4 and file5 and allowed_file(file1.filename) and allowed_file(file2.filename) and allowed_file(file3.filename) and allowed_file(file4.filename) and allowed_file(file5.filename):
            filename1 = secure_filename(file1.filename)
            filename2 = secure_filename(file2.filename)
            filename3 = secure_filename(file3.filename)
            filename4 = secure_filename(file4.filename)
            filename5 = secure_filename(file5.filename)
            file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
            file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
            file3.save(os.path.join(app.config['UPLOAD_FOLDER'], filename3))
            file4.save(os.path.join(app.config['UPLOAD_FOLDER'], filename4))
            file5.save(os.path.join(app.config['UPLOAD_FOLDER'], filename5))
            print('----------Files uploaded----------: \n')
            emotions = emotion_detection()
            data = {filename1: emotions[0], filename2: emotions[1],
                    filename3: emotions[2], filename4: emotions[3], filename5: emotions[4]}
            delete_files()
            return render_template('results.html', emotion_detection=data)
    return render_template('forms.html')

# Extract features (mfcc, chroma, mel) from a sound file


def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(
                y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(
                S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(
                X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        return result


def emotion_detection():
    model = joblib.load(os.path.join(os.path.dirname(
        os.path.realpath(__file__)), 'model/finalized_model.sav'))
    testfile = []
    for file in glob.glob("uploads/*.wav"):
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        testfile.append(feature)
    emotions = model.predict(testfile)
    print(emotions)
    return emotions


def delete_files():
    for filename in glob.glob(os.path.join(folder_path, '*.wav')):
        print(filename)
        if os.path.exists(filename):
            os.remove(filename)


if __name__ == "__main__":
    app.run(debug=True)
