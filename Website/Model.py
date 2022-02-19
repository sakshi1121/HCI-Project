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
import os
import glob
import pickle
import numpy as np
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


@app.route('/user', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # if 'file' not in request.files:
        #     print('No file attached in request')
        #     return redirect(request.url)
        file = request.files['q1']
        file1 = request.files['q2']
        file2 = request.files['q3']
        file3 = request.files['q4']
        file4 = request.files['q5']

        if file.filename == '' or file1.filename == '' or file2.filename == '' or file3.filename == '' or file4.filename == '':
            print('No file selected')
            return redirect(request.url)

        if file and file1 and file2 and file3 and file4 and allowed_file(file.filename) and allowed_file(file1.filename) and allowed_file(file2.filename) and allowed_file(file3.filename) and allowed_file(file4.filename):
            filename = secure_filename(file.filename)
            filename1 = secure_filename(file1.filename)
            filename2 = secure_filename(file2.filename)
            filename3 = secure_filename(file3.filename)
            filename4 = secure_filename(file4.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
            file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
            file3.save(os.path.join(app.config['UPLOAD_FOLDER'], filename3))
            file4.save(os.path.join(app.config['UPLOAD_FOLDER'], filename4))
            print('----------Files uploaded----------: \n')
            emotion1 = emotion_detection()
            data = {filename: emotion1}
            final_ranks = dict(
                sorted(data.items(), key=operator.itemgetter(1), reverse=True))
            return render_template('results.html', emotion_detection=final_ranks)
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
    for file in glob.glob("/uploads/q*/*.wav"):
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        testfile.append(feature)
    result1 = model.predict(testfile)
    return result1


if __name__ == "__main__":
    app.run(debug=True)
