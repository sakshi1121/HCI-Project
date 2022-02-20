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
        file1 = request.files['q1']
        file2 = request.files['q2']
        file3 = request.files['q3']
        file4 = request.files['q4']
        file5 = request.files['q5']
        file6 = request.files['q6']
        file7 = request.files['q7']
        file8 = request.files['q8']
        file9 = request.files['q9']
        file10 = request.files['q10']
        if file1.filename == '' or file2.filename == '' or file3.filename == '' or file4.filename == '' or file5.filename == '' or file6.filename == '' or file7.filename == '' or file8.filename == '' or file9.filename == '' or file10.filename == '':
            print('No file selected')
            return redirect(request.url)

        if file1 and file2 and file3 and file4 and file5 and file6 and file7 and file8 and file9 and file10 and allowed_file(file1.filename) and allowed_file(file2.filename) and allowed_file(file3.filename) and allowed_file(file4.filename) and allowed_file(file5.filename) and allowed_file(file6.filename) and allowed_file(file7.filename) and allowed_file(file8.filename) and allowed_file(file9.filename) and allowed_file(file10.filename):
            filename1 = secure_filename(file1.filename)
            filename2 = secure_filename(file2.filename)
            filename3 = secure_filename(file3.filename)
            filename4 = secure_filename(file4.filename)
            filename5 = secure_filename(file5.filename)
            filename6 = secure_filename(file6.filename)
            filename7 = secure_filename(file7.filename)
            filename8 = secure_filename(file8.filename)
            filename9 = secure_filename(file9.filename)
            filename10 = secure_filename(file10.filename)
            file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
            file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
            file3.save(os.path.join(app.config['UPLOAD_FOLDER'], filename3))
            file4.save(os.path.join(app.config['UPLOAD_FOLDER'], filename4))
            file5.save(os.path.join(app.config['UPLOAD_FOLDER'], filename5))
            file6.save(os.path.join(app.config['UPLOAD_FOLDER'], filename6))
            file7.save(os.path.join(app.config['UPLOAD_FOLDER'], filename7))
            file8.save(os.path.join(app.config['UPLOAD_FOLDER'], filename8))
            file9.save(os.path.join(app.config['UPLOAD_FOLDER'], filename9))
            file10.save(os.path.join(app.config['UPLOAD_FOLDER'], filename10))
            print('----------Files uploaded----------: \n')
            emotions = emotion_detection()
            data = {filename1: emotions[0], filename2: emotions[1],
                    filename3: emotions[2], filename4: emotions[3], filename5: emotions[4],
                    filename6: emotions[5], filename7: emotions[6],
                    filename8: emotions[7], filename9: emotions[8], filename10: emotions[9]}
            percentages = emotion_percentages(emotions)
            data2 = {'Emotions': 'percentages', 'calm': percentages[0], 'happy': percentages[1],
                     'sad': percentages[2], 'angry': percentages[3], 'fearful': percentages[4]}
            delete_files()
            return render_template('results.html', emotion_detection=data, percentages_data=data2)
    return render_template('forms.html')


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
    emotions = []
    model = joblib.load(os.path.join(os.path.dirname(
        os.path.realpath(__file__)), 'model/finalized_model.sav'))
    testfile = []
    for file in glob.glob("uploads/*.wav"):
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        testfile.append(feature)
    emotions = model.predict(testfile)
    print(emotions)
    return emotions


def emotion_percentages(emotions):
    percentages = []
    percentages = [0 for i in range(5)]
    for i in range(0, 10):
        if emotions[i] == 'calm':
            percentages[0] = percentages[0]+1
        elif emotions[i] == 'happy':
            percentages[1] = percentages[1]+1
        elif emotions[i] == 'sad':
            percentages[2] = percentages[2]+1
        elif emotions[i] == 'angry':
            percentages[3] = percentages[3]+1
        elif emotions[i] == 'fearful':
            percentages[4] = percentages[4]+1
    for i in range(0, 5):
        percentages[i] = percentages[i]*10
    return percentages


def delete_files():
    for filename in glob.glob(os.path.join(folder_path, '*.wav')):
        print(filename)
        if os.path.exists(filename):
            os.remove(filename)


if __name__ == "__main__":
    app.run(debug=True)
