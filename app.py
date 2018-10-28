#!flask/bin/python
from flask import Flask, jsonify
from flask import request, send_file
import io

import base64
import tempfile
import os
# from PIL import Image

import numpy as np
import librosa
import pickle
import keras
from keras.models import load_model
import keras.preprocessing.text
app = Flask(__name__, static_folder='public/app', static_url_path='')


model = load_model("../data/wr_72e_30.h5")
tokenizer = pickle.load(open("../data/tokenizer_30.pkl", "rb"))


index_word = {tokenizer.word_index[key]:key for key in tokenizer.word_index}

def melspec(file):
    data, fs = librosa.load(file, mono=True, sr=44e3)
    D = np.abs(librosa.stft(data))**2
    S = librosa.feature.melspectrogram(S=D)
    return S.reshape(-1)


def data_generation(file, batch_size):
    'Generates data containing batch_size samples'

    kernel_size=(256,43,1)
    X = np.empty((batch_size, 256, 43, 1), dtype=np.ndarray)
    # y = np.empty((batch_size), dtype=object)

    # Generate data
        #for i, file in enumerate(files):
        # load mfcc
    data = melspec(file)
    if len(data) >= 11008:
        data=data[:11008]
    else:
        data = np.concatenate( (data, np.zeros(11008 - len(data))), axis=0)

    X = data.reshape(kernel_size)
        # Store class
        #y[i] = file_to_label[file]
        # print(y)

    return X #, tokenizer.texts_to_matrix(y)


@app.route('/')
def index():
    return send_file("public/app/index.html")


@app.route('/api/predict', methods=['POST'])
def post():
    f = request.files['data']
    f.save('temp.wav')
    # class_name, accuracy = model.predict('temp.bin')
    x = data_generation('temp.wav', 1)

    pp = model.predict(np.array([x]))

    class_id = np.argmax(pp)
    accuracy = np.amax(pp)

    class_name = index_word[class_id]

    print(class_name, accuracy)

    return jsonify({"class_name": class_name, "accuracy": float(accuracy)}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT')) if None else 8080
    app.run(host='0.0.0.0', port=port, debug=False, threaded=False, ssl_context='adhoc')