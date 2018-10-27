#!flask/bin/python
from flask import Flask, jsonify
from flask import request, send_file
import io

import base64
import tempfile
import os
from PIL import Image


app = Flask(__name__, static_folder='public/app', static_url_path='')

@app.route('/')
def index():
    return send_file("public/app/index.html")


@app.route('/api/predict', methods=['POST'])
def post():
    f = request.files['data']
    f.save('temp.wav')
    # class_name, accuracy = model.predict('temp.bin')
    
    return jsonify({"class_name": 'hello', "accuracy": 0.88}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT')) if None else 8080
    app.run(host='0.0.0.0', port=port, debug=True)