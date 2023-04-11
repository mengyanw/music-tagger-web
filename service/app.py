from flask import Flask, request
from flask_cors import CORS, cross_origin

import yaml

import sys
sys.path.append('ml/run')
from dataset import *
from train import *
from models import *
from music_tagger_app import convert, predict

import torch
import pandas as pd

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
UPLOAD_FOLDER = '/userUploads'
ALLOWED_EXTENSIONS = {'mp3'}

torch.manual_seed(0)
np.random.seed(0)
feature_extractor_type = 'raw'
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(config)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/hello/', methods=['GET', 'POST'])
@cross_origin()
def welcome():
    return "Hello World!"


@app.route('/api/predict/', methods=['GET', 'POST'])
@cross_origin()
def _predict():
    if request.method == 'POST':
        if not request.files.get('uploadedAudio'):
            file_path = request.form['audioPath']
            model_path = request.form['modelPath']
        else:
            file_path = request.files['uploadedAudio']
            model_path = request.form['modelPath']
        print(file_path, model_path)
        mel = convert(file_path)
        print(mel)
        result = predict(model_path, mel, config)
        print(result.reset_index().to_json(orient="records"))
        
        return result.reset_index().to_json(orient="records")



if __name__ == '__main__':
    app.run(debug=True)
