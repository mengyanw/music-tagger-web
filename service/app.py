from flask import Flask, request
from flask import jsonify

from flask_cors import CORS, cross_origin

import yaml
from dataset import *
import torch
import pandas as pd

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(config)

def convert(filename):

    waveform, _ = librosa.load(filename, sr=config['sample_rate'], mono=True)
    mel = librosa.feature.melspectrogram(y=waveform,
                                         sr=config['sample_rate'],
                                         n_fft=config['n_fft'],
                                         hop_length=config['hop_length'],
                                         n_mels=config['n_mels'],
                                         fmin=config['fmin'],
                                         fmax=config['fmax'])
    length = int(
            (10 * config['sample_rate'] + config['hop_length'] - 1) // config['hop_length'])
    mel = clip(mel, length)
    return mel


def predict(model_path, mel, config):
    model = torch.load(model_path)
    model.eval()
    mel = torch.tensor(mel).unsqueeze(0)
    pred = model(mel)
    prob, indices = pred.topk(k=config['topk'])
    category = []
    tags = []
    probs = []
    for i in range(config['topk']):
        category.append(TAGS[i].split('---')[0].capitalize())
        tags.append(TAGS[i].split('---')[1].capitalize())
        probs.append(np.round(prob[0][i].detach().numpy(), 4))
    output_df = pd.DataFrame(list(zip(category, tags, probs)), columns=['Category', 'Tag', 'Probability'])
    output_df = output_df.sort_values(by='Probability', ascending=False)
    return output_df

# def process_uploaded_file(file_path):
#     mel = convert(file_path)
#     output_df = predict(mel, config)
#     st.write("Result:")
#     st.dataframe(output_df)


@app.route('/hello/', methods=['GET', 'POST'])
@cross_origin()
def welcome():
    return "Hello World!"


@app.route('/predict/', methods=['GET', 'POST'])
@cross_origin()
def predict1():
    return jsonify(data=[1, 2, 3])


@app.route('/predict2/', methods=['GET', 'POST'])
@cross_origin()
def predict2():
    if request.method == 'POST':
        print(request.form['audioPath'])
        print(request.form['modelPath'])
        file_path = request.form['audioPath']
        model_path = request.form['modelPath']
        mel = convert(file_path)
        print(mel)
        result = predict(model_path, mel, config)
        print(result.reset_index().to_json(orient="records"))
        
        return result.reset_index().to_json(orient="records")


if __name__ == '__main__':
    app.run(debug=True)
