# In-browser music tagger
## Implementation 1:  React + Flask website (main branch)
### Setup
#### For UI (react)
install dependencies `npm install`

run `npm start`

The website will run on http://localhost:3000/

#### For flask service
go to service folder `cd service`

install dependencies `pip install -r requirements.txt`

start server `python app.py`

The server will run on http://localhost:5000/

#### Use ngrok to put the app on internet
`ngrok http --domain=xxx 3000 --host-header="localhost:5000"`


## Implementation 2: static React + ONNX website (onnx branch)
### Setup
install dependencies `npm install`

run `npm start`

The website will run on http://localhost:3000/

#### Use ngrok to put the app on internet
`ngrok http --domain=xxx 3000`


## Implementation 3: Streamlit
see https://github.com/Jenniferlyx/si699-music-tagging

