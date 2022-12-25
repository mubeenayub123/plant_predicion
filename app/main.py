from flask import Flask
from flask import jsonify
import pandas as pd
from flask import request
from keras.models import load_model
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
     json_ = request.json
     query_df = pd.DataFrame(json_, index=[0])
     query = pd.get_dummies(query_df)
     model = load_model('model.h5')
     prediction = model.predict(query)
     return jsonify({'prediction': prediction.tolist()})
