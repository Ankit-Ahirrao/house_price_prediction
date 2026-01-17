import pickle
from  flask import Flask, request, jsonify, app, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('regmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = regmodel.transform(np.array(data).reshape(1,-1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    # Get form values and convert to float
    data = [float(x) for x in request.form.values()]

    # Convert to numpy array with correct shape
    final_input = np.array(data).reshape(1, -1)

    # Predict directly (NO scaling)
    output = regmodel.predict(final_input)[0]

    return render_template(
        "home.html",
        prediction_text=f"The predicted house price is {output}"
    )



if __name__ == "__main__":
    app.run(debug=True)
