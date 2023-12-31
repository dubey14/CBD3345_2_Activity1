from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from src.data_preparation import generate_regression_data
from src.model import create_regression_model
from src.train import train_regression_model
from src.predict import predict_regression
import os

app = Flask(__name__)

# Define your fixtures and model loading here

# Define the model globally so it can be accessed by the predict function
model = None
X_train, y_train = generate_regression_data()
model = create_regression_model(X_train.shape[1:])
train_regression_model(model, X_train, y_train, epochs=1, batch_size=32)
@app.route('/')
def hello():
    return 'Welcome to the Regression Model API!'

@app.route('/predict', methods=['GET'])
def predict():
    global model  # Access the globally defined model
    if model is None:
        return "Model is not loaded. Please load the model first."

    # sample_input = [[0.5]]
    sample_input = np.random.rand(1, 1)  # Example input for prediction
    predicted_value = predict_regression(model, sample_input)
    print(predicted_value)
    # return jsonify({'predicted_value': predicted_value.tolist()})
    html_response = f"<h1>Predicted Value: {predicted_value[0][0]}</h1>"

    return html_response

if __name__ == '__main__':
    # Load your regression model here
    

    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
