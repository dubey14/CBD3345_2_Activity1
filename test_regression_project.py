import pytest
import tensorflow as tf
import numpy as np
from src.data_preparation import generate_regression_data
from src.model import create_regression_model
from src.train import train_regression_model
from src.predict import predict_regression

@pytest.fixture
def regression_data():
    X_train, y_train = generate_regression_data()
    return X_train, y_train

@pytest.fixture
def regression_model(regression_data):
    X_train, y_train = regression_data
    model = create_regression_model(X_train.shape[1:])
    train_regression_model(model, X_train, y_train, epochs=1, batch_size=32)
    return model

def test_regression_data_preparation(regression_data):
    X_train, y_train = regression_data
    assert X_train.shape[1] == 1
    assert y_train.shape[1] == 1

def test_regression_model_creation(regression_model):
    assert isinstance(regression_model, tf.keras.models.Sequential)

def test_regression_model_training(regression_data, regression_model):
    X_train, y_train = regression_data
    loss = regression_model.evaluate(X_train, y_train)
    assert loss < 10  # Adjust as needed

def test_regression_model_prediction(regression_model):
    sample_input = np.array([[0.5]])  # Example input for prediction
    predicted_value = predict_regression(regression_model, sample_input)
    # print(predicted_value)
    assert isinstance(predicted_value, np.ndarray)

if __name__ == '__main__':
    pytest.main()
