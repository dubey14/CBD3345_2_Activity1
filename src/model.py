import tensorflow as tf
from tensorflow.keras import layers, models

# Create a simple linear regression model
def create_regression_model(input_shape):
    model = models.Sequential()
    model.add(layers.Dense(1, input_shape=input_shape))
    return model
