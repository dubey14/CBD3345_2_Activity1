from src.data_preparation import generate_regression_data
from src.model import create_regression_model
from src.train import train_regression_model
from src.predict import predict_regression

# Generate synthetic regression data
X_train, y_train = generate_regression_data()

# Create and train the regression model
regression_model = create_regression_model(X_train.shape[1:])
train_regression_model(regression_model, X_train, y_train, epochs=100, batch_size=32)

# Perform regression prediction
sample_input = [[0.5]]  # Example input for prediction
predicted_value = predict_regression(regression_model, sample_input)
print(f"Predicted value: {predicted_value[0][0]}")
