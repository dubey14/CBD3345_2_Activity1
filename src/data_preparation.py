import numpy as np

# Generate a synthetic regression dataset
def generate_regression_data(num_samples=1000):
    np.random.seed(0)
    X = np.random.rand(num_samples, 1)  # Feature (input)
    y = 2 * X + 1 + 0.1 * np.random.randn(num_samples, 1)  # Target (output)
    return X, y
