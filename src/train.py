import tensorflow as tf

# Train the regression model
def train_regression_model(model, X_train, y_train, epochs=10, batch_size=32):
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
