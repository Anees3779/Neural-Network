import numpy as np
import pandas as pd
from src.RNN.RNN import TemperaturePredictionModel  # Assuming the class is in a file named temperature_prediction_model.py

# Generate a sample dataset
np.random.seed(0)
dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
temperatures = np.random.normal(loc=25, scale=5, size=len(dates))  # Random daily temperatures
df = pd.DataFrame({'Date': dates, 'Temperature': temperatures})

# Create an instance of TemperaturePredictionModel
model = TemperaturePredictionModel()

# Preprocess the data
X_train, X_test, y_train, y_test = model.preprocess_data(df)

# Train the model
model.train_model(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
y_true, predictions = model.evaluate(y_test, predictions)

# Plot the results
model.plot_results(df, y_true, predictions)
