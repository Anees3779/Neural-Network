import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

class TemperaturePredictionModel:
    def __init__(self, sequence_length=10, epochs=20, batch_size=32):
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self._build_model()
        self.scaler = None  # Initialize scaler as None

    def _build_model(self):
        model = Sequential()
        model.add(SimpleRNN(units=50, input_shape=(self.sequence_length, 1)))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def preprocess_data(self, df):
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # Assign scaler to the class attribute
        df['Scaled_Temperature'] = self.scaler.fit_transform(df['Temperature'].values.reshape(-1, 1))

        X, y = [], []
        for i in range(len(df) - self.sequence_length):
            X.append(df['Scaled_Temperature'][i:i + self.sequence_length])
            y.append(df['Scaled_Temperature'][i + self.sequence_length])

        X, y = np.array(X), np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_true, predictions):
        predictions = self.scaler.inverse_transform(predictions)
        y_true = self.scaler.inverse_transform(y_true.reshape(-1, 1))

        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        print(f"Root Mean Squared Error (RMSE): {rmse}")

        return y_true, predictions

    def plot_results(self, df, y_true, predictions):
        plt.figure(figsize=(12, 6))
        plt.plot(df['Date'][-len(y_true):], y_true, label='Actual Temperatures')
        plt.plot(df['Date'][-len(y_true):], predictions, label='Predicted Temperatures')
        plt.title('Temperature Prediction using RNN')
        plt.xlabel('Date')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.show()
