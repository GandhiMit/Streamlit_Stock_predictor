import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Attention

st.title("Stock Price Prediction")

# Default values
DEFAULT_COMPANY = "TCS.NS"
DEFAULT_START_DATE = date(2020, 1, 1)
DEFAULT_END_DATE = date.today()
DEFAULT_FACTOR = 28
DEFAULT_PREDICTION_DAYS = 20

# User inputs
asset_type = st.selectbox("Asset Type", ["Stock", "Cryptocurrency"])
company = st.text_input("Company/Crypto Symbol", value=DEFAULT_COMPANY)
start_date = DEFAULT_START_DATE
end_date =  DEFAULT_END_DATE
factor = DEFAULT_FACTOR
price_type = st.selectbox("Price Type", ["Open", "Close", "High", "Low"])
prediction_days = st.number_input("Number of days to predict", value=DEFAULT_PREDICTION_DAYS, min_value=1, max_value=365)

def generate_prediction_dates(start_date, num_days):
    dates = []
    current_date = start_date
    while len(dates) < num_days:
        if asset_type == "Cryptocurrency" or current_date.weekday() < 5:
            dates.append(current_date)
        current_date += timedelta(days=1)
    return dates

def run_model():
    data = yf.download(company, start=start_date, end=end_date, progress=False)
    if data.empty:
        st.error("No data available for the selected company and date range.")
        return

    if data.isnull().sum().any():
        data.fillna(method="ffill", inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[price_type].values.reshape(-1, 1))

    X, y = [], []
    for i in range(factor, len(scaled_data)):
        X.append(scaled_data[i - factor: i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    input_layer = Input(shape=(X_train.shape[1], 1))
    lstm_out = LSTM(50, return_sequences=True)(input_layer)
    lstm_out = LSTM(50, return_sequences=True)(lstm_out)
    attention_out = Attention()([lstm_out, lstm_out])
    dense_out = Dense(1)(attention_out)
    model = Model(inputs=input_layer, outputs=dense_out)
    model.compile(optimizer="adam", loss="mean_squared_error")

    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    st.subheader("Model Evaluation")
    st.write(f"Mean Absolute Error: {mae}")
    st.write(f"Root Mean Square Error: {rmse}")

    # Prediction for future dates
    last_sequence = scaled_data[-factor:]
    predicted_prices = []

    for _ in range(prediction_days):
        next_day_prediction = model.predict(last_sequence.reshape(1, factor, 1))
        predicted_prices.append(next_day_prediction[0, 0])
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = next_day_prediction

    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

    last_date = data.index[-1]
    prediction_dates = generate_prediction_dates(last_date + timedelta(days=1), prediction_days)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-30:], data[price_type][-30:], label="Actual Prices")
    plt.plot(prediction_dates, predicted_prices, label="Predicted Prices")
    plt.title(f"{company} Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt)

    # Display predicted prices
    predictions_df = pd.DataFrame({"Date": prediction_dates, "Predicted Price": predicted_prices.flatten()})
    st.subheader("Predicted Prices")
    st.dataframe(predictions_df)

if st.button("Run Prediction"):
    with st.spinner('Training the model...'):
        run_model()
