import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, AdditiveAttention, Flatten, Multiply
from tensorflow.keras.callbacks import EarlyStopping
from huggingface_hub import HfApi, HfFolder
from huggingface_hub import login
from transformers import TFAutoModel
import os

# Hugging Face login
hf_token = st.secrets["HF_TOKEN"]  # Store your Hugging Face token in Streamlit secrets
login(token=hf_token)

st.title("Stock Price Prediction")

# Default values
DEFAULT_COMPANY = "TCS.NS"
DEFAULT_START_DATE = date(2020, 1, 1)
DEFAULT_END_DATE = date.today()
DEFAULT_START_DATE_PREDICTION = date(2024, 3, 7)
DEFAULT_END_DATE_PREDICTION = date.today()
DEFAULT_FACTOR = 28
DEFAULT_PREDICTION_DAYS = 20

# User inputs
asset_type = st.selectbox("Asset Type", ["Stock", "Cryptocurrency"])
company = st.text_input("Company/Crypto Symbol", value=DEFAULT_COMPANY)
start_date = DEFAULT_START_DATE
end_date = DEFAULT_END_DATE
start_date_prediction = DEFAULT_START_DATE_PREDICTION
end_date_prediction = DEFAULT_END_DATE_PREDICTION

st.write("Validate that the batch_size is not more than 28 units")
factor = st.number_input("Training Batch size", value=DEFAULT_FACTOR)

price_type = st.selectbox("Price Type", ["Open", "Close", "High", "Low"])
save_model = st.checkbox("Save model after training", value=True)
prediction_days = st.number_input("Number of days to predict", value=DEFAULT_PREDICTION_DAYS, min_value=1, max_value=365)

def generate_prediction_dates(start_date, num_days):
    dates = []
    current_date = start_date
    while len(dates) < num_days:
        if asset_type == "Cryptocurrency" or current_date.weekday() < 5:
            dates.append(current_date)
        current_date += timedelta(days=1)
    return dates

def save_model_to_huggingface(model, model_name):
    # Save the model locally first
    model.save(model_name)
    
    # Upload the model to Hugging Face
    api = HfApi()
    api.upload_folder(
        folder_path=model_name,
        repo_id=f"your-huggingface-username/{model_name}",
        repo_type="model",
    )
    st.success(f"Model uploaded to Hugging Face: your-huggingface-username/{model_name}")

def load_model_from_huggingface(model_name):
    try:
        model = TFAutoModel.from_pretrained(f"your-huggingface-username/{model_name}")
        st.success(f"Model loaded from Hugging Face: your-huggingface-username/{model_name}")
        return model
    except Exception as e:
        st.error(f"Error loading model from Hugging Face: {e}")
        return None

def run_model():
    # Data Loader
    data = yf.download(company, start=start_date, end=end_date)

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

    model_name = f"{company}_model_PT_{price_type}"
    
    # Try to load the model from Hugging Face
    model = load_model_from_huggingface(model_name)
    
    if model is None:
        # If the model doesn't exist on Hugging Face, create and train a new one
        input_layer = Input(shape=(X_train.shape[1], 1))
        lstm_out = LSTM(50, return_sequences=True)(input_layer)
        lstm_out = LSTM(50, return_sequences=True)(lstm_out)

        query = Dense(50)(lstm_out)
        value = Dense(50)(lstm_out)
        attention_out = AdditiveAttention()([query, value])

        multiply_layer = Multiply()([lstm_out, attention_out])

        flatten_layer = Flatten()(multiply_layer)
        output_layer = Dense(1)(flatten_layer)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.summary()

        early_stopping = EarlyStopping(monitor="val_loss", patience=10)
        history = model.fit(X_train, y_train, epochs=100, batch_size=25, validation_split=0.2,
                            callbacks=[early_stopping])

        if save_model:
            save_model_to_huggingface(model, model_name)

    test_loss = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    st.subheader("Model Evaluation")
    st.write(f"Test Loss: {test_loss}")
    st.write(f"Mean Absolute Error: {mae}")
    st.write(f"Root Mean Square Error: {rmse}")

    prediction_data = yf.download(company, start=start_date_prediction, end=end_date_prediction)
    st.write("Latest Stock/Crypto Data for Prediction")
    st.write(prediction_data.tail())

    closing_prices = prediction_data[price_type].values
    scaled_data = scaler.fit_transform(closing_prices.reshape(-1, 1))
    X_latest = np.array([scaled_data[-factor:].reshape(factor)])
    X_latest = np.reshape(X_latest, (X_latest.shape[0], X_latest.shape[1], 1))

    predicted_stock_price = model.predict(X_latest)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

    st.write("Predicted Price for the next day: ", predicted_stock_price[0][0])

    predicted_prices = []
    current_batch = scaled_data[-factor:].reshape(1, factor, 1)

    for i in range(prediction_days):
        next_prediction = model.predict(current_batch)
        next_prediction_reshaped = next_prediction.reshape(1, 1, 1)
        current_batch = np.append(current_batch[:, 1:, :], next_prediction_reshaped, axis=1)
        predicted_prices.append(scaler.inverse_transform(next_prediction)[0, 0])

    last_date = prediction_data.index[-1]
    next_day = last_date + timedelta(days=1)
    prediction_dates = generate_prediction_dates(next_day, prediction_days)
    predictions_df = pd.DataFrame(index=prediction_dates, data=predicted_prices, columns=[price_type])

    plt.figure(figsize=(12, 6))
    plt.plot(prediction_data.index[-factor:], prediction_data[price_type][-factor:], linestyle="-", marker="o", color="blue", label="Actual Data")
    plt.plot(prediction_dates, predicted_prices, linestyle="-", marker="o", color="red", label="Predicted Data")

    for i, price in enumerate(prediction_data[price_type][-factor:]):
        plt.annotate(f'{price:.2f}', (prediction_data.index[-factor:][i], price), textcoords="offset points", xytext=(0, 10), ha='center')

    for i, price in enumerate(predicted_prices):
        plt.annotate(f'{price:.2f}', (prediction_dates[i], price), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.title(f"{company} Price: Last {factor} Days and Next {prediction_days} Days Predicted")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

    plt.figure(figsize=(12, 6))
    plt.plot(prediction_dates, predicted_prices, linestyle="-", marker="o", color="red", label="Predicted Data")

    for i, price in enumerate(predicted_prices):
        plt.annotate(f'{price:.2f}', (prediction_dates[i], price), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.title(f"{company} Predicted Prices for Next {prediction_days} Days")
    plt.xlabel("Date")
    plt.ylabel("Predicted Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

if st.button("Run Prediction"):
    with st.spinner('Training the model...'):
        run_model()
