import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# App Title
st.title("📈 Stock Price Predictor App (12-Month Forecast)")

# User input
stock = st.text_input("Enter the Stock Ticker (e.g., AAPL, GOOG, TSLA)", "GOOG")

try:
    # Load historical stock data
    end = datetime.now()
    start = datetime(end.year - 10, end.month, end.day)
    df = yf.download(stock, start=start, end=end)

    if df.empty:
        st.error("No data found for the provided ticker.")
    else:
        # Show historical data
        st.subheader("📊 Historical Stock Data")
        st.write(df.tail())

        # Calculate moving averages
        df['MA100'] = df['Close'].rolling(100).mean()
        df['MA200'] = df['Close'].rolling(200).mean()
        df['MA250'] = df['Close'].rolling(250).mean()

        # Plot function
        def plot_graph(title, *lines):
            fig, ax = plt.subplots(figsize=(14,6))
            for line in lines:
                ax.plot(df.index, line[0], label=line[1])
            ax.plot(df.index, df['Close'], label='Close Price', color='blue')
            ax.set_title(title)
            ax.legend()
            st.pyplot(fig)

        plot_graph("Close vs MA100", (df['MA100'], 'MA100'))
        plot_graph("Close vs MA200", (df['MA200'], 'MA200'))
        plot_graph("Close vs MA250", (df['MA250'], 'MA250'))
        plot_graph("Close vs MA100 & MA250", (df['MA100'], 'MA100'), (df['MA250'], 'MA250'))

        # Load model
        try:
            model = load_model("Latest_stock_price_model.h5")
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            raise

        # Preprocess data
        data = df[['Close']].dropna()  # Ensure no NaN values
        dataset = data.values

        if dataset.shape[0] < 100:
            st.error("Not enough data points to make prediction (need at least 100).")
        else:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(dataset)

            # Prepare input
            look_back = 100
            x_input = scaled_data[-look_back:].reshape(1, look_back, 1)

            # Predict next 365 days
            predicted_prices = []
            for _ in range(365):
                pred = model.predict(x_input, verbose=0)
                predicted_prices.append(pred[0][0])
                x_input = np.append(x_input[:, 1:, :], [[[pred[0][0]]]], axis=1)

            # Convert back to original scale
            predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

            # Create date range
            future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=365)
            pred_df = pd.DataFrame(predicted_prices, index=future_dates, columns=["Predicted Price"])

            # Show prediction
            st.subheader("📈 Predicted Stock Prices (Next 12 Months)")
            st.line_chart(pred_df)

except Exception as e:
    st.error(f"❌ An error occurred: {e}")
