
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Function to load data
@st.cache_data # Cache data to improve performance
def load_data(filepath):
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        st.error(f"Error: File not found at {filepath}")
        return None

# Load the data
df_weather = load_data('farmer_weather_data.csv')
df_prophet = load_data('prophet_forecast.csv')
df_arima = load_data('arima_forecast.csv')

# Check if data loaded successfully
if df_weather is not None and df_prophet is not None and df_arima is not None:
    st.title("Melbourne Temperature Forecast Dashboard")

    st.write("This dashboard displays actual temperature data and forecasts from three models: Linear Regression, Prophet, and ARIMA.")

    # Data Preparation for Plotting
    df_weather['Date'] = pd.to_datetime(df_weather['Date'])
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    df_arima['Date'] = pd.to_datetime(df_arima['Date'])

    # Combine actual and LR predictions for plotting
    df_actual_lr = df_weather[['Date', 'Day Temp (°C)', 'LR_Pred']].set_index('Date')

    # Prepare Prophet forecast for plotting
    df_prophet_plot = df_prophet.rename(columns={'ds': 'Date', 'yhat': 'Prophet Forecast'}).set_index('Date')

    # Prepare ARIMA forecast for plotting
    df_arima_plot = df_arima.rename(columns={'ARIMA_Pred': 'ARIMA Forecast'}).set_index('Date')

    # Display Actual Temperature and Linear Regression Predictions
    st.subheader("Actual Temperature and Linear Regression Predictions")
    st.line_chart(df_actual_lr)
    st.write("Linear Regression models the relationship between temperature, day index, and humidity.")

    # Display Prophet Forecast
    st.subheader("Prophet Forecast")
    st.line_chart(df_prophet_plot)
    st.write("Prophet is a time series forecasting model developed by Facebook, designed for time series with strong seasonal effects.")

    # Display ARIMA Forecast
    st.subheader("ARIMA Forecast")
    st.line_chart(df_arima_plot)
    st.write("ARIMA (AutoRegressive Integrated Moving Average) is a statistical model for time series forecasting.")

    # Model Evaluation Metrics (from previous notebook output)
    st.subheader("Model Evaluation (on Training Data)")
    st.write("📊 Model Evaluation:")
    st.write("- Linear Regression RMSE: 3.28") # Hardcoded from previous output
    st.write("- Prophet RMSE (Training Data): 2.38") # Hardcoded from previous output
    st.write("- ARIMA RMSE on training data can be calculated from model residuals but not directly compared to forecast plot.")

else:
    st.error("Could not load all necessary data files. Please ensure 'farmer_weather_data.csv', 'prophet_forecast.csv', and 'arima_forecast.csv' are in the same directory.")
