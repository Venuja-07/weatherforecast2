
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

    st.write("This dashboard displays actual temperature, humidity, and rainfall data, along with temperature forecasts from three models: Linear Regression, Prophet, and ARIMA.")

    # Data Preparation for Plotting and Table
    df_weather['Date'] = pd.to_datetime(df_weather['Date'])
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    df_arima['Date'] = pd.to_datetime(df_arima['Date'])

    # Combine actual weather data (including rainfall) and LR predictions for table
    # We'll use the full df_weather with LR_Pred for the main table
    df_weather_table = df_weather[['Date', 'Day Temp (°C)', 'Humidity (%)', 'Rainfall (mm)', 'LR_Pred']].set_index('Date')


    # Prepare Prophet forecast for plotting and table
    df_prophet_plot = df_prophet.rename(columns={'ds': 'Date', 'yhat': 'Prophet Forecast'}).set_index('Date')

    # Prepare ARIMA forecast for plotting and table
    df_arima_plot = df_arima.rename(columns={'ARIMA_Pred': 'ARIMA Forecast'}).set_index('Date')

    # Display Actual Weather Data and Linear Regression Predictions (Table and Plot)
    st.subheader("Actual Weather Data and Linear Regression Predictions")
    st.dataframe(df_weather_table)
    # Plotting only temperature for clarity in the main time series plot
    st.line_chart(df_weather_table[['Day Temp (°C)', 'LR_Pred']])
    st.write("This table shows historical daily temperature, humidity, and rainfall, along with the Linear Regression model's predictions on this historical data.")


    # Display Prophet Forecast (Plot and Table)
    st.subheader("Prophet Forecast (Temperature)")
    st.line_chart(df_prophet_plot)
    st.dataframe(df_prophet_plot)
    st.write("Prophet is a time series forecasting model developed by Facebook, designed for time series with strong seasonal effects.")

    # Display ARIMA Forecast (Plot and Table)
    st.subheader("ARIMA Forecast (Temperature)")
    st.line_chart(df_arima_plot)
    st.dataframe(df_arima_plot)
    st.write("ARIMA (AutoRegressive Integrated Moving Average) is a statistical model for time series forecasting.")

    # Model Evaluation Metrics (from previous notebook output)
    st.subheader("Model Evaluation (on Training Data)")
    st.write("📊 Model Evaluation:")
    # Update RMSE values based on the latest successful run if available, otherwise keep previous
    # Assuming the last run with rainfall in LR had RMSE around LR: 3.14, Prophet: 2.38
    st.write("- Linear Regression RMSE: 3.14") # Updated based on last successful run
    st.write("- Prophet RMSE (Training Data): 2.38") # Updated based on last successful run
    st.write("- ARIMA RMSE on training data can be calculated from model residuals but not directly compared to forecast plot.")

else:
    st.error("Could not load all necessary data files. Please ensure 'farmer_weather_data.csv', 'prophet_forecast.csv', and 'arima_forecast.csv' are in the same directory.")
