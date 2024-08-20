import pandas as pd
import yfinance as yf
import numpy as np
import datetime as dt
import streamlit as st

# Load the CPI data
def load_cpi_data(cpi_csv_path):
    cpi_data = pd.read_csv(cpi_csv_path, parse_dates=['Date'])
    cpi_data = cpi_data.sort_values('Date')
    return cpi_data

# Convert monthly CPI to daily CPI
def convert_monthly_to_daily(cpi_data):
    daily_cpi = []
    for i in range(len(cpi_data) - 1):
        start_date = cpi_data['Date'].iloc[i]
        end_date = cpi_data['Date'].iloc[i + 1]
        inflation_rate = cpi_data['CPI_MOM'].iloc[i]  # No conversion needed
        
        # Adjust date_range without 'closed' parameter
        date_range = pd.date_range(start_date, end_date - pd.Timedelta(days=1), freq='D')
        daily_cpi.extend([(date, inflation_rate) for date in date_range])
    
    # Append the last month data
    last_date = cpi_data['Date'].iloc[-1]
    last_inflation_rate = cpi_data['CPI_MOM'].iloc[-1]  # No conversion needed
    daily_cpi.extend([(date, last_inflation_rate) for date in pd.date_range(last_date, dt.datetime.today())])
    
    daily_cpi_df = pd.DataFrame(daily_cpi, columns=['Date', 'Daily_CPI'])
    return daily_cpi_df

# Adjust historical prices based on cumulative inflation calculated backwards
def adjust_prices_for_inflation(prices_df, daily_cpi_df):
    # Merge daily CPI into the stock data
    prices_df = prices_df.merge(daily_cpi_df, on='Date', how='left')
    prices_df['Daily_CPI'].fillna(method='ffill', inplace=True)  # Forward fill missing CPI values

    # Sort by date to calculate cumulative inflation backwards
    prices_df = prices_df.sort_values('Date')

    # Calculate cumulative inflation backwards
    prices_df['Cumulative_Inflation'] = 1.0
    for i in range(len(prices_df) - 2, -1, -1):
        prices_df['Cumulative_Inflation'].iloc[i] = prices_df['Cumulative_Inflation'].iloc[i + 1] * (1 + prices_df['Daily_CPI'].iloc[i])
    
    # Adjust prices based on cumulative inflation
    prices_df['Adjusted_Price'] = prices_df['Price'] * prices_df['Cumulative_Inflation']
    
    return prices_df

# Fetch historical stock prices
def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
        if stock_data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        stock_data.reset_index(inplace=True)
        stock_data.rename(columns={'Date': 'Date', 'Adj Close': 'Price'}, inplace=True)
        return stock_data[['Date', 'Price']]
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame(columns=['Date', 'Price'])

# Main function to adjust historical stock prices for inflation
def main(ticker, start_date, end_date, cpi_csv_path):
    # Load CPI data and convert to daily
    cpi_data = load_cpi_data(cpi_csv_path)
    daily_cpi_df = convert_monthly_to_daily(cpi_data)
    
    # Fetch stock data
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    
    # Adjust stock prices for inflation
    if not stock_data.empty:
        adjusted_stock_data = adjust_prices_for_inflation(stock_data, daily_cpi_df)
        return adjusted_stock_data
    else:
        return pd.DataFrame(columns=['Date', 'Price', 'Adjusted_Price'])

# Streamlit UI
st.title("Stock Price Adjustment for Inflation")

ticker = st.text_input("Enter stock ticker:", 'YPF.BA')
start_date = st.date_input("Start date:", dt.datetime(2023, 1, 1))
end_date = st.date_input("End date:", dt.datetime.today())

if st.button("Get Data and Plot"):
    cpi_csv_path = 'cpi_mom_data.csv'
    adjusted_stock_data = main(ticker, start_date, end_date, cpi_csv_path)
    
    if not adjusted_stock_data.empty:
        st.write(adjusted_stock_data.head())
        st.line_chart(adjusted_stock_data.set_index('Date')[['Price', 'Adjusted_Price']])
    else:
        st.write("No data available for the selected ticker and date range.")
