import pandas as pd
import yfinance as yf
import numpy as np
import datetime as dt
import streamlit as st
import os

# Define the path for the CPI CSV file
cpi_csv_path = os.path.join(os.path.dirname(__file__), 'cpi_mom_data.csv')

# Function to convert monthly CPI to daily CPI
def convert_monthly_to_daily(cpi_data):
    daily_cpi = []
    for i in range(len(cpi_data) - 1):
        start_date = cpi_data['Date'].iloc[i]
        end_date = cpi_data['Date'].iloc[i + 1]
        inflation_rate = cpi_data['CPI_MOM'].iloc[i]
        date_range = pd.date_range(start_date, end_date - pd.Timedelta(days=1))  # Subtract one day from end_date
        daily_cpi.extend([(date, inflation_rate) for date in date_range])
    
    # Append the last month data
    last_date = cpi_data['Date'].iloc[-1]
    last_inflation_rate = cpi_data['CPI_MOM'].iloc[-1]
    daily_cpi.extend([(date, last_inflation_rate) for date in pd.date_range(cpi_data['Date'].iloc[-2], last_date)])
    
    daily_cpi_df = pd.DataFrame(daily_cpi, columns=['Date', 'Daily_CPI'])
    return daily_cpi_df

# Function to adjust historical prices based on daily CPI
def adjust_prices_for_inflation(prices_df, daily_cpi_df):
    prices_df = prices_df.merge(daily_cpi_df, on='Date', how='left')
    prices_df['Adjusted_Price'] = prices_df['Price'] * (1 + prices_df['Daily_CPI']).cumprod()
    return prices_df

# Function to fetch historical stock prices
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    stock_data.reset_index(inplace=True)
    stock_data.rename(columns={'Date': 'Date', 'Adj Close': 'Price'}, inplace=True)
    return stock_data[['Date', 'Price']]

# Streamlit application
def main():
    st.title("Adjust Historical Stock Prices for Inflation")

    # Load CPI data from the file in the same directory as the script
    cpi_data = pd.read_csv(cpi_csv_path, parse_dates=['Date'])
    cpi_data = cpi_data.sort_values('Date')

    # Date input for stock data
    start_date = st.date_input("Start Date", value=pd.to_datetime('2023-01-01'))
    end_date = st.date_input("End Date", value=pd.to_datetime('2024-08-18'))
    ticker = st.text_input("Stock Ticker", value='YPF.BA')

    # Button to perform calculations
    if st.button("Adjust Prices"):
        # Convert CPI data to daily
        daily_cpi_df = convert_monthly_to_daily(cpi_data)
        
        # Fetch stock data
        stock_data = fetch_stock_data(ticker, start_date, end_date)
        
        # Adjust stock prices for inflation
        adjusted_stock_data = adjust_prices_for_inflation(stock_data, daily_cpi_df)
        
        # Display results
        st.write("Adjusted Stock Data")
        st.dataframe(adjusted_stock_data.head())

        # Plotting the adjusted prices
        st.line_chart(adjusted_stock_data.set_index('Date')['Adjusted_Price'])

if __name__ == "__main__":
    main()
