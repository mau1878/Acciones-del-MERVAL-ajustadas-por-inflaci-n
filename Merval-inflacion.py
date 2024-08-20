import pandas as pd
import yfinance as yf
import numpy as np
import datetime as dt
import streamlit as st
import re

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
        monthly_rate = cpi_data['CPI_MOM'].iloc[i]
        
        # Convert monthly rate to daily rate
        daily_rate = (1 + monthly_rate) ** (1/30) - 1
        
        # Generate daily dates and apply daily inflation rate
        date_range = pd.date_range(start_date, end_date - pd.Timedelta(days=1), freq='D')
        daily_cpi.extend([(date, daily_rate) for date in date_range])
    
    # Append the last month data
    last_date = cpi_data['Date'].iloc[-1]
    last_monthly_rate = cpi_data['CPI_MOM'].iloc[-1]
    last_daily_rate = (1 + last_monthly_rate) ** (1/30) - 1
    daily_cpi.extend([(date, last_daily_rate) for date in pd.date_range(last_date, dt.datetime.today())])
    
    daily_cpi_df = pd.DataFrame(daily_cpi, columns=['Date', 'Daily_CPI'])
    return daily_cpi_df

# Adjust historical prices based on daily CPI
def adjust_prices_for_inflation(prices_df, daily_cpi_df):
    # Merge daily CPI into the stock data
    prices_df = prices_df.merge(daily_cpi_df, on='Date', how='left')
    prices_df['Daily_CPI'].fillna(method='ffill', inplace=True)  # Forward fill missing CPI values
    
    # Ensure proper data type for inflation calculations
    prices_df['Daily_CPI'] = prices_df['Daily_CPI'].astype(np.float64)
    
    # Calculate cumulative product of daily inflation rates
    prices_df['Daily_CPI'] = prices_df['Daily_CPI'].apply(lambda x: x + 1)  # Convert inflation rates to growth factors
    prices_df['Cumulative_Inflation'] = (prices_df['Daily_CPI'].cumprod()).astype(np.float64)  # Calculate cumulative inflation
    
    # Find the cumulative inflation factor for the earliest date
    earliest_cumulative_inflation = prices_df['Cumulative_Inflation'].iloc[0]
    
    # Adjust prices based on cumulative inflation
    prices_df['Adjusted_Price'] = prices_df['Price'] * (earliest_cumulative_inflation / prices_df['Cumulative_Inflation'])
    
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

# Parse and evaluate the stock ratio expression
def compute_stock_ratios(ratio_expr, start_date, end_date):
    # Extract tickers and operators from the ratio expression
    tickers = re.findall(r'[A-Za-z0-9._-]+', ratio_expr)
    operators = re.findall(r'[*/]', ratio_expr)
    
    # Fetch price data for each ticker
    price_data = {}
    for ticker in tickers:
        if ticker not in price_data:
            price_data[ticker] = fetch_stock_data(ticker, start_date, end_date)
    
    # Create an empty DataFrame for the ratios
    ratio_df = pd.DataFrame()
    
    # Compute the ratio expression
    for i, ticker in enumerate(tickers):
        if ratio_df.empty:
            ratio_df = price_data[ticker].copy()
            ratio_df.rename(columns={'Price': f'Price_{ticker}'}, inplace=True)
        else:
            if operators[i-1] == '*':
                ratio_df[f'Price_{ticker}'] = price_data[ticker]['Price'] * ratio_df[f'Price_{tickers[i-1]}']
            elif operators[i-1] == '/':
                ratio_df[f'Price_{ticker}'] = price_data[ticker]['Price'] / ratio_df[f'Price_{tickers[i-1]}']
    
    ratio_df['Date'] = price_data[tickers[0]]['Date']
    ratio_df.set_index('Date', inplace=True)
    ratio_df = ratio_df[[f'Price_{tickers[-1]}']]
    ratio_df.rename(columns={f'Price_{tickers[-1]}': 'Ratio'}, inplace=True)
    
    return ratio_df

# Main function to adjust stock ratios for inflation
def main(ratio_expr, start_date, end_date, cpi_csv_path):
    # Load CPI data and convert to daily
    cpi_data = load_cpi_data(cpi_csv_path)
    daily_cpi_df = convert_monthly_to_daily(cpi_data)
    
    # Compute stock ratios
    ratio_data = compute_stock_ratios(ratio_expr, start_date, end_date)
    
    # Adjust stock ratios for inflation
    if not ratio_data.empty:
        adjusted_ratio_data = adjust_prices_for_inflation(ratio_data, daily_cpi_df)
        return adjusted_ratio_data
    else:
        return pd.DataFrame(columns=['Date', 'Ratio', 'Adjusted_Ratio'])

# Streamlit UI
st.title("Stock Ratio Adjustment for Inflation")

ratio_expr = st.text_input("Enter stock ratio (e.g., YPFD.BA/YPF or GGAL.BA*10/GGAL):", 'YPF.BA/YPF')
start_date = st.date_input("Start date:", dt.datetime(2023, 1, 1))
end_date = st.date_input("End date:", dt.datetime.today())

if st.button("Get Data and Plot"):
    cpi_csv_path = 'cpi_mom_data.csv'
    adjusted_ratio_data = main(ratio_expr, start_date, end_date, cpi_csv_path)
    
    if not adjusted_ratio_data.empty:
        st.write(adjusted_ratio_data.head())
        st.line_chart(adjusted_ratio_data.set_index('Date')[['Ratio', 'Adjusted_Ratio']])
    else:
        st.write("No data available for the selected ratio and date range.")
