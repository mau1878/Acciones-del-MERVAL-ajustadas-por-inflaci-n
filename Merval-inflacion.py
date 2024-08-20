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

# Fetch historical stock prices
def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    if stock_data.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    stock_data.reset_index(inplace=True)
    if 'Adj Close' in stock_data.columns:
        stock_data.rename(columns={'Adj Close': 'Price'}, inplace=True)
    elif 'Close' in stock_data.columns:
        stock_data.rename(columns={'Close': 'Price'}, inplace=True)
    else:
        raise ValueError(f"Neither 'Adj Close' nor 'Close' columns found for ticker {ticker}")
    return stock_data[['Date', 'Price']]

# Parse ratio expressions and fetch data
def parse_and_fetch_ratios(ratio_expr: str, start_date: str, end_date: str) -> pd.DataFrame:
    # Split the ratio expression
    ratio_expr = ratio_expr.upper()
    parts = re.split(r'[\/*]', ratio_expr)
    operators = re.findall(r'[\/*]', ratio_expr)
    
    # Fetch individual stock data
    stock_dfs = {}
    for part in parts:
        ticker = re.findall(r"[A-Z\.]+", part.strip())[0]
        multiplier = float(re.findall(r"\d+", part.strip())[0]) if re.findall(r"\d+", part.strip()) else 1.0
        if ticker:
            stock_dfs[ticker] = fetch_stock_data(ticker, start_date, end_date)
            stock_dfs[ticker]['Price'] *= multiplier  # Apply the multiplier
    
    # Create a DataFrame for the ratio
    ratio_df = pd.DataFrame()
    for ticker, df in stock_dfs.items():
        if ratio_df.empty:
            ratio_df = df[['Date']].copy()
            ratio_df.set_index('Date', inplace=True)
        
        df.set_index('Date', inplace=True)
        ratio_df[ticker] = df['Price']
    
    # Forward fill missing values for all stocks
    ratio_df = ratio_df.ffill()  # Use ffill instead of deprecated method
    
    # Calculate the ratio
    ratio_df['Ratio'] = ratio_df[parts[0]]
    for i, op in enumerate(operators):
        ticker = parts[i + 1]
        if op == '*':
            ratio_df['Ratio'] *= ratio_df[ticker]
        elif op == '/':
            ratio_df['Ratio'] /= ratio_df[ticker]
    
    # Reset index to get 'Date' back as a column
    ratio_df.reset_index(inplace=True)
    return ratio_df[['Date', 'Ratio']]

# Adjust historical prices based on daily CPI
def adjust_prices_for_inflation(prices_df: pd.DataFrame, daily_cpi_df: pd.DataFrame) -> pd.DataFrame:
    # Merge daily CPI into the stock data
    prices_df = prices_df.merge(daily_cpi_df, on='Date', how='left')
    prices_df['Daily_CPI'] = prices_df['Daily_CPI'].fillna(method='ffill')  # Forward fill missing CPI values
    
    # Ensure proper data type for inflation calculations
    prices_df['Daily_CPI'] = prices_df['Daily_CPI'].astype(np.float64)
    
    # Calculate cumulative inflation rate backward
    prices_df['Cumulative_Inflation'] = (1 + prices_df['Daily_CPI'][::-1]).cumprod()[::-1]
    
    # Adjust prices based on cumulative inflation
    prices_df['Adjusted_Price'] = prices_df['Ratio'] * prices_df['Cumulative_Inflation']
    
    return prices_df

# Main function to adjust historical stock prices for inflation
def main(ratio_expr: str, start_date: str, end_date: str, cpi_csv_path: str) -> pd.DataFrame:
    # Load CPI data and convert to daily
    cpi_data = load_cpi_data(cpi_csv_path)
    daily_cpi_df = convert_monthly_to_daily(cpi_data)
    
    # Parse and fetch ratio data
    ratio_data = parse_and_fetch_ratios(ratio_expr, start_date, end_date)
    
    # Adjust ratio prices for inflation
    if not ratio_data.empty:
        adjusted_ratio_data = adjust_prices_for_inflation(ratio_data, daily_cpi_df)
        return adjusted_ratio_data
    else:
        return pd.DataFrame(columns=['Date', 'Ratio', 'Adjusted_Price'])

# Streamlit UI
st.title("Stock Price Adjustment for Inflation")

ratio_expr = st.text_input("Enter stock ratio (e.g., YPFD.BA/YPF or GGAL.BA*10/GGAL):", 'YPF.BA/YPF')
start_date = st.date_input("Start date:", dt.datetime(2023, 1, 1))
end_date = st.date_input("End date:", dt.datetime.today())

if st.button("Get Data and Plot"):
    cpi_csv_path = 'cpi_mom_data.csv'
    adjusted_ratio_data = main(ratio_expr, start_date, end_date, cpi_csv_path)
    
    if not adjusted_ratio_data.empty:
        st.write(adjusted_ratio_data.head())
        st.line_chart(adjusted_ratio_data.set_index('Date')[['Ratio', 'Adjusted_Price']])
    else:
        st.write("No data available for the selected ratio and date range.")
