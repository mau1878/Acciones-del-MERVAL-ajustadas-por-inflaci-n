import pandas as pd
import yfinance as yf
import numpy as np
import datetime as dt
import streamlit as st
import re

# Load the CPI data
def load_cpi_data(cpi_csv_path):
    try:
        cpi_data = pd.read_csv(cpi_csv_path, parse_dates=['Date'])
        cpi_data = cpi_data.sort_values('Date')
        return cpi_data
    except Exception as e:
        st.error(f"Error loading CPI data: {e}")
        return pd.DataFrame()

# Calculate cumulative inflation based on monthly CPI
def calculate_monthly_cumulative_inflation(cpi_data):
    try:
        cpi_data['Cumulative_Inflation'] = (1 + cpi_data['CPI_MOM']).cumprod()
        return cpi_data
    except Exception as e:
        st.error(f"Error calculating monthly cumulative inflation: {e}")
        return cpi_data

# Convert monthly cumulative inflation to daily cumulative inflation
def convert_monthly_cumulative_to_daily(cpi_data):
    try:
        daily_cpi = []
        for i in range(len(cpi_data) - 1):
            start_date = cpi_data['Date'].iloc[i]
            end_date = cpi_data['Date'].iloc[i + 1]
            cumulative_inflation_start = cpi_data['Cumulative_Inflation'].iloc[i]
            cumulative_inflation_end = cpi_data['Cumulative_Inflation'].iloc[i + 1]
            
            date_range = pd.date_range(start_date, end_date - pd.Timedelta(days=1), freq='D')
            inflation_diff = cumulative_inflation_end / cumulative_inflation_start
            daily_inflation_growth_factor = inflation_diff ** (1/len(date_range))
            
            for date in date_range:
                daily_cpi.append((date, cumulative_inflation_start * (daily_inflation_growth_factor ** (len(date_range) - (date_range[-1] - date).days))))
        
        last_date = cpi_data['Date'].iloc[-1]
        last_cumulative_inflation = cpi_data['Cumulative_Inflation'].iloc[-1]
        daily_cpi.extend([(date, last_cumulative_inflation) for date in pd.date_range(last_date, dt.datetime.today())])
        
        daily_cpi_df = pd.DataFrame(daily_cpi, columns=['Date', 'Daily_Cumulative_Inflation'])
        return daily_cpi_df
    except Exception as e:
        st.error(f"Error converting monthly to daily cumulative inflation: {e}")
        return pd.DataFrame(columns=['Date', 'Daily_Cumulative_Inflation'])

# Adjust historical prices based on daily cumulative inflation
def adjust_prices_for_inflation(prices_df: pd.DataFrame, daily_cpi_df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Check for necessary columns
        if 'Price' not in prices_df.columns:
            st.error("'Price' column is missing from the DataFrame")
            raise KeyError("'Price' column is missing from the DataFrame")
        if 'Date' not in prices_df.columns:
            st.error("'Date' column is missing from the DataFrame")
            raise KeyError("'Date' column is missing from the DataFrame")
        
        # Merge dataframes
        prices_df = prices_df.merge(daily_cpi_df, on='Date', how='left')
        prices_df['Daily_Cumulative_Inflation'] = prices_df['Daily_Cumulative_Inflation'].fillna(method='ffill')
        
        # Check if 'Daily_Cumulative_Inflation' is present
        if 'Daily_Cumulative_Inflation' not in prices_df.columns:
            st.error("'Daily_Cumulative_Inflation' column is missing from the DataFrame")
            raise KeyError("'Daily_Cumulative_Inflation' column is missing from the DataFrame")
        
        # Calculate adjusted price
        prices_df['Adjusted_Price'] = prices_df['Price'] * (prices_df['Daily_Cumulative_Inflation'].iloc[-1] / prices_df['Daily_Cumulative_Inflation'])
        return prices_df
    except KeyError as e:
        st.error(f"KeyError: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error adjusting prices for inflation: {e}")
        return pd.DataFrame()

# Fetch historical stock prices
def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
        if stock_data.empty:
            st.error(f"No data found for ticker {ticker}")
            raise ValueError(f"No data found for ticker {ticker}")
        stock_data.reset_index(inplace=True)
        if 'Adj Close' in stock_data.columns:
            stock_data.rename(columns={'Adj Close': 'Price'}, inplace=True)
        elif 'Close' in stock_data.columns:
            stock_data.rename(columns={'Close': 'Price'}, inplace=True)
        else:
            st.error(f"Neither 'Adj Close' nor 'Close' columns found for ticker {ticker}")
            raise ValueError(f"Neither 'Adj Close' nor 'Close' columns found for ticker {ticker}")
        return stock_data[['Date', 'Price']]
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame(columns=['Date', 'Price'])

# Parse ratio expressions and fetch data
def parse_and_fetch_ratios(ratio_expr: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        ratio_expr = ratio_expr.upper()
        parts = re.split(r'[\/*]', ratio_expr)
        operators = re.findall(r'[\/*]', ratio_expr)
        
        stock_dfs = {}
        for part in parts:
            ticker = part.strip()
            if ticker:
                stock_dfs[ticker] = fetch_stock_data(ticker, start_date, end_date)
        
        ratio_df = pd.DataFrame()
        for ticker, df in stock_dfs.items():
            if ratio_df.empty:
                ratio_df = df[['Date']].copy()
                ratio_df.set_index('Date', inplace=True)
            
            df.set_index('Date', inplace=True)
            ratio_df[ticker] = df['Price']
        
        ratio_df = ratio_df.ffill()
        
        ratio_df['Ratio'] = ratio_df[parts[0]]
        for i, op in enumerate(operators):
            ticker = parts[i + 1]
            if op == '*':
                ratio_df['Ratio'] *= ratio_df[ticker]
            elif op == '/':
                ratio_df['Ratio'] /= ratio_df[ticker]
        
        ratio_df.reset_index(inplace=True)
        return ratio_df[['Date', 'Ratio']]
    except Exception as e:
        st.error(f"Error parsing and fetching ratios: {e}")
        return pd.DataFrame(columns=['Date', 'Ratio'])

# Main function to adjust historical stock prices for inflation
def main(ratio_expr: str, start_date: str, end_date: str, cpi_csv_path: str) -> pd.DataFrame:
    try:
        cpi_data = load_cpi_data(cpi_csv_path)
        cpi_data = calculate_monthly_cumulative_inflation(cpi_data)
        daily_cpi_df = convert_monthly_cumulative_to_daily(cpi_data)
        
        ratio_data = parse_and_fetch_ratios(ratio_expr, start_date, end_date)
        
        if not ratio_data.empty:
            adjusted_ratio_data = adjust_prices_for_inflation(ratio_data, daily_cpi_df)
            return adjusted_ratio_data
        else:
            st.error("No data available for the selected ratio and date range.")
            return pd.DataFrame(columns=['Date', 'Ratio', 'Adjusted_Price'])
    except Exception as e:
        st.error(f"Error in main function: {e}")
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
