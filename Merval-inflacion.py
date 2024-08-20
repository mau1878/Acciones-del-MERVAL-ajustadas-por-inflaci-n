import pandas as pd
import yfinance as yf
import numpy as np
import datetime as dt
import streamlit as st
import re
import plotly.express as px

# Load the CPI data
def load_cpi_data(cpi_csv_path):
    try:
        cpi_data = pd.read_csv(cpi_csv_path, parse_dates=['Date'])
        cpi_data = cpi_data.sort_values('Date')
        st.write("CPI Data Columns:", cpi_data.columns.tolist())  # Debugging line
        return cpi_data
    except Exception as e:
        st.error(f"Error loading CPI data: {e}")
        return pd.DataFrame()

# Convert monthly cumulative CPI to daily inflation factors
def convert_monthly_to_daily(cpi_data):
    daily_cpi = []
    if 'Cumulative_Inflation' not in cpi_data.columns:
        st.error("Cumulative_Inflation column is missing from CPI data.")
        return pd.DataFrame()
    
    for i in range(len(cpi_data) - 1):
        start_date = cpi_data['Date'].iloc[i]
        end_date = cpi_data['Date'].iloc[i + 1]
        cumulative_inflation = cpi_data['Cumulative_Inflation'].iloc[i]
        date_range = pd.date_range(start_date, end_date - pd.Timedelta(days=1), freq='D')
        daily_cpi.extend([(date, cumulative_inflation) for date in date_range])
    
    last_date = cpi_data['Date'].iloc[-1]
    last_cumulative_inflation = cpi_data['Cumulative_Inflation'].iloc[-1]
    daily_cpi.extend([(date, last_cumulative_inflation) for date in pd.date_range(last_date, dt.datetime.today())])
    
    daily_cpi_df = pd.DataFrame(daily_cpi, columns=['Date', 'Cumulative_Inflation'])
    daily_cpi_df['Daily_Inflation_Factor'] = daily_cpi_df['Cumulative_Inflation'].pct_change().fillna(0) + 1
    daily_cpi_df['Daily_Inflation_Factor'] = daily_cpi_df['Daily_Inflation_Factor'].cumprod()
    
    return daily_cpi_df

# Adjust historical prices based on daily CPI
def adjust_prices_for_inflation(prices_df: pd.DataFrame, daily_cpi_df: pd.DataFrame) -> pd.DataFrame:
    try:
        st.write("Stock Data Columns:", prices_df.columns.tolist())  # Debugging line
        st.write("Daily CPI Data Columns:", daily_cpi_df.columns.tolist())  # Debugging line
        
        # Ensure 'Date' column is in datetime format
        prices_df['Date'] = pd.to_datetime(prices_df['Date'])
        daily_cpi_df['Date'] = pd.to_datetime(daily_cpi_df['Date'])
        
        # Merge daily CPI into the stock data
        prices_df = prices_df.merge(daily_cpi_df, on='Date', how='left')
        prices_df['Daily_Inflation_Factor'] = prices_df['Daily_Inflation_Factor'].fillna(method='ffill')  # Forward fill missing CPI values
        
        if 'Price' not in prices_df.columns:
            st.error("The 'Price' column is missing in the stock data DataFrame.")
            return pd.DataFrame()

        # Adjust prices based on daily inflation factors
        prices_df['Adjusted_Price'] = prices_df['Price'] / prices_df['Daily_Inflation_Factor']
        
        # Set up hover data for Plotly
        prices_df['Cumulative_Inflation_Hover'] = prices_df['Daily_Inflation_Factor'].round(4)
        prices_df['Unadjusted_Price_Hover'] = prices_df['Price'].round(2)
        prices_df['Adjusted_Price_Hover'] = prices_df['Adjusted_Price'].round(2)
        
        return prices_df
    except Exception as e:
        st.error(f"Error adjusting prices for inflation: {e}")
        return pd.DataFrame()

# Fetch historical stock prices
def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
        if stock_data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        stock_data.reset_index(inplace=True)
        
        if 'Adj Close' in stock_data.columns:
            stock_data.rename(columns={'Adj Close': 'Price'}, inplace=True)
        elif 'Close' in stock_data.columns:
            stock_data.rename(columns={'Close': 'Price'}, inplace=True)
        else:
            st.error(f"Neither 'Adj Close' nor 'Close' columns found for ticker {ticker}")
            return pd.DataFrame(columns=['Date', 'Price'])
        
        st.write(f"Data for {ticker}:", stock_data.head())  # Debugging line
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
            if any(char.isdigit() for char in part):
                stock_dfs[part] = pd.DataFrame({'Date': pd.date_range(start=start_date, end=end_date), 'Price': float(part)})
            else:
                ticker = part.strip()
                if ticker:
                    stock_dfs[ticker] = fetch_stock_data(ticker, start_date, end_date)
        
        ratio_df = pd.DataFrame()
        for ticker, df in stock_dfs.items():
            if df.empty:
                continue
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
        st.write("Ratio Data:", ratio_df.head())  # Debugging line
        return ratio_df[['Date', 'Ratio']]
    except Exception as e:
        st.error(f"Error parsing and fetching ratios: {e}")
        return pd.DataFrame(columns=['Date', 'Ratio'])

# Main function to adjust historical stock prices for inflation
def main(ratio_expr: str, start_date: str, end_date: str, cpi_csv_path: str) -> pd.DataFrame:
    try:
        cpi_data = load_cpi_data(cpi_csv_path)
        if cpi_data.empty:
            st.error("No CPI data available.")
            return pd.DataFrame(columns=['Date', 'Ratio', 'Adjusted_Price'])
        
        daily_cpi_df = convert_monthly_to_daily(cpi_data)
        if daily_cpi_df.empty:
            st.error("No daily CPI data available.")
            return pd.DataFrame(columns=['Date', 'Ratio', 'Adjusted_Price'])
        
        ratio_data = parse_and_fetch_ratios(ratio_expr, start_date, end_date)
        if ratio_data.empty:
            st.error("No ratio data available.")
            return pd.DataFrame(columns=['Date', 'Ratio', 'Adjusted_Price'])
        
        adjusted_ratio_data = adjust_prices_for_inflation(ratio_data, daily_cpi_df)
        return adjusted_ratio_data
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
        fig = px.line(adjusted_ratio_data, x='Date', y='Adjusted_Price', title='Adjusted Ratio Prices',
                      labels={'Date': 'Date', 'Adjusted_Price': 'Adjusted Price'},
                      hover_data=['Date', 'Ratio', 'Adjusted_Price_Hover'])
        st.plotly_chart(fig)
    else:
        st.write("No data available for the selected ratio and date range.")
