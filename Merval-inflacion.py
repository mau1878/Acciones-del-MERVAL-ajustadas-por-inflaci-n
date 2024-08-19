import pandas as pd
import yfinance as yf
import numpy as np
import datetime as dt

# Load the CPI data
cpi_data = pd.read_csv('cpi_mom_data.csv', parse_dates=['Date'])
cpi_data = cpi_data.sort_values('Date')

# Convert monthly CPI to daily CPI
def convert_monthly_to_daily(cpi_data):
    daily_cpi = []
    for i in range(len(cpi_data) - 1):
        start_date = cpi_data['Date'].iloc[i]
        end_date = cpi_data['Date'].iloc[i + 1]
        inflation_rate = cpi_data['CPI_MOM'].iloc[i]
        date_range = pd.date_range(start_date, end_date, closed='left')
        daily_cpi.extend([(date, inflation_rate) for date in date_range])
    
    # Append the last month data
    last_date = cpi_data['Date'].iloc[-1]
    last_inflation_rate = cpi_data['CPI_MOM'].iloc[-1]
    daily_cpi.extend([(date, last_inflation_rate) for date in pd.date_range(last_date, dt.datetime.today())])
    
    daily_cpi_df = pd.DataFrame(daily_cpi, columns=['Date', 'Daily_CPI'])
    return daily_cpi_df

# Adjust historical prices based on daily CPI
def adjust_prices_for_inflation(prices_df, daily_cpi_df):
    prices_df = prices_df.merge(daily_cpi_df, on='Date', how='left')
    prices_df['Adjusted_Price'] = prices_df['Price'] * (1 + prices_df['Daily_CPI']).cumprod()
    return prices_df

# Fetch historical stock prices
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    stock_data.reset_index(inplace=True)
    stock_data.rename(columns={'Date': 'Date', 'Adj Close': 'Price'}, inplace=True)
    return stock_data[['Date', 'Price']]

# Main function to adjust historical stock prices for inflation
def main(ticker, start_date, end_date, cpi_csv_path):
    # Load CPI data and convert to daily
    daily_cpi_df = convert_monthly_to_daily(pd.read_csv(cpi_csv_path, parse_dates=['Date']))
    
    # Fetch stock data
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    
    # Adjust stock prices for inflation
    adjusted_stock_data = adjust_prices_for_inflation(stock_data, daily_cpi_df)
    
    return adjusted_stock_data

# Parameters
ticker = 'YPF.BA'  # Example ticker
start_date = '2023-01-01'
end_date = '2024-08-18'
cpi_csv_path = 'cpi_mom_data.csv'

# Run the main function
adjusted_stock_data = main(ticker, start_date, end_date, cpi_csv_path)

# Output adjusted stock data
print(adjusted_stock_data.head())
