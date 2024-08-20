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
        # Load data with correct date parsing
        cpi_data = pd.read_csv(cpi_csv_path, parse_dates=['Date'], dayfirst=False)
        cpi_data['Date'] = pd.to_datetime(cpi_data['Date'], format='%m/%d/%Y')
        cpi_data = cpi_data.sort_values('Date')
        return cpi_data
    except Exception as e:
        st.error(f"Error loading CPI data: {e}")
        return pd.DataFrame()

# Adjust cumulative CPI values so that the present day is 1
def adjust_cumulative_cpi(cpi_data):
    try:
        # Reverse the data to get correct cumulative inflation values
        cpi_data = cpi_data[::-1].reset_index(drop=True)

        # Normalize the cumulative CPI values
        cpi_data['Adjusted_Cumulative_CPI'] = cpi_data['Cumulative_CPI'] / cpi_data['Cumulative_CPI'].iloc[0]

        # Reverse data back to original order
        cpi_data = cpi_data[::-1].reset_index(drop=True)
        
        daily_cpi_df = cpi_data[['Date', 'Adjusted_Cumulative_CPI']]
        return daily_cpi_df
    except Exception as e:
        st.error(f"Error adjusting cumulative CPI: {e}")
        return pd.DataFrame(columns=['Date', 'Adjusted_Cumulative_CPI'])

# Convert adjusted cumulative CPI to daily cumulative CPI
def convert_cumulative_to_daily(cpi_data):
    try:
        # Calculate daily inflation values based on the scaled cumulative CPI
        cpi_data['Daily_Cumulative_Inflation'] = cpi_data['Adjusted_Cumulative_CPI'] / cpi_data['Adjusted_Cumulative_CPI'].shift(-1)
        cpi_data['Daily_Cumulative_Inflation'] = cpi_data['Daily_Cumulative_Inflation'].fillna(1)

        # Make cumulative product to get daily inflation values
        cpi_data['Daily_Cumulative_Inflation'] = cpi_data['Daily_Cumulative_Inflation'].cumprod()

        # Reverse data back to original order
        cpi_data = cpi_data[::-1].reset_index(drop=True)

        daily_cpi_df = cpi_data[['Date', 'Daily_Cumulative_Inflation']]
        return daily_cpi_df
    except Exception as e:
        st.error(f"Error converting cumulative CPI to daily CPI: {e}")
        return pd.DataFrame(columns=['Date', 'Daily_Cumulative_Inflation'])

# Main function to adjust historical stock prices for inflation
def main(ratio_expr: str, start_date: str, end_date: str, cpi_csv_path: str) -> pd.DataFrame:
    try:
        # Load CPI data and adjust for inflation
        cpi_data = load_cpi_data(cpi_csv_path)
        adjusted_cpi_df = adjust_cumulative_cpi(cpi_data)
        daily_cpi_df = convert_cumulative_to_daily(adjusted_cpi_df)

        # Parse and fetch ratio data
        ratio_data = parse_and_fetch_ratios(ratio_expr, start_date, end_date)
        
        # Adjust ratio prices for inflation
        if not ratio_data.empty:
            adjusted_ratio_data = adjust_prices_for_inflation(ratio_data, daily_cpi_df)
            return adjusted_ratio_data
        else:
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
        fig = px.line(adjusted_ratio_data, x='Date', y=['Ratio', 'Adjusted_Price'], 
                      labels={'value': 'Price', 'variable': 'Type'},
                      title="Adjusted vs Unadjusted Prices")
        fig.update_traces(
            hovertemplate=
            '<b>Date:</b> %{x|%b %d, %Y}<br>' +
            '<b>Unadjusted Price:</b> %{customdata[1]:.2f}<br>' +
            '<b>Adjusted Price:</b> %{customdata[2]:.2f}<br>' +
            '<b>Cumulative Inflation:</b> %{customdata[0]:.4f}<br>' +
            '<extra></extra>'
        )
        fig.update_traces(customdata=adjusted_ratio_data[['Daily_Cumulative_Inflation', 'Unadjusted_Price_Hover', 'Adjusted_Price_Hover']])
        st.plotly_chart(fig)
    else:
        st.write("No data available for the selected ratio and date range.")
