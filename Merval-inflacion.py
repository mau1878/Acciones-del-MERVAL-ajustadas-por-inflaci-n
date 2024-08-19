import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Predefined tickers list
tickers = [
    "GGAL.BA", "YPFD.BA", "PAMP.BA", "TXAR.BA", "ALUA.BA", "CRES.BA", "SUPV.BA", "CEPU.BA", "BMA.BA", 
    "TGSU2.BA", "TRAN.BA", "EDN.BA", "LOMA.BA", "MIRG.BA", "DGCU2.BA", "BBAR.BA", "MOLI.BA", "TGNO4.BA", 
    "CGPA2.BA", "COME.BA", "IRSA.BA", "BYMA.BA", "TECO2.BA", "METR.BA", "CECO2.BA", "BHIP.BA", "AGRO.BA", 
    "LEDE.BA", "CVH.BA", "HAVA.BA", "AUSO.BA", "VALO.BA", "SEMI.BA", "INVJ.BA", "CTIO.BA", "MORI.BA", 
    "HARG.BA", "GCLA.BA", "SAMI.BA", "BOLT.BA", "MOLA.BA", "CAPX.BA", "OEST.BA", "LONG.BA", "GCDI.BA", 
    "GBAN.BA", "CELU.BA", "FERR.BA", "CADO.BA", "GAMI.BA", "PATA.BA", "CARC.BA", "BPAT.BA", "RICH.BA", 
    "INTR.BA", "GARO.BA", "FIPL.BA", "GRIM.BA", "DYCA.BA", "POLL.BA", "DOME.BA", "ROSE.BA", "MTR.BA"
]

# Helper function to find the most recent valid trading date
def get_recent_valid_date(start_date, end_date):
    while end_date.weekday() >= 5:  # Skip weekends
        end_date -= pd.Timedelta(days=1)
    return end_date

# Streamlit UI
st.title('Análisis de Ratios de Activos del MERVAL. De MTAURUS - X: https://x.com/MTaurus_ok')

# Sidebar inputs
with st.sidebar:
    # Main stock selection
    main_stock_input = st.text_input('Ingresar manualmente un ticker principal (si no está en la lista):', '').upper()
    main_stock = st.selectbox(
        'Seleccionar el ticker principal:',
        options=[main_stock_input] + tickers if main_stock_input else tickers,
        index=0 if main_stock_input else 0
    )

    # Additional tickers selection
    extra_stocks_input = st.text_input('Ingresar manualmente tickers adicionales (separados por comas):', '').upper()
    extra_stocks_manual = [ticker.strip() for ticker in extra_stocks_input.split(',') if ticker.strip()]
    extra_stocks_options = extra_stocks_manual + tickers
    extra_stocks = st.multiselect(
        'Seleccionar hasta 6 tickers adicionales:',
        options=extra_stocks_options,
        default=extra_stocks_manual[:6]
    )

    # Date inputs
    start_date = st.date_input("Fecha de inicio", pd.to_datetime("2023-01-01"))
    end_date = st.date_input("Fecha de finalización", pd.to_datetime("today"))

    # Determine the most recent valid date for the reference date
    today = pd.to_datetime("today")
    most_recent_valid_date = get_recent_valid_date(start_date, today)
    reference_date = st.date_input("Fecha de referencia para visualizar como porcentajes:", most_recent_valid_date)

    # Checkbox to choose percentage view
    view_as_percentages = st.checkbox('Ver como porcentajes en vez de ratios')

    # SMA input field
    sma_period = st.number_input('Periodo de SMA', min_value=1, value=20, key='sma_period')

# Load the inflation data from the CSV file
cpi_file_path = 'argentine_inflation_data.csv'

# Fetch and process data
if st.button('Obtener Datos y Graficar'):

    try:
        # Load CPI data
        cpi_data = pd.read_csv(cpi_file_path, parse_dates=['Date'], dayfirst=True)
        cpi_data.set_index('Date', inplace=True)

        # Calculate daily inflation factors
        cpi_data.sort_index(inplace=True)
        daily_cpi = cpi_data.resample('D').ffill()
        daily_cpi['InflationFactor'] = (1 + daily_cpi['CPI_MoM'] / 100).cumprod()

        # Fetch stock data
        raw_data = yf.download([main_stock] + extra_stocks, start=start_date, end=end_date)['Adj Close']

        # Ensure raw_data is a DataFrame
        if isinstance(raw_data, pd.Series):
            raw_data = raw_data.to_frame()
            raw_data.columns = [main_stock] + extra_stocks

        # Forward fill missing values
        raw_data.ffill(inplace=True)

        # Align stock data with inflation data
        aligned_data = pd.merge(raw_data, daily_cpi, left_index=True, right_index=True, how='left')
        if aligned_data.isna().any().any():
            st.warning('Algunos datos de inflación no están disponibles para todas las fechas. Los datos se ajustarán usando el valor más reciente disponible.')

        # Adjust prices for inflation
        inflation_adjusted_data = aligned_data[[main_stock] + extra_stocks].mul(aligned_data['InflationFactor'], axis=0)

        # Plot setup
        fig = go.Figure()
        for stock in extra_stocks:
            if stock not in inflation_adjusted_data.columns:
                st.warning(f"No se encontró el ticker '{stock}' en los datos.")
                continue
            
            ratio = inflation_adjusted_data[main_stock] / inflation_adjusted_data[stock]

            # Ensure ratio is a DataFrame
            if isinstance(ratio, pd.Series):
                ratio = ratio.to_frame()
                ratio.columns = [f'{main_stock} / {stock}']

            # If viewing as percentages
            if view_as_percentages:
                reference_date = pd.Timestamp(reference_date)

                # Find the nearest available date to the reference_date
                if reference_date not in ratio.index:
                    closest_date = ratio.index.get_loc(reference_date, method='nearest')
                    reference_date = ratio.index[closest_date]
                    st.warning(f"La fecha de referencia ha sido ajustada a la fecha más cercana disponible: {reference_date.date()}")

                reference_value = ratio.loc[reference_date].values[0]
                ratio = (ratio / reference_value - 1) * 100
                ratio.columns = [f'{main_stock} / {stock} ({reference_value:.2f})']

                # Add vertical reference line
                fig.add_shape(
                    type="line",
                    x0=reference_date, y0=ratio.min().values[0], x1=reference_date, y1=ratio.max().values[0],
                    line=dict(color="yellow", dash="dash"),
                    xref="x", yref="y"
                )
            else:
                ratio.columns = [f'{main_stock} / {stock}']

            fig.add_trace(go.Scatter(
                x=ratio.index,
                y=ratio.iloc[:, 0],
                mode='lines',
                name=ratio.columns[0]
            ))

            # If only one additional ticker is selected, show the SMA and histogram
            if len(extra_stocks) == 1:
                # Calculate SMA
                sma = ratio.rolling(window=sma_period).mean()

                # Create figure with SMA
                fig_sma = go.Figure()
                fig_sma.add_trace(go.Scatter(
                    x=ratio.index,
                    y=ratio.iloc[:, 0],
                    mode='lines',
                    name=f'{main_stock} / {stock}'
                ))
                fig_sma.add_trace(go.Scatter(
                    x=sma.index,
                    y=sma,
                    mode='lines',
                    name=f'SMA {sma_period}',
                    line=dict(color='orange')
                ))

                # Average value line
                average_value = ratio.mean().values[0]
                fig_sma.add_trace(go.Scatter(
                    x=[ratio.index.min(), ratio.index.max()],
                    y=[average_value, average_value],
                    mode='lines',
                    name=f'Promedio ({average_value:.2f})',
                    line=dict(color='purple', dash='dot')
                ))

                fig_sma.update_layout(
                    title=f'Ratio de {main_stock} con {stock} y SMA ({sma_period} días)',
                    xaxis_title='Fecha',
                    yaxis_title='Ratio' if not view_as_percentages else 'Porcentaje',
                    xaxis_rangeslider_visible=False,
                    yaxis=dict(showgrid=True),
                    xaxis=dict(showgrid=True)
                )

                st.plotly_chart(fig_sma, use_container_width=True)

                # Histogram of dispersion
                dispersion = ratio.iloc[:, 0] - sma
                dispersion = dispersion.dropna()

                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=dispersion,
                    nbinsx=30,
                    marker=dict(color='blue')
                ))

                percentiles = [5, 25, 50, 75, 95]
                for perc in percentiles:
                    perc_value = np.percentile(dispersion, perc)
                    fig_hist.add_shape(
                        type='line',
                        x0=perc_value, y0=0, x1=perc_value, y1=dispersion.max(),
                        line=dict(color='red', dash='dash'),
                        xref="x", yref="y"
                    )
                    fig_hist.add_annotation(
                        x=perc_value,
                        y=dispersion.max() * 0.95,
                        text=f'{perc}th percentile',
                        showarrow=True,
                        arrowhead=2
                    )

                fig_hist.update_layout(
                    title='Histograma de Dispersión del Ratio',
                    xaxis_title='Dispersión',
                    yaxis_title='Frecuencia',
                    xaxis=dict(showgrid=True),
                    yaxis=dict(showgrid=True)
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)

        fig.update_layout(
            title=f'Ratios de {main_stock} con otras acciones',
            xaxis_title='Fecha',
            yaxis_title='Ratio' if not view_as_percentages else 'Porcentaje',
            xaxis_rangeslider_visible=False,
            yaxis=dict(showgrid=True),
            xaxis=dict(showgrid=True)
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error al obtener datos o graficar: {e}")
