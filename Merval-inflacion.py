import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Predefined tickers list
tickers = [
    # List of tickers
]

# Helper function to find the most recent valid trading date
def get_recent_valid_date(start_date, end_date):
    while end_date.weekday() >= 5:  # Skip weekends
        end_date -= pd.Timedelta(days=1)
    return end_date

# Streamlit UI
st.title('Análisis de Ratios de Activos del MERVAL')

# Sidebar inputs
with st.sidebar:
    main_stock_input = st.text_input('Ingresar manualmente un ticker principal (si no está en la lista):', '').upper()
    main_stock = st.selectbox(
        'Seleccionar el ticker principal:',
        options=[main_stock_input] + tickers if main_stock_input else tickers,
        index=0 if main_stock_input else 0
    )

    extra_stocks_input = st.text_input('Ingresar manualmente tickers adicionales (separados por comas):', '').upper()
    extra_stocks_manual = [ticker.strip() for ticker in extra_stocks_input.split(',') if ticker.strip()]
    extra_stocks_options = extra_stocks_manual + tickers
    extra_stocks = st.multiselect(
        'Seleccionar hasta 6 tickers adicionales:',
        options=extra_stocks_options,
        default=extra_stocks_manual[:6]
    )

    start_date = st.date_input("Fecha de inicio", pd.to_datetime("2023-01-01"))
    end_date = st.date_input("Fecha de finalización", pd.to_datetime("today"))

    today = pd.to_datetime("today")
    most_recent_valid_date = get_recent_valid_date(start_date, today)
    reference_date = st.date_input("Fecha de referencia para visualizar como porcentajes:", most_recent_valid_date)

    view_as_percentages = st.checkbox('Ver como porcentajes en vez de ratios')

    sma_period = st.number_input('Periodo de SMA', min_value=1, value=20, key='sma_period')

# Load inflation data from CSV file
cpi_file_path = 'argentine_inflation_data.csv'

# Fetch and process data
if st.button('Obtener Datos y Graficar'):
    try:
        cpi_data = pd.read_csv(cpi_file_path, parse_dates=['Date'], dayfirst=True)
        cpi_data.set_index('Date', inplace=True)

        daily_cpi = cpi_data.resample('D').interpolate(method='linear')
        daily_cpi['InflationFactor'] = (1 + daily_cpi['CPI_MoM'] / 100).cumprod()

        data = yf.download([main_stock] + extra_stocks, start=start_date, end=end_date)['Adj Close']
        st.write(data.head())  # Debugging statement
        data.fillna(method='ffill', inplace=True)

        if not isinstance(data, pd.DataFrame) or main_stock not in data.columns:
            st.error(f"No se encontró el ticker principal '{main_stock}' en los datos.")
        else:
            inflation_adjusted_data = data.mul(daily_cpi['InflationFactor'], axis=0)

            fig = go.Figure()
            for stock in extra_stocks:
                if stock not in inflation_adjusted_data.columns:
                    st.warning(f"No se encontró el ticker '{stock}' en los datos.")
                    continue
                
                ratio = inflation_adjusted_data[main_stock] / inflation_adjusted_data[stock]
                
                if view_as_percentages:
                    reference_date = pd.Timestamp(reference_date)

                    if reference_date not in ratio.index:
                        differences = abs(ratio.index - reference_date)
                        closest_date = ratio.index[differences.argmin()]
                        reference_date = closest_date
                        st.warning(f"La fecha de referencia ha sido ajustada a la fecha más cercana disponible: {reference_date.date()}")

                    reference_value = ratio.loc[reference_date]
                    ratio = (ratio / reference_value - 1) * 100
                    name_suffix = f"({reference_value:.2f})"
                    
                    fig.add_shape(
                        type="line",
                        x0=reference_date, y0=ratio.min(), x1=reference_date, y1=ratio.max(),
                        line=dict(color="yellow", dash="dash"),
                        xref="x", yref="y"
                    )
                else:
                    name_suffix = ""

                fig.add_trace(go.Scatter(
                    x=ratio.index,
                    y=ratio,
                    mode='lines',
                    name=f'{main_stock} / {stock} {name_suffix}'
                ))

                if len(extra_stocks) == 1:
                    sma = ratio.rolling(window=sma_period).mean()
                    fig_sma = go.Figure()
                    fig_sma.add_trace(go.Scatter(
                        x=ratio.index,
                        y=ratio,
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

                    average_value = ratio.mean()
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

                    dispersion = ratio - sma
                    dispersion = dispersion.dropna()

                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=dispersion,
                        nbinsx=50,
                        marker=dict(color='lightblue')
                    ))

                    percentiles = [5, 25, 50, 75, 95]
                    percentile_values = np.percentile(dispersion, percentiles)

                    for p, value in zip(percentiles, percentile_values):
                        fig_hist.add_shape(
                            type="line",
                            x0=value, x1=value, y0=0, y1=max(np.histogram(dispersion, bins=50)[0]),
                            line=dict(color="red" if p == 50 else "green", dash="dash"),
                            xref="x", yref="y"
                        )
                        fig_hist.add_annotation(
                            x=value,
                            y=max(np.histogram(dispersion, bins=50)[0]),
                            text=f"P{p}: {value:.2f}",
                            showarrow=False,
                            yshift=10
                        )
                    
                    fig_hist.update_layout(
                        title=f'Histograma de la Dispersión ({main_stock} / {stock}) - SMA {sma_period}',
                        xaxis_title='Dispersión',
                        yaxis_title='Frecuencia',
                        bargap=0.1
                    )
                    
                    st.plotly_chart(fig_hist, use_container_width=True)

            fig.update_layout(
                title=f'Comparación de Ratios entre {main_stock} y Otros Tickers',
                xaxis_title='Fecha',
                yaxis_title='Ratio' if not view_as_percentages else 'Porcentaje',
                xaxis_rangeslider_visible=False,
                yaxis=dict(showgrid=True),
                xaxis=dict(showgrid=True)
            )
            if view_as_percentages:
                fig.add_hline
