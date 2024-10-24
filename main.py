import os
import sys

import streamlit as st

# Data handling imports
import pandas as pd
import plotly.express as px
import random
from pygwalker.api.streamlit import StreamlitRenderer

# Configuration imports
import config as cfg
# =================================================================================================================================================
def get_data_directory() -> str:
    """Get the path to the data directory."""
    return os.path.join(os.getcwd(), 'data')

def load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV file from the data directory."""
    return pd.read_csv(os.path.join(get_data_directory(), filename))

def get_data() -> list:
    """Load all financial data from CSV files."""
    annual_files = [
        'annually_income_statement.csv',
        'annually_balance_sheet.csv',
        'annually_cash_flow.csv'
    ]
    quarterly_files = [
        'quarterly_income_statement.csv',
        'quarterly_balance_sheet.csv',
        'quarterly_cash_flow.csv'
    ]
    return [load_csv(file) for file in annual_files + quarterly_files]

@st.cache_data
def get_cached_data() -> list:
    """Get cached financial data."""
    return get_data()

def plot_line_chart(df: pd.DataFrame, metrics: list, period: str = 'annually', show_percentage: bool = False, tickers: list = None) -> None:
    """Plot a line chart for the given metrics."""
    filtered_df = df.copy()
    if tickers:
        filtered_df = filtered_df[filtered_df['Symbol'].isin(tickers)]
    
    filtered_df['Year'] = pd.to_datetime(filtered_df['Date']).dt.year
    if period == 'quarterly':
        filtered_df['Quarter'] = pd.to_datetime(filtered_df['Date']).dt.to_period('Q').astype(str)
        filtered_df['Quarter'] = filtered_df['Year'].astype(str) + ' ' + filtered_df['Quarter'].str[-2:]

    # Remove rows with NaN values in the metric columns
    filtered_df = filtered_df.dropna(subset=metrics)

    # Sort the dataframe by Year and Quarter
    sort_columns = ['Year'] if period == 'annually' else ['Year', 'Quarter']
    filtered_df = filtered_df.sort_values(sort_columns)

    # Round the metrics to 2 decimal places
    filtered_df[metrics] = filtered_df[metrics].round(2)

    if show_percentage:
        filtered_df[metrics] *= 100

    # Ensure each year has all four quarters
    if period == 'quarterly':
        all_quarters = ['Q1', 'Q2', 'Q3', 'Q4']
        filtered_df = filtered_df[filtered_df['Quarter'].str[-2:].isin(all_quarters)]

    fig = px.line(filtered_df, x='Year' if period == 'annually' else 'Quarter', y=metrics, color='Symbol',
                  title=f'{", ".join(metrics)} Trends ({period.capitalize()})',
                  color_discrete_map= cfg.COLOR_THEME)
    
    # Set the x-axis properties
    if period == 'annually':
        fig.update_xaxes(title_text='Year', tickmode='linear', dtick=1, range=[filtered_df['Year'].min() - 0.5, filtered_df['Year'].max() + 0.5])
    else:
        fig.update_xaxes(title_text='Quarter', tickmode='linear', dtick=1, categoryorder='category ascending')
    
    fig.update_yaxes(title_text=', '.join(metrics))

    # Set y-axis range
    y_min, y_max = filtered_df[metrics].min().min(), filtered_df[metrics].max().max()
    fig.update_yaxes(range=[y_min * 1.1, y_max * 1.1] if y_min < 0 else [0, y_max * 1.1])
    
    # Add hover data
    hover_template = '%{y:.2f}%' if show_percentage else '%{y:.2f}'
    fig.update_traces(hovertemplate=hover_template)

    st.plotly_chart(fig, use_container_width=True)

def create_custom_plotly_chart(df: pd.DataFrame) -> None:
    """Create a custom Plotly chart based on user selections."""
    selected_symbols = st.multiselect('Select one or more symbols', cfg.tickers)
    numeric_columns = df.select_dtypes(include=['number']).columns.difference(['Cik', 'Calendar Year']).tolist()
    selected_features = st.multiselect('Select one or more features to plot', numeric_columns)

    if not selected_symbols or not selected_features:
        st.warning("Please select at least one symbol and one feature.")
        return

    filtered_df = df[df['Symbol'].isin(selected_symbols)].copy()
    filtered_df['Year'] = pd.to_datetime(filtered_df['Date']).dt.year

    fig = px.line(filtered_df, x='Year', y=selected_features, color='Symbol',
                  title='Selected features over years for chosen symbols',
                  color_discrete_map=cfg.COLOR_THEME)
    fig.update_xaxes(title_text='Year', tickmode='linear', dtick=1)
    fig.update_yaxes(title_text='Value')

    st.plotly_chart(fig, use_container_width=True)

def create_custom_chart(df: pd.DataFrame) -> None:
    """Create a custom chart based on user input."""
    col1, col2 = st.columns(2)
    with col1:
        create_own_chart = st.checkbox("Create your own chart")
    with col2:
        chart_builder = st.radio("Select a chart builder", ["Pygwalker", "Plotly"]) if create_own_chart else None
    
    if create_own_chart:
        if chart_builder == "Pygwalker":
            st.write("You selected Pygwalker")
            pyg_app = get_pyg_app(df)
            pyg_app.explorer()
        else:
            st.write("You selected Plotly")
            create_custom_plotly_chart(df)

def display_income_statement_tab(annual_income_statement_df: pd.DataFrame, quarterly_income_statement_df: pd.DataFrame) -> None:
    """Display the income statement analysis tab."""

    st.title("💰 Annual Income Statement Analysis")
    create_custom_chart(annual_income_statement_df)
    
    selected_tickers = st.multiselect('Select tickers to analyze', sorted(annual_income_statement_df['Symbol'].unique()), default=['ALC'])

    col1, col2 = st.columns(2)
    with col1:
        for metric in ['Total Revenue', 'Normalized EBITDA', 'Normalized Income']:
            plot_line_chart(annual_income_statement_df, [metric], tickers=selected_tickers)
    with col2:
        for metric in ['Net Income', 'Basic EPS', 'Operating Expense']:
            plot_line_chart(annual_income_statement_df, [metric], tickers=selected_tickers)

    st.markdown("### Quarterly Income Statement Analysis")
    col1, col2 = st.columns(2)
    with col1:
        for metric in ['Total Revenue', 'Normalized EBITDA', 'Normalized Income']:
            plot_line_chart(quarterly_income_statement_df, [metric], period='quarterly', tickers=selected_tickers)
    with col2:
        for metric in ['Net Income', 'Basic EPS', 'Operating Expense']:
            plot_line_chart(quarterly_income_statement_df, [metric], period='quarterly', tickers=selected_tickers)

def display_cash_flow_tab(annual_cash_flow_df: pd.DataFrame, quarterly_cash_flow_df: pd.DataFrame) -> None:
    """Display the cash flow analysis tab."""

    st.title("💵 Annual Cash Flow Analysis")
    create_custom_chart(annual_cash_flow_df)

    selected_tickers = st.multiselect('Select tickers to analyze', sorted(annual_cash_flow_df['Symbol'].unique()), default=['ALC'])

    col1, col2 = st.columns(2)
    with col1:
        for metric in ['Changes In Cash', 'Financing Cash Flow']:
            plot_line_chart(annual_cash_flow_df, [metric], tickers=selected_tickers)
    with col2:
        for metric in ['Investing Cash Flow', 'Operating Cash Flow']:
            plot_line_chart(annual_cash_flow_df, [metric], tickers=selected_tickers)

    st.markdown("### Quarterly Cash Flow Analysis")
    col1, col2 = st.columns(2)
    with col1:
        for metric in ['Changes In Cash', 'Financing Cash Flow']:
            plot_line_chart(quarterly_cash_flow_df, [metric], period='quarterly', tickers=selected_tickers)
    with col2:
        for metric in ['Investing Cash Flow', 'Operating Cash Flow']:
            plot_line_chart(quarterly_cash_flow_df, [metric], period='quarterly', tickers=selected_tickers)

def display_balance_sheet_tab(annual_balance_sheet_df: pd.DataFrame, quarterly_balance_sheet_df: pd.DataFrame) -> None:
    """Display the balance sheet analysis tab."""

    st.title("💲 Annual Balance Sheet Analysis")
    create_custom_chart(annual_balance_sheet_df)

    selected_tickers = st.multiselect('Select tickers to analyze', sorted(annual_balance_sheet_df['Symbol'].unique()), default=['ALC'])

    col1, col2 = st.columns(2)
    with col1:
        plot_line_chart(annual_balance_sheet_df, ['Working Capital'], tickers=selected_tickers)
    with col2:
        plot_line_chart(annual_balance_sheet_df, ["Current Ratio"], tickers=selected_tickers)

    st.markdown("### Quarterly Balance Sheet Analysis")
    col1, col2 = st.columns(2)
    with col1:
        plot_line_chart(quarterly_balance_sheet_df, ['Working Capital'], period='quarterly', tickers=selected_tickers)
    with col2:
        plot_line_chart(quarterly_balance_sheet_df, ["Current Ratio"], period='quarterly', tickers=selected_tickers)
    
@st.cache_resource
def get_pyg_app(df: pd.DataFrame) -> StreamlitRenderer:
    """Get the Pygwalker app instance."""
    return StreamlitRenderer(df)

def main() -> None:
    """Main function to run the Streamlit app."""
    st.set_page_config(layout="wide", page_title="Alcon Chatbot", page_icon="💬")
    col1, col2 = st.columns(2)
    with col1:
        st.image("images/Alcon.png", width=300)
    with col2:
        st.image("images/TCU.png", width=250)
    
    # Get the data
    (
        annual_income_statement_df,
        annual_balance_sheet_df,
        annual_cash_flow_df,
        quarterly_income_statement_df,
        quarterly_balance_sheet_df,
        quarterly_cash_flow_df
    ) = get_data()
    
    topic = st.sidebar.radio("Select an option", ["Income Statement", "Balance Sheet", "Cash Flow"])

    if topic == "Income Statement":
        display_income_statement_tab(annual_income_statement_df, quarterly_income_statement_df)
    elif topic == "Balance Sheet":
        display_balance_sheet_tab(annual_balance_sheet_df, quarterly_balance_sheet_df)
    elif topic == "Cash Flow":
        display_cash_flow_tab(annual_cash_flow_df, quarterly_cash_flow_df)

if __name__ == "__main__":
    main()
