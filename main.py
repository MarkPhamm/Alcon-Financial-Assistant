# Streamlit and environment imports
import streamlit as st
import os
import sys
from dotenv import load_dotenv

# OpenAI and LangChain imports
from openai import AsyncOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Data handling imports
import pandas as pd
import csv
import plotly.express as px

# Utility imports
from datetime import datetime
from functools import lru_cache
import asyncio
import time
import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer

import config as cfg
sys.path.append(os.path.join(os.path.dirname(__file__), 'etl'))
# Define a color theme for each ticker
COLOR_THEME = cfg.COLOR_THEME

from etl import etl_scripts
import populate_vectordb_files as pvf 

# Load environment variables from .env file
load_dotenv('.env')

# Initialize OpenAI client with API key
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# Define the chatbot model
chatbot_model = "gpt-3.5-turbo"

# Initialize embeddings with OpenAI's text-embedding-3-large model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Define the persist directory and collection name for the Chroma database
persist_directory = "./chroma_langchain_db"
collection_name = "alcon_collection_financial_statements"

# Create a Chroma instance with the specified embeddings and collection name
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name=collection_name)

# Function to find relevant entries from the Chroma database
@lru_cache(maxsize=100)
async def find_relevant_entries_from_chroma_db(query):
    """
    Searches the Chroma database for entries relevant to the given query.

    This function uses the Chroma instance to perform a similarity search based on the user's query.
    It embeds the query and compares it to the existing vector embeddings in the database.

    For more information, see:
    https://python.langchain.com/v0.2/api_reference/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html

    Parameters:
        query (str): The user's input query.

    Returns:
        list: A list of tuples, each containing a Document object and its similarity score.
    """
    # Perform similarity search with score
    results = await asyncio.to_thread(vectordb.similarity_search_with_score, query, k=5)
    
    # Print results for debugging
    for doc, score in results:
        print(f"Similarity: {score:.3f}")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("---")

    return results

# Function to generate a GPT response
@lru_cache(maxsize=100)
async def generate_gpt_response(user_query, chroma_result):
    '''
    Generate a response using GPT model based on user query and related information.

    See the link below for further information on crafting prompts:
    https://github.com/openai/openai-python    

    Parameters:
        user_query (str): The user's input query
        chroma_result (str): Related documents retrieved from the database based on the user query

    Returns:
        str: A formatted string containing both naive and augmented responses
    '''    
    
    # Generate both naive and augmented responses in a single API call
    combined_prompt = f"""User query: {user_query}

    Please provide two responses:
    1. A naive response without any additional context.
    2. An augmented response considering the following related information from our database:
    {chroma_result}

    Format your response as follows:
    **Naive Response**

    [Your naive response here]

    -------------------------

    **Augmented Response**

    [Your augmented response here]
    """
    response = await client.chat.completions.create(
        model=chatbot_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant capable of providing both naive and context-aware responses."},
            {"role": "user", "content": combined_prompt}
        ]
    )

    return response.choices[0].message.content

# Function to handle user queries
async def query_interface(user_query, is_first_prompt):
    '''
    Process user query and generate a response using GPT model and relevant information from the database.

    For more information on crafting prompts, see:
    https://github.com/openai/openai-python

    Parameters:
        user_query (str): The query input by the user
        is_first_prompt (bool): Whether this is the first prompt in the conversation

    Returns:
        str: A formatted response from the chatbot, including both naive and augmented answers
    '''
    start_time = time.time()

    # Step 1 and 2: Find relevant information and generate response concurrently
    chroma_task = asyncio.create_task(find_relevant_entries_from_chroma_db(user_query))
    chroma_result = await chroma_task
    gpt_response = await generate_gpt_response(user_query, str(chroma_result))

    end_time = time.time()
    response_time = end_time - start_time

    # Log the response time to a CSV file asynchronously
    asyncio.create_task(log_response_time(user_query, response_time, is_first_prompt))

    # Step 3: Return the generated response
    return gpt_response

async def log_response_time(query, response_time, is_first_prompt):
    """
    Log the response time for a query to a CSV file asynchronously.

    Parameters:
        query (str): The user's query
        response_time (float): The time taken to generate the response
        is_first_prompt (bool): Whether this is the first prompt in the conversation
    """
    csv_file = 'response_times.csv'
    file_exists = os.path.isfile(csv_file)

    async with asyncio.Lock():
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Timestamp', 'Query', 'Response Time (seconds)', 'Is First Prompt'])
            writer.writerow([datetime.now(), query, f"{response_time:.2f}", "Yes" if is_first_prompt else "No"])

def chatbot():
    st.title("Welcome to Alcon Chatbot")
    st.markdown("### How can we assist you today?")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Type your question here..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Show a loading spinner while waiting for the response
        with st.spinner("Thinking..."):
            # Get bot response
            is_first_prompt = len(st.session_state.messages) == 1
            
            # Use asyncio.run in a separate thread to avoid blocking Streamlit
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(query_interface(prompt, is_first_prompt))
            loop.close()

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Add a button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# =================================================================================================================================================

def get_data_directory() -> str:
    """Get the path to the data directory."""
    return os.path.join(os.getcwd(), 'data')

def load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV file from the data directory."""
    return pd.read_csv(os.path.join(get_data_directory(), filename))

def get_data():
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

# def plot_bar_chart(income_statement_df, metric, period='annually', tickers=None):
#     filtered_df = income_statement_df.copy()
#     if tickers:
#         filtered_df = filtered_df[filtered_df['Symbol'].isin(tickers)]
    
#     if period == 'annually':
#         filtered_df['Year'] = pd.to_datetime(filtered_df['Date']).dt.year
#         x_axis = 'Year'
#     else:  # quarterly
#         filtered_df['Quarter'] = pd.to_datetime(filtered_df['Date']).dt.to_period('Q').astype(str)
#         x_axis = 'Quarter'

#     # Remove rows with NaN values in the metric column
#     filtered_df = filtered_df.dropna(subset=[metric])

#     # Sort the dataframe by the selected metric in descending order within each Year/Quarter
#     filtered_df = filtered_df.sort_values([x_axis, metric], ascending=[True, False])

#     # Get the min and max values for the x-axis
#     x_min = filtered_df[x_axis].min()
#     x_max = filtered_df[x_axis].max()

#     fig = px.bar(filtered_df, x=x_axis, y=metric, color='Symbol',
#                  title=f'{metric} for Selected Companies ({period.capitalize()})',
#                  barmode='group',
#                  color_discrete_map=COLOR_THEME,
#                  category_orders={'Symbol': filtered_df.groupby(x_axis)[metric].sum().sort_values(ascending=False).index.tolist()})
    
#     # Set the x-axis range to include full years, extending slightly beyond the data range
#     try:
#         fig.update_xaxes(title_text=x_axis, tickmode='linear', dtick=1, range=[x_min - 0.5, x_max + 0.5])
#     except TypeError:
#         st.warning(f"Unable to set x-axis range due to incompatible data types. x_min: {type(x_min)}, x_max: {type(x_max)}")
#         fig.update_xaxes(title_text=x_axis, tickmode='linear', dtick=1)
#     fig.update_yaxes(title_text=metric)

#     st.plotly_chart(fig, use_container_width=True)

def plot_line_chart(df, metrics, period='annually', show_percentage=False, tickers=None):
    filtered_df = df.copy()
    if tickers:
        filtered_df = filtered_df[filtered_df['Symbol'].isin(tickers)]
    
    if period == 'annually':
        filtered_df['Year'] = pd.to_datetime(filtered_df['Date']).dt.year
        x_axis = 'Year'
    else:  # quarterly
        filtered_df['Year'] = pd.to_datetime(filtered_df['Date']).dt.year
        filtered_df['Quarter'] = pd.to_datetime(filtered_df['Date']).dt.to_period('Q').astype(str)
        filtered_df['Quarter'] = filtered_df['Year'].astype(str) + ' ' + filtered_df['Quarter'].str[-2:]
        x_axis = 'Quarter'

    # Remove rows with NaN values in the metric columns
    filtered_df = filtered_df.dropna(subset=metrics)

    # Sort the dataframe by Year and Quarter
    if period == 'annually':
        filtered_df = filtered_df.sort_values(['Year'])
    else:
        filtered_df = filtered_df.sort_values(['Year', 'Quarter'])

    # Round the metrics to 2 decimal places
    for metric in metrics:
        filtered_df[metric] = filtered_df[metric].round(2)

    if show_percentage:
        for metric in metrics:
            filtered_df[metric] = filtered_df[metric] * 100

    # Ensure each year has all four quarters
    if period == 'quarterly':
        all_quarters = ['Q1', 'Q2', 'Q3', 'Q4']
        filtered_df = filtered_df[filtered_df['Quarter'].str[-2:].isin(all_quarters)]

    fig = px.line(filtered_df, x=x_axis, y=metrics, color='Symbol',
                  title=f'{", ".join(metrics)} Trends ({period.capitalize()})',
                  color_discrete_map=COLOR_THEME)
    
    # Set the x-axis properties
    if period == 'annually':
        x_min = filtered_df[x_axis].min()
        x_max = filtered_df[x_axis].max()
        fig.update_xaxes(title_text=x_axis, tickmode='linear', dtick=1, range=[x_min - 0.5, x_max + 0.5])
    else:
        fig.update_xaxes(title_text=x_axis, tickmode='linear', dtick=1, categoryorder='category ascending')
    
    fig.update_yaxes(title_text=', '.join(metrics))

    # Set y-axis range
    y_min = filtered_df[metrics].min().min()
    y_max = filtered_df[metrics].max().max()
    if y_min >= 0:
        fig.update_yaxes(range=[0, y_max * 1.1])
    else:
        fig.update_yaxes(range=[y_min * 1.1, y_max * 1.1])
    
    # Add hover data
    hover_template = '%{y:.2f}%' if show_percentage else '%{y:.2f}'
    fig.update_traces(hovertemplate=hover_template)

    st.plotly_chart(fig, use_container_width=True)

def create_custom_plotly_chart(df):
    symbols = cfg.tickers
    selected_symbols = st.multiselect('Select one or more symbols', symbols)
    numeric_columns = df.select_dtypes(include=['number']).columns
    numeric_columns = [col for col in numeric_columns if col not in ['Cik', 'Calendar Year']]
    selected_features = st.multiselect('Select one or more features to plot', numeric_columns)

    if not selected_symbols or not selected_features:
        st.warning("Please select at least one symbol and one feature.")
        return

    filtered_df = df[df['Symbol'].isin(selected_symbols)].copy()
    filtered_df['Year'] = pd.to_datetime(filtered_df['Date']).dt.year

    fig = px.line(filtered_df, x='Year', y=selected_features, color='Symbol',
                  title='Selected features over years for chosen symbols',
                  color_discrete_map=COLOR_THEME)
    fig.update_xaxes(title_text='Year', tickmode='linear', dtick=1)
    fig.update_yaxes(title_text='Value')

    st.plotly_chart(fig, use_container_width=True)

def create_custom_chart(df):
    col1, col2 = st.columns(2)
    with col1:
        create_own_chart = st.checkbox("Create your own chart")
    with col2:
        if create_own_chart:
            chart_builder = st.radio("Select a chart builder", ["Pygwalker", "Plotly"])
    
    if create_own_chart:
        if chart_builder == "Pygwalker":
            st.write("You selected Pygwalker")
            pyg_app = get_pyg_app(df)
            pyg_app.explorer()
        else:
            st.write("You selected Plotly")
            create_custom_plotly_chart(df)

def display_income_statment_tab(annual_income_statement_df, quarterly_income_statement_df):
    create_custom_chart(annual_income_statement_df)
    
    st.markdown("### Annual Income Statement Analysis")
    # Let users select tickers at the beginning
    all_tickers = sorted(annual_income_statement_df['Symbol'].unique())
    selected_tickers = st.multiselect('Select tickers to analyze', all_tickers, default=['ALC'])

    col1, col2 = st.columns(2)
    with col1:
        plot_line_chart(annual_income_statement_df, ['Total Revenue'], tickers=selected_tickers)
        plot_line_chart(annual_income_statement_df, ['Normalized EBITDA'], tickers=selected_tickers)
        plot_line_chart(annual_income_statement_df, ['Normalized Income'], tickers=selected_tickers)
    with col2:
        plot_line_chart(annual_income_statement_df, ['Net Income'], tickers=selected_tickers)
        plot_line_chart(annual_income_statement_df, ['Basic EPS'], tickers=selected_tickers)
        plot_line_chart(annual_income_statement_df, ['Operating Expense'], tickers=selected_tickers)

    st.markdown("### Quarterly Income Statement Analysis")
    col1, col2 = st.columns(2)
    with col1:
        plot_line_chart(quarterly_income_statement_df, ['Total Revenue'], period='quarterly', tickers=selected_tickers)
        plot_line_chart(quarterly_income_statement_df, ['Normalized EBITDA'], period='quarterly', tickers=selected_tickers)
        plot_line_chart(quarterly_income_statement_df, ['Normalized Income'], period='quarterly', tickers=selected_tickers)
    with col2:
        plot_line_chart(quarterly_income_statement_df, ['Net Income'], period='quarterly', tickers=selected_tickers)
        plot_line_chart(quarterly_income_statement_df, ['Basic EPS'], show_percentage=True, period='quarterly', tickers=selected_tickers)
        plot_line_chart(quarterly_income_statement_df, ['Operating Expense'], period='quarterly', tickers=selected_tickers)

def display_cash_flow_tab(annual_cash_flow_df, quarterly_cash_flow_df):
    create_custom_chart(annual_cash_flow_df)

    st.markdown("### Annual Cash Flow Analysis")
    # Let users select tickers at the beginning
    all_tickers = sorted(annual_cash_flow_df['Symbol'].unique())
    selected_tickers = st.multiselect('Select tickers to analyze', all_tickers, default=['ALC'])

    col1, col2 = st.columns(2)
    with col1:
        # plot_bar_chart(annual_cash_flow_df, 'Changes In Cash', tickers=selected_tickers)
        plot_line_chart(annual_cash_flow_df, ['Changes In Cash'], tickers=selected_tickers)
        plot_line_chart(annual_cash_flow_df, ['Financing Cash Flow'], tickers=selected_tickers)
    with col2:
        plot_line_chart(annual_cash_flow_df, ['Investing Cash Flow'], tickers=selected_tickers)
        plot_line_chart(annual_cash_flow_df, ['Operating Cash Flow'], tickers=selected_tickers)

    st.markdown("### Quarterly Cash Flow Analysis")
    col1, col2 = st.columns(2)
    with col1:
        plot_line_chart(quarterly_cash_flow_df, ['Changes In Cash'], period='quarterly', tickers=selected_tickers)
        plot_line_chart(quarterly_cash_flow_df, ['Financing Cash Flow'], period='quarterly', tickers=selected_tickers)
    with col2:
        plot_line_chart(quarterly_cash_flow_df, ['Investing Cash Flow'], period='quarterly', tickers=selected_tickers)
        plot_line_chart(quarterly_cash_flow_df, ['Operating Cash Flow'], period='quarterly', tickers=selected_tickers)
    
def display_balance_sheet_tab(annual_balance_sheet_df, quarterly_balance_sheet_df):
    create_custom_chart(annual_balance_sheet_df)

    st.markdown("### Annual Balance Sheet Analysis")
    # Let users select tickers at the beginning
    all_tickers = sorted(annual_balance_sheet_df['Symbol'].unique())
    selected_tickers = st.multiselect('Select tickers to analyze', all_tickers, default=['ALC'])

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
        plot_line_chart(quarterly_balance_sheet_df, ["Current Ratio"],period='Quarterly', tickers=selected_tickers)

@st.cache_resource
def get_pyg_app(df):
    return StreamlitRenderer(df)
        
@st.cache_data
def get_cached_data():
    return get_data()


def main():
    st.set_page_config(layout="wide", page_title="Alcon Financial Competitors Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("images/Alcon.png", width=300)
    with col2:
        st.image("images/TCU.png", width=250)
    
    # get the data
    (
        annual_income_statement_df,
        annual_balance_sheet_df,
        annual_cash_flow_df,
        quarterly_income_statement_df,
        quarterly_balance_sheet_df,
        quarterly_cash_flow_df
    ) = get_data()
    
    topic = st.sidebar.radio("Select an option", ["Income Statement", "Balance Sheet", "Cash Flow", "Chatbot", "Configs"])
    
    st.title('Financial Data Visualization')
    if topic == "Income Statement":
        display_income_statment_tab(annual_income_statement_df, quarterly_income_statement_df)
    
    elif topic == "Balance Sheet":
        display_balance_sheet_tab(annual_balance_sheet_df, quarterly_balance_sheet_df)

    elif topic == "Cash Flow":
        display_cash_flow_tab(annual_cash_flow_df, quarterly_cash_flow_df)
    
    elif topic == "Chatbot":
        chatbot()
    elif topic == "Configs":
        st.header('Changing Configurations')
        
        st.subheader("Current Tickers")
        st.markdown("Here are the tickers currently being tracked: " + ", ".join(cfg.tickers))
        
        st.subheader("Modify Tickers")
        col1, col2 = st.columns(2)
        
        action = st.radio("Choose an action", ("Add a New Ticker", "Remove an Existing Ticker"))

        if action == "Add a New Ticker":
            st.markdown("**Add a New Ticker**")
            new_ticker = st.text_input("Enter a new ticker symbol (e.g., AAPL):")
            if st.button("Add Ticker"):
                if new_ticker and new_ticker not in cfg.tickers:
                    cfg.tickers.append(new_ticker)
                    
                    # Generate a random color for the new ticker
                    import random
                    random_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
                    cfg.COLOR_THEME[new_ticker] = random_color
                    
                    with open('config.py', 'r') as f:
                        lines = f.readlines()
                    with open('config.py', 'w') as f:
                        for line in lines:
                            if line.strip() == "tickers = [":
                                f.write(line)
                            elif line.strip() == "]":
                                f.write(f'    "{new_ticker}",\n{line}')
                            elif line.strip().startswith("COLOR_THEME = {"):
                                f.write(line)
                                f.write(f'    "{new_ticker}": "{random_color}",\n')
                            else:
                                f.write(line)
                    st.success(f"Ticker {new_ticker} has been added with color {random_color}.")
                else:
                    st.warning("The ticker is either empty or already exists.")
        
        elif action == "Remove an Existing Ticker":
            st.markdown("**Remove an Existing Ticker**")
            ticker_to_remove = st.selectbox("Select a ticker to remove", cfg.tickers)
            if st.button("Remove Ticker"):
                if ticker_to_remove in cfg.tickers:
                    cfg.tickers.remove(ticker_to_remove)
                    if ticker_to_remove in cfg.COLOR_THEME:  # Check if the ticker exists in COLOR_THEME
                        del cfg.COLOR_THEME[ticker_to_remove]  # Remove the color associated with the ticker
                    with open('config.py', 'w') as f:
                        f.write("# Configuration for tickers\n")
                        f.write("tickers = [\n")
                        for ticker in cfg.tickers:
                            f.write(f'    "{ticker}",\n')
                        f.write("]\n")
                        f.write("COLOR_THEME = {\n")
                        for ticker, color in cfg.COLOR_THEME.items():
                            f.write(f'    "{ticker}": "{color}",\n')
                        f.write("}\n")
                    st.success(f"Ticker {ticker_to_remove} has been removed.")
                else:
                    st.warning("The ticker was not found.")

        st.subheader("Step 1: Run ETL Pipeline")
        st.text("This will run the ETL pipeline, which will take about 10 seconds to complete...")
        if st.button("Run ETL Pipeline"):
            etl_scripts.main()
            st.success("ETL pipeline completed successfully!")
        
        st.subheader("Step 2: Run Vector Database Population")
        st.text("This will run the Vector Database Population, which will take about 70 seconds to complete...")
        if st.button("Run Vector Database Population"):
            pvf.main()
            st.success("Vector Database Population completed successfully!")

if __name__ == "__main__":
    main()
