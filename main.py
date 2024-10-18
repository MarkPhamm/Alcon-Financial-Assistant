import os
import sys

import streamlit as st
from openai import OpenAI

deploy = True
if deploy:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    CONFIG_PASSWORD = st.secrets["CONFIG_PASSWORD"]

else:
    from dotenv import load_dotenv
    load_dotenv('.env')
    CONFIG_PASSWORD = os.getenv("CONFIG_PASSWORD")

# OpenAI and LangChain imports
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Data handling imports
import pandas as pd
import csv
import plotly.express as px
import random

# Utility imports
from datetime import datetime
import time
import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer

# Configuration imports
import config as cfg

# ETL and Vector Database Population imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'etl'))
from etl import etl_scripts
import populate_vectordb as pvf

# Define the chatbot model
chatbot_model = "gpt-3.5-turbo"

# Initialize embeddings with OpenAI's text-embedding-3-large model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Define the persist directory for the Chroma database
persist_directory = "./chroma_langchain_db"

# Create a Chroma instance for annual and quarterly collections
annual_vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name='alcon_collection_financial_statements_annually')
quarterly_vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name='alcon_collection_financial_statements_quarterly')

# Function to find relevant entries from the Chroma database
def find_relevant_entries_from_chroma_db(query, selected_collection):
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
    # Determine which collection to use based on the selected collection
    vectordb = annual_vectordb if selected_collection == "Annually" else quarterly_vectordb

    # Perform similarity search with score
    results = vectordb.similarity_search_with_score(query, k=5)
    
    # Print results for debugging
    for doc, score in results:
        print(f"Similarity: {score:.3f}")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("---")

    return results

# Function to generate a GPT response
def generate_gpt_response(user_query, chroma_result, client):
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
    response = client.chat.completions.create(
        model=chatbot_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant capable of providing both naive and context-aware responses."},
            {"role": "user", "content": combined_prompt}
        ]
    )

    return response.choices[0].message.content

# Function to handle user queries
def query_interface(user_query, is_first_prompt, selected_collection, client):
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

    # Step 1 and 2: Find relevant information and generate response
    chroma_result = find_relevant_entries_from_chroma_db(user_query, selected_collection)
    gpt_response = generate_gpt_response(user_query, str(chroma_result), client)

    # Step 3: Log the response time
    log_response_time = True
    if log_response_time:
        end_time = time.time()
        response_time = end_time - start_time

        # Log the response time to a CSV file
        log_response_time(user_query, response_time, is_first_prompt)

    # Step 4: Return the generated response
    return gpt_response

def log_response_time(query, response_time, is_first_prompt):
    """
    Log the response time for a query to a CSV file.

    Parameters:
        query (str): The user's query
        response_time (float): The time taken to generate the response
        is_first_prompt (bool): Whether this is the first prompt in the conversation
    """
    csv_file = 'responses.csv'
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'Query', 'Response Time (seconds)', 'Is First Prompt'])
        writer.writerow([datetime.now(), query, f"{response_time:.2f}", "Yes" if is_first_prompt else "No"])

def display_chatbot():
    st.markdown("#### How can we assist you today?")
    # Prompt user for OpenAI API key
    api_key = st.text_input("Enter your OpenAI API key:", type="password")

    if api_key:
        client = OpenAI(api_key=api_key)
        # Initialize a variable to store the selected collection
        selected_collection = st.radio("Select Time Period", ("Annually", "Quarterly"))
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)

        # React to user input
        if prompt := st.chat_input("Type your question here..."):
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt, unsafe_allow_html=True)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Show a loading spinner while waiting for the response
            with st.spinner("Thinking..."):
                # Get bot response
                is_first_prompt = len(st.session_state.messages) == 1
                
                response = query_interface(prompt, is_first_prompt, selected_collection, client)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response, unsafe_allow_html=True)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Add a button to clear chat history at the bottom
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    else:
        st.warning("Please enter your OpenAI API key to proceed.")
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
    create_custom_chart(annual_income_statement_df)
    
    st.markdown("### Annual Income Statement Analysis")
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
    create_custom_chart(annual_cash_flow_df)

    st.markdown("### Annual Cash Flow Analysis")
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
    create_custom_chart(annual_balance_sheet_df)

    st.markdown("### Annual Balance Sheet Analysis")
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

def add_ticker():
    """Add a new ticker to the configuration."""

    new_ticker = st.text_input("Enter a new ticker symbol (e.g., AAPL):")
    if st.button("Add Ticker"):
        if new_ticker and new_ticker not in cfg.tickers:
            cfg.tickers.append(new_ticker)
            
            # Generate a random color for the new ticker
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

def remove_ticker():
    """Remove a ticker from the configuration."""

    ticker_to_remove = st.selectbox("Select a ticker to remove", cfg.tickers)
    if st.button("Remove Ticker"):
        if ticker_to_remove in cfg.tickers:
            cfg.tickers.remove(ticker_to_remove)
            cfg.COLOR_THEME.pop(ticker_to_remove, None)  # Remove the color associated with the ticker
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

def add_new_data():
    """Add new data to the database."""
    uploaded_files = st.file_uploader("Choose CSV files to add new data to the Chabot.", type='csv', accept_multiple_files=True)

    if st.button("Add New Data"):
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join('data', uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success("New data added successfully!")
        else:
            st.warning("No file was uploaded.")

def display_configs_tab():
    st.header('Changing Configurations')
    st.text("This is where you can change the tickers that are being tracked and run the ETL pipeline and Vector Database Population.")
    config_password = st.text_input("Enter your config password:", type="password")
    if config_password == CONFIG_PASSWORD:
        st.subheader("Add New Data")
        # add_new_data()
        st.text("Feature coming soon...")
        
        st.subheader("Modify Tickers")
        st.markdown("Here are the tickers currently being tracked: " + ", ".join(cfg.tickers))
        action = st.radio("Choose an action", ("Add a New Ticker", "Remove an Existing Ticker"))

        if action == "Add a New Ticker":
            st.markdown("**Add a New Ticker**")
            add_ticker()
        
        elif action == "Remove an Existing Ticker":
            st.markdown("**Remove an Existing Ticker**")
            remove_ticker()

        st.subheader("Step 1: Run ETL Pipeline")
        st.text("This will run the ETL pipeline, which will take about 10 seconds to complete...")
        if st.button("Run ETL Pipeline"):
            etl_scripts.main()
            st.success("ETL pipeline completed successfully!")

        st.subheader("Step 2: Run Vector Database Population")
        st.text("This populates the vector database with new data, which will take about 5 seconds to complete...")
        if st.button("Run Vector Database Population"):
            pvf.main()
            st.success("Vector Database Population completed successfully!")
    else:
        st.warning("Incorrect password.")

@st.cache_resource
def get_pyg_app(df: pd.DataFrame) -> StreamlitRenderer:
    """Get the Pygwalker app instance."""
    return StreamlitRenderer(df)
        
@st.cache_data
def get_cached_data() -> list:
    """Get cached financial data."""
    return get_data()

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
    
    topic = st.sidebar.radio("Select an option", ["Income Statement", "Balance Sheet", "Cash Flow", "Chatbot", "Configs"])

    if topic == "Income Statement":
        display_income_statement_tab(annual_income_statement_df, quarterly_income_statement_df)
    elif topic == "Balance Sheet":
        display_balance_sheet_tab(annual_balance_sheet_df, quarterly_balance_sheet_df)
    elif topic == "Cash Flow":
        display_cash_flow_tab(annual_cash_flow_df, quarterly_cash_flow_df)
    elif topic == "Chatbot":
        display_chatbot()
    elif topic == "Configs":
        display_configs_tab()

if __name__ == "__main__":
    main()
