# OpenAI and LangChain imports
from openai import AsyncOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Streamlit and environment imports
import streamlit as st
from dotenv import load_dotenv
import os

# Data handling imports
import pandas as pd
import csv
import plotly.express as px

# Utility imports
from datetime import datetime
from functools import lru_cache
import asyncio
import time

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
    results = await asyncio.to_thread(vectordb.similarity_search_with_score, query, k=1)
    
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
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    return os.path.join(parent_dir, 'data')

def get_data():
    data_dir = get_data_directory()
    income_statement_df = pd.read_csv(os.path.join(data_dir, 'income_statement.csv'))
    balance_sheet_df = pd.read_csv(os.path.join(data_dir, 'balance_sheet.csv'))
    cash_flow_df = pd.read_csv(os.path.join(data_dir, 'cash_flow.csv'))
    key_metrics_df = pd.read_csv(os.path.join(data_dir, 'key_metrics.csv'))
    return income_statement_df, balance_sheet_df, cash_flow_df, key_metrics_df 


def analyze_income_statement(income_statement_df):
    st.write('Income Statement Analysis')

    # Get unique symbols
    symbols = income_statement_df['Symbol'].unique()

    # User inputs
    selected_symbols = st.multiselect('Select one or more symbols', symbols)
    features = income_statement_df.select_dtypes(include=['number']).columns  # Excluding non-numeric columns
    selected_features = st.multiselect('Select one or more features to plot', features)

    if not selected_symbols or not selected_features:
        st.warning("Please select at least one symbol and one feature.")
        return

    # Filter data for the selected symbols
    filtered_df = income_statement_df[income_statement_df['Symbol'].isin(selected_symbols)]

    # Create the interactive chart
    fig = px.line(filtered_df, x='Date', y=selected_features, color='Symbol',
                  title=f'Selected features over time for chosen symbols')
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Value')

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

    # Display the dataframe
    st.write('Income Statement Data')
    st.write(filtered_df)

def main():
    st.set_page_config(layout="wide", page_title="Financial Data Visualization")
    
    # Load data
    income_statement_df, balance_sheet_df, cash_flow_df, key_metrics_df = get_data()
    
    topic = st.sidebar.radio("Select an option", ["Income Statement", "Balance Sheet", "Cash Flow", "Key Metrics", "Chatbot"])
    
    st.title('Financial Data Visualization')
    if topic == "Income Statement":
        analyze_income_statement(income_statement_df)
    elif topic == "Balance Sheet":
        st.write('Balance Sheet')
        st.write(balance_sheet_df)
    elif topic == "Cash Flow":
        st.write('Cash Flow')
        st.write(cash_flow_df)
    elif topic == "Chatbot":
        chatbot()

if __name__ == "__main__":
    main()





