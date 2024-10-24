import os
import sys

import streamlit as st
from openai import OpenAI

# Data handling imports
import csv

# Utility imports
from datetime import datetime
import time

# Configuration imports
import config as cfg

deploy = cfg.deploy
if deploy:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
else:
    from dotenv import load_dotenv
    load_dotenv('.env')

# OpenAI and LangChain imports
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

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
    results = vectordb.similarity_search_with_score(query, k=3)
    
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
        str: A formatted string containing the augmented response
    '''    
    
    current_year = datetime.now().year
    last_quarter = 2

    # Generate an augmented response in a single API call
    combined_prompt = f"""User query: {user_query}

    You are a financial advisor at ALCON Inc. Please provide an augmented response considering the following related information from our database:
    {chroma_result}

    The current year is {current_year} and the last available quarter is {last_quarter}.

    Format your response as follows:
    **Augmented Response**

    [Your augmented response here]
    """
    response = client.chat.completions.create(
        model=chatbot_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant capable of providing context-aware responses."},
            {"role": "user", "content": combined_prompt}
        ]
    )

    return response.choices[0].message.content

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

    # Check if the user query includes the word 'competitors'
    if 'competitors' in user_query.lower():
        user_query = user_query.replace('competitors', f'competitors including {", ".join(ticker for ticker in cfg.tickers if ticker != "ALC")}')

    # Step 1 and 2: Find relevant information and generate response
    chroma_result = find_relevant_entries_from_chroma_db(user_query, selected_collection)
    gpt_response = generate_gpt_response(user_query, str(chroma_result), client)

    # Step 3: Log the response time
    logging_response_time = True
    if logging_response_time:
        end_time = time.time()
        response_time = end_time - start_time

        # Log the response time to a CSV file
        log_response_time(user_query, response_time, is_first_prompt)

    # Step 4: Return the generated response
    return gpt_response

def display_chatbot():
    st.title("💬 Chatbot")
    st.write(
    "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    )
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
                # Replace $ with \$
                safe_content = message["content"].replace('$', '\\$')
                st.markdown(safe_content, unsafe_allow_html=True)  

        # React to user input
        if prompt := st.chat_input("Type your question here..."):
            # Display user message in chat message container
            with st.chat_message("user"):
                safe_prompt = prompt.replace('$', '\\$')
                st.markdown(safe_prompt) 
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Show a loading spinner while waiting for the response
            with st.spinner("Thinking..."):
                # Get bot response
                is_first_prompt = len(st.session_state.messages) == 1
                
                response = query_interface(prompt, is_first_prompt, selected_collection, client)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                safe_response = response.replace('$', '\\$')
                st.markdown(safe_response, unsafe_allow_html=True) 
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Add a button to clear chat history at the bottom
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("Please add your OpenAI API key to continue.", icon="🗝️")

def main():
    display_chatbot()   

if __name__ == "__main__":
    main()

