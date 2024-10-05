from openai import OpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import gradio as gr
from dotenv import load_dotenv
import os
import time
import csv
from datetime import datetime
from functools import lru_cache
import asyncio
from openai import AsyncOpenAI


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
collection_name = "uw_collection"

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

    # Step 1: Find relevant information from the Chroma database
    chroma_result = await find_relevant_entries_from_chroma_db(user_query)

    # Step 2: Generate a response using GPT model, considering both the query and relevant information
    gpt_response = await generate_gpt_response(user_query, str(chroma_result))

    end_time = time.time()
    response_time = end_time - start_time

    # Log the response time to a CSV file
    await asyncio.to_thread(log_response_time, user_query, response_time, is_first_prompt)

    # Step 3: Return the generated response
    return gpt_response

def log_response_time(query, response_time, is_first_prompt):
    """
    Log the response time for a query to a CSV file.

    Parameters:
        query (str): The user's query
        response_time (float): The time taken to generate the response
        is_first_prompt (bool): Whether this is the first prompt in the conversation
    """
    csv_file = 'response_times.csv'
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'Query', 'Response Time (seconds)', 'Is First Prompt'])
        writer.writerow([datetime.now(), query, f"{response_time:.2f}", "Yes" if is_first_prompt else "No"])

# Create a Gradio interface for the chatbot
with gr.Blocks(title="Tarrant County United Way") as interface:
    gr.Markdown("# Tarrant County United Way")
    gr.Markdown("How can we help you?")
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Enter your message here...")
    clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    async def bot(history):
        user_message = history[-1][0]
        is_first_prompt = len(history) == 1
        bot_message = await query_interface(user_message, is_first_prompt)
        history[-1][1] = bot_message
        return history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

# Launch the Gradio interface
if __name__ == "__main__":
    interface.launch(share=True)

