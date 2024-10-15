from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import chromadb

import os
from dotenv import load_dotenv
import pandas as pd
import logging

load_dotenv('.env') # looks for .env in Python script directory unless path is provided
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Document locations (relative to this py file)
folder_paths = ['data']

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_csv_files(folders):
    """
    Process CSV files from specified folders and convert them to a list of LangChain documents.
    """
    all_docs = []
    for folder in folders:
        if not os.path.exists(folder):
            logging.warning(f"Folder '{folder}' does not exist.")
            continue
        
        logging.info(f"Processing folder: {folder}")
        csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
        
        for file in csv_files:
            file_path = os.path.join(folder, file)
            logging.info(f"Processing CSV file: {file_path}")
            
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Process each row as a separate document
            for _, row in df.iterrows():
                content = " ".join([f"{col}: {val}" for col, val in row.items()])
                metadata = {
                    "source": file,
                    "row_index": _,
                }
                doc = Document(page_content=content, metadata=metadata)
                all_docs.append(doc)
            
            logging.info(f"Processed {len(df)} rows from {file}")
    
    logging.info(f"Total documents created: {len(all_docs)}")
    return all_docs

def insert_into_vector_db(docs):
    """
    Inserts documents into a vector database.
    """
    try:
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model='text-embedding-3-large')

        client = chromadb.PersistentClient(path="./chroma_langchain_db")
        collection = client.get_or_create_collection(
            name='alcon_collection_financial_statements',
            metadata={'hnsw:space': 'cosine'}
        )

        alcon_vectorstore = Chroma(
            client=client,
            collection_name='alcon_collection_financial_statements',
            embedding_function=embeddings,
        )

        # Insert documents into the vector store
        alcon_vectorstore.add_documents(documents=docs)
        logging.info(f"Inserted {len(docs)} documents into the vector store")
    except Exception as e:
        logging.error(f"Error inserting documents into vector store: {str(e)}")
        raise

def delete_vector_db():
    """
    Deletes the entire vector database.
    """
    try:
        client = chromadb.PersistentClient(path="./chroma_langchain_db")
        client.delete_collection(name='alcon_collection_financial_statements')
        logging.info("Vector database collection deleted successfully")
    except Exception as e:
        logging.error(f"Error deleting vector database collection: {str(e)}")
        raise

def main():
    """
    Main function to process CSV files and populate the vector database.
    """
   
    try:
        # Delete existing vector database
        # delete_vector_db()
        logging.info("Existing vector database deleted")

        # Process CSV files
        all_docs = process_csv_files(folder_paths)

        # Insert documents into the vector database
        insert_into_vector_db(all_docs)

        logging.info("Vector database population completed successfully")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

# Execute the main function when the script is run
if __name__ == "__main__":
    main()

# Add this function call to the main function if you want to delete the database before repopulating
# Uncomment the following line in the main() function to use it:
# delete_vector_db()

