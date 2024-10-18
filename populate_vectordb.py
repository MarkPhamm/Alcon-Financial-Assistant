from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import chromadb

import os
from dotenv import load_dotenv
import pandas as pd
import logging
import shutil

load_dotenv('.env')  # looks for .env in Python script directory unless path is provided
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Document locations (relative to this py file)
folder_paths = ['data']

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_csv_files(folders):
    """
    Process CSV files from specified folders and convert them to a list of LangChain documents.
    """
    all_docs_annually = []
    all_docs_quarterly = []
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
                
                # Determine if the document is annual or quarterly based on the file name
                if 'annual' in file.lower():
                    all_docs_annually.append(doc)
                elif 'quarterly' in file.lower():
                    all_docs_quarterly.append(doc)
            
            logging.info(f"Processed {len(df)} rows from {file}")
    
    logging.info(f"Total annual documents created: {len(all_docs_annually)}")
    logging.info(f"Total quarterly documents created: {len(all_docs_quarterly)}")
    return all_docs_annually, all_docs_quarterly

def insert_into_vector_db(docs_annually, docs_quarterly):
    """
    Inserts documents into two separate vector databases for annual and quarterly data.
    """
    try:
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model='text-embedding-3-large')

        client = chromadb.PersistentClient(path="./chroma_langchain_db")
        
        # Create or get collections for annual and quarterly data
        annual_collection = client.get_or_create_collection(
            name='alcon_collection_financial_statements_annually',
            metadata={'hnsw:space': 'cosine'}
        )
        quarterly_collection = client.get_or_create_collection(
            name='alcon_collection_financial_statements_quarterly',
            metadata={'hnsw:space': 'cosine'}
        )

        alcon_vectorstore_annually = Chroma(
            client=client,
            collection_name='alcon_collection_financial_statements_annually',
            embedding_function=embeddings,
        )
        alcon_vectorstore_quarterly = Chroma(
            client=client,
            collection_name='alcon_collection_financial_statements_quarterly',
            embedding_function=embeddings,
        )

        # Insert documents into the respective vector stores
        alcon_vectorstore_annually.add_documents(documents=docs_annually)
        alcon_vectorstore_quarterly.add_documents(documents=docs_quarterly)
        
        logging.info(f"Inserted {len(docs_annually)} annual documents into the vector store")
        logging.info(f"Inserted {len(docs_quarterly)} quarterly documents into the vector store")
    except Exception as e:
        logging.error(f"Error inserting documents into vector store: {str(e)}")
        raise

def delete_vector_db():
    """
    Deletes everything in the chroma_langchain_db directory except for the chroma.sqlite3 file.
    """
    try:
        # Remove all files except for chroma.sqlite3
        for filename in os.listdir("./chroma_langchain_db"):
            if filename != "chroma.sqlite3":
                file_path = os.path.join("./chroma_langchain_db", filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        
        logging.info("All contents in the chroma_langchain_db directory deleted successfully except for chroma.sqlite3")
    except Exception as e:
        logging.error(f"Error deleting contents of chroma_langchain_db directory: {str(e)}")
        raise

def main():
    """
    Main function to process CSV files and populate the vector databases.
    """
   
    try:
        # Delete existing vector databases
        delete_vector_db()
        logging.info("Existing vector databases deleted")

        # Process CSV files
        all_docs_annually, all_docs_quarterly = process_csv_files(folder_paths)

        # Insert documents into the vector databases
        insert_into_vector_db(all_docs_annually, all_docs_quarterly)

        logging.info("Vector database population completed successfully")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

# Execute the main function when the script is run
if __name__ == "__main__":
    main()
