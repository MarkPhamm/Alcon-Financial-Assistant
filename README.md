# Alcon Financial Competitors Analysis App
Access our Streamlit app [here](https://alcon-financial-assistant.streamlit.app/)

## Overview
This Streamlit application provides a comprehensive financial analysis tool for Alcon and its competitors. It offers various features including financial data visualization, a chatbot for queries, and configuration management.

## Features
1. **Financial Data Visualization**
   - Income Statement Analysis (Annual and Quarterly)
     ![image](https://github.com/user-attachments/assets/01971893-86b8-4061-9138-8f57518d8979)
   - Balance Sheet Analysis (Annual and Quarterly)
     ![image](https://github.com/user-attachments/assets/8e0d49d9-c5e9-486f-98ef-b8375d0dcdc1)
   - Cash Flow Analysis (Annual and Quarterly)
     ![image](https://github.com/user-attachments/assets/32b3d48c-aa1c-41ff-a063-222ae9f0da2f)
   - Interactive charts using Plotly and Pygwalker
     ![image](https://github.com/user-attachments/assets/f183164c-2b8d-44ea-9f2f-cfb909867f92)

2. **RAG-powered Chatbot**
   ![image](https://github.com/user-attachments/assets/a8cd78bf-dd55-4986-bad7-a3fda456ecd0)
   - AI-powered assistant for financial queries
   - Utilizes OpenAI's GPT model and Chroma vector database for retrieval-augmented generation
   - Provides context-aware responses based on up-to-date financial data
  
   - **RAG Architecture**
   ![image](https://github.com/user-attachments/assets/82fe2c8f-ee92-4b63-81b4-f3f185d97d88)

4. **Configuration Management**
   - Add or remove ticker symbols for analysis

   
     ![image](https://github.com/user-attachments/assets/b53d0722-e8f3-4690-b82a-121414015fd3)
   - Run ETL (Extract, Transform, Load) pipeline
  
     
     ![image](https://github.com/user-attachments/assets/ea618c7b-c98a-4dde-bf20-2098ded081e6)
   - Populate Vector Database for enhanced chatbot performance
  
     
     ![image](https://github.com/user-attachments/assets/558fb045-e678-4734-997c-94ab353d2282)

## Installation

1. Clone the repository:
   ```
   git clone [repository-url]
   ```

2. Navigate to the project directory:
   ```
   cd [project-directory]
   ```

3. Create a virtual environment:
   ```
   python -m venv .venv
   ```

4. Activate the virtual environment:
   - On Windows:
     ```
     .venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source .venv/bin/activate
     ```

5. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

6. Set up environment variables:
   - Create a `.env` file in the root directory
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Navigate through different sections using the sidebar:
   - Income Statement
   - Balance Sheet
   - Cash Flow
   - Chatbot
   - Configs

3. In the visualization sections, you can:
   - Select specific companies for comparison
   - Choose between annual and quarterly data
   - Create custom charts using Plotly or Pygwalker

4. Use the Chatbot for financial queries related to the loaded data

5. In the Configs section, you can:
   - Add or remove ticker symbols
   - Run the ETL pipeline to update data
   - Populate the vector database for improved chatbot responses

## Data Sources

The application uses financial data stored in CSV files:
- `annually_income_statement.csv`
- `annually_balance_sheet.csv`
- `annually_cash_flow.csv`
- `quarterly_income_statement.csv`
- `quarterly_balance_sheet.csv`
- `quarterly_cash_flow.csv`

## Dependencies

- Streamlit
- OpenAI
- LangChain
- Pandas
- Plotly
- Pygwalker
- Chroma

## Customization

You can modify the `config.py` file to change the list of tracked tickers and update the color theme for visualizations.

## Note

Ensure that you have the necessary permissions and comply with the terms of service for all data sources and APIs used in this application.
