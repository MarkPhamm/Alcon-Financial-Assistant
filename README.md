# Alcon Financial Competitors Analysis App

## Overview

This Streamlit application provides a comprehensive financial analysis tool for Alcon and its competitors. It offers various features including financial data visualization, a chatbot for queries, and configuration management.

## Features

1. **Financial Data Visualization**
   - Income Statement Analysis (Annual and Quarterly)
   - Balance Sheet Analysis (Annual and Quarterly)
   - Cash Flow Analysis (Annual and Quarterly)
   - Interactive charts using Plotly and Pygwalker

2. **Chatbot**
   - AI-powered assistant for financial queries
   - Utilizes OpenAI's GPT model and Chroma vector database for context-aware responses

3. **Configuration Management**
   - Add or remove ticker symbols for analysis
   - Run ETL (Extract, Transform, Load) pipeline
   - Populate Vector Database for enhanced chatbot performance

## Installation

1. Clone the repository:
   ```
   git clone [repository-url]
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
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