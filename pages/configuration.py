import config as cfg
import random
import streamlit as st
import sys
import os

deploy = cfg.deploy
if deploy:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    CONFIG_PASSWORD = st.secrets["CONFIG_PASSWORD"]
else:
    from dotenv import load_dotenv
    load_dotenv('.env')
    CONFIG_PASSWORD = os.getenv("CONFIG_PASSWORD")

# ETL and Vector Database population imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'etl'))
from etl import etl_scripts
import populate_vectordb as pvf

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
                f.write("deploy = False\n")  # Retain the deploy variable
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
    st.title("⚙️ Configurations")
    st.write(
        "This is where you can change the tickers that are being tracked and run the ETL pipeline and Vector Database Population. "
        "To learn more about how to manage your tickers and run the ETL process, please refer to our documentation."
    )
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
        st.info("Please add your config key to continue.", icon="🗝️")

def main():
    display_configs_tab()   

if __name__ == "__main__":
    main()
