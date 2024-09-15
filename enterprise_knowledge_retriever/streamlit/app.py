import os
import sys
import logging
import yaml
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
from dateutil import parser

current_dir = os.path.dirname(os.path.abspath(_file_))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from enterprise_knowledge_retriever.src.document_retrieval import DocumentRetrieval
from utils.visual.env_utils import env_input_fields, initialize_env_variables, are_credentials_set, save_credentials
from utils.vectordb.vector_db import VectorDb

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')
PERSIST_DIRECTORY = os.path.join(kit_dir, "data/my-vector-db")

logging.basicConfig(level=logging.INFO)
logging.info("URL: http://localhost:8501")



def handle_userinput(user_question, stock_data=None):
    # Ensure 'conversation' object is initialized
    if 'conversation' not in st.session_state or st.session_state.conversation is None:
        if stock_data is not None:
            # Handle questions related to the uploaded stock data
            answer = search_stock_data(stock_data, user_question)
            if answer:
                response = {"answer": answer}
            else:
                response = {"answer": "I'm sorry, I couldn't find an answer to your question based on the uploaded stock data."}
        else:
            st.error("Conversation system not initialized and no stock data uploaded.")
            return
    else:
        # Existing conversation logic
        if user_question:
            try:
                with st.spinner("Processing..."):
                    response = st.session_state.conversation.retrieve({"question": user_question})

                # Append the question and response to the chat history
                st.session_state.chat_history.append(user_question)
                st.session_state.chat_history.append(response["answer"])

                # Process the sources from the response
                sources = set([f'{sd.metadata["filename"]}' for sd in response["source_documents"]])
                sources_text = ""
                for index, source in enumerate(sources, start=1):
                    source_link = source
                    sources_text += f'<font size="2" color="grey">{index}. {source_link}</font>  \n'
                st.session_state.sources_history.append(sources_text)

            except Exception as e:
                st.error(f"An error occurred while processing your question: {str(e)}")

    # Display the chat and sources history
    """for ques, ans, source in zip(st.session_state.chat_history[::2], st.session_state.chat_history[1::2], st.session_state.sources_history):
        with st.chat_message("user"):
            st.write(f"{ques}")

        with st.chat_message("ai", avatar="https://sambanova.ai/hubfs/logotype_sambanova_orange.png"):
            st.write(f"{ans}")
            if st.session_state.show_sources:
                with st.expander("Sources"):
                    st.markdown(f'<font size="2" color="grey">{source}</font>', unsafe_allow_html=True)
"""
def search_stock_data(stock_data, user_question):
    # Simple logic to extract information from the stock_data DataFrame
    user_question_lower = user_question.lower()

    if "average" in user_question_lower:
        if "open" in user_question_lower:
            avg_open = stock_data['Open'].mean()
            return f"The average opening price is {avg_open:.2f}."
        elif "close" in user_question_lower:
            avg_close = stock_data['Close'].mean()
            return f"The average closing price is {avg_close:.2f}."
    elif "date" in user_question_lower:
        return f"The data covers the period from {stock_data['Date'].min()} to {stock_data['Date'].max()}."
    elif "total" in user_question_lower and "volume" in user_question_lower:
        total_volume = stock_data['Volume'].sum()
        return f"The total trading volume is {total_volume:,}."
    else:
        return None

def initialize_document_retrieval():
    # Initialize the conversation object if credentials are set
    if are_credentials_set():
        try:
            st.session_state.conversation = DocumentRetrieval()  # Store in session state
        except Exception as e:
            st.error(f"Failed to initialize DocumentRetrieval: {str(e)}")
            st.session_state.conversation = None  # Ensure conversation is set to None on failure
    else:
        st.session_state.conversation = None





def visualize_data(stock_data):
    # Check for required columns
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in stock_data.columns for col in required_columns):
        st.error("The uploaded data must contain 'Date', 'Open', 'High', 'Low', 'Close', and 'Volume' columns for visualization.")
        return

    st.subheader("Enhanced Data Visualization")

    # Convert Date column to datetime, use 'dayfirst' to handle dd-mm-yyyy format
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], dayfirst=True, errors='coerce')

    # Check for NaT values in 'Date' after conversion
    if stock_data['Date'].isnull().any():
        st.error("There are invalid dates in the data. Please check the 'Date' column.")
        return

    # Plotly Candlestick Chart
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=stock_data['Date'],
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name='Candlestick'
    ))

    # Add Moving Averages
    short_window = 20
    long_window = 50
    stock_data['SMA20'] = stock_data['Close'].rolling(window=short_window, min_periods=1).mean()
    stock_data['SMA50'] = stock_data['Close'].rolling(window=long_window, min_periods=1).mean()

    fig.add_trace(go.Scatter(
        x=stock_data['Date'],
        y=stock_data['SMA20'],
        mode='lines',
        name=f'SMA {short_window}',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=stock_data['Date'],
        y=stock_data['SMA50'],
        mode='lines',
        name=f'SMA {long_window}',
        line=dict(color='red')
    ))

    # Volume Chart
    volume_fig = go.Figure()
    volume_fig.add_trace(go.Bar(
        x=stock_data['Date'],
        y=stock_data['Volume'],
        name='Volume',
        marker_color='rgba(17, 157, 255, 0.6)'
    ))

    # Layout and Annotations for Candlestick Chart
    fig.update_layout(
        title="Stock Price and Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )

    # Layout and Annotations for Volume Chart
    volume_fig.update_layout(
        title="Trading Volume",
        xaxis_title="Date",
        yaxis_title="Volume"
    )

    # Display the plots
    st.plotly_chart(fig)
    st.plotly_chart(volume_fig)

    # Save the plots as PNG files
    if st.button("Save Charts"):
        fig.write_image("stock_price_candlestick_chart.png")
        volume_fig.write_image("stock_volume_chart.png")
        st.success("Charts saved as stock_price_candlestick_chart.png and stock_volume_chart.png")


def analyze_data(stock_data):
    # Check if 'Symbol' column exists
    if 'Symbol' in stock_data.columns:
        st.write(f"Total Symbols: {stock_data['Symbol'].nunique()}")
    else:
        st.warning("'Symbol' column not found in the dataset.")

    # General insights on stock data
    st.write(f"Date range: {stock_data['Date'].min()} to {stock_data['Date'].max()}")
    st.write(f"Total data points: {len(stock_data)}")

    # Summary statistics
    st.write("Summary statistics:")
    st.write(stock_data[['Open', 'High', 'Low', 'Close']].describe())

    # Display trends for Close prices over time
    st.line_chart(stock_data.set_index('Date')['Close'])




def safe_parse_date(date_string):
    if isinstance(date_string, str):
        try:
            return parser.parse(date_string)
        except (ValueError, TypeError):
            return None  # or a default date, e.g., datetime.datetime(1970, 1, 1)
    return None

def predict_stock_prices(stock_data, days_ahead=7):
    # Ensure the 'Date' column is parsed correctly
    stock_data['Date'] = stock_data['Date'].apply(safe_parse_date)

    # Check for NaT values in 'Date' after conversion
    if stock_data['Date'].isnull().any():
        st.error("There are invalid dates in the data. Please check the 'Date' column.")
        return

    # Check for NaN values in 'Close' column
    if stock_data['Close'].isnull().any():
        stock_data = stock_data.dropna(subset=['Close'])

    # Prepare the data for prediction
    stock_data['Days'] = (stock_data['Date'] - stock_data['Date'].min()).dt.days
    X = np.array(stock_data['Days']).reshape(-1, 1)
    y = stock_data['Close']  # Assuming 'Close' column exists and is the target

    # Check if y has any NaN values after dropping
    if y.isnull().any():
        st.error("Target variable contains NaN values after processing. Please check your data.")
        return

    # Train a simple linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Generate future days for prediction
    future_days = np.array([stock_data['Days'].max() + i for i in range(1, days_ahead + 1)]).reshape(-1, 1)
    future_prices = model.predict(future_days)

    # Generate future dates
    future_dates = pd.date_range(stock_data['Date'].max(), periods=days_ahead + 1)[1:]  # Exclude the first date

    # Ensure both have the same length
    if len(future_dates) == len(future_prices):
        predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted Close Price': future_prices})
        st.subheader("Predicted Stock Prices")
        st.write(predictions_df)

        # Plotting the predictions

        # Visualization 2: Interactive Line Chart with Plotly
        line_fig = go.Figure()
        line_fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], mode='lines', name='Historical Prices', line=dict(color='blue')))
        line_fig.add_trace(go.Scatter(x=predictions_df['Date'], y=predictions_df['Predicted Close Price'], mode='lines', name='Predicted Prices', line=dict(color='orange', dash='dash')))
        line_fig.update_layout(title='Stock Price Prediction - Plotly', xaxis_title='Date', yaxis_title='Close Price', legend=dict(x=0, y=1))
        st.plotly_chart(line_fig)

        # Visualization 3: Bar Chart of Historical vs Predicted Prices
        combined_df = pd.concat([stock_data[['Date', 'Close']],
                                  predictions_df.rename(columns={'Predicted Close Price': 'Close'})],
                                 ignore_index=True)
        combined_df['Type'] = np.where(combined_df['Close'].isna(), 'Predicted', 'Historical')

        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(x=combined_df['Date'], y=combined_df['Close'], name='Prices', marker_color='lightblue'))
        bar_fig.update_layout(title='Historical and Predicted Stock Prices', xaxis_title='Date', yaxis_title='Close Price', barmode='group')
        st.plotly_chart(bar_fig)
    else:
        st.error("The lengths of future dates and predicted prices do not match.")

def main():
    os.environ['SAMBANOVA_API_KEY'] = 'cf430214-2555-48f2-848d-6e823dd07679'

    # Store the API key in the session state
    if 'SAMBANOVA_API_KEY' not in st.session_state:
        st.session_state.SAMBANOVA_API_KEY = os.environ['SAMBANOVA_API_KEY']

    with open(CONFIG_PATH, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    prod_mode = config.get('prod_mode', False)
    default_collection = 'ekr_default_collection'

    initialize_env_variables(prod_mode)

    st.set_page_config(
        page_title="Enterprise Knowledge retriever",
        page_icon="https://sambanova.ai/hubfs/logotype_sambanova_orange.png",
    )

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True
    if "sources_history" not in st.session_state:
        st.session_state.sources_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if 'input_disabled' not in st.session_state:
        st.session_state.input_disabled = True
    if 'document_retrieval' not in st.session_state:
        st.session_state.document_retrieval = None

    st.title(":orange[StockVista] Insight Stream")

    # Sidebar for setup and document retrieval
    with st.sidebar:
        st.title("Welcome to Stockvista!")
        st.write("Your one-stop solution for stock insights and predictions.")
        st.write("Dive into the World of Stock Analysis with Us!")
        st.image("https://cdn.sologo.ai/temp24h/logo/5e4e4122-9613-49e4-a6fb-4b17cccdc510.jpeg", caption="Discover your stock potential", use_column_width=True)

        # Existing sidebar code...

    # Display the chat and sources history at the top
    if st.session_state.chat_history:
        for ques, ans, source in zip(st.session_state.chat_history[::2], st.session_state.chat_history[1::2], st.session_state.sources_history):
            with st.chat_message("user"):
                st.write(f"{ques}")

            with st.chat_message("ai", avatar="https://sambanova.ai/hubfs/logotype_sambanova_orange.png"):
                st.write(f"{ans}")
                if st.session_state.show_sources:
                    with st.expander("Sources"):
                        st.markdown(f'<font size="2" color="grey">{source}</font>', unsafe_allow_html=True)

    # Upload stock data for visualization and predictions
    st.markdown("### Upload Stock Data for Visualization and Prediction")
    stock_file = st.file_uploader("Upload a CSV file containing stock data", type=["csv"])

    if stock_file is not None:
        stock_data = pd.read_csv(stock_file)

        if st.button("Visualize Data"):
            visualize_data(stock_data)
            analyze_data(stock_data)  # Call the analysis function to provide insights

        # Allow user to input number of days to predict
        days_ahead = st.slider("Predict how many days ahead?", min_value=1, max_value=360, value=7)

        if st.button("Predict Future Prices"):
            predict_stock_prices(stock_data, days_ahead=days_ahead)



if _name_ == "_main_":
    main()
