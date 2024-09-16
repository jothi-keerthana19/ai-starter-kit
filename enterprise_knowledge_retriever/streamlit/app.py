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
import altair as alt
import requests
import json
from transformers import pipeline


current_dir = os.path.dirname(os.path.abspath(__file__))
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
# Add this function to compare multiple stock dataframes
def compare_stock_data(stock_data_list):
    # Combine the stock data into a single DataFrame with a multi-index
    combined_df = pd.concat(stock_data_list, keys=[f'Stock {i+1}' for i in range(len(stock_data_list))])

    # Aggregate summary statistics
    combined_summary = combined_df.groupby(level=0).agg(
        Total_Volume=('Volume', 'sum'),
        Avg_Open=('Open', 'mean'),
        Avg_Close=('Close', 'mean')
    ).reset_index()

    # Display the comparison summary
    st.write("Comparison Summary of Uploaded Stock Data:")
    st.write(combined_summary)

    # Remove the visualization code to focus on the comparison table
    # If needed, you can add further analysis or insights here.

# Function for the Q&A section
nlp = pipeline("question-answering")

# Function to load dataset (accepts CSV or Excel)
def load_dataset(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)  # Pass the uploaded file directly
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)  # Pass the uploaded file directly
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")

        # Check the columns
        st.write("Columns in DataFrame:", df.columns.tolist())
        return df
    else:
        raise ValueError("No file uploaded.")

# Sample context generation from dataset for answering queries
def generate_context_from_df(df):
    context = ""
    for index, row in df.iterrows():
        context += f"On {row['Date']}, {row['Symbol']} had an opening price of {row['Open']}, a high of {row['High']}, a low of {row['Low']}, and a closing price of {row['Close']}. The volume traded was {row['Volume']}. "
    return context

# Function to generate recommendations based on stock trends
def generate_recommendation(df):
    avg_close = df['Close'].mean()
    latest_close = df['Close'].iloc[-1]
    if latest_close > avg_close:
        return "The latest closing price is above the average closing price. Consider holding or buying more shares."
    else:
        return "The latest closing price is below the average closing price. It might be a good time to evaluate selling options."

# Function to generate insights based on stock performance
def generate_insight(df):
    df['Price Change'] = df['Close'].diff()
    volatility = df['Price Change'].std()
    max_close = df['Close'].max()
    max_close_date = df[df['Close'] == max_close]['Date'].iloc[0]

    insights = (
        f"The average closing price for the last month is {df['Close'].mean():.2f}.\n"
        f"The highest closing price was {max_close} on {max_close_date}.\n"
        f"The stock shows a volatility of {volatility:.2f}, indicating fluctuations in price."
    )
    return insights

# Function to generate the best time to invest
def best_time_to_invest(df):
    df['Date'] = pd.to_datetime(df['Date'])  # Ensure the 'Date' column is in datetime format
    buy_signal = df[df['Close'] < df['Close'].rolling(window=5).mean()]  # Buy when current close is below the 5-day average
    if not buy_signal.empty:
        best_dates = buy_signal['Date'].dt.date.unique()  # Get unique dates to buy
        return f"Consider investing on the following dates: {', '.join(map(str, best_dates))}."
    else:
        return "No specific dates are suggested for investment based on the analysis."

# Function to process user query
def answer_user_query(user_query, context, df):
    user_query_lower = user_query.lower()

    if "suggest" in user_query_lower or "recommend" in user_query_lower:
        return generate_recommendation(df)
    elif "insight" in user_query_lower or "insights" in user_query_lower:
        return generate_insight(df)
    elif "best time to invest" in user_query_lower:
        return best_time_to_invest(df)
    else:
        query = {
            'question': user_query,
            'context': context
        }
        result = nlp(query)
        return result['answer']

# Main fun

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
        st.title("Stock Data Comparison Tool")

    # File uploader for CSV files
    uploaded_files = st.file_uploader("Upload Stock Data CSV Files", type="csv", accept_multiple_files=True)

    stock_data_list = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Read the uploaded CSV file into a DataFrame
            stock_data = pd.read_csv(uploaded_file)
            stock_data_list.append(stock_data)

        # Call the function to compare stock data
        compare_stock_data(stock_data_list)

        st.title("Stock Data Question Answering System")

    # Upload CSV or Excel file
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        try:
            # Load the dataset
            df = load_dataset(uploaded_file)

            # Generate context from the dataset
            context = generate_context_from_df(df)

            # Input for user query
            user_query = st.text_input("Please enter your question:")

            if user_query:
                # Get the answer based on the dataset
                answer = answer_user_query(user_query, context, df)
                st.write(f"**Answer:** {answer}")

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Entry point for the script

# Entry point for the script

        # Additional content for stock data comparison goes here
        # Remove the visualization button
        # Uncomment or remove the following lines if you had a visualization button
        # if st.button("Visualize Data"):
        #     visualize_stock_data(stock_data_list)

        # Ensure there are no leftover visualization button statements
        # Any reference to visualizations should be removed or commented out


if __name__ == "__main__":
    main()
