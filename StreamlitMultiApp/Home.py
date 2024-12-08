import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Quotes, quotes, quotes...",
    page_icon="👋",
)

st.sidebar.success("Select a demo above.")

st.write("# Welcome to the Quote App! 👋")

st.markdown(
    """
    This app contains various uses for a dataset of English quotes. The 
    **Quote Search** page allows you to search for quotes with a particular 
    meaning by entering a text string, and the **Quote Visualiser** page allows 
    you to see the distribution of quotes by topic.

    ### About the dataset
    The dataset contains almost 500,000 quotes scraped from popular websites 
    such as goodreads, BrainyQuote, Famous Quotes & Authors, and Curated 
    Quotes. The dataset was made publicly available in 2018 by the authors 
    Shivali Goel, Rishi Madhok, Shweta Garg. Further details can be found 
    [here](https://github.com/ShivaliGoel/Quotes-500K).
    """
)

quotes_df = pd.read_csv('quotes_clean.csv')
st.write(quotes_df.head())

st.markdown(
    """
    ### What is Streamlit?
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    """
)