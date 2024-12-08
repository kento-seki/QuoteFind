from sentence_transformers import SentenceTransformer, util
import streamlit as st
import pandas as pd
import numpy as np
import torch
import os

############################### HELPER FUNCTIONS ###############################

def get_quote_by_index(idx, df):
    """
    Given the index number and the dataframe of quotes, returns the quote along
    with the author.
    """
    quote = df.iloc[idx].quote
    author = df.iloc[idx].author
    return quote, author


def display_quote(quote, author):
    """
    Function to format and display the given quote.
    """
    st.markdown(f"""
        <style>
        .quote {{
            font-family: "Georgia", serif;
            font-size: 16px;
            font-style: italic;
            color: #333333;
            padding: 10px;
            margin: 20px;
            background-color: #f9f9f9;
            border-left: 8px solid #cccccc;
        }}
        .author {{
            font-family: "Arial", sans-serif;
            font-size: 14rpx;
            color: #666666;
            text-align: right;
            margin-right: 30px;
        }}
        </style>
        <div class="quote">"{quote}"</div>
        <div class="author">- {author}</div>
        """, unsafe_allow_html=True)


@st.cache_data
def construct_embedding_matrix(EMBEDDINGS_DIRECTORY):
    """
    Construct the embedding matrix from the pieces stored as pickle files in
    the embeddings directory.
    """
    embedding_files = sorted(os.listdir(EMBEDDINGS_DIRECTORY))
    embedding_matrix = None

    for file in embedding_files:
        piece = np.load(EMBEDDINGS_DIRECTORY + '/' + file, allow_pickle=True)

        if embedding_matrix is None:
            embedding_matrix = piece
        else:
            embedding_matrix = np.vstack((embedding_matrix, piece))

    return embedding_matrix

################################# MAIN #########################################

QUOTE_FILE_PATH = 'quotes_clean.csv'
EMBEDDINGS_DIRECTORY = 'SentenceBERT_embeddings'
SENTENCEBERT_MODEL_NAME = 'all-MiniLM-L6-v2'

st.title('Semantic Quote Search 🔎')
st.markdown(
    """
    Hi there!

    This app enables **highly intuitive and accurate quote-searching by
    harnessing the power of Sentence-BERT**, a state-of-the-art machine learning
    model. To utilize this, simply input a sentence or phrase in the box below.
    The model then processes your input, understanding the context and semantic
    meaning behind your words. It compares this against the dataset of quotes,
    identifying those with the closest matching meanings.

    This advanced algorithm ensures that **the quotes you receive aren't just
    keyword-based matches** but deeply resonate with the sentiment or idea
    you're expressing. This is especially useful for finding quotes that align
    with specific emotions, thoughts, or situations, providing you with a
    powerful tool for inspiration, reflection, or expression.
    """
)

# Get inputs
quotes_df = pd.read_csv(QUOTE_FILE_PATH)
model = SentenceTransformer(SENTENCEBERT_MODEL_NAME)
embedding_matrix = construct_embedding_matrix(EMBEDDINGS_DIRECTORY)
input_string = st.text_input('Enter your search string here')
if input_string == '':
    st.warning('Please enter more than the empty string.')
    exit()
num_results = int(st.number_input(
        label = 'Enter the number of results to return',
        min_value = 1,
        value = 10
))

# Get the quotes most similar to the input string
input_embedding = model.encode(input_string)
similarity_scores = util.cos_sim(input_embedding, embedding_matrix)
values, indices = torch.topk(similarity_scores, k=num_results)

for idx in indices.tolist()[0]:
    quote, author = get_quote_by_index(idx, quotes_df)

    # Display each quote
    display_quote(quote, author)
