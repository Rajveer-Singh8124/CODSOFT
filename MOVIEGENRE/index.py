import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import streamlit as st
import pickle
import nltk

st.set_page_config(page_title='MOVIE GENRE CLASSIFICATION', layout = 'wide', page_icon = 'movie.png', initial_sidebar_state = 'auto')
@st.cache_data
def download_nltk_data():
    nltk.download('stopwords')
    print("NLTK stopwords downloaded successfully.")

download_nltk_data()

ps= PorterStemmer()


def remove_whitespace(text):
    return text.str.strip()

def remove_punctuation(text):
    return text.str.replace('[^\w\s]', '', regex=True)

def word_token(text):
    return text.apply(lambda x: x.split())

def remove_stopwords(text):
    return text.apply(lambda x: [word for word in x if word not in stopwords.words("english")])

def stemming(text):
    return text.apply(lambda x: [ps.stem(word) for word in x ]) 

def string_fun(text):
    return text.astype(str).agg(''.join, axis=0)


def sparse_to_csr(sparse_matrix):
    if isinstance(sparse_matrix, csr_matrix):
        return sparse_matrix
    return csr_matrix(sparse_matrix)

pipe =pickle.load(open("pipe.pkl","rb"))

st.header("Predict  The Genre Of A Movie",divider="rainbow")

text = st.text_area("Movie Overview")

df = pd.DataFrame([text], columns=['text'])

pridect = pipe.predict(df)[0]

print(type(pridect))

if st.button("Pridect",type="primary"):
    if len(text)!=0:
        st.write(pridect)
