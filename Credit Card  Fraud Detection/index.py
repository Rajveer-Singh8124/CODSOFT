import numpy as np
import streamlit as st
import pandas as pd 
import pickle 
from sklearn.preprocessing import LabelEncoder
import time
st.set_page_config(page_title="Credit Card Fraud Detection",layout="wide",page_icon="fraud.png")
lb = LabelEncoder()


def text_col(df_pipe):
    df_pipe["category"] = lb.fit_transform(df_pipe["category"])
    
    return df_pipe


pipe = pickle.load(open("pipe.pkl","rb"))
category = pickle.load(open("category.pkl","rb"))



st.header("Credit Card Fraud Detection",divider="blue")
	
with st.form("input-form"):
    col = st.columns(2)
    
    with st.container():
        with col[0]:
            cc_num = st.number_input("Credit card number*",key=1,step=1);
        with col[1]:
            amt = st.number_input("Amount of the transaction*",key=2);
        
    
    col = st.columns(2)
    with st.container():
        with col[0]:
            long = st.number_input("Longitude of the cardholder's address*",key=4);
        with col[1]:
            lat = st.number_input("Latitude of the cardholder's address*",key=3);

    with st.container():
        with col[0]:
            merch_lat = st.number_input("Latitude of the merchant's location*")        
        with col[1]:
            merch_long = st.number_input("Longitude of the merchant's location*")

        category = st.selectbox("Category of the transaction",category["category"])
        pd = pd.DataFrame({"cc_num":cc_num,"lat":lat,"long":long,"amt":amt,"merch_lat":merch_lat,"merch_long":merch_long,"category":category},index=["a"])
        predict = pipe.predict(pd)

    if st.form_submit_button("Predict"):
        if (cc_num == 0 or lat == 0 or long==0 or amt == 0 or merch_lat == 0 or merch_long == 0):
            st.warning("Please enter the required field.",icon="⚠")
        else:
            with st.spinner("Predict..."):
                time.sleep(2)
            st.success("Done")
            if predict[0] == 0:
                st.subheader("Non-Fraud")    
            else:
                st.subheader("Fraud")
        
