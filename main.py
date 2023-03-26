import numpy as np # linear algebra
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import streamlit as st
import pickle

import base64

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('White.png')

def sidebar_bg(side_bg):

   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )
side_bg = 'Black.png'
sidebar_bg(side_bg)
menu = ['Home','About','Email Prediction','From dataset']
choice = st.sidebar.radio("Menu", menu)

email = pd.read_csv("emails")
X_train,X_test,y_train,y_test=train_test_split(email.content,email.Class,test_size=0.25)
clf=pickle.load(open('nlp_model.pkl', 'rb'))

if choice == "Home":
         st.subheader("Home")

         email.set_index('Class', inplace=True)
         st.subheader('Business Objective:')
         st.write('Inappropriate emails would demotivates and spoil the positive environment that would lead to more attrition rate and low productivity and Inappropriate emails could be on form of bullying, racism, sexual favourtism and hate in the gender or culture, in today’s world so dominated by email no organization is immune to these hate emails.-The goal of the project is to identify such emails in the given day based on the above inappropriate content.')

         st.button('Show Dataset')
         st.header('email Data')
         st.write(email)

         st.button("Shape of Dataset")
         p = email.shape
         st.subheader(p)

         st.subheader("""Dataset Information """)

         col1, col2, col3, col4, col5 = st.columns(5)
         col1.metric("No. of Rows", 48076)
         col2.metric("No. of Columns", 5)
         col3.metric("No. of Duplicate values", 00)
         col4.metric("No. of Null Values", 00)
         col5.metric("No. of Missing Values", 00)
elif choice == "About":
    st.subheader("About")

    st.write("""P171 Email Prediction :-

            Group - 4       

            Ms. Gurule Jayashri

            Ms. Arkhade Rohini

            Mr. Bhosale Pranav

            Mr. Sai Praneeth

            Mr. Subin Nair

            Mr Nitin S

             """)
    st.write("""Under the Guidance :-

              Mr. Whatever
              """)

elif choice=='Email Prediction':
    text = st.text_input('Enter your Email:',"")
    if st.button ('Predict'):
      A=clf.predict([text])
      st.subheader(A)

elif choice=='From dataset':
    st.subheader('Enter the row number of email:-')
    st.write('Choose between 0 to 48076')
    s = st.number_input('', min_value=0, max_value=48076)
    if st.button('Analyse'):
        text1 = email['content'][s]
        st.write(text1)
        st.subheader('prediction of your Class:-')
        st.subheader(clf.predict([text1]))













