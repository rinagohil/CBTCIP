# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:14:00 2024

@author: Rina Gohil
"""

import streamlit as st
import numpy as np
import pickle

# Title of the web application
st.title("IRIS Flower Classification WebApp")

# Dropdown menu for selecting the model
activities = ['SVM', 'KNN', 'DT', 'RF', 'NB']
option = st.sidebar.selectbox('Which model would you like to use?', activities)

# Sliders for inputting flower measurements
sl = st.slider('Select SepalLengthCm', 0.0, 10.0)
sw = st.slider('Select SepalWidthCm ', 0.0, 5.0)
pl = st.slider('Select PetalLengthCm', 0.0, 10.0)
pw = st.slider('Select PetalWidthCm', 0.0, 5.0)

# Create a feature list from the slider inputs
feature_list = [sl, sw, pl, pw]
single_pred = np.array(feature_list).reshape(1, -1)

# Class names
clas = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# Load the appropriate model and make a prediction
if st.button('Predict'):
    if option == 'SVM':
        model = pickle.load(open('SVM.pkl', 'rb'))
    elif option == 'KNN':   
        model = pickle.load(open('KNN.pkl', 'rb'))
    elif option == 'DT':   
        model = pickle.load(open('DT.pkl', 'rb'))
    elif option == 'RF':   
        model = pickle.load(open('RF.pkl', 'rb'))
    else:
        model = pickle.load(open('NB.pkl', 'rb'))

    prediction = model.predict(single_pred)
    st.success(f'The predicted class is: {clas[int(prediction)]}')