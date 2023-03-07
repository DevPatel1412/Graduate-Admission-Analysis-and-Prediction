import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures

# Load the saved models
gre_model = pickle.load(open("C:/Users/Dev Patel/Documents/Collage_files/Team_4331 (COE) Project/trained_gre_model.sav", 'rb'))
toefl_model = pickle.load(open("C:/Users/Dev Patel/Documents/Collage_files/Team_4331 (COE) Project/trained_toefl_model.sav", 'rb'))

# Define a function to predict the admission chance for gre
def predict_admission_chance_gre(model, gre_score, cgpa, university_rating, sop, lor, research):
    # Create a NumPy array with the input values
    input_data = np.array([[gre_score, university_rating, sop, lor, cgpa,research]])
    #polynomial model required polynomial features
    poly = PolynomialFeatures(degree=2,include_bias=False)
    X_poly = poly.fit_transform(input_data)
    # Use the model to make a prediction
    prediction = model.predict(X_poly)
    # Return the predicted admission chance (as a float)
    return float(prediction[0])

# Define a function to predict the admission chance
def predict_admission_chance_toefl(model, toefl_score, cgpa, university_rating, sop, lor, research):
    # Create a NumPy array with the input values
    input_data = np.array([[toefl_score, university_rating, sop, lor, cgpa,research]])
    #polynomial model required polynomial features
    poly = PolynomialFeatures(degree=2,include_bias=False)
    X_poly = poly.fit_transform(input_data)
    # Use the model to make a prediction
    prediction = model.predict(X_poly)
    # Return the predicted admission chance (as a float)
    return float(prediction[0])

# Define the input form for GRE score
st.write('# Admission Chance Prediction (GRE Score)')
gre_score = st.slider('GRE Score (out of 340)', min_value=0, max_value=340, value=300, step=1)
cgpa = st.slider('CGPA (out of 10)', min_value=0.0, max_value=10.0, value=8.0, step=0.1)
university_rating = st.slider('University Rating (out of 5)', min_value=1, max_value=5, value=3, step=1)
sop = st.slider('Statement of Purpose (out of 5)', min_value=1, max_value=5, value=3, step=1)
lor = st.slider('Letter of Recommendation (out of 5)', min_value=1, max_value=5, value=3, step=1)
research = st.radio('Research Experience', options=[0, 1], index=0)

# Use the predict_admission_chance function to make a prediction with the GRE model
gre_prediction = predict_admission_chance_gre(gre_model, gre_score, cgpa, university_rating, sop, lor, research)

# Display the predicted admission chance to the user
st.write('## Predicted Admission Chance (GRE Score)')
st.write(f'{gre_prediction:.2%}')

# Define the input form for TOEFL score
st.write('# Admission Chance Prediction (TOEFL Score)')
toefl_score = st.slider('TOEFL Score (out of 120)', min_value=0, max_value=120, value=100, step=1)
cgpa = st.slider('CGPA (out of 10)', min_value=0.0, max_value=10.0, value=8.0, step=0.1,key='CGPA slider')
university_rating = st.slider('University Rating (out of 5)', min_value=1, max_value=5, value=3, step=1,key='university_rating slider')
sop = st.slider('Statement of Purpose (out of 5)', min_value=1, max_value=5, value=3, step=1,key='sop slider')
lor = st.slider('Letter of Recommendation (out of 5)', min_value=1, max_value=5, value=3, step=1,key='lor slider')
research = st.radio('Research Experience', options=[0, 1], index=0,key='research slider')

# Use the predict_admission_chance function to make a prediction with the TOEFL model
toefl_prediction = predict_admission_chance_toefl(toefl_model, toefl_score, cgpa, university_rating, sop, lor, research)

# Display the predicted admission chance to the user
st.write('## Predicted Admission Chance (TOEFL Score)')
st.write(f'{toefl_prediction:.2%}')