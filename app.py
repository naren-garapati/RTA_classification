import streamlit as st
import pandas as pd
from joblib import load
import json

# Load JSON files and model
with open("nominal_feature_options.json", "r") as f:
    nominal_feature_options = json.load(f)

with open("ordinal_feature_options.json", "r") as f:
    ordinal_feature_options = json.load(f)

model = load('rf_model.joblib')
label_encoders = load('label_encoders.joblib')
ordinal_encoders = load('ordinal_encoders.joblib')

# Collect user input
user_input = {}

# Streamlit widgets for nominal features
st.sidebar.header("Nominal Features")
for feature, options in nominal_feature_options.items():
    user_input[feature] = st.sidebar.selectbox(feature, options=options)

# Streamlit widgets for ordinal features
st.sidebar.header("Ordinal Features")
for feature, options in ordinal_feature_options.items():
    user_input[feature] = st.sidebar.selectbox(feature, options=options)

# Streamlit widgets for 'hour' and 'minute'
user_input['hour'] = st.sidebar.slider("Hour", 0, 23, 12)
user_input['minute'] = st.sidebar.slider("Minute", 0, 59, 30)

# Convert user input data into a DataFrame
user_input_df = pd.DataFrame([user_input])

# Transform the input data
for feature in nominal_feature_options.keys():
    if feature in label_encoders:  # Skip features like 'hour', 'minute' which are not encoded
        user_input_df[feature] = label_encoders[feature].transform(user_input_df[feature])

for feature in ordinal_feature_options.keys():
    if feature in ordinal_encoders:  # Skip features like 'hour', 'minute' which are not encoded
        user_input_df[feature] = ordinal_encoders[feature].transform(user_input_df[[feature]])

# Make prediction
if st.button('Predict Accident Severity'):
    prediction = model.predict(user_input_df)
    st.write(f'The predicted accident severity is: {prediction[0]}')

