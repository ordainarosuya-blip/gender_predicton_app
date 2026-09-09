import streamlit as st
import joblib
import numpy as np

# Load your trained model
model = joblib.load('gender_model.pkl')
label_encoder = joblib.load('gender_label_encoder.pkl')

st.set_page_config(page_title="Gender Prediction AI")
st.title("🎤 Gender Prediction from Voice - 96% Accuracy")
st.write("Fill the voice features below and click Predict")

# Change these to match the features you used to train
col1, col2 = st.columns(2)
with col1:
    meanfreq = st.number_input("Mean Frequency", value=0.0)
    sd = st.number_input("Standard Deviation", value=0.0)
    median = st.number_input("Median Frequency", value=0.0)
with col2:
    Q25 = st.number_input("Q25 Frequency", value=0.0)
    Q75 = st.number_input("Q75 Frequency", value=0.0)
    IQR = st.number_input("IQR", value=0.0)
    mode = st.number_input("Mode Frequency", value=0.0)
    
    
if st.button("🚀 Predict Gender"):
    features = np.array([[meanfreq, sd, median, Q25, Q75, IQR, mode]])
    prediction = model.predict(features)
    gender = label_encoder.inverse_transform(prediction)
    st.success(f"### Predicted Gender: **{gender[0]}**")