import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.title("Melbourne Housing Price Predictor")

with open('financial_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

suburb = st.selectbox("Suburb", ["Hawthorn", "Camberwell", "Ringwood_North"])
bedrooms = st.slider("Bedroom", 1, 6, 3)
bathrooms = st.slider("Bathroom", 1, 4, 2)
parking = st.slider("Parking Spaces", 0, 5, 2)
property_type = st.selectbox("Property Type", ["House", "Apartment", "Townhouse", "Villa", "Unit"])
has_study = st.checkbox("Has Study")

input_data = pd.DataFrame({
    'property_type': [property_type],
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'parking_spaces': [parking],
    'has_study': [int(has_study)],
    'size': [100],
    'suburb': [suburb],
    'total_rooms': [bedrooms + bathrooms],
    'bed_to_bath_ration': [bedrooms / bathrooms],
    "parking_per_bedroom": [parking / bedrooms]
})

if st.button("Predict Price"):
    input_transformed = preprocessor.transform(input_data)
    prediction = model.predict(input_transformed)[0]
    st.success(f"Estiamted Price: ${prediction:,.0f}")