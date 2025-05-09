import pandas as pd
import joblib
import streamlit as st

model = joblib.load("housing_model.pkl")

# UI related code
st.title("Housing Price Prediction: Batch 07")

st.sidebar.header('User Input Features')


def get_input_features():
    longitude = st.sidebar.slider("Longitude", -124.0, -113.0, -118.0)
    latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 37.0)
    housing_median_age = st.sidebar.slider("Housing Median Age", 1, 52, 20)
    total_rooms = st.sidebar.slider("Total Rooms", 1, 10000, 5000)
    total_bedrooms = st.sidebar.slider("Total Bedrooms", 1, 10000, 1000)
    population = st.sidebar.slider("Population", 1, 10000, 500)
    households = st.sidebar.slider("Households", 1, 10000, 200)
    median_income = st.sidebar.slider("Median Income", 0.0, 15.0, 3.5)
    ocean_proximity = st.sidebar.selectbox(
        "Ocean Proximity",
        ("NEAR BAY", "NEAR OCEAN", "ISLAND", "INLAND", "<1H OCEAN")
    )

    return {
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'ocean_proximity': ocean_proximity
    }

data = get_input_features()

input_df = pd.DataFrame(data, index=[0])
st.write("Input Data:")
st.write(input_df)

prediction = model.predict(input_df)

st.subheader("Prediction:")
st.header(f'${prediction[0]:,.2f}')