import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('models/random_forest_model.pkl')

st.title('Housing Price Prediction App')

st.sidebar.header('User Input Features')

# Create a function to take input from the user
def user_input_features():
    longitude = st.sidebar.slider('longitude', -180.0, 180.0, -122.23)
    latitude = st.sidebar.slider('latitude', -90.0, 90.0, 37.88)
    housing_median_age = st.sidebar.slider('housing_median_age', 0, 100, 41)
    total_rooms = st.sidebar.slider('total_rooms', 0, 10000, 880)
    total_bedrooms = st.sidebar.slider('total_bedrooms', 0, 5000, 129)
    population = st.sidebar.slider('population', 0, 50000, 322)
    households = st.sidebar.slider('households', 0, 5000, 126)
    median_income = st.sidebar.slider('median_income', 0.0, 15.0, 8.3252)
    ocean_proximity = st.sidebar.selectbox('ocean_proximity', ('NEAR BAY', 'INLAND', '<1H OCEAN', 'NEAR OCEAN', 'ISLAND'))

    data = {
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
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input Features')
st.write(input_df)

# Make predictions
prediction = model.predict(input_df)

st.subheader('Predicted Price')
st.header(f'${prediction[0]:,.2f}')