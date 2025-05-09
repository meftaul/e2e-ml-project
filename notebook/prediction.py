import pandas as pd
import joblib

model = joblib.load("housing_model.pkl")

example_data = pd.DataFrame({
    'longitude': [-122.23],
    'latitude': [37.88],
    'housing_median_age': [41],
    'total_rooms': [880],
    'total_bedrooms': [129],
    'population': [322],
    'households': [126],
    'median_income': [8.3252],
    'ocean_proximity': ['NEAR BAY']
})
print("Example data:")
print(example_data)

predictions = model.predict(example_data)

print("Predictions:")
print(predictions)