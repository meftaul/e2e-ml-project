import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

data_url = '/home/meftaul/Documents/abp/e2e-ml-project/dataset/housing.csv'

housing = pd.read_csv(data_url)

housing['income_cat'] = pd.cut(housing['median_income'],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

X = housing.drop('median_house_value', axis=1)
y = housing['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=housing['income_cat'])

num_attributes = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
cat_attributes = ['ocean_proximity']

num_pipeline = Pipeline([
    ('num_imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('cat_imputer', SimpleImputer(strategy='most_frequent')),
    ('cat_encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessing_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attributes),
    ('cat', cat_pipeline, cat_attributes)
])

# linear_regression_pipeline = Pipeline([
#     ('preprocessing', preprocessing_pipeline),
#     ('linear_regression', LinearRegression())
# ])

random_forest_pipe = Pipeline([
    ('preprocessing', preprocessing_pipeline),
    ('random_forest', RandomForestRegressor())
])

# tree_pipe = Pipeline([
#     ('preprocessing', preprocessing_pipeline),
#     ('linear_regression', LinearRegression())
# ])

model = random_forest_pipe.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)


# Save the model
import joblib

print("Saving the model...")
joblib.dump(model, 'housing_model.pkl')
print("Model saved successfully.")