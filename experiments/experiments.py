import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor

import mlflow


def evaluate_model(model, data, labels):
    predictions = model.predict(data)
    r2 = r2_score(labels, predictions)
    rmse = root_mean_squared_error(labels, predictions)
    # rerun as dict
    return {'r2': r2, 'rmse': rmse}


housing = pd.read_csv('../datasets/housing.csv')

housing['income_cat'] = pd.cut(housing['median_income'],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

X = housing.drop('median_house_value', axis=1)
y = housing['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    stratify=housing['income_cat'],
                                                    random_state=42)

num_attributes = ['longitude', 'latitude', 'housing_median_age', 
                  'total_rooms', 'total_bedrooms', 'population', 
                  'households', 'median_income']
cat_attributes = ['ocean_proximity']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('cat_imputer', SimpleImputer(strategy='most_frequent')),
    ('one_hot', OneHotEncoder())
])

preprocessing_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attributes),
    ('cat', cat_pipeline, cat_attributes)
])

mlflow.set_tracking_uri('http://127.0.0.1:8080')
mlflow.set_experiment(experiment_name='housing_experiment_linear_regression_with_autolog')

with mlflow.start_run():
    
    alpha = 0.1
    l1_ratio = 0.5

    model_pipeline = Pipeline([
        ('preprocessing', preprocessing_pipeline),
        ('linear_regression', LinearRegression())
        # ('decision_tree', DecisionTreeRegressor())
        # ('random_forest', RandomForestRegressor(n_estimators=100, random_state=42))
        # ('elastic_net', ElasticNet(alpha=alpha, l1_ratio=l1_ratio))
    ])

    final_model = model_pipeline.fit(X_train, y_train)

    result = evaluate_model(final_model, X_test, y_test)

    # mlflow.log_param('alpha', alpha)
    # mlflow.log_param('l1_ratio', l1_ratio)
    mlflow.log_metric('r2', result['r2'])
    mlflow.log_metric('rmse', result['rmse'])
    mlflow.sklearn.log_model(final_model, 'model_linear_regression')

    print(f"R2: {result['r2']}")
    print(f"RMSE: {result['rmse']}")