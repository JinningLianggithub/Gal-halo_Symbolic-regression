import os
default_n_threads = 8
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

######################## set up the environment #########################


import numpy as np
import os

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import shap
import joblib


"""Hyperparameter grid for Random Forest Regressor"""
param_grid = {
    'n_estimators': [200, 300,400,500,600,800,1000,1200],
    'max_depth': [7, 10, 13, 16, 19,22,25],
    'min_samples_leaf': [10,20, 50, 70, 95, 110],
    'max_features': ['sqrt', 'log2', None]
    }


frac_test = 0.2  # fraction of data used as test sample



"""Function to run Random Forest Regressor with optional hyperparameter search, and SHAP analysis"""
def run_RF(X,y,frac_test=frac_test,param_grid=param_grid,search=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=frac_test, random_state=0)

    if search==True:
        regressor = RandomForestRegressor(random_state=11)

        grid_search = GridSearchCV(
        estimator=regressor,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='neg_mean_squared_error',
        n_jobs=150,  # Use all available cores
        verbose=1,)

        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        print(f"Best Parameters: {best_params}")

        best_rf_model = grid_search.best_estimator_
    else:
        regressor = RandomForestRegressor(random_state=11,n_estimators=param_grid["n_estimators"][0],
                                               max_depth=param_grid["max_depth"][0],
                                               min_samples_leaf=param_grid["min_samples_leaf"][0],
                                               max_features=param_grid["max_features"][0],
                                               n_jobs=150)
        best_rf_model = regressor.fit(X_train, y_train)

    print("RF model is done")
    #explainer = shap.Explainer(best_rf_model,  X)
    explainer = shap.TreeExplainer(best_rf_model, X)
    shap_values = explainer(X) 

    return X_train, X_test, y_train, y_test,best_rf_model,shap_values


"""define your X, halo property, and y, galaxy property here"""
X_train, X_test, y_train, y_test,best_rf_model,shap_values=run_RF(X,y)
joblib.dump(best_rf_model, './RFtrees/XXX.pkl')
np.savez('./SHAPexplainers/XXX', shap_values.values)