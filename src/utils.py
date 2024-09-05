'''
utils will have all the common things we will import or use in our project

'''
import os 
import sys

import numpy as np
import pandas as pd 

from src.exception import CustomException
import dill 

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        best_score = -float('inf')
        best_model_name = None
        best_model_params = None

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            param = params[model_name]

            gs = GridSearchCV(model, param, cv=3, scoring='r2')
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = {
                "test_r2_score": test_model_score,
                "train_r2_score": train_model_score,
                "best_params": gs.best_params_
            }

            if test_model_score > best_score:
                best_score = test_model_score
                best_model_name = model_name
                best_model_params = gs.best_params_

        return {
            "detailed_report": report,
            "best_model": {
                "name": best_model_name,
                "r2_score": best_score,
                "parameters": best_model_params
            }
        }
    
    except Exception as e:
        raise CustomException(e, sys)
    


