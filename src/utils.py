import os
import sys
from src.logger import logging
from src.exception import CustomException
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train,X_test,y_test,models):
    report = {}
    for name,model in models.items():
        model.fit(X_train,y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        train_r2_score = r2_score(y_train,y_pred_train)
        test_r2_score = r2_score(y_test,y_pred_test)
        report[name] = test_r2_score

    return report
        