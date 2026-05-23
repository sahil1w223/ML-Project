import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.model_selection import GridSearchCV

from src.utils import save_object,evaluate_model

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

# from src.componetns.data_transformation import data_transformations

@dataclass
class modeltransformationconfig:
    train_model_path = os.path.join('artifacts', 'model_trainer.pkl')

class modeltransformation:
    def __init__(self):
        self.model_trainer_config = modeltransformationconfig

    def intial_model_trainer(self,train_array,test_array):
        try:
            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'SVR': SVR(),
                'RandomForestRegressor': RandomForestRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor()
            }
            x_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            x_test = test_array[:, :-1]
            y_test = test_array[:, -1]
            
            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'SVR': SVR(),
                'RandomForestRegressor': RandomForestRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor()
            }

            model_data = evaluate_model(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                             models=models)
            
            best_model_score = max(sorted(model_data.values()))
            best_model_name = [model_name for model_name in model_data.keys() if model_data[model_name] == best_model_score]

            best_model_name = best_model_name[0]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No Best Model Is Found")
            
            save_object(
                file_path=modeltransformationconfig.train_model_path,
                obj = best_model
            )

            y_pred = best_model.predict(x_test)
            return r2_score(y_test,y_pred)
            

        except Exception as e:
            raise CustomException(e,sys)