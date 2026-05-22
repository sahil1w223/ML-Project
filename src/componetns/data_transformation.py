import sys
import os
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.utils import save_object


class datatransformationconfig:
    data_transformatiion_path = os.path.join('artifacts', 'data_transformation.pkl')

@dataclass
class datatransformation:
    def __init__(self):
        path = datatransformationconfig()

    try:
        def transformation(self):
            num_fetures = ['reading_score', 'writing_score']
            cat_fetures = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            logging.info("numerical clumns and categorical columns are defined")

            num_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())]
            )

            logging.info("numerical pipeline is created")
            logging.info("categorical pipeline is created")

            cat_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('encoding', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ]
            )
            logging.info("numerical and categorical pipeline is created")

            logging.info("numerical and categorical columns are defined")
            preproceser = ColumnTransformer([
                ('num_pipeline', num_pipeline, num_fetures),
                ('cat_pipeline', cat_pipeline, cat_fetures)
            ])
            logging.info("preprocessor is created")


            return preproceser
    except Exception as e:
        raise CustomException(e,sys)
    
    def data_transformation(self,train_data_path, test_data_path):
        try:

            logging.info("reading the train and test data")

            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)

            logging.info("train and test data is read")


            model_preprocessor = self.transformation()
            target_columns = 'math_score'

            input_feature_train_df = train_data.drop(target_columns, axis = 1)
            target_feature_train_df = train_data[target_columns]

            input_feature_test_df = test_data.drop(target_columns, axis = 1)
            target_feature_test_df = test_data[target_columns]

            input_features_train_df = model_preprocessor.fit_transform(input_feature_train_df)
            input_features_test_df = model_preprocessor.transform(input_feature_test_df)

            train_arr = np.column_stack((input_features_train_df, np.array(target_feature_train_df)))
            test_arr = np.column_stack((input_features_test_df, np.array(target_feature_test_df)))

            logging.info('transformation is completed')

            save_object(
                file_path = datatransformationconfig.data_transformatiion_path,
                obj = model_preprocessor
            )

            return(
                train_arr,
                test_arr,
                model_preprocessor
            )
        
        except Exception as e:
            raise CustomException(e,sys)

