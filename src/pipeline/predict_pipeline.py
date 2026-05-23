import sys
import os
import pandas as pd


from src.exception import CustomException
from src.utils import load_obj

class predictionpipeline:
    def __init__(self):
        pass

    def predict(self,feature):
        try:
            model_path = 'artifacts/model_trainer.pkl'
            preproceser = 'artifacts/data_transformation.pkl'

            load_model = load_obj(model_path)
            load_proceser = load_obj(preproceser)

            data_scaler = load_proceser.transform(feature)
            score = load_model.predict(data_scaler)
            return score

        except Exception as e:
            raise CustomException(e,sys)




class customdata:
    def __init__(self,gender,eace_ethnicity,parental_level_of_education,lunch_type,test_preparation_course,writing_score,reading_score):
        self.gender = gender
        self.eace_ethnicity = eace_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch_type = lunch_type
        self.test_preparation_course = test_preparation_course
        self.writing_score = writing_score
        self.reading_score = reading_score

    def conver_frame(self):
        try:
            data = {
                'gender': [self.gender],
                'race_ethnicity': [self.eace_ethnicity],  
                'parental_level_of_education': [self.parental_level_of_education],
                'lunch': [self.lunch_type],          
                'test_preparation_course': [self.test_preparation_course],
                'writing_score': [self.writing_score],
                'reading_score': [self.reading_score]
            }

            return pd.DataFrame(data)
        except Exception as e:
            CustomException(e,sys)

