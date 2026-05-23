from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import customdata,predictionpipeline

from src.componetns.model_train import modeltransformation

application = Flask("__main__")
app = application

@app.route("/") 
def Home():
    return render_template("home.html")

@app.route("/prediction", methods=['GET','POST'])
def predict_datapoint():
    if request.method=="GET":
        return render_template("index.html")
    else:
        gender = request.form.get('gender')
        eace_ethnicity = request.form.get('ethnicity')
        parental_level_of_education = request.form.get('parental_level_of_education')
        lunch_type = request.form.get('lunch')
        test_preparation_course = request.form.get('test_preparation_course')
        writing_score = float(request.form.get('writing_score'))
        reading_score = float(request.form.get('reading_score'))

        data = customdata(
                    gender=gender,
                    eace_ethnicity=eace_ethnicity,
                    parental_level_of_education=parental_level_of_education,
                    lunch_type=lunch_type,
                    test_preparation_course=test_preparation_course,
                    writing_score=writing_score,
                    reading_score=reading_score,
        )        
    
        data_db = data.conver_frame()

        predict = predictionpipeline()
        results = predict.predict(data_db)
        return render_template("index.html", results = results[0])


if __name__ == "__main__":
    app.run(host='0.0.0.0')