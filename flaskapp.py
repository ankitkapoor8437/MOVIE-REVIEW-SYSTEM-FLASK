from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
import joblib

loaded_model=joblib.load("./pkl_objects/model.pkl")
loaded_stop=joblib.load("./pkl_objects/stopwords.pkl")
loaded_vec=joblib.load("./pkl_objects/vectorizer.pkl")


app = Flask(__name__)

picFolder = os.path.join('static', 'pics')

app.config['UPLOAD_FOLDER'] = picFolder

def classify(document):
    label = {0: 'negative', 1: 'positive'}
    X = loaded_vec.transform([document])
    y = loaded_model.predict(X)[0]
    proba = np.max(loaded_model.predict_proba(X))
    return label[y], proba



class ReviewForm(Form):
    moviereview = TextAreaField('',[validators.DataRequired(),validators.length(min=0)])

@app.route('/')
def index():
    pic1= os.path.join(app.config['UPLOAD_FOLDER'], 'wandavision.jpeg')
    pic2= os.path.join(app.config['UPLOAD_FOLDER'], 'parasite.jpg')
    pic3= os.path.join(app.config['UPLOAD_FOLDER'], 'darisyam.jpg')
    pic4= os.path.join(app.config['UPLOAD_FOLDER'], 'suits.jpg')
    pic5= os.path.join(app.config['UPLOAD_FOLDER'], 'andha.jpg')
    pic6= os.path.join(app.config['UPLOAD_FOLDER'], 'gangs.jpg')


    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form, movie_image1= pic1, movie_image2= pic2, movie_image3=pic3, movie_image4=pic4, movie_image5=pic5, movie_image6=pic6)

@app.route('/results', methods=['POST'])
def results():
    pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'wandavision.jpeg')
    pic2 = os.path.join(app.config['UPLOAD_FOLDER'], 'parasite.jpg')
    pic3 = os.path.join(app.config['UPLOAD_FOLDER'], 'darisyam.jpg')
    pic4 = os.path.join(app.config['UPLOAD_FOLDER'], 'suits.jpg')
    pic5 = os.path.join(app.config['UPLOAD_FOLDER'], 'andha.jpg')
    pic6 = os.path.join(app.config['UPLOAD_FOLDER'], 'gangs.jpg')


    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['moviereview']
        y, proba = classify(review)
        return render_template('results.html',content=review,prediction=y,probability=round(proba*100, 2), movie_image1= pic1, movie_image2= pic2, movie_image3=pic3, movie_image4=pic4, movie_image5=pic5, movie_image6=pic6)
    return render_template('reviewform.html', form=form )


if __name__ == '__main__':
    app.run(debug=True)

# hello