from pathlib import Path

import joblib
from flask import Flask, render_template, request, url_for
from wtforms import Form, TextAreaField, validators

app = Flask(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "pkl_objects"
MOVIE_IMAGE_NAMES = (
    "wandavision.jpeg",
    "ludo.jpg",
    "karwan.jpg",
    "suits.jpg",
    "andha.jpg",
    "parasite.jpg",
    "falcon.jpg",
    "widow.jpg",
    "gangs.jpg",
)

loaded_model = joblib.load(MODEL_DIR / "model.pkl")
loaded_vec = joblib.load(MODEL_DIR / "vectorizer.pkl")

transformer = getattr(loaded_vec, "_tfidf", None)
if transformer is not None and not hasattr(transformer, "n_features_in_"):
    # Older saved vectorizers predate this fitted-state attribute.
    transformer.n_features_in_ = len(loaded_vec.vocabulary_)


def classify(document):
    labels = {0: "negative", 1: "positive"}
    features = loaded_vec.transform([document])
    prediction = int(loaded_model.predict(features)[0])
    probability = float(max(loaded_model.predict_proba(features)[0]))
    return labels[prediction], probability


def movie_image_context():
    return {
        f"movie_image{index}": url_for("static", filename=f"pics/{filename}")
        for index, filename in enumerate(MOVIE_IMAGE_NAMES, start=1)
    }


class ReviewForm(Form):
    moviereview = TextAreaField("", [validators.DataRequired()])


@app.route("/")
def index():
    form = ReviewForm(request.form)
    return render_template(
        "reviewform.html", form=form, **movie_image_context()
    )


@app.route("/results", methods=["POST"])
def results():
    form = ReviewForm(request.form)
    if form.validate():
        review = form.moviereview.data
        prediction, probability = classify(review)
        return render_template(
            "results.html",
            content=review,
            prediction=prediction,
            probability=round(probability * 100, 2),
            **movie_image_context(),
        )
    return render_template(
        "reviewform.html", form=form, **movie_image_context()
    )


if __name__ == "__main__":
    app.run()
