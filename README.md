# MOVIE-REVIEW-SYSTEM-FLASK

Movie review sentiment analysis web app built with Flask and a saved scikit-learn classifier.

Original deployment URL (availability unverified): <https://movie-review-system-flask.herokuapp.com/>

## Run locally

Python 3.9 or newer is required; the app was tested on Python 3.9.6 and 3.14.8. Create and
activate a virtual environment, then install the dependencies and start the app:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python flaskapp.py
```

Open <http://127.0.0.1:5000/> in a browser. On Windows, activate the environment with
`.venv\Scripts\activate`.

The model and vectorizer are persisted in `pkl_objects/`. They were created with scikit-learn
0.24.1; scikit-learn does not guarantee compatibility across versions, so startup may show
version-mismatch warnings. The app includes a compatibility adjustment for the saved vectorizer,
but predictions should be revalidated against the original training data before relying on them.
Only load model files from trusted sources.

## Deploy to Render

Create a **Web Service** connected to this repository and set:

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn flaskapp:app --bind 0.0.0.0:$PORT`

The included `Procfile` has the same Gunicorn command and defaults to port `10000` when `PORT`
isn't set locally. Render requires a web service to bind to `0.0.0.0` and its assigned port.
Keep `pkl_objects/` in the repository; the app loads those model files at startup.

The saved model was created with scikit-learn 0.24.1, while the current requirements install a
newer release. Startup may show model-version warnings, so revalidate prediction quality before
relying on a deployment.

Pull requests are welcome. For major changes, open an issue first to discuss the proposal, and
update tests as appropriate.

Created and designed by Ankit Kapoor and Ayushi Jain.
