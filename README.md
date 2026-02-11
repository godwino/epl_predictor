# EPL Win/Draw/Loss Predictor (Local)

## Live App
- Streamlit: https://share.streamlit.io/user/godwino

## Setup
1) Create a virtual env (optional)
2) Install deps:

```
pip install -r requirements.txt
```

## Run baseline training
```
python train.py
```

## Advanced train + save model
```
python -B advanced_train.py
```
This writes `advanced_model.pkl` for inference.

## Run Streamlit app locally
```
streamlit run app.py
```

## Predict one fixture
```
python predict_match.py --home Arsenal --away Chelsea
```

## Predict many fixtures (CSV)
Create a CSV with columns `HomeTeam,AwayTeam`, then run:
```
python predict_match.py --fixtures-csv fixtures.csv
```
Optional output file:
```
python predict_match.py --fixtures-csv fixtures.csv --output-csv predictions.csv
```

## Data
Place season CSVs in this folder (e.g., `season-2324.csv`).
The scripts use the last 10 full seasons and hold out `2324` for test.

## Model
- Baseline: Logistic Regression
- Improved: XGBoost with regularization + early stopping

Expected accuracy for W/D/L is typically in the 50-65% range; higher claims usually mean leakage.

## Deploy to GitHub + Streamlit Cloud
1) Initialize git and commit:
```
git init
git add .
git commit -m "Initial EPL predictor app"
```

2) Create a GitHub repo, then connect and push:
```
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

3) Deploy on Streamlit Cloud:
- Go to https://share.streamlit.io/
- Click "New app"
- Select your GitHub repo
- Branch: `main`
- Main file path: `app.py`
- Click "Deploy"
