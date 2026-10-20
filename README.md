# AI-Based Customer Spending Prediction

A beginner-friendly machine-learning project that classifies customers into **Low**, **Medium**, or **High** spending categories using demographic and purchasing-related features.

## Project contents
- `data/mall_customers.csv` — reproducible synthetic customer dataset (600 rows).
- `notebooks/customer_spending_prediction.ipynb` — complete EDA, preprocessing, model training, evaluation, visualizations, feature importance, and sample prediction.
- `src/train_model.py` — runnable Python pipeline that trains and evaluates Logistic Regression and Random Forest.
- `reports/project_report.pdf` — project report with methodology, results context, limitations, and usage.
- `outputs/` — evaluation outputs created by the training script.
- `requirements.txt` — required Python packages.

## Run locally
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python src/train_model.py
jupyter notebook notebooks/customer_spending_prediction.ipynb
```

## Target and features
The target is `AnnualSpendingCategory` (Low/Medium/High). The dataset is synthetic and designed for learning; it is not a real customer database. `CustomerID` is excluded from model features. Numeric missing values are median-imputed and categorical values are most-frequent-imputed; categorical predictors are one-hot encoded and numeric predictors are standardized for Logistic Regression.

## Models and evaluation
- Logistic Regression (multinomial classification)
- Random Forest Classifier
- Stratified train/test split (80/20)
- Accuracy, precision, recall, F1-score, confusion matrix, and feature importance

Results may vary slightly with library versions. Do not interpret the scores as real-world business performance because the data is synthetic and the labels are derived from a spending proxy.

## Responsible use
Customer predictions can affect marketing and service decisions. Validate on representative, consented real data before any operational use; review fairness across groups and avoid using sensitive attributes for consequential decisions.
