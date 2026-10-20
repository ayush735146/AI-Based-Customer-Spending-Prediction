"""Train and evaluate customer spending category classifiers."""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "mall_customers.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
df = pd.read_csv(DATA)
target = "AnnualSpendingCategory"
X = df.drop(columns=[target, "CustomerID"])
y = df[target]
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42,
                                            class_weight="balanced", min_samples_leaf=2)
}
results = {}
fitted = {}
for name, model in models.items():
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results[name] = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0)
    }
    fitted[name] = (pipe, pred)
    print(f"\\n{name}: accuracy={results[name]['accuracy']:.3f}")
    print(classification_report(y_test, pred, zero_division=0))
    disp = ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap=None, xticks_rotation=25)
    disp.ax_.set_title(f"{name} — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUT / (name.lower().replace(" ", "_") + "_confusion_matrix.png"), dpi=160)
    plt.close()
with open(OUT / "model_metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
best_name = max(results, key=lambda name: results[name]["accuracy"])
best_pipe, best_pred = fitted[best_name]
print(f"\\nBest model by test accuracy: {best_name}")
print("Sample prediction:", best_pipe.predict(X_test.iloc[[0]])[0])
# Random forest feature importance after preprocessing
rf_pipe = fitted["Random Forest"][0]
feature_names = rf_pipe.named_steps["preprocess"].get_feature_names_out()
importance = rf_pipe.named_steps["model"].feature_importances_
imp = pd.DataFrame({"Feature": feature_names, "Importance": importance}).sort_values("Importance", ascending=False)
imp.to_csv(OUT / "feature_importance.csv", index=False)
plt.figure(figsize=(9, 6))
plot_data = imp.head(12).sort_values("Importance")
plt.barh(plot_data["Feature"], plot_data["Importance"])
plt.title("Top 12 Random Forest Feature Importances")
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=160)
plt.close()
print(f"Outputs saved to: {OUT}")
