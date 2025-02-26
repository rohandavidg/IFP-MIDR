import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from config import N_SPLITS, RANDOM_STATE, MODELS_DIR

def train_cv(model, X, y, method, model_name):
    """Train model using cross-validation."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    metrics = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics.append({
            'fold': fold, 'accuracy': (y_test == y_pred).mean(),
            'roc_auc': roc_auc_score(y_test, y_pred_proba), 
            'method': method, 'model_name': model_name
        })

    return pd.DataFrame(metrics)

def train_models(X, y):
    """Trains multiple models and saves them."""
    models = {
        "XGB": XGBClassifier(**XGB_PARAMS),
        "RF": RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
        "NB": GaussianNB(),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=RANDOM_STATE)
    }

    results = []
    for name, model in models.items():
        result = train_cv(model, X, y, method="Hadamard", model_name=name)
        results.append(result)
        joblib.dump(model, f"{MODELS_DIR}/{name}_model.pkl")

    return pd.concat(results, ignore_index=True)

if __name__ == "__main__":
    df = pd.read_json(PROCESSED_JSON, lines=True)
    X = df.drop(columns=['Class'])
    y = df['Class'].apply(lambda x: 1 if x == "Deleterious" else 0)
    
    results = train_models(X, y)
    results.to_csv(f"{MODELS_DIR}/model_results.csv", index=False)
    print("Model training completed.")
