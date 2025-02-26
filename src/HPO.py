import pandas as pd
import optuna
from xgboost import XGBClassifier
from config import RESULTS_DIR

def objective(trial, X, y):
    """Hyperparameter optimization for XGBoost."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'tree_method': "gpu_hist"
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    return scores.mean()

def optimize_xgboost(X, y):
    """Runs Optuna hyperparameter optimization."""
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50)
    
    best_params = study.best_params
    pd.DataFrame([best_params]).to_csv(f"{RESULTS_DIR}/best_xgboost_params.csv", index=False)

if __name__ == "__main__":
    df = pd.read_json(PROCESSED_JSON, lines=True)
    X = df.drop(columns=['Class'])
    y = df['Class'].apply(lambda x: 1 if x == "Deleterious" else 0)
    
    optimize_xgboost(X, y)
    print("Hyperparameter tuning completed.")
