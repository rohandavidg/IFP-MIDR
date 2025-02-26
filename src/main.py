import pandas as pd
import joblib
import shap
from config import PROCESSED_JSON, MODELS_DIR, RESULTS_DIR
from data_preprocessing import load_data
from prottrans_features import compute_embedding_features
from model_training import train_models
from HPO import optimize_xgboost
from visualization import plot_auc_results, plot_feature_importance
from utils import load_dataframe

# Load dataset
print("Loading dataset...")
df = load_data()

# Compute embeddings and save processed features
print("Computing embeddings...")
compute_embedding_features(df)

# Load processed data
print("Loading processed feature data...")
processed_df = pd.read_json(PROCESSED_JSON, lines=True)
X = processed_df.drop(columns=['Class'])
y = processed_df['Class'].apply(lambda x: 1 if x == "Deleterious" else 0)

# Train models
print("Training models...")
results = train_models(X, y)
results.to_csv(f"{MODELS_DIR}/model_results.csv", index=False)

# Optimize XGBoost
print("Optimizing XGBoost hyperparameters...")
optimize_xgboost(X, y)

# Load trained model for SHAP analysis
print("Loading best trained model...")
model_path = f"{MODELS_DIR}/XGB_model.pkl"
model_xg = joblib.load(model_path)

# Compute SHAP values
explainer = shap.Explainer(model_xg, X)
shap_values = explainer(X)

# Save feature importance plot
plot_feature_importance(shap_values, X.columns, 'feature_importance_model.png')

# Load model evaluation results
results_df = load_dataframe(f"{MODELS_DIR}/model_results.csv")

# Generate AUC plots
plot_auc_results(results_df, 'final_results_emb_box.png')

print("Pipeline execution complete. All results saved.")
