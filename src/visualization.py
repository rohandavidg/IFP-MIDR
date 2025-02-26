import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config import RESULTS_DIR
import numpy as np
from config import DATA_PATHS

def plot_auc_barplot(df, filename):
    """Plot AUC scores for different predictors."""
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=df, x='name', y='AUC', hue='Clinvar', dodge=False, errorbar=None,
        palette={"No": "red", "Yes": "grey"}
    )

    x_coords = np.arange(len(df))
    y_values = df['AUC'].values
    y_err_lower = (df['AUC'] - df['AUC_CI_low']).values
    y_err_upper = (df['AUC_CI_high'] - df['AUC']).values

    plt.errorbar(x=x_coords, y=y_values, yerr=[y_err_lower, y_err_upper], fmt='none', 
                 ecolor='black', capsize=3, elinewidth=1.2, capthick=1.2)

    plt.xticks(rotation=90, ha="right")
    plt.xlabel("Predictor Name")
    plt.ylabel("AUC")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="ClinVar")
    sns.despine()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

def plot_feature_importance(shap_values, feature_names, output_file):
    """Plots SHAP feature importance."""
    shap_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(shap_values.values).mean(axis=0)
    }).sort_values(by='Importance', ascending=False)

    # Normalize importance values to get probabilities
    shap_importance['Probability'] = shap_importance['Importance'] / shap_importance['Importance'].sum()
    shap_importance = shap_importance.head(10)

    # Define colors for different feature groups
    group_colors = {
        'red': ['abs_delta_gravy', 'abs_delta_IEP'],
        'blue': ['delta_emb_hadamard_value_217', 'delta_emb_hadamard_value_536'],
        'green': ['abs_delta_end_to_end_distance']
    }
    color_map = {feature: color for color, features in group_colors.items() for feature in features}

    plt.figure(figsize=(10, 6))
    plt.bar(shap_importance['Feature'], shap_importance['Probability'],
            color=[color_map.get(f, 'black') for f in shap_importance['Feature']])
    
    plt.xlabel('Features', fontsize=14, weight='bold')
    plt.ylabel('Probability', fontsize=14, weight='bold')
    plt.xticks(rotation=90, fontsize=12)
    plt.tight_layout()

    # Add legend
    from matplotlib.patches import Patch
    legend_labels = [Patch(color='red', label='Phase Separation'),
                     Patch(color='blue', label='ProtTransTM'),
                     Patch(color='green', label='gIDRc')]
    plt.legend(handles=legend_labels, loc='upper right')
    
    plt.savefig(f"{RESULTS_DIR}/{output_file}", dpi=300)
    plt.show()

def plot_auc_results(results_df, output_file):
    """Creates box plots for accuracy and AUC scores."""
    diverse_palette = sns.color_palette("tab10")
    plt.figure(figsize=(14, 8))

    # Accuracy box plot
    plt.subplot(1, 2, 1)
    sns.boxplot(data=results_df, x='method', y='accuracy', hue='model_name', palette=diverse_palette)
    plt.xticks(rotation=90)
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)

    # AUC box plot
    plt.subplot(1, 2, 2)
    sns.boxplot(data=results_df, x='method', y='roc_auc', hue='model_name', palette=diverse_palette)
    plt.xticks(rotation=90)
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('AUC', fontsize=14)

    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{output_file}", dpi=300)
    plt.show()
