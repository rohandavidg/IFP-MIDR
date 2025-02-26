import pandas as pd
from data_processing import load_data, merge_transcript_info, preprocess_dbNSFP
from visualization import plot_auc_barplot
from config import DATA_PATHS

def main():
    # Load Data
    data = load_data()
    
    # Merge AF2 disorder dataset with transcript mapping
    AF2_df = merge_transcript_info(data["AF2_df"], data["mane_df"])
    
    # Prepare dbNSFP predictor data
    dbnsfp_predictors = [col for col in data["dbnsfp_gene_df"].columns if col.endswith('rankscore')]
    
    # Merge AF2 disorder with dbNSFP gene dataset
    AF2_disorder_dbnsfp_insilico_ps_df = pd.merge(
        data["AF2_df"], data["dbnsfp_gene_df"], left_on=['GENES', 'mutation'], right_on=['genename', 'mutations'], how='inner'
    )
    
    # Save final processed dataset
    AF2_disorder_dbnsfp_insilico_ps_df.to_csv(DATA_PATHS["output_csv"], index=False)
    print(f"Saved processed dataset to: {DATA_PATHS['output_csv']}")
    
    # Compute AUC metrics (assuming calculate_metrics_and_store is defined in another module)
    result_df = calculate_metrics_and_store(AF2_disorder_dbnsfp_insilico_ps_df, dbnsfp_predictors, 'label', 'dbNSFP Predictor')
    
    # Plot results
    plot_auc_barplot(result_df, DATA_PATHS["auc_plot"])

if __name__ == "__main__":
    main()
