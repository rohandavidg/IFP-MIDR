import pandas as pd
from config import DATA_PATH

def load_data_model():
    """Load dataset and preprocess columns."""
    data_df = pd.read_csv(DATA_PATH)
    data_df['seq_mut'] = data_df['mut_seq']
    data_df = data_df.loc[:, ~data_df.columns.str.startswith(('ref', 'mut'))]
    return data_df


def load_data_dbnsfp():
    """Load all required datasets into Pandas DataFrames."""
    return {
        "AF2_df": pd.read_csv(DATA_PATHS["AF2_disorder"]),
        "mane_df": pd.read_csv(DATA_PATHS["refseq_enst"], sep='\t')[['Transcript_stable_ID_version', 'MANE_Select']],
        "dbnsfp_df": pd.read_csv(DATA_PATHS["dbnsfp"], sep='\t'),
        "dbnsfp_gene_df": pd.read_csv(DATA_PATHS["dbnsfp_processed"], sep='\t')
    }

# ----------------------------
# PREPROCESSING FUNCTIONS
# ----------------------------

def merge_transcript_info(AF2_df, mane_df):
    """Merge MANE transcript mapping with AF2 disorder dataset."""
    return pd.merge(AF2_df, mane_df, left_on='CAVA_TRANSCRIPT', right_on='MANE_Select', how='left')

def preprocess_dbNSFP(df, gene, transcript):
    """Filter and preprocess dbNSFP dataset for a given gene and transcript."""
    required_columns = ['aaref', 'aaalt', 'aapos', 'genename', 'Ensembl_transcriptid',
                        'gnomAD_exomes_AC', 'gnomAD_exomes_AN', 'gnomAD_exomes_AF',
                        'clinvar_clnsig']
    
    rankscore_columns = [col for col in df.columns if col.endswith('rankscore')]
    required_columns += rankscore_columns

    df_filtered = df[required_columns].copy()
    
    # Explode semicolon-separated columns
    for col in ['aapos', 'Ensembl_transcriptid', 'genename']:
        df_filtered[col] = df_filtered[col].str.split(';')
        df_filtered = df_filtered.explode(col)
    
    df_filtered = df_filtered[(df_filtered['genename'] == gene) & 
                              (df_filtered['Ensembl_transcriptid'] == transcript)]
    
    df_filtered = df_filtered.dropna(subset=['aapos'])
    df_filtered['aapos'] = df_filtered['aapos'].astype(int)
    df_filtered['mutations'] = df_filtered['aaref'] + df_filtered['aapos'].astype(str) + df_filtered['aaalt']
    
    df_filtered = df_filtered.replace(".", np.nan)
    df_filtered[rankscore_columns] = df_filtered[rankscore_columns].fillna(0)

    return df_filtered.drop_duplicates().reset_index(drop=True)


    
