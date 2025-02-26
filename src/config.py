import torch
import os

# Paths
BASE_DIR = "../../"
DATA_DIR = os.path.join(BASE_DIR, "data")
UNIPROT_ID_FILE = os.path.join(BASE_DIR, "uniprot_disorder_genes/idmapping_2024_09_19.tsv")
CLINVAR_FILE = os.path.join(BASE_DIR, "ClinVar/clinvar.snp.header.chr.nopfam.noalt.missense.cava.tsv")
ENSEMBL_REFSEQ_FILE = os.path.join(BASE_DIR, "gene2refseq/gene2ensembl.tsv")
WGS_BED_FILE = os.path.join(BASE_DIR, "gene2refseq/MANE_WGS.tsv")
DISPROT_FILE = os.path.join(BASE_DIR, "disprot/DisProt_release_2023_123_with_ambiguous_evidences.tsv")
AF2_DISORDER_FILE = os.path.join(BASE_DIR, "alphafold/AlphaFold_disorder.human.tsv")

DATA_PATH = '../alphafold/AF2_disorder_all_features.csv'
EMBEDDINGS_JSON = 'AF2_data_with_emb.json'
PROCESSED_JSON = 'AF2_data_with_emb_l1_l2_average.json'
RESULTS_DIR = './results/'
MODELS_DIR = './models/'


# Output Paths
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DISORDER_REGIONS_FILE = os.path.join(OUTPUT_DIR, "AF2_disorder_regions.bed")
CLINVAR_RESULTS_FILE = os.path.join(OUTPUT_DIR, "AF2_Disorder_deleterious_neutral_vus_results.tsv")
STUDY_GENE_LIST_FILE = os.path.join(OUTPUT_DIR, "aim2_gene_list.csv")
STUDY_REGIONS_FILE = os.path.join(OUTPUT_DIR, "aim2_regions_genes.tsv")
PLOT_FILE = os.path.join(OUTPUT_DIR, "disorder_idr_mutation_counts.png")


BASE_DIR = "/projects/wangc/rohan/missense_prediction/disorder/disorder"
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

DATA_PATHS = {
    "dbnsfp": os.path.join(BASE_DIR, "dbNSFP/dbNSFP4.8a_variant.disorder_genes.tsv"),
    "AF2_disorder": os.path.join(BASE_DIR, "alphafold/AF2_disorder_albatross_features.csv"),
    "refseq_enst": os.path.join(BASE_DIR, "gene2refseq/mart_export.ensemble_refseq.mapping.tsv"),
    "ref_ps": os.path.join(BASE_DIR, "alphafold/ref_disorder_phase_separation.tsv"),
    "mut_ps": os.path.join(BASE_DIR, "alphafold/mut_disorder_phase_separation.tsv"),
    "dbnsfp_processed": os.path.join(OUTPUT_DIR, "dbNSFP_disorder_genes.tsv"),
    "output_csv": os.path.join(OUTPUT_DIR, "AF2_disorder_all_features.csv"),
    "auc_plot": os.path.join(OUTPUT_DIR, "AUC_dbsnsfp_disorder.png")
}

# ----------------------------
# CLINICAL SIGNIFICANCE CATEGORIES
# ----------------------------

PATHOGENIC_CLASSES = ['Likely_pathogenic', 'Pathogenic', 'Pathogenic/Likely_pathogenic', 
                      'Pathogenic/Likely_pathogenic|other', 'Likely_pathogenic|other']
BENIGN_CLASSES = ['Likely_benign', 'Benign', 'Benign/Likely_benign']
VUS_CLASSES = ['Uncertain_significance']



# Model Parameters
N_SPLITS = 10
RANDOM_STATE = 42
USE_CUDA = torch.cuda.is_available()

# Feature Selection
FEATURES_NOCON = [
    'abs_delta_asphericity', 'abs_delta_radius_of_gyration',
    'abs_delta_end_to_end_distance', 'abs_delta_scaling_exponent',
    'abs_delta_prefactor', 'abs_delta_IEP', 'abs_delta_molecular_weight',
    'abs_delta_gravy', 'abs_delta_Pos_charge', 'abs_delta_Neg_charge'
]

# Model Hyperparameters
XGB_PARAMS = {
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'random_state': RANDOM_STATE,
    'tree_method': "gpu_hist"
}
