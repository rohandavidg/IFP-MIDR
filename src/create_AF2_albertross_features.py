import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis, ProtParamData
from pysam import FastaFile
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from scipy.stats import mannwhitneyu, t, sem
import metapredict as meta
from sparrow import Protein
from quantiprot.utils.io import load_fasta_file  # if needed
from sklearn.utils import resample
from sklearn.metrics import average_precision_score

# ----------------------------
# DATA LOADING & SEQUENCE PROCESSING
# ----------------------------

def load_mutation_data(filepath):
    df = pd.read_csv(filepath, sep='\t')
    df = df[df['Class'] != 'VUS'].copy()
    df['mutation'] = df['CAVA_PROTREF'] + df['CAVA_PROTPOS'].astype(str) + df['CAVA_PROTALT']
    return df

def load_fasta_region(fasta_path, start, end):
    with open(fasta_path) as handle:
        record = next(SeqIO.parse(handle, 'fasta'))
        # Adjust start index from 1-indexed to 0-indexed
        return str(record.seq)[start - 1:end]

# ----------------------------
# SEQUENCE MUTATION CLASS
# ----------------------------

class SequenceMutator:
    
    def __init__(self, seq, ref, alt, seq_start, mutation_pos):
        """
        :param seq: Wildtype sequence (string)
        :param ref: Reference residue (string)
        :param alt: Alternate residue (string)
        :param seq_start: Start position of the sequence (1-indexed)
        :param mutation_pos: Mutation position (1-indexed)
        """
        self.seq = seq
        self.ref = ref
        self.alt = alt
        self.seq_start = int(seq_start)
        self.mutation_index = int(mutation_pos) - 1  # Convert to 0-index
        self.ref_size = len(ref)

    def verify_ref(self):
        """Verify that the sequence at the mutation position matches the reference."""
        ref_index = self.mutation_index - (self.seq_start - 1)
        if self.seq[ref_index] == self.ref:
            return True
        else:
            print(f"Warning: Sequence base {self.seq[ref_index]} does not match reference {self.ref} at position {self.mutation_index}")
            return False

    def generate_mutant(self):
        """Generate the mutant sequence by replacing the reference residue with the alternate."""
        pos = self.mutation_index - (self.seq_start - 1)
        mutant = self.seq[:pos] + self.alt + self.seq[pos+1:]
        return mutant

    def generate_deletion(self):
        """Generate the mutant sequence with the reference residue(s) deleted."""
        pos = self.mutation_index - (self.seq_start - 1)
        mutant = self.seq[:pos] + self.seq[pos + self.ref_size:]
        return mutant

# ----------------------------
# FEATURE CALCULATION (using Sparrow Protein predictors)
# ----------------------------

def get_protein_features(seq):
    """
    Compute several protein structure features using Sparrow's predictors.
    Returns a dict of features.
    """
    p = Protein(seq)
    predictors = p.predictor
    features = {
        'asphericity': predictors.asphericity(),
        'radius_of_gyration': predictors.radius_of_gyration(),
        'end_to_end_distance': predictors.end_to_end_distance(),
        'scaling_exponent': predictors.scaling_exponent(),
        'prefactor': predictors.prefactor()
    }
    return features

# ----------------------------
# DISORDER SCORING
# ----------------------------

def count_low_disorder_residues(sequence, threshold=50):
    """
    Count residues with predicted pLDDT score below a threshold.
    Uses metapredict to predict pLDDT scores.
    """
    disorder_scores = meta.predict_pLDDT(sequence)
    return sum(score < threshold for score in disorder_scores)

# ----------------------------
# PLOTTING FUNCTIONS
# ----------------------------

def plot_boxplot(data, feature, ylabel, filename, title=None):
    """
    Plot a boxplot of the square-root transformed absolute deltas for a given feature.
    Performs a Mann-Whitney U test and annotates the p-value.
    """
    # Square-root transformation for visualization
    col_scaled = f"scaled_{feature}"
    data[col_scaled] = np.sqrt(data[f"abs_delta_{feature}"])
    
    # Prepare classes for test
    class_deleterious = data[data['Class'] == 'Deleterious'][f"abs_delta_{feature}"]
    class_neutral = data[data['Class'] == 'Neutral'][f"abs_delta_{feature}"]
    stat, p_value = mannwhitneyu(class_deleterious, class_neutral, alternative='two-sided')
    
    # Plot
    plt.figure(figsize=(6, 5))
    ax = sns.boxplot(x='Class', y=col_scaled, data=data, width=0.5, fliersize=3, boxprops={'facecolor': 'lightblue'})
    
    # Adjust y-axis
    y_min, y_max = data[col_scaled].min(), data[col_scaled].max()
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.2 * (y_max - y_min))
    
    # Annotation
    x1, x2 = 0, 1
    y = y_max + 0.1 * (y_max - y_min)
    h = 0.05 * (y_max - y_min)
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
    plt.text((x1 + x2) * 0.5, y+h, f'p = {p_value:.3e}', ha='center', va='bottom', color='k')
    
    plt.xlabel('Class', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    if title:
        plt.title(title)
    sns.despine()
    plt.savefig(filename, dpi=300)
    plt.show()

def plot_roc_curves(data, features, label_col, filename):
    """
    Plot ROC curves for the specified features.
    """
    plt.figure(figsize=(8, 6))
    true_labels = data[label_col]
    
    for feature in features:
        predicted = data[feature]
        fpr, tpr, _ = roc_curve(true_labels, predicted)
        auc_val = roc_auc_score(true_labels, predicted)
        plt.plot(fpr, tpr, label=f'{feature} (AUC = {auc_val:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(loc='lower right')
    sns.despine()
    plt.savefig(filename, dpi=300)
    plt.show()

def plot_precision_recall(data, features, label_col, filename):
    """
    Plot Precision-Recall curves for the specified features.
    """
    plt.figure(figsize=(8, 6))
    true_labels = data[label_col]
    
    for feature in features:
        predicted = data[feature]
        precision, recall, _ = precision_recall_curve(true_labels, predicted)
        pr_auc_val = auc(recall, precision)
        plt.plot(recall, precision, label=f'{feature} (AUC-PR = {pr_auc_val:.4f})')
    
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.legend(loc='upper right')
    sns.despine()
    plt.savefig(filename, dpi=300)
    plt.show()

# ----------------------------
# MAIN WORKFLOW
# ----------------------------

def main():
    # --- Settings & Filepaths ---
    mutation_filepath = "output/AF2_Disorder_deleterious_neutral_vus_results_filtered.tsv"
    fasta_dir = 'disorder/alphafold/fasta_download'
    output_features_csv = "output/AF2_disorder_albertross_prediction_all.tsv"
    
    # --- Load Mutation Data ---
    mut_df = load_mutation_data(mutation_filepath)
    
    # --- Get Wildtype and Mutant Sequences ---
    # Assume UNIPROT_ID is used to locate the FASTA file and GSTART, GEND define the region.
    def get_wt_sequence(row):
        fasta_path = os.path.join(fasta_dir, f"{row['UNIPROT_ID']}.fasta")
        return load_fasta_region(fasta_path, row['GSTART'], row['GEND'])
    
    mut_df['WT'] = mut_df.apply(get_wt_sequence, axis=1)
    
    def get_mutant_sequence(row):
        mutator = SequenceMutator(
            seq=row['WT'],
            ref=row['CAVA_PROTREF'],
            alt=row['CAVA_PROTALT'],
            seq_start=row['GSTART'],
            mutation_pos=row['CAVA_PROTPOS']
        )
        # Uncomment verify_ref() if you wish to check the reference residue.
        # mutator.verify_ref()
        return mutator.generate_mutant()
    
    mut_df['mut_seq'] = mut_df.apply(get_mutant_sequence, axis=1)
    
    # --- Compute Structural Features ---
    # For both wildtype and mutant sequences, compute features and record differences.
    for col in ['asphericity', 'radius_of_gyration', 'end_to_end_distance', 'scaling_exponent', 'prefactor']:
        wt_feature = mut_df['WT'].apply(lambda s: get_protein_features(s)[col])
        mut_feature = mut_df['mut_seq'].apply(lambda s: get_protein_features(s)[col])
        mut_df[f'wt_{col}'] = wt_feature
        mut_df[f'mut_{col}'] = mut_feature
        mut_df[f'delta_{col}'] = mut_df[f'mut_{col}'] - mut_df[f'wt_{col}']
        mut_df[f'abs_delta_{col}'] = mut_df[f'delta_{col}'].abs()
    
    # Save feature table
    mut_df.to_csv(output_features_csv, sep='\t', index=False)
    
    # --- Filtering for Further Analysis ---
    data_df = pd.read_csv(output_features_csv, sep='\t')
    data_keep = data_df[data_df['Class'] != 'VUS'].copy()
    
    # --- Plotting Boxplots for Each Feature ---
    plot_boxplot(data_keep, 'asphericity', 'Asphericity', 'association_asphericity_sqrt.png')
    plot_boxplot(data_keep, 'radius_of_gyration', 'Radius of Gyration', 'association_radiusofgyration.png')
    plot_boxplot(data_keep, 'end_to_end_distance', 'End-to-End Distance', 'association_endtoenddistance.png')
    plot_boxplot(data_keep, 'scaling_exponent', 'Scaling Exponent', 'association_scaling_exponent.png')
    plot_boxplot(data_keep, 'prefactor', 'Prefactor', 'association_prefactor.png')
    
    # --- ROC and Precision-Recall Curves ---
    # Convert Class to binary label: Deleterious = 1, Neutral = 0.
    data_keep['label'] = data_keep['Class'].map({'Deleterious': 1, 'Neutral': 0})
    feature_list = [
        'abs_delta_asphericity',
        'abs_delta_radius_of_gyration',
        'abs_delta_scaling_exponent',
        'abs_delta_end_to_end_distance',
        'abs_delta_prefactor'
    ]
    
    # Plot ROC curves and PR curves
    plot_roc_curves(data_keep, feature_list, 'label', "AUC_of_pidrc.png")
    plot_precision_recall(data_keep, feature_list, 'label', "AUCPR_of_pidrc.png")
    
    # --- Save Additional Outputs ---
    # Save gene chromosome mapping and reference/mutant sequences.
    data_keep[['CHROM', 'GENES']].drop_duplicates().to_csv("AF2_disorder_genes_chrom.tsv", sep='\t', index=False)
    data_df.to_csv('output/AF2_disorder_albatross_features.csv', index=False)
    
    # Save unique wildtype and mutant sequences to files.
    for col, fname in [('WT', 'disorder_ref_seq.list'), ('mut_seq', 'disorder_mut_seq.list')]:
        unique_seqs = data_df[col].unique()
        with open(fname, 'w') as fout:
            fout.write("\n".join(unique_seqs))
    
if __name__ == "__main__":
    main()
