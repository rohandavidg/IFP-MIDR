import os
import pandas as pd
import numpy as np
from config import (
    DISPROT_FILE, UNIPROT_ID_FILE, CLINVAR_FILE, ENSEMBL_REFSEQ_FILE, WGS_BED_FILE, AF2_DISORDER_FILE,
    PATHOGENIC_CLASSES, BENIGN_CLASSES, VUS_CLASSES, CLINVAR_RESULTS_FILE, STUDY_GENE_LIST_FILE, STUDY_REGIONS_FILE,
    DISORDER_REGIONS_FILE
)

def load_data():
    """Loads all required datasets."""
    df = pd.read_csv(DISPROT_FILE, sep='\t')
    uniprot_id_df = pd.read_csv(UNIPROT_ID_FILE, sep='\t')
    clinvar_df = pd.read_csv(CLINVAR_FILE, sep='\t',low_memory=False)
    ensemble_to_refseq_df = pd.read_csv(ENSEMBL_REFSEQ_FILE, sep='\t')
    wgs_bed_df = pd.read_csv(WGS_BED_FILE, sep='\t')

    af2_disorder_df = pd.read_csv(AF2_DISORDER_FILE, sep='\t', header=None, 
                                  names=['PID', 'START', "STOP", "GENE", 'ENSEMBLE_ID']).dropna(subset=['ENSEMBLE_ID'])

    return df, uniprot_id_df, clinvar_df, ensemble_to_refseq_df, wgs_bed_df, af2_disorder_df


def generate_gene_bed(df):
    """Generates a BED format file from disorder gene data."""
    df_filtered = df[~df['Gene Names (primary)'].str.contains(';', na=False)]
    bed_df = df_filtered[['START', 'STOP', 'Gene Names (primary)', 'RNA_nucleotide_accession.version', 'From']].copy()
    bed_df.columns = ['GSTART', "GEND", 'GENES', "TRANSCRIPT", "UNIPROT_ID"]
    bed_df['TRANSCRIPT_NOV'] = bed_df['TRANSCRIPT'].str.split('.').str[0]
    return bed_df.drop_duplicates().reset_index(drop=True)


def process_data():
    """Main function that processes and saves all required datasets."""
    df, uniprot_id_df, clinvar_df, ensemble_to_refseq_df, wgs_bed_df, af2_disorder_df = load_data()

    # Human-Only Data
    df_human = df[df['ncbi_taxon_id'] == 9606]

    af2_disorder_uniprot_df = af2_disorder_df.merge(uniprot_id_df, left_on='PID', right_on='From', how='left')

    ensemble_to_refseq_df = ensemble_to_refseq_df[
        (ensemble_to_refseq_df['RNA_nucleotide_accession.version'].str.startswith('NM_')) &
        (ensemble_to_refseq_df['Ensembl_rna_identifier'].str.startswith('ENST')) &
        (ensemble_to_refseq_df['#tax_id'] == 9606)
    ]
    ensemble_to_refseq_df = ensemble_to_refseq_df[['RNA_nucleotide_accession.version', 'Ensembl_rna_identifier']].drop_duplicates()

    disorder_uniprot_ref_df = af2_disorder_uniprot_df.merge(
        ensemble_to_refseq_df, left_on='ENSEMBLE_ID', right_on='Ensembl_rna_identifier', how='left'
    )

    disorder_uniprot_ref_df.loc[
        (disorder_uniprot_ref_df['GENE'] == 'RYBP') & disorder_uniprot_ref_df['RNA_nucleotide_accession.version'].isna(),
        'RNA_nucleotide_accession.version'
    ] = 'NM_012234.7'

    disorder_bed_df = generate_gene_bed(disorder_uniprot_ref_df)

    # Ensure wgs_bed_df has the TRANSCRIPT_NOV column
    wgs_bed_df['TRANSCRIPT_NOV'] = wgs_bed_df['TRANSCRIPT'].str.split('.').str[0]

    disorder_bed_wgs_df = pd.merge(
        disorder_bed_df, wgs_bed_df, left_on='TRANSCRIPT_NOV', right_on='TRANSCRIPT_NOV', how='left'
    )

    disorder_genes = list(set(disorder_bed_wgs_df.GENES.tolist()))

    disorder_bed_wgs_df[['CHROM', 'GSTART', 'GEND', 'GENES', 'TRANSCRIPT_x', 'TRANSCRIPT_NOV', 'UNIPROT_ID']].to_csv(
        DISORDER_REGIONS_FILE, sep='\t', index=False, header=None
    )

    disorder_disprot_df = disorder_bed_wgs_df[['CHROM', 'GSTART', 'GEND', 'GENES', 'TRANSCRIPT_x', 'TRANSCRIPT_NOV', 'UNIPROT_ID']]

    # Extract ClinVar Data
    clinvar_data_df = clinvar_df[clinvar_df['CAVA_GENE'].isin(disorder_genes) & (clinvar_df['CAVA_SO'] == 'missense_variant')]
    clinvar_pathogenic_df = clinvar_data_df[clinvar_data_df['CLNSIG'].isin(PATHOGENIC_CLASSES)]
    clinvar_benign_df = clinvar_data_df[clinvar_data_df['CLNSIG'].isin(BENIGN_CLASSES)]
    clinvar_vus_df = clinvar_data_df[clinvar_data_df['CLNSIG'].isin(VUS_CLASSES)]
    clinvar_pathogenic_df["Class"] = "Deleterious"
    clinvar_benign_df["Class"] = "Neutral"
    clinvar_vus_df["Class"] = "VUS"
    clinvar_results_df = pd.concat([clinvar_pathogenic_df, clinvar_benign_df, clinvar_vus_df])
    clinvar_results_df['GENES'] = clinvar_results_df['CAVA_GENE']
    clinvar_results_df.to_csv(CLINVAR_RESULTS_FILE, sep='\t', index=False)

    clinvar_AF2_disorder_results_merge_df = pd.merge(
        clinvar_results_df, disorder_disprot_df, on="GENES", how="inner"
    )
    clinvar_AF2_disorder_results_merge_df["CAVA_PROTPOS"] = clinvar_AF2_disorder_results_merge_df["CAVA_PROTPOS"].astype(int)
#    print(clinvar_AF2_disorder_results_merge_df["GSTART"])
#    print(clinvar_AF2_disorder_results_merge_df["GEND"])    
    clinvar_AF2_disorder_results_merge_df["within_interval"] = (
        (clinvar_AF2_disorder_results_merge_df["CAVA_PROTPOS"] >= clinvar_AF2_disorder_results_merge_df["GSTART"]) &
        (clinvar_AF2_disorder_results_merge_df["CAVA_PROTPOS"] <= clinvar_AF2_disorder_results_merge_df["GEND"]))

    clinvar_AF2_disorder_results_merge_df = clinvar_AF2_disorder_results_merge_df[
        clinvar_AF2_disorder_results_merge_df["within_interval"]]

    # Save Final Filtered Dataset
    clinvar_AF2_disorder_results_merge_df.to_csv(CLINVAR_RESULTS_FILE.replace(".tsv", "_filtered.tsv"), sep='\t', index=False)


#    study_regions_df = wgs_bed_df[wgs_bed_df['GENE'].isin(study_gene_list)].drop_duplicates()
#    study_regions_df.to_csv(STUDY_REGIONS_FILE, sep='\t', index=False)

#    return clinvar_AF2_disorder_results_merge_df


if __name__ == "__main__":
    clinvar_AF2_disorder_results_merge_df = process_data()
