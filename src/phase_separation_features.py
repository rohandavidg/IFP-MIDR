"""Main module for peptide sequence analysis."""
import argparse
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from tqdm.auto import tqdm
from scipy import signal
from loguru import logger
from datetime import time
import datetime
import time

# Constants
RESIDUES = [
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"
]
HP = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, 
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, 
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2, 
    "U": 0.0
}
COV_WINDOW = 30


class HydroPhobicIndex:
    """Handles hydrophobicity index calculations."""
    def __init__(self, hpi_list):
        self.hpi_list = hpi_list


class PeptideAnalyzer:
    """Analyzes peptide sequences and extracts biochemical features."""
    
    def __init__(self, peptide_file: str):
        """
        Initialize the analyzer with a list of peptide sequences.

        Parameters:
        peptide_sequences (list): A list of peptide strings for analysis.
        """
        self.df = self.read_sequences(peptide_file)
        self.run_analysis()

    @staticmethod
    def read_sequences(peptide_file: str) -> pd.DataFrame:
        """Reads peptide sequences from a file and returns a DataFrame."""
        logger.debug(f"Reading sequences from file: {peptide_file}")
        with open(peptide_file, "r") as f:
            sequences = [line.strip() for line in f if line.strip()]
        return pd.DataFrame(sequences, columns=["sequence"])
        
    def run_analysis(self):
        """Run the full pipeline for feature extraction."""
        analysis_steps = [
            self.add_hydrophobic_features,
            self.amino_acid_analysis,
            self.add_biochemical_combinations,
            self.add_low_complexity_features,
        ]
        for step in analysis_steps:
            start = time.time()
            step()
            logger.debug(f"{round(time.time() - start, 2)}s - {step.__name__}")

    def add_hydrophobic_features(self):
        """Adds hydrophobicity-related features."""
        logger.debug("Calculating hydrophobic features.")
        self.df["HydroPhobicIndex"] = self.df["sequence"].apply(
            lambda seq: HydroPhobicIndex([HP.get(res, 0) for res in seq])
        )

        for index, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            win = signal.hann(COV_WINDOW)
            smoothed = signal.convolve(
                row["HydroPhobicIndex"].hpi_list, win, mode="same"
            ) / sum(win)
            for threshold in [-1.5, -2.0, -2.5]:
                key = f"hpi_<{threshold}"
                self.df.loc[index, key + "_frac"] = sum(i < threshold for i in smoothed) / len(smoothed)
                self.df.loc[index, key] = sum(i < threshold for i in smoothed)

    def amino_acid_analysis(self):
        """Adds amino acid fraction and other sequence properties."""
        logger.debug("Analyzing amino acid composition.")
        self.df["length"] = self.df["sequence"].str.len()
        for res in RESIDUES:
            self.df[f"fraction_{res}"] = self.df["sequence"].apply(lambda seq: seq.count(res) / len(seq))

        for index, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            seq = row["sequence"]
            analysis = ProteinAnalysis(seq)
            self.df.loc[index, "IEP"] = analysis.isoelectric_point()
            if "X" not in seq and "B" not in seq:
                self.df.loc[index, "molecular_weight"] = analysis.molecular_weight()
            if set(seq).isdisjoint({"X", "B", "U"}):
                self.df.loc[index, "gravy"] = analysis.gravy()

    def add_biochemical_combinations(self):
        """Adds combined biochemical features."""
        logger.debug("Calculating biochemical combinations.")
        self.df = self.df.assign(
            Asx=self.df["sequence"].apply(lambda seq: seq.count("D") + seq.count("N")),
            Glx=self.df["sequence"].apply(lambda seq: seq.count("E") + seq.count("Q")),
            Xle=self.df["sequence"].apply(lambda seq: seq.count("I") + seq.count("L")),
            Pos_charge=self.df["sequence"].apply(lambda seq: seq.count("K") + seq.count("R") + seq.count("H")),
            Neg_charge=self.df["sequence"].apply(lambda seq: seq.count("D") + seq.count("E")),
            Aromatic=self.df["sequence"].apply(lambda seq: seq.count("F") + seq.count("W") + seq.count("Y") + seq.count("H")),
            Alipatic=self.df["sequence"].apply(lambda seq: seq.count("V") + seq.count("I") + seq.count("L") + seq.count("M")),
        )

    def add_low_complexity_features(self):
        """Adds low complexity features to the DataFrame."""
        logger.debug("Calculating low complexity features.")
        window = 20
        cutoff = 7
        for index, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            seq = row["sequence"]
            complexity_scores = [
                len(set(seq[i:i + window]))
                for i in range(len(seq) - window + 1)
            ]
            low_complexity_count = sum(1 for score in complexity_scores if score <= cutoff)
            self.df.loc[index, "lcs_score"] = low_complexity_count
            self.df.loc[index, "lcs_fraction"] = low_complexity_count / max(len(complexity_scores), 1)


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze peptide sequences from a file.")
    parser.add_argument(
        "-i", "--input", required=True, help="Input file containing peptide sequences (one per line)."
    )
    parser.add_argument(
        "-o", "--output", default="peptide_analysis.tsv", help="Output file to save results (default: peptide_analysis.tsv)."
    )
    args = parser.parse_args()

    # Analyze sequences
    analyzer = PeptideAnalyzer(args.input)
    analyzer.df.to_csv(args.output, sep="\t", index=False)
    print(f"Analysis complete! Results saved to {args.output}.")
#    sequences = ["MKWVTFISLLFLFSSAYS", "AGGAVLTALLA", "MVTLSKLV"]
#    analyzer = PeptideAnalyzer(sequences)
#    analyzer.df.to_csv("test_ps.tsv", sep='\t', index=None)
