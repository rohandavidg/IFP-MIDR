import numpy as np
import pandas as pd
from bio_embeddings.embed import ProtTransBertBFDEmbedder
from config import EMBEDDINGS_JSON, PROCESSED_JSON

embedder = ProtTransBertBFDEmbedder()

def get_embedding(sequence):
    """Computes protein sequence embeddings."""
    embedding = embedder.embed(sequence)
    return embedder.reduce_per_protein(embedding)

def compute_embedding_features(df):
    """Compute embeddings and feature transformations."""
    df['wt_emb'] = df['WT'].apply(get_embedding)
    df['mut_emb'] = df['seq_mut'].apply(get_embedding)

    df['delta_emb_L1'] = df.apply(lambda row: np.abs(row['wt_emb'] - row['mut_emb']), axis=1)
    df['delta_emb_L2'] = df.apply(lambda row: np.square(row['wt_emb'] - row['mut_emb']), axis=1)
    df['delta_emb_average'] = df.apply(lambda row: (row['wt_emb'] + row['mut_emb']) / 2, axis=1)
    df['delta_emb_hadamard'] = df.apply(lambda row: row['wt_emb'] * row['mut_emb'], axis=1)

    df.to_json(PROCESSED_JSON, orient='records', lines=True)

if __name__ == "__main__":
    df = pd.read_json(EMBEDDINGS_JSON, lines=True)
    compute_embedding_features(df)
    print("Embeddings processed and saved.")
