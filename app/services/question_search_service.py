import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from utils.config import INDEX_FILE, METADATA_FILE, MODEL_NAME
import os

_model = SentenceTransformer(MODEL_NAME)
_index = None
_df = None

def _initialize_index_and_metadata():
    global _index, _df

    if not (os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE)):
        raise FileNotFoundError("Index or metadata not ExIst. Pls run `train.py` first.")
    
    print("LoadIng saved Index and metadata...")
    _index = faiss.read_index(INDEX_FILE)
    _df = pd.read_pickle(METADATA_FILE)

_initialize_index_and_metadata()

def query_embedding(prompt):
    return _model.encode(prompt)

def search_questions(prompt_embedding, top_k=10):
    if _index is None or _df is None:
        raise ValueError("Index and metadata not initialized.")

    D, I = _index.search(np.array([prompt_embedding]), top_k)
    results = _df.iloc[I[0]]
    return results
