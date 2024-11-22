import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from utils.config import INDEX_FILE, METADATA_FILE, MODEL_NAME,DATA_FILE

def generate_embeddings_and_index():
    model = SentenceTransformer(MODEL_NAME)

    if not (os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE)):
        print("Generating embeddings and creating index...")
        df = pd.read_json(DATA_FILE)
        df.fillna("", inplace=True)
        df['combined'] = df['subject'] + " " + df['topic'] + " " + df['difficulty']
        
        question_embeddings = model.encode(df['combined'].tolist(), show_progress_bar=True)

        index = faiss.IndexFlatL2(question_embeddings.shape[1])
        index.add(np.array(question_embeddings))
        faiss.write_index(index, INDEX_FILE)

        df.to_pickle(METADATA_FILE)

        print("Index and metadata saved")
    else:
        print("Index and metadata already exist")

if __name__ == "__main__":
    generate_embeddings_and_index()
