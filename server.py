from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import numpy as np
import os
from flask import Flask, request, jsonify

# Load dataset and SentenceTransformer model
df = pd.read_json('questions.json')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Check if the FAISS index and metadata file already exist
if not os.path.exists("question_index.faiss") or not os.path.exists("questions_with_metadata.pkl"):
    # If index or metadata do not exist, encode and save them
    print("Generating embeddings and creating index...")
    question_embeddings = model.encode(df['question'].tolist(), show_progress_bar=True)
    
    # Create and save FAISS index
    index = faiss.IndexFlatL2(question_embeddings.shape[1])
    index.add(np.array(question_embeddings))
    faiss.write_index(index, "question_index.faiss")

    # Save question metadata
    df.to_pickle("questions_with_metadata.pkl")
else:
    # Load FAISS index and question metadata from disk
    print("Loading saved index and metadata...")
    index = faiss.read_index("question_index.faiss")
    df = pd.read_pickle("questions_with_metadata.pkl")


# Define functions for querying
def query_embedding(prompt):
    return model.encode(prompt)


def search_questions(prompt_embedding, top_k=10):
    D, I = index.search(np.array([prompt_embedding]), top_k)
    results = df.iloc[I[0]]
    return results


# Set up the Flask app
app = Flask(__name__)

@app.route('/get_questions', methods=['POST'])
def get_questions_route():
    data = request.json
    prompt = data.get("prompt", "")
    
    # Convert prompt to embedding
    prompt_embedding = query_embedding(prompt)
    
    # Perform a search in the vector database
    results = search_questions(prompt_embedding, top_k=10)
    
    return jsonify({"questions": results.to_dict(orient="records")})

if __name__ == '__main__':
    app.run(debug=True)