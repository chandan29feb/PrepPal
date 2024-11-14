import spacy
import re
from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import numpy as np
from flask import Flask, request, jsonify
import os

nlp = spacy.load("en_core_web_sm")

subjects = ["math", "biology", "chemistry", "physics"]
chapters = ["algebra", "geometry", "calculus", "matrices", "vectors", "newton's law"]
topics = ["practice", "questions", "examples", "introduction", "applications"]
difficulties = ["easy", "medium", "hard", "advanced"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

df = pd.read_json('questions.json')

def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

if not os.path.exists("question_index.faiss") or not os.path.exists("questions_with_metadata.pkl"):
    print("Generating embeddings and creating index...")
    question_embeddings = model.encode(df['question'].tolist(), show_progress_bar=True)
    normalized_embeddings = normalize_embeddings(question_embeddings)

    index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
    index.add(np.array(normalized_embeddings))
    faiss.write_index(index, "question_index.faiss")

    df.to_pickle("questions_with_metadata.pkl")
else:
    print("Loading saved index and metadata...")
    index = faiss.read_index("question_index.faiss")
    df = pd.read_pickle("questions_with_metadata.pkl")

def query_embedding(prompt):
    prompt_embedding = model.encode([prompt])[0]
    normalized_prompt_embedding = normalize_embeddings(np.array([prompt_embedding]))[0]
    return normalized_prompt_embedding


def extract_keywords_from_prompt(prompt):
    prompt = prompt.lower()

    subject = next((sub for sub in subjects if sub in prompt), None)
    chapter = next((ch for ch in chapters if ch in prompt), None)
    topic = next((t for t in topics if t in prompt), None)
    difficulty = next((dif for dif in difficulties if dif in prompt), None)

    count_match = re.search(r'\b(\d+)\b', prompt)
    count = int(count_match.group(1)) if count_match else 10

    return subject, chapter, topic, difficulty, count

def filter_questions(subject, chapter, topic):
    filtered_df = df.copy()
    if subject:
        filtered_df = filtered_df[filtered_df['subject'].str.contains(subject, case=False, na=False)]
    if chapter:
        filtered_df = filtered_df[filtered_df['chapter'].str.contains(chapter, case=False, na=False)]
    if topic:
        filtered_df = filtered_df[filtered_df['topic'].str.contains(topic, case=False, na=False)]
    return filtered_df

def search_questions(prompt_embedding, top_k=10):
    D, I = index.search(np.array([prompt_embedding]), top_k)
    return df.iloc[I[0]].copy()

app = Flask(__name__)

@app.route('/get_questions', methods=['POST'])
def get_questions_route():
    data = request.json
    prompt = data.get("prompt", "")

    subject, chapter, topic, difficulty, count = extract_keywords_from_prompt(prompt)

    filtered_df = filter_questions(subject, chapter, topic)

    if not filtered_df.empty:
        results = filtered_df.head(count)
    else:
        prompt_embedding = query_embedding(prompt)
        results = search_questions(prompt_embedding, top_k=count)

    return jsonify({"questions": results.to_dict(orient="records")})

if __name__ == '__main__':
    app.run(debug=True)
