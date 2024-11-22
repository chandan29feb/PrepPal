import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY =os.getenv("GROQ_API_KEY")
DATA_FILE = "data/questions.json"
INDEX_FILE = "data/question_index.faiss"
METADATA_FILE = "data/questions_with_metadata.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
