from utils.prompt_template import extract_academic_info_prompt
from langchain_groq import ChatGroq
from utils.config import GROQ_API_KEY
import os 
import json

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not PResent in .env file.")

os.environ["GROQ_API_KEY"] =GROQ_API_KEY
llm = ChatGroq(model="llama-3.1-70b-versatile")

def extract_information(input_text):
    chain_extract = extract_academic_info_prompt | llm
    
    res = chain_extract.invoke({"user_query": input_text})
    try:
        extracted_values = json.loads(res.content)
        return extracted_values
    except json.JSONDecodeError:
        raise ValueError("Response is not valid JSON.")
