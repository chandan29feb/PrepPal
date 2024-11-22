from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from app.config import GROQ_API_KEY
import os 
import json

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not PResent in .env file.")

os.environ["GROQ_API_KEY"] =GROQ_API_KEY
llm = ChatGroq(model="llama-3.1-70b-versatile")

def extract_information(input_text):
    prompt_extract = PromptTemplate.from_template(
        """
        You are an advanced language model tasked with extracting academic subject, topic, and difficulty from the user input string.

        Please extract the following details:

        1. **Subject**: Academic subject.
        2. **Topic**: Academic topic name.
        3. **Difficulty**: Hard, Medium, Easy
        
        ### INPUT

        User input string:
        {user_query}

        Please return the academic subject, topic, and difficulty as a structured JSON object.
        Return only valid JSON, no other extra information like explanation.

        ## VALID JSON (NO PREAMBLE):
        """
    )

    chain_extract = prompt_extract | llm
    res = chain_extract.invoke({"user_query": input_text})
    try:
        extracted_values = json.loads(res.content)
        return extracted_values
    except json.JSONDecodeError:
        raise ValueError("Response is not valid JSON.")
