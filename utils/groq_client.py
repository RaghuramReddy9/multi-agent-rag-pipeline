import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def get_groq_client():
    """Initialize Groq client using environment variable."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in environment")
    return Groq(api_key=api_key)

