# src/config.py
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

class Config:
    """Centralized configuration for the application."""
    
    # Azure OpenAI Settings
    AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")

    # Hugging Face Settings
    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
    
    # Provider Toggle: 'azure' or 'huggingface'
    ACTIVE_LLM_PROVIDER = os.getenv("ACTIVE_LLM_PROVIDER", "huggingface")