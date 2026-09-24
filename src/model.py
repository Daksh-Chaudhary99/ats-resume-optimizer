# src/model.py
import json
import re
from abc import ABC, abstractmethod
from openai import AzureOpenAI
from huggingface_hub import InferenceClient

class LLMProvider(ABC):
    """
    Abstract base class defining the standard interface for all LLM providers.
    Any new provider must implement the generate_json method.
    """
    
    @abstractmethod
    def generate_json(self, prompt: str, system_prompt: str = "") -> dict:
        """Takes a prompt and returns a parsed Python dictionary."""
        pass


class AzureOpenAIProvider(LLMProvider):
    """Implementation for Azure OpenAI."""
    
    def __init__(self, endpoint: str, api_key: str, api_version: str, deployment_name: str):
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )
        self.deployment_name = deployment_name

    def generate_json(self, prompt: str, system_prompt: str = "You output strictly valid JSON.") -> dict:
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        return json.loads(response.choices[0].message.content)


class HuggingFaceProvider(LLMProvider):
    """Implementation for Hugging Face Serverless API using Chat Completion."""
    
    def __init__(self, token: str, model_name: str):
        self.client = InferenceClient(token=token)
        self.model_name = model_name

    def generate_json(self, prompt: str, system_prompt: str = "") -> dict:
        messages = [
            {
                "role": "system", 
                "content": system_prompt + " You output strictly valid JSON without markdown."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        try:
            response = self.client.chat_completion(
                messages=messages,
                model=self.model_name,
                max_tokens=1000,
                temperature=0.2
            )
            
            response_text = response.choices[0].message.content
            
            # Print the raw response to your terminal for debugging
            print("\n--- RAW MODEL OUTPUT ---")
            print(response_text)
            print("------------------------\n")
            
            # Robust JSON extraction: Find the first '{' and the last '}'
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                cleaned_text = response_text[start_idx:end_idx+1]
                return json.loads(cleaned_text)
            else:
                print("Error: No JSON brackets found in the response.")
                return {}
                
        except json.JSONDecodeError as e:
            print(f"JSON Parsing Error: {e}")
            return {}
        except Exception as e:
            print(f"Hugging Face API Error: {str(e)}")
            return {}