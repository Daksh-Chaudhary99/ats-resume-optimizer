# src/keyword_analyzer.py
from .model import LLMProvider
from .prompts import KEYWORD_ANALYSIS_PROMPT, KEYWORD_SYSTEM_PROMPT

class KeywordAnalyzer:
    """Handles keyword extraction and rewrite suggestions using an abstract LLM provider."""
    
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def analyze(self, resume_text: str, jd_text: str) -> dict:
        """Formats the prompt and delegates JSON generation to the configured LLM provider."""
        prompt = KEYWORD_ANALYSIS_PROMPT.format(jd_text=jd_text, resume_text=resume_text)
        
        return self.provider.generate_json(
            prompt=prompt, 
            system_prompt=KEYWORD_SYSTEM_PROMPT
        )