# src/interview_predictor.py
from .model import LLMProvider
from .prompts import INTERVIEW_PREDICTION_PROMPT, INTERVIEW_SYSTEM_PROMPT

class InterviewPredictor:
    """Evaluates candidate interview likelihood using an abstract LLM provider."""
    
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def assess_likelihood(self, resume_text: str, jd_text: str) -> dict:
        """Formats the prompt and delegates matrix evaluation to the configured LLM provider."""
        prompt = INTERVIEW_PREDICTION_PROMPT.format(jd_text=jd_text, resume_text=resume_text)
        
        return self.provider.generate_json(
            prompt=prompt, 
            system_prompt=INTERVIEW_SYSTEM_PROMPT
        )