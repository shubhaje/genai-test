import pytest
import ollama
import json
import re
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM

class OllamaJudge(DeepEvalBaseLLM):
    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        # 1. Use 'format': 'json' to force Ollama to output valid JSON
        response = ollama.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            format='json' 
        )
        content = response['message']['content']
        
        # 2. SDET Safety: Strip any leading/trailing whitespace or markdown backticks
        # Some models still wrap JSON in ```json { ... } ```
        content = re.sub(r'```json\s?|```', '', content).strip()
        return content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

def test_hallucination_check():
    local_judge = OllamaJudge(model_name="llama3:8b")
    
    # We purposefully make this "Unfaithful" to see the test fail correctly
    context = ["The 2026 SDET roadmap requires learning local LLM orchestration."]
    actual_output = "The 2026 roadmap says you only need to learn Manual Testing."
    
    test_case = LLMTestCase(
        input="What does the 2026 roadmap require?",
        actual_output=actual_output,
        retrieval_context=context
    )

    # Note: Llama 3.2 might need a lower threshold to be 'sure' of its judgment
    metric = FaithfulnessMetric(threshold=0.5, model=local_judge)
    assert_test(test_case, [metric])