import pytest
import ollama
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM

# 1. Define a Custom Wrapper (The SDET Adapter Pattern)
class OllamaJudge(DeepEvalBaseLLM):
    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        # Direct call to the ollama python library
        response = ollama.chat(model=self.model_name, messages=[
            {'role': 'user', 'content': prompt},
        ])
        return response['message']['content']

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

# 2. The Test Function
def test_hallucination_check():
    # Initialize our custom local judge
    local_judge = OllamaJudge(model_name="llama3.2")
    
    # Context vs. Hallucinated Output
    context = ["The 2026 SDET roadmap requires learning local LLM orchestration."]
    actual_output = "The 2026 roadmap says you only need to learn Manual Testing."
    
    test_case = LLMTestCase(
        input="What does the 2026 roadmap require?",
        actual_output=actual_output,
        retrieval_context=context
    )

    # Use the FaithfulnessMetric with our Custom Judge
    metric = FaithfulnessMetric(threshold=0.5, model=local_judge)
    
    assert_test(test_case, [metric])