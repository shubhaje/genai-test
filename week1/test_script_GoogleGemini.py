import os
import asyncio
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.models import GeminiModel

# 1. Silence the noise
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
#MY_KEY = "YOUR_API_KEY_HERE"
import os
from deepeval.models import GeminiModel

# This looks for the variable you set in the terminal/Windows settings
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("API Key not found! Run the PowerShell command to set it.")


# 2. Setup the Judge
cloud_judge = GeminiModel(
    model="gemini-2.5-flash", 
    api_key=api_key
)
# 3. Define your scenarios
test_cases = [
    LLMTestCase(
        input="What is the refund window?",
        actual_output="You have 90 days to return your item.",
        expected_output="Refunds are accepted within 30 days only."
    ),
    LLMTestCase(
        input="Do you offer international shipping?",
        actual_output="Yes, we ship to over 50 countries worldwide.",
        expected_output="Yes, we ship to over 50 countries worldwide."
    )
]

metric = GEval(
    name="Hallucination Check",
    model=cloud_judge,
    criteria="Factual consistency between actual and expected output.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.5
)

# 4. The Direct Run (This bypasses Pytest entirely)
if __name__ == "__main__":
    os.environ["DEEPEVAL_VERBOSE_MODE"] = "YES"
    evaluate(test_cases=test_cases, metrics=[metric])