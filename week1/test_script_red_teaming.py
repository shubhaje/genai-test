import os
import time
from google import genai
from google.genai import errors
from dotenv import load_dotenv

# 1. Force environment reload
load_dotenv(override=True)

# 2. CLEAR the problematic global keys that are overriding your code
os.environ.pop("GOOGLE_API_KEY", None)
os.environ.pop("GEMINI_API_KEY", None)

class BankingAIManager:
    def __init__(self):
        # Explicitly grabbing your custom keys from .env
        self.keys = [os.getenv("GEMINI_KEY_A"), os.getenv("GEMINI_KEY_B")]
        self.keys = [k for k in self.keys if k]
        self.current_idx = 0
        
        if not self.keys:
            raise ValueError("❌ No keys found in .env! Ensure GEMINI_KEY_A is set.")
        
        self.client = self._get_client()

    def _get_client(self):
        # We explicitly pass the api_key here to bypass the SDK's auto-pick logic
        print(f"🔑 Switching to API Key {self.current_idx + 1}...")
        return genai.Client(api_key=self.keys[self.current_idx])

    def generate_iso_xml(self, attack_type):
        prompt = f"Act as a Banking Expert. Generate a 'pain.001' snippet for: {attack_type}. Return XML ONLY."
        
        # Try each key once before giving up
        for _ in range(len(self.keys)):
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                return response.text
            except errors.ClientError as e:
                if "429" in str(e):
                    print(f"⚠️ Key {self.current_idx + 1} exhausted. Rotating...")
                    self.current_idx = (self.current_idx + 1) % len(self.keys)
                    self.client = self._get_client()
                    time.sleep(10) # Wait for rotation cooldown
                else:
                    raise e
        return "❌ All keys exhausted."

# --- Execution ---
ai_manager = BankingAIManager()
scenarios = ["Buffer Overflow in Creditor", "SQL injection in Remittance"]

for scenario in scenarios:
    print(f"\n🚀 Running: {scenario}")
    result = ai_manager.generate_iso_xml(scenario)
    print(result)
    # 2026 FREE TIER TIP: 20 seconds is the safest interval for Flash 2.0
    print("⏳ Cooldown 20s...")
    time.sleep(20)