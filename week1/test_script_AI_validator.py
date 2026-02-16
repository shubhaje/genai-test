import os
from pathlib import Path
from dotenv import load_dotenv

# This finds the directory where your SCRIPT is located
script_dir = Path(__file__).resolve().parent
env_path = script_dir / ".env"

print(f"DEBUG: Looking for .env at: {env_path}")
print(f"DEBUG: Does .env exist? {env_path.exists()}")

# Load it explicitly
load_dotenv(dotenv_path=env_path, override=True)

# Remove the system variables that the SDK is "auto-picking"
os.environ.pop("GEMINI_KEY_A", None)
os.environ.pop("GEMINI_KEY_B", None)

class BankingAISuite:
    def __init__(self):
        self.keys = [os.getenv("GEMINI_KEY_A"), os.getenv("GEMINI_KEY_B")]
        self.keys = [k for k in self.keys if k]
        if not self.keys:
            raise ValueError("❌ No keys found in .env! Add GEMINI_KEY_A=your_key")
        
        self.current_idx = 0
        self.client = self._get_client()

    def _get_client(self):
        return genai.Client(api_key=self.keys[self.current_idx])

    def generate_batch(self, scenarios):
        """Generates all XMLs in one request to avoid 429 errors."""
        scenario_list = "\n".join([f"- {s}" for s in scenarios])
        prompt = f"""
        Act as an ISO 20022 Expert. Generate raw XML snippets for 'pain.001' based on:
        {scenario_list}
        
        Format your response exactly like this for my parser:
        ---SCENARIO: [Name]---
        [XML HERE]
        """
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            return response.text
        except errors.ClientError as e:
            print(f"⚠️ API Error: {e}. Rotating key...")
            self.current_idx = (self.current_idx + 1) % len(self.keys)
            self.client = self._get_client()
            return None

    def local_validate_xml(self, xml_str):
        """Checks if the XML is well-formed without using API quota."""
        try:
            parser = etree.XMLParser(recover=False)
            etree.fromstring(xml_str.encode('utf-8'), parser=parser)
            return "✅ Well-formed XML"
        except etree.XMLSyntaxError as e:
            return f"❌ XML Syntax Error: {e}"

# --- 2. EXECUTION ---
tester = BankingAISuite()
my_scenarios = [
    "Buffer Overflow in Creditor Name",
    "SQL Injection in Remittance Info",
    "Invalid characters in IBAN"
]

print("📡 Requesting batch from Gemini...")
raw_output = tester.generate_batch(my_scenarios)

if raw_output:
    # Split the batch output into individual test cases
    parts = re.split(r'---SCENARIO: (.*?)---', raw_output)
    
    for i in range(1, len(parts), 2):
        name = parts[i].strip()
        xml_content = parts[i+1].strip()
        
        print(f"\n📂 Testing Scenario: {name}")
        validation_status = tester.local_validate_xml(xml_content)
        print(validation_status)
        
        # Save to file for your records
        filename = f"test_{name.replace(' ', '_').lower()}.xml"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(xml_content)
        print(f"💾 Saved to {filename}")
else:
    print("❌ Batch generation failed. Check your API dashboard for quota limits.")