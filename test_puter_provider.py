import os
import json
import logging
from pipeline.providers.puter import PuterProvider

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_puter():
    print("Testing PuterProvider...")
    try:
        response = PuterProvider.chat(
            model="claude-opus-4-6", # Testing claude-opus-4-6 as requested
            messages=[{"role": "user", "content": "Respond with a simple JSON object: {\"status\": \"ok\", \"message\": \"hello world\"}"}],
            max_tokens=100,
            temperature=0.1,
            json_mode=True
        )
        print("Response received:")
        print(response)
        
        # Verify JSON
        try:
            data = json.loads(response)
            if data.get("status") == "ok":
                print("JSON parsing successful.")
            else:
                print("JSON parsing failed, unexpected response:", data)
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e)
            print("Raw response:", response)
            
    except Exception as e:
        print("Puter API call failed:", e)

if __name__ == "__main__":
    test_puter()
