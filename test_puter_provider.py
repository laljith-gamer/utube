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
        if "error" in response:
            print("Puter returned an error (expected if no API key):", response["error"])
        else:
            if response.get("status") == "ok" or "text" in response:
                print("JSON response successful.")
            else:
                print("JSON parsing failed, unexpected response:", response)
            
    except Exception as e:
        print("Puter API call failed:", e)

if __name__ == "__main__":
    test_puter()
