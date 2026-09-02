import sys
from pathlib import Path

sys.path.append(str(Path().resolve()))

from pipeline.providers.brave import BraveProvider
from pipeline.providers.llm import LLMRouter, ProviderStatus
import requests
import base64
from dotenv import load_dotenv

def test_brave_images_and_vision():
    load_dotenv()
    print("Testing Brave Images...")
    cands = BraveProvider.search_images("cybersecurity hackers matrix", count=3)
    print(f"Got {len(cands)} candidates.")
    
    if not cands:
        print("No candidates found.")
        return
        
    llm = LLMRouter("llm_vision")
    
    for cand in cands:
        img_url = cand.get("url")
        print(f"Testing URL: {img_url}")
        
        try:
            resp = requests.get(img_url, timeout=10)
            resp.raise_for_status()
            mime = resp.headers.get("content-type", "image/jpeg")
            b64 = base64.b64encode(resp.content).decode("utf-8")
            data_uri = f"data:{mime};base64,{b64}"
            
            sys_prompt = "You are a visual investigator. Given the image and a scene description, rate the image relevance from 0 to 100. Return JSON: {'relevance': int, 'reason': 'str'}."
            user_prompt = "Scene Description: A cinematic shot of a hacker typing quickly in the dark."
            
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}}
                ]}
            ]
            
            print("Calling Vision LLM...")
            res = llm.chat_json_structured(messages, max_tokens=150)
            print(f"LLM Response Status: {res.status}")
            if res.status == ProviderStatus.SUCCESS:
                print(f"Parsed response: {res.parsed}")
                break
            else:
                print(f"LLM Error: {res.raw_response}")
        except Exception as e:
            print(f"Error checking image: {e}")

if __name__ == '__main__':
    test_brave_images_and_vision()
