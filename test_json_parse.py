import json
import re

text = """Here's a thinking process:
1. **Analyze the Request:**
   - **Channel:** TechByte Daily
   - We should use {"example": 1} in our reasoning.

```json
{
  "audience_fit": 80,
  "curiosity_gap": 90
}
```
"""

def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()

clean = _strip_fences(text)
m = re.search(r"\{.*\}", clean, flags=re.DOTALL)
if m:
    print("Greedy match:", m.group(0))
    try:
        print("Parsed greedy:", json.loads(m.group(0)))
    except Exception as e:
        print("Greedy parse failed:", e)

m2 = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
if m2:
    print("Fenced match:", m2.group(1))
    try:
        print("Parsed fenced:", json.loads(m2.group(1)))
    except Exception as e:
        print("Fenced parse failed:", e)
