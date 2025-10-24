import os
import requests
from dotenv import load_dotenv

load_dotenv()

GROK_API_KEY = os.getenv("GROK_API_KEY")

def ask_grok(prompt):
    url = "https://api.x.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROK_API_KEY}"}
    data = {
        "model": "grok-beta",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0
    }
    response = requests.post(url, json=data, headers=headers)
    return response.json()["choices"][0]["message"]["content"]
