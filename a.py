import os
import requests
import json
api_key = os.environ["GROQ_API_KEY"]
models = requests.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {api_key}"}).json()
print([m['id'] for m in models.get('data', [])])
