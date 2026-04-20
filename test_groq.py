import os
import requests
import base64

api_key = os.environ["GROQ_API_KEY"]
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

payload = {
    "model": "llama-3.2-11b-vision-preview",
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe this clothing. Return ONLY JSON: {\"prompt\": \"...\"}"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="}}
    ]}]
}

r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
print(r.status_code)
print(r.text)
