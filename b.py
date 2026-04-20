import os
import requests

api_key = os.environ["GROQ_API_KEY"]
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
payload = {
    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe this."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="}}
    ]}]
}
r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
print(r.status_code)
print(r.text)
