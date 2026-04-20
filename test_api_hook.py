import requests
import json
import time

api_key = "bf727677774148425720f24b229ae9e9"
external_url = "https://api.nanobananaapi.ai/api/v1/nanobanana/generate"

webhook_init = requests.post("https://webhook.site/token")
uuid_str = webhook_init.json()["uuid"]
callBackUrl = f"https://webhook.site/{uuid_str}"

payload = {
    "prompt": "Perform a flawless virtual try-on.",
    "numImages": 1,
    "type": "TEXTTOIAMGE",
    "image_size": "9:16",
    "person_image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
    "clothing_image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
    "callBackUrl": callBackUrl
}

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

print("Firing API...")
r = requests.post(external_url, headers=headers, json=payload)
print(r.status_code, r.text)

print("Polling...")
for _ in range(30):
    time.sleep(2)
    poll_resp = requests.get(f"https://webhook.site/token/{uuid_str}/requests")
    if poll_resp.status_code == 200:
        data = poll_resp.json()
        if len(data['data']) > 0:
            print("\nGOT REQUEST:", data['data'][0]['content'])
            break
    print(".", end="", flush=True)
print("\nDone")
