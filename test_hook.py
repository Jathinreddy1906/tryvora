import requests
import time

print("Creating webhook...")
webhook_init = requests.post("https://webhook.site/token")
uuid = webhook_init.json()["uuid"]
callBackUrl = f"https://webhook.site/{uuid}"
get_url = f"https://webhook.site/token/{uuid}/requests"

print(f"Callback URL: {callBackUrl}")
print(f"Waiting 5 seconds to test polling...")
time.sleep(5)
r = requests.get(get_url)
print("Poll status:", r.status_code)
print("Poll response:", r.text[:200])
