import requests
taskId = "6a067034d929fbffe4bf9479214918f4"
headers = {"Authorization": "Bearer bf727677774148425720f24b229ae9e9"}
endpoints = [
    f"https://api.nanobananaapi.ai/api/v1/nanobanana/result/{taskId}",
    f"https://api.nanobananaapi.ai/api/v1/nanobanana/status/{taskId}",
    f"https://api.nanobananaapi.ai/api/v1/nanobanana/fetch/{taskId}",
    f"https://api.nanobananaapi.ai/api/v1/nanobanana/task/{taskId}",
    f"https://api.nanobananaapi.ai/api/v1/nanobanana/{taskId}",
    f"https://api.nanobananaapi.ai/api/v1/nanobanana/generate/{taskId}",
]
for e in endpoints:
    try:
        r = requests.get(e, headers=headers)
        print(f"{e} -> {r.status_code}")
        if r.status_code == 200:
            print("Response:", r.text[:200])
    except Exception as ex:
        print(ex)
