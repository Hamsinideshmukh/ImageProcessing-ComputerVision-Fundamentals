import requests 
url = "http://127.0.0.1:8000/classify"

with open("input.png", "rb") as f:
    response = requests.post(
        url,
        files={"file":f}
    )
print(response.json())
