
import time
import requests

url = "http://localhost:11434/api/generate"

payload = {
    "model": "qwen2.5:1.5b",
    "prompt": "Explain artificial intelligence in three simple sentences.",
    "stream": False
}

start = time.perf_counter()

response = requests.post(url, json=payload, timeout=180)
response.raise_for_status()

result = response.json()

print("\nAI Response:")
print(result["response"])

print("\nResponse time:", round(time.perf_counter() - start, 2), "seconds")
