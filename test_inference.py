import urllib.request
import json
import time

url = "http://ai-inference-alb-481845a4-42841387.us-east-1.elb.amazonaws.com/v1/chat/completions"

payload = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [
        {"role": "system", "content": "You are a helpful AI cloud architect assistant."},
        {"role": "user", "content": "Explain what a Bastion Host in AWS is and why it is placed in a Public Subnet while the GPU node is in a Private Subnet."}
    ],
    "max_tokens": 200,
    "temperature": 0.7
}

headers = {
    "Content-Type": "application/json"
}

print(f"Sending inference request to: {url}")
start_time = time.time()

req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers)
with urllib.request.urlopen(req) as response:
    latency = time.time() - start_time
    result = json.loads(response.read().decode('utf-8'))
    
print(f"Inference Latency: {latency:.3f} seconds\n")
print("Response JSON:")
print(json.dumps(result, indent=2))
print("\nGenerated Content:")
print(result["choices"][0]["message"]["content"])
