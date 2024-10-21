import requests
import time
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "who won cricket world cup 2011",
})

print(output)