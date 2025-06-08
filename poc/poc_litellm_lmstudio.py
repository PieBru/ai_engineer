from litellm import completion
import os

os.environ['LM_STUDIO_API_BASE'] = "http://localhost:1234/v1" # Ensure this points to your LM Studio server with /v1

response = completion(
    model="lm_studio/deepseek-r1-0528-qwen3-8b@q8_0",
    api_key="dummy", # Add a dummy API key
    messages=[
        {
            "role": "user",
            "content": "What's the weather like in Boston today in Fahrenheit?",
        }
    ]
)
print(response)