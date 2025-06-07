from litellm import completion
import os

os.environ['GEMINI_API_KEY'] = "AI**************************"
response = completion(
    model="gemini/gemini-1.5-pro-latest", 
    messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}]
)

