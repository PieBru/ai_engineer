from litellm import completion
import os

#LITELLM_MODEL = os.environ['LITELLM_MODEL'] = "cerebras/llama3.1-8b"
#LITELLM_MODEL = os.environ['LITELLM_MODEL'] = "cerebras/llama-3.3-70b"
#LITELLM_MODEL = os.environ['LITELLM_MODEL'] = "cerebras/qwen-3-32b"
LITELLM_MODEL = os.environ['LITELLM_MODEL'] = "cerebras/llama-4-scout-17b-16e-instruct"
os.environ['CEREBRAS_API_KEY'] = "csk-*****************************"
response = completion(
    model=LITELLM_MODEL,
    messages=[
        {
            "role": "user",
            "content": "What's the weather like in Boston today in Fahrenheit? (Write in JSON)",
        }
    ],
    max_tokens=10,
        
    # The prompt should include JSON if 'json_object' is selected; otherwise, you will get error code 400.
    response_format={ "type": "json_object" },
    seed=123,
    stop=["\n\n"],
    temperature=0.2,
    top_p=0.9,
    tool_choice="auto",
    tools=[],
    user="user",
)
print(response)
