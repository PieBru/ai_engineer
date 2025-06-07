import os
import litellm
from litellm import completion

# model_name = "ollama_chat/llama3.2"
# model_name = "ollama_chat/qwq"
model_name = "ollama_chat/deepcoder:14b-preview-q8_0"

# os.environ['DEEPSEEK_API_KEY'] = "sk-******************"
# model_name = "deepseek/deepseek-reasoner"
# model_name = "deepseek/deepseek-chat"
# model_name = "deepseek/deepseek-coder"

def chat_with_model():
    print("Chat with LiteLLM (type 'exit' to quit)")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        try:
            response = completion(
                model=model_name,
                messages=[{
                    "content": user_input,
                    "role": "user"
                }]
            )
            print(f"\nModel: {response.choices[0].message.content}")
            
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    chat_with_model()
