* **`/set [parameter] [value]`** - Change configuration parameters for the current session. Call /set without arguments to list available parameters and see usage. Example: `/set model "ollama/qwen3:32b"`.
Available parameters:
   * **`model`**: The language model to use.
   * **`api_base`**: The API base URL for the LLM provider.
   * **`reasoning_style`**: Controls display of AI's reasoning (full, compact, silent).
   * **`max_tokens`**: Maximum number of tokens for the LLM response.
   * **`reasoning_effort`**: Controls AI's internal thinking depth (low, medium, high).
   * **`reply_effort`**: Controls verbosity of AI's final reply (low, medium, high).
   * **`temperature`**: Controls randomness/creativity (0.0 to 2.0).
   * **`system_prompt`**: Path to a file whose content will replace the current system prompt.
