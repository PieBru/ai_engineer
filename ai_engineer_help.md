 🎯 **Commands:**

*   **`/add`** `[path/to/file_or_folder]` - Add the content of a specific file or all files in a folder to the conversation context.
    Call `/add` without arguments for detailed usage and examples.
    *(Note: The AI can often read files automatically by mentioning them in your conversation.)*

*   **`/ask`** `<text>` - Treat the following text as a direct user prompt to the LLM.
    Call `/ask` without arguments for detailed usage.
    Example: `/ask Explain the concept of recursion.`

*   **`/context`** `[subcommand] [name/path]` - Manage conversation context (save, load, list, summarize).
    Call `/context` without arguments for detailed usage.
    Subcommands:
    - `save <name>`: Save current context to a file.
    - `load <name>`: Load context from a file.
    - `list [path]`: List saved contexts in a directory.
    - `summarize`: Summarize current context using the LLM.

*   **`/exit`** or **`/quit`** - End the current session.

*   **`/help`** - Display this detailed help information.

*   **`/prompt`** `<subcommand> <text>` - Tools to help craft better prompts for AI Engineer using the LLM.
    Call `/prompt` without arguments for detailed usage.
    Subcommands:
    - `refine <text>`: Optimizes `<text>` into a clearer and more effective prompt.
    - `detail <text>`: Expands `<text>` into a more comprehensive and detailed prompt.

*   **`/rules`** `<subcommand> [arguments]` - Manage the AI's guiding rules (system prompt).
    Call `/rules` without arguments for detailed usage.
    Subcommands:
    - `show`: Display the current system prompt (rules) being used.
    - `list`: List available rule files in the `./.aie_rules/` directory.
    - `add <rule-file>`: Add rules from a specified markdown file to the current session's system prompt.
    - `reset`: Empties the current system prompt, then asks for confirmation to load default rules from `./.aie_rules/_default.md`.

*   **`/script`** `<script_path>` - Execute a sequence of AI Engineer commands from the specified script file.
    Call `/script` without arguments for detailed usage.
    Example: `/script ./my_setup_script.aiescript`
    The script file contains AI Engineer commands, one per line. Lines starting with `#` are comments.

*   **`/session`** `[...]` - Alias for the `/context` command.

*   **`/set`** `[parameter] [value]` - Change configuration parameters for the current session.
    Call `/set` without arguments to list available parameters and see usage.
    Example: `/set model gpt-4o`
    Available parameters:
    - `model`: The language model to use.
    - `api_base`: The API base URL for the LLM provider.
    - `reasoning_style`: Controls display of AI's reasoning (`full`, `compact`, `silent`).
    - `max_tokens`: Maximum number of tokens for the LLM response.
    - `reasoning_effort`: Controls AI's internal thinking depth (`low`, `medium`, `high`).
    - `reply_effort`: Controls verbosity of AI's final reply (`low`, `medium`, `high`).
    - `temperature`: Controls randomness/creativity (0.0 to 2.0).
    - `system_prompt`: Path to a file whose content will replace the current system prompt.

*   **`/shell`** `[command]` - Execute a shell command and add its output to the conversation history.
    Call `/shell` without arguments for usage and examples.
    *(Warning: Executing arbitrary shell commands can be risky.)*


*   **`/test`** `<subcommand> [arguments]` - Run diagnostic tests.
    Call `/test` without arguments for detailed usage.
    Subcommands:
    - `all`: Run all available tests (currently runs 'inference').
    - `inference`: Test the LLM inference endpoint configuration and connectivity.

*   **`/time`** - Toggle the display of a timestamp in the user prompt.
    Example: `[Ctx: 5%, 16:04:52] You>`

*   **Just ask naturally:** The AI will handle file operations automatically by mentioning files in your conversation.