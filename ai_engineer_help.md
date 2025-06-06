 🎯 **Commands:**
*   `/exit` or `/quit` - End the session.
*   `/add [path/to/file_or_folder]` - Add file/folder to context.
    *   Call `/add` without arguments for detailed usage.
    *   The AI can automatically read and create files using function calls.
*   `/set [parameter] [value]` - Change configuration for the current session.
    *   Call `/set` without arguments to list available parameters and see usage.
    *   Example: `/set reasoning_style compact`
    *   Available: `model`, `api_base`, `reasoning_style`, `max_tokens`, `reasoning_effort`, `reply_effort`, `temperature`.
*   `/help` - Display this detailed help information.
*   `/shell [command]` - Execute a shell command and add output to history.
    *   Call `/shell` without arguments for usage.
    *   Example: `/shell ls -l`
*   `/context [subcommand] [name/path]` - Manage conversation context.
    *   Call `/context` without arguments for detailed usage.
    *   Subcommands:
        *   `save <name>`     - Save current context to a file.
        *   `load <name>`     - Load context from a file.
        *   `list [path]`     - List saved contexts in a directory.
        *   `summarize`       - Summarize current context using the LLM.
*   `/session [...]` - Alias for `/context [...]`.
*   **Just ask naturally** - The AI will handle file operations automatically!
