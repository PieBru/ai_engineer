
---
description: "Rules and available tools for file system operations."
author: "AI Engineer Team"
version: "1.0"
---
2. **File Operations:**
   The following tools are available for interacting with the file system:
   - read_file: Read a single file's content.
   - read_multiple_files: Read multiple files at once.
   - create_file: Create or overwrite a single file.
   - create_multiple_files: Create multiple files at once.
   - edit_file: Edit an existing file by replacing a specific snippet with new content.

**File Operation Specific Guidelines:**
- Always try to read files first (e.g., using `read_file`) before editing them to understand the context.
- For `edit_file`, use precise snippet matching.
- Clearly explain what changes you're making and why before making a tool call.
- Consider the impact of any changes on the overall codebase.
