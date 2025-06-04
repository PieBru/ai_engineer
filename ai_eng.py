#!/usr/bin/env python3
"""
AI Engineer: An AI-powered coding assistant.

This script provides an interactive terminal interface for code development,
leveraging AI's reasoning models for intelligent file operations,
code analysis, and development assistance via natural conversation and function calling.
"""

import os
import sys
import json
from pathlib import Path
from textwrap import dedent # Used for system_PROMPT
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.style import Style
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style as PromptStyle
import time
import copy # For deepcopy
import httpx # For new MCP tools

# Import litellm
from litellm import completion

# Initialize Rich console and prompt session
console = Console()
prompt_session = PromptSession(
    style=PromptStyle.from_dict({
        'prompt': '#0066ff bold',  # Bright blue prompt
        'completion-menu.completion': 'bg:#1e3a8a fg:#ffffff',
        'completion-menu.completion.current': 'bg:#3b82f6 fg:#ffffff bold',
    })
)

# --------------------------------------------------------------------------------
# 1. Configure LLM client settings and load environment variables
# --------------------------------------------------------------------------------
load_dotenv()  # Load environment variables from .env file

# Define module-level constants for limits
MAX_FILES_TO_PROCESS_IN_DIR = 1000
MAX_FILE_SIZE_BYTES = 5_000_000  # 5MB


# --------------------------------------------------------------------------------
# 2. Define our schema using Pydantic for type safety
# --------------------------------------------------------------------------------
class FileToCreate(BaseModel):
    path: str
    content: str

    model_config = ConfigDict(extra='ignore', frozen=True)

class FileToEdit(BaseModel):
    path: str
    original_snippet: str
    new_snippet: str

    model_config = ConfigDict(extra='ignore', frozen=True)

# Remove AssistantResponse as we're using function calling now

# --------------------------------------------------------------------------------
# 2.1. Define Function Calling Tools
# --------------------------------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the content of a single file from the filesystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to read (relative or absolute)",
                    }
                },
                "required": ["file_path"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_multiple_files",
            "description": "Read the content of multiple files from the filesystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of file paths to read (relative or absolute)",
                    }
                },
                "required": ["file_paths"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a new file or overwrite an existing file with the provided content",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path where the file should be created",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file",
                    }
                },
                "required": ["file_path", "content"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_multiple_files",
            "description": "Create multiple files at once",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"}
                            },
                            "required": ["path", "content"]
                        },
                        "description": "Array of files to create with their paths and content",
                    }
                },
                "required": ["files"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit an existing file by replacing a specific snippet with new content",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to edit",
                    },
                    "original_snippet": {
                        "type": "string",
                        "description": "The exact text snippet to find and replace",
                    },
                    "new_snippet": {
                        "type": "string",
                        "description": "The new text to replace the original snippet with",
                    }
                },
                "required": ["file_path", "original_snippet", "new_snippet"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "connect_local_mcp_stream",
            "description": "Connects to a local MCP server endpoint that provides a streaming response. Reads the stream and returns the aggregated data. Primarily for localhost or 127.0.0.1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "endpoint_url": {
                        "type": "string",
                        "description": "The full URL of the local MCP streaming endpoint (e.g., http://localhost:8000/stream)."
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Timeout in seconds for the connection and for reading the entire stream (default: 30).",
                        "default": 30
                    },
                    "max_data_chars": {
                        "type": "integer",
                        "description": "Maximum number of characters to read from the stream before truncating (default: 10000).",
                        "default": 10000
                    }
                },
                "required": ["endpoint_url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "connect_remote_mcp_sse",
            "description": "Connects to a remote MCP server endpoint using Server-Sent Events (SSE). Listens for events and returns a summary of received events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "endpoint_url": {
                        "type": "string",
                        "description": "The full URL of the remote MCP SSE endpoint."
                    },
                    "max_events": {
                        "type": "integer",
                        "description": "Maximum number of SSE events to process before closing the connection (default: 10).",
                        "default": 10
                    },
                    "listen_timeout_seconds": {
                        "type": "integer",
                        "description": "Timeout in seconds for the connection and for listening to events (default: 60).",
                        "default": 60
                    }
                },
                "required": ["endpoint_url"]
            }
        }
    }
]

# --------------------------------------------------------------------------------
# 3. system prompt
# --------------------------------------------------------------------------------

# """
# Original system_PROMPT (before Code Citation and refined Guidelines):
# system_PROMPT = dedent(\"""\
#     You are DeepSeek Engineer, a helpful and elite software engineering assistant.
#     Your expertise spans system design, algorithms, testing, and best practices.
#     You provide thoughtful, well-structured solutions while explaining your reasoning.
#
#     Core capabilities:
#     0. Conversational Interaction:
#        - Engage in natural conversation. For simple greetings or chit-chat, respond conversationally without using tools.
#     1. Code Analysis & Discussion
#        - Analyze code with expert-level insight
#        - Explain complex concepts clearly
#        - Suggest optimizations and best practices
#        - Debug issues with precision
#
#     2. File Operations (via function calls):
#        - read_file: Read a single file's content
#        - read_multiple_files: Read multiple files at once
#        - create_file: Create or overwrite a single file
#        - create_multiple_files: Create multiple files at once
#        - edit_file: Make precise edits to existing files using snippet replacement
#
#     Guidelines:
#     1. Provide natural, conversational responses explaining your reasoning
#     2. Use function calls *only when necessary* to read or modify files, or to perform other specified tool actions.
#     3. For file operations:
#        - Always read files first before editing them to understand the context
#        - Use precise snippet matching for edits
#        - Explain what changes you're making and why
#        - Consider the impact of changes on the overall codebase
#     4. Follow language-specific best practices
#     5. Suggest tests or validation steps when appropriate
#     6. Be thorough in your analysis and recommendations
#
#     IMPORTANT: If a user's request clearly requires a file operation or another tool, proceed to the tool call. For ambiguous or simple conversational inputs (like a greeting), prioritize a direct conversational response.
#
#     Remember: You're a senior engineer - be thoughtful, precise, and explain your reasoning clearly.
# \""")
# """

system_PROMPT = dedent("""\
    You are AI Engineer, a helpful and elite software engineering assistant.
    Your expertise spans system design, algorithms, testing, and best practices.
    You provide thoughtful, well-structured solutions while explaining your reasoning.

    Core capabilities:
    0. Conversational Interaction:
       - Engage in natural conversation. For simple greetings or chit-chat, respond conversationally without using tools.
    1. Code Analysis & Discussion:
       - Analyze code with expert-level insight.
       - Explain complex concepts clearly.
       - Suggest optimizations and best practices.
       - Debug issues with precision.

    Code Citation Format:
    When citing code regions or blocks in your responses, you MUST use the following format:
    ```startLine:endLine:filepath
    // ... existing code ...
    ```
    Example:
    ```12:15:app/components/Todo.tsx
    // ... existing code ...
    ```
    - startLine: The starting line number (inclusive)
    - endLine: The ending line number (inclusive)
    - filepath: The complete path to the file
    - The code block should be enclosed in triple backticks.
    - Use "// ... existing code ..." to indicate omitted code sections.

    2. File Operations (via function calls):
       The following tools are available for interacting with the file system:
       - read_file: Read a single file's content.
       - read_multiple_files: Read multiple files at once.
       - create_file: Create or overwrite a single file.
       - create_multiple_files: Create multiple files at once.
       - edit_file: Make precise edits to existing files using snippet replacement.

    3. Network Operations (via function calls):
       The following tools are available for network interactions:
       - connect_local_mcp_stream: Connects to a local (localhost or 127.0.0.1) MCP server endpoint that provides a streaming HTTP response. Returns the aggregated data from the stream.
         Example: Fetching logs or real-time metrics from a local development server.
         The 'endpoint_url' must start with 'http://localhost' or 'http://127.0.0.1'.
       - connect_remote_mcp_sse: Connects to a remote MCP server endpoint using Server-Sent Events (SSE) over HTTP/HTTPS. Returns a summary of received events.
         Example: Monitoring status updates or notifications from a remote service.
         The 'endpoint_url' must be a valid HTTP or HTTPS URL.


    General Guidelines:
    1. Provide natural, conversational responses, always explaining your reasoning.
    2. Use function calls (tools) *only when necessary* and after explaining your intent.
    3. For file operations:
       - Always try to read files first (e.g., using `read_file`) before editing them to understand the context.
       - For `edit_file`, use precise snippet matching.
       - Clearly explain what changes you're making and why.
       - Consider the impact of any changes on the overall codebase.
    4. All file paths provided to tools can be relative or absolute.
    5. Explanations for tool use should be clear and concise (ideally one sentence).
    6. Tool calls must include all required parameters. Optional parameters should only be included when necessary.
    7. When tool parameters require values provided by the user (e.g., a file path), use the exact values given.
    8. Follow language-specific best practices in your code suggestions and analysis.
    9. Suggest tests or validation steps when appropriate.
    10. Be thorough in your analysis and recommendations.
    11. For Network Operations:
        - Clearly state the purpose of connecting to an endpoint.
        - Use `connect_local_mcp_stream` only for `http://localhost...` or `http://127.0.0.1...` URLs.
        - Be mindful of potential timeouts or if the service is not running.
        - The data returned will be a text summary or aggregation.
        - When `connect_local_mcp_stream` returns data, if it appears to be structured (e.g., JSON lines, logs), try to parse and summarize it meaningfully. If it's unstructured text, summarize its main content.
        - After `connect_remote_mcp_sse` provides a summary of events, analyze these events in the context of the user's original request. For example, if the user asked about a service's status, try to infer the status from the events.
    12. If a tool operation is cancelled by the user (indicated by a tool message like 'User cancelled execution...'), acknowledge the cancellation and ask the user for new instructions or how they would like to proceed. Do not re-attempt the cancelled operation unless explicitly asked to by the user.

    IMPORTANT: If a user's request clearly requires a file operation or another tool, proceed to the tool call. For ambiguous or simple conversational inputs (like a greeting), prioritize a direct conversational response.

    Remember: You're a senior engineer - be thoughtful, precise, and explain your reasoning clearly.
""")

# --------------------------------------------------------------------------------
# 4. Helper functions 
# --------------------------------------------------------------------------------

def _handle_local_mcp_stream(endpoint_url: str, timeout_seconds: int, max_data_chars: int) -> str:
    """
    Connects to a local MCP server endpoint that provides a streaming response.
    Reads the stream and returns the aggregated data.
    """
    # URL validation is expected to be done by the caller (execute_function_call_dict)
    # based on the schema, but an extra check here is fine.
    try:
        parsed_url = httpx.URL(endpoint_url)
        if not (parsed_url.host.lower() in ("localhost", "127.0.0.1") and parsed_url.scheme.lower() in ("http", "https")):
             return f"Error: For connect_local_mcp_stream, endpoint_url must be for localhost (http or https). Provided: {endpoint_url}"
    except httpx.UnsupportedProtocol:
         return f"Error: Invalid or unsupported URL scheme for local MCP stream: {endpoint_url}"
    except Exception as e_val: # Catch other potential parsing errors
        return f"Error validating local MCP stream URL '{endpoint_url}': {str(e_val)}"

    data_buffer = []
    chars_read = 0
    timeout_config = httpx.Timeout(timeout_seconds, read=timeout_seconds)

    try:
        with httpx.Client(timeout=timeout_config) as client:
            with client.stream("GET", endpoint_url) as response:
                response.raise_for_status() # Check for HTTP errors like 4xx/5xx
                for chunk in response.iter_text():
                    if chars_read + len(chunk) > max_data_chars:
                        remaining_len = max_data_chars - chars_read
                        data_buffer.append(chunk[:remaining_len])
                        data_buffer.append("... (data truncated)")
                        chars_read = max_data_chars
                        break
                    data_buffer.append(chunk)
                    chars_read += len(chunk)
        return "".join(data_buffer)
    except httpx.HTTPStatusError as e:
        error_detail = f"HTTP {e.response.status_code}"
        try:
            # Ensure the response body is read before accessing .text for streaming responses
            e.response.read()
            error_detail += f" - {e.response.text[:200]}"
        except httpx.ResponseNotRead:
            error_detail += " - (Error response body could not be read for details)"
        except Exception: # Catch any other issues during error body reading
            error_detail += " - (Failed to retrieve error response body details)"
        return f"Error connecting to local MCP stream '{endpoint_url}': {error_detail}"
    except httpx.RequestError as e:
        return f"Error connecting to local MCP stream '{endpoint_url}': {str(e)}"
    except Exception as e:
        # This will catch unexpected errors, including ResponseNotRead if it happens outside HTTPStatusError handling
        return f"An unexpected error occurred with local MCP stream '{endpoint_url}': {str(e)}"

def _handle_remote_mcp_sse(endpoint_url: str, max_events: int, listen_timeout_seconds: int) -> str:
    """
    Connects to a remote MCP server endpoint using Server-Sent Events (SSE).
    Listens for events and returns a summary of received events.
    """
    # URL validation is expected to be done by the caller (execute_function_call_dict)
    try:
        parsed_url = httpx.URL(endpoint_url)
        if not parsed_url.scheme.lower() in ("http", "https"):
            return f"Error: For connect_remote_mcp_sse, endpoint_url must be a valid HTTP/HTTPS URL. Provided: {endpoint_url}"
    except httpx.UnsupportedProtocol:
        return f"Error: Invalid or unsupported URL scheme for remote MCP SSE: {endpoint_url}"
    except Exception as e_val:
        return f"Error validating remote MCP SSE URL '{endpoint_url}': {str(e_val)}"

    events_received = []
    timeout_config = httpx.Timeout(listen_timeout_seconds, read=listen_timeout_seconds)

    try:
        with httpx.Client(timeout=timeout_config) as client:
            with client.stream("GET", endpoint_url, headers={"Accept": "text/event-stream"}) as response:
                response.raise_for_status()
                if "text/event-stream" not in response.headers.get("Content-Type", "").lower():
                    return f"Error: Endpoint '{endpoint_url}' did not return 'text/event-stream' content type. Got: {response.headers.get('Content-Type')}"

                current_event_data = []
                for line in response.iter_lines():
                    if not line: # Empty line signifies end of an event
                        if current_event_data:
                            events_received.append("".join(current_event_data))
                            current_event_data = []
                            if len(events_received) >= max_events:
                                break
                    elif line.startswith("data:"):
                        current_event_data.append(line[5:].strip() + "\n")
                
                if current_event_data and len(events_received) < max_events: # Process any trailing data
                    events_received.append("".join(current_event_data).strip())

        summary = f"Received {len(events_received)} SSE event(s) from '{endpoint_url}'.\n"
        summary += "Last few events:\n" + "\n---\n".join(events_received[-5:]) # Show last 5 events
        return summary
    except httpx.HTTPStatusError as e:
        error_detail = f"HTTP {e.response.status_code}"
        try:
            # Ensure the response body is read before accessing .text for streaming responses
            e.response.read()
            error_detail += f" - {e.response.text[:200]}"
        except httpx.ResponseNotRead:
            error_detail += " - (Error response body could not be read for details)"
        except Exception: # Catch any other issues during error body reading
            error_detail += " - (Failed to retrieve error response body details)"
        return f"Error connecting to remote MCP SSE '{endpoint_url}': {error_detail}"
    except httpx.RequestError as e:
        return f"Error connecting to remote MCP SSE '{endpoint_url}': {str(e)}"
    except Exception as e:
        # This will catch unexpected errors
        return f"An unexpected error occurred with remote MCP SSE '{endpoint_url}': {str(e)}"

def read_local_file(file_path: str) -> str:
    """Return the text content of a local file.
    Raises FileNotFoundError or OSError on issues.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise
    except OSError:
        raise

def create_file(path: str, content: str):
    """Create (or overwrite) a file at 'path' with the given 'content'."""
    # Policy: Disallow direct tilde usage in paths for creation for security/explicitness.
    # normalize_path would expand it, but we want to prevent it at this stage for this function.
    if path.lstrip().startswith('~'):
        raise ValueError("Home directory references not allowed")

    try:
        absolute_normalized_path_str = normalize_path(path)
    except ValueError as e: # Catch errors from normalize_path (e.g., ".." or other Path issues)
        console.print(f"[bold red]✗[/bold red] Could not create file. Invalid path: '[bright_cyan]{path}[/bright_cyan]'. Error: {e}")
        raise ValueError(f"Invalid path for create_file: {path}. Details: {e}") from e

    normalized_file_path_obj = Path(absolute_normalized_path_str)

    if len(content) > MAX_FILE_SIZE_BYTES:
        err_msg = f"File content exceeds {MAX_FILE_SIZE_BYTES // (1024*1024)}MB size limit"
        console.print(f"[bold red]✗[/bold red] {err_msg}")
        raise ValueError(err_msg)
    
    try:
        normalized_file_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(normalized_file_path_obj, "w", encoding="utf-8") as f:
            f.write(content)
        console.print(f"[bold blue]✓[/bold blue] Created/updated file at '[bright_cyan]{normalized_file_path_obj}[/bright_cyan]'")
    except OSError as e:
        err_msg = f"Failed to write file '{normalized_file_path_obj}': {e}"
        console.print(f"[bold red]✗[/bold red] {err_msg}")
        raise OSError(err_msg) from e

def show_diff_table(files_to_edit: List[FileToEdit]) -> None:
    if not files_to_edit:
        return
    
    table = Table(title="📝 Proposed Edits", show_header=True, header_style="bold bright_blue", show_lines=True, border_style="blue")
    table.add_column("File Path", style="bright_cyan", no_wrap=True)
    table.add_column("Original", style="red dim")
    table.add_column("New", style="bright_green")

    for edit in files_to_edit:
        table.add_row(edit.path, edit.original_snippet, edit.new_snippet)
    
    console.print(table)

def apply_diff_edit(path: str, original_snippet: str, new_snippet: str):
    """Reads the file at 'path', replaces the first occurrence of 'original_snippet' with 'new_snippet', then overwrites."""
    content = "" # Initialize content
    try:
        content = read_local_file(path) # Can raise FileNotFoundError, OSError
        
        # Verify we're replacing the exact intended occurrence
        occurrences = content.count(original_snippet)
        if occurrences == 0:
            raise ValueError("Original snippet not found")
        if occurrences > 1:
            console.print(f"[bold yellow]⚠ Multiple matches ({occurrences}) found - requiring line numbers for safety. Snippet found in '{path}'[/bold yellow]")
            console.print("[dim]Use format:\n--- original.py (lines X-Y)\n+++ modified.py[/dim]")
            raise ValueError(f"Ambiguous edit: {occurrences} matches")
        
        updated_content = content.replace(original_snippet, new_snippet, 1)
        
        # create_file can raise ValueError (bad path, size limit) or OSError (write error)
        create_file(path, updated_content)

        console.print(f"[bold blue]✓[/bold blue] Applied diff edit to '[bright_cyan]{path}[/bright_cyan]'")

    except FileNotFoundError: # From read_local_file
        msg = f"File not found for diff editing: '{path}'"
        console.print(f"[bold red]✗[/bold red] {msg}")
        raise FileNotFoundError(msg) from None # Propagate for tool
    except ValueError as e: # From snippet check, or from create_file (via normalize_path or size check)
        err_msg = str(e)
        console.print(f"[bold yellow]⚠[/bold yellow] {err_msg} in '[bright_cyan]{path}[/bright_cyan]'. No changes made.")
        if "Original snippet not found" in err_msg or "Ambiguous edit" in err_msg:
            # Avoid printing panels if snippet wasn't found (content might be empty) or if content is not available
            if "Original snippet not found" not in err_msg and content:
                console.print("\n[bold blue]Expected snippet:[/bold blue]")
                console.print(Panel(original_snippet, title="Expected", border_style="blue", title_align="left"))
                console.print("\n[bold blue]Actual file content:[/bold blue]")
                console.print(Panel(content, title="Actual", border_style="yellow", title_align="left"))
        # Always re-raise ValueError so the tool call reports an error
        raise ValueError(f"Failed to apply diff to '{path}': {err_msg}") from e
    except OSError as e: # From read_local_file or create_file
        err_msg = str(e)
        console.print(f"[bold red]✗[/bold red] OS error during diff edit for '{path}': {err_msg}")
        raise OSError(f"OS error during diff edit for '{path}': {err_msg}") from e

def try_handle_add_command(user_input: str) -> bool:
    prefix = "/add "
    if user_input.strip().lower().startswith(prefix):
        path_to_add = user_input[len(prefix):].strip()
        try:
            normalized_path = normalize_path(path_to_add)
            if os.path.isdir(normalized_path):
                # Handle entire directory
                add_directory_to_conversation(normalized_path)
            else:
                # Handle a single file as before
                content = read_local_file(normalized_path)
                conversation_history.append({
                    "role": "system",
                    "content": f"Content of file '{normalized_path}':\n\n{content}"
                })
                console.print(f"[bold blue]✓[/bold blue] Added file '[bright_cyan]{normalized_path}[/bright_cyan]' to conversation.\n")
        except OSError as e:
            console.print(f"[bold red]✗[/bold red] Could not add path '[bright_cyan]{path_to_add}[/bright_cyan]': {e}\n")
        return True
    return False

def add_directory_to_conversation(directory_path: str):
    with console.status("[bold bright_blue]🔍 Scanning directory...[/bold bright_blue]") as status:
        excluded_files = {
            # Python specific
            ".DS_Store", "Thumbs.db", ".gitignore", ".python-version",
            "uv.lock", ".uv", "uvenv", ".uvenv", ".venv", "venv",
            "__pycache__", ".pytest_cache", ".coverage", ".mypy_cache",
            # Node.js / Web specific
            "node_modules", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
            ".next", ".nuxt", "dist", "build", ".cache", ".parcel-cache",
            ".turbo", ".vercel", ".output", ".contentlayer",
            # Build outputs
            "out", "coverage", ".nyc_output", "storybook-static",
            # Environment and config
            ".env", ".env.local", ".env.development", ".env.production",
            # Misc
            ".git", ".svn", ".hg", "CVS"
        }
        excluded_extensions = {
            # Binary and media files
            ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp", ".avif",
            ".mp4", ".webm", ".mov", ".mp3", ".wav", ".ogg",
            ".zip", ".tar", ".gz", ".7z", ".rar",
            ".exe", ".dll", ".so", ".dylib", ".bin",
            # Documents
            ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
            # Python specific
            ".pyc", ".pyo", ".pyd", ".egg", ".whl",
            # UV specific
            ".uv", ".uvenv",
            # Database and logs
            ".db", ".sqlite", ".sqlite3", ".log",
            # IDE specific
            ".idea", ".vscode",
            # Web specific
            ".map", ".chunk.js", ".chunk.css",
            ".min.js", ".min.css", ".bundle.js", ".bundle.css",
            # Cache and temp files
            ".cache", ".tmp", ".temp",
            # Font files
            ".ttf", ".otf", ".woff", ".woff2", ".eot"
        }
        skipped_files = []
        added_files = []
        total_files_processed = 0

        for root, dirs, files in os.walk(directory_path):
            if total_files_processed >= MAX_FILES_TO_PROCESS_IN_DIR:
                console.print(f"[bold yellow]⚠[/bold yellow] Reached maximum file limit ({MAX_FILES_TO_PROCESS_IN_DIR})")
                break

            status.update(f"[bold bright_blue]🔍 Scanning {root}...[/bold bright_blue]")
            # Skip hidden directories and excluded directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in excluded_files]

            for file in files:
                if total_files_processed >= MAX_FILES_TO_PROCESS_IN_DIR:
                    break

                if file.startswith('.') or file in excluded_files:
                    skipped_files.append(str(Path(root) / file)) # Use Path for consistency
                    continue

                _, ext = os.path.splitext(file)
                if ext.lower() in excluded_extensions:
                    skipped_files.append(os.path.join(root, file))
                    continue

                full_path = str(Path(root) / file) # Use Path for consistency

                try:
                    # Check file size before processing
                    if os.path.getsize(full_path) > MAX_FILE_SIZE_BYTES:
                        skipped_files.append(f"{full_path} (exceeds size limit)")
                        continue

                    # Check if it's binary
                    if is_binary_file(full_path):
                        skipped_files.append(full_path)
                        continue

                    normalized_path = normalize_path(full_path)
                    content = read_local_file(normalized_path)
                    conversation_history.append({
                        "role": "system",
                        "content": f"Content of file '{normalized_path}':\n\n{content}"
                    })
                    added_files.append(normalized_path)
                    total_files_processed += 1

                except OSError:
                    skipped_files.append(str(full_path)) # Ensure it's a string

        console.print(f"[bold blue]✓[/bold blue] Added folder '[bright_cyan]{directory_path}[/bright_cyan]' to conversation.")
        if added_files:
            console.print(f"\n[bold bright_blue]📁 Added files:[/bold bright_blue] [dim]({len(added_files)} of {total_files_processed})[/dim]")
            for f in added_files:
                console.print(f"  [bright_cyan]📄 {f}[/bright_cyan]")
        if skipped_files:
            console.print(f"\n[bold yellow]⏭ Skipped files:[/bold yellow] [dim]({len(skipped_files)})[/dim]")
            for f in skipped_files[:10]:  # Show only first 10 to avoid clutter
                console.print(f"  [yellow dim]⚠ {f}[/yellow dim]")
            if len(skipped_files) > 10:
                console.print(f"  [dim]... and {len(skipped_files) - 10} more[/dim]")
        console.print()

def is_binary_file(file_path: str, peek_size: int = 1024) -> bool:
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(peek_size)
        # If there is a null byte in the sample, treat it as binary
        if b'\0' in chunk:
            return True
        return False
    except Exception:
        # If we fail to read, just treat it as binary to be safe
        return True

def ensure_file_in_context(file_path: str) -> bool:
    try:
        normalized_path = normalize_path(file_path)
        content = read_local_file(normalized_path)
        file_marker = f"Content of file '{normalized_path}'"
        if not any(file_marker in msg["content"] for msg in conversation_history):
            conversation_history.append({
                "role": "system",
                "content": f"{file_marker}:\n\n{content}"
            })
        return True
    except OSError:
        console.print(f"[bold red]✗[/bold red] Could not read file '[bright_cyan]{file_path}[/bright_cyan]' for editing context")
        return False

def normalize_path(path_str: str) -> str:
    """Return a canonical, absolute version of the path with security checks."""
    try:
        if not path_str:
            raise ValueError("Path cannot be empty.")
        # Create a Path object and expand tilde (e.g., ~ -> /home/user)
        expanded_path = Path(path_str).expanduser()

        # Policy: Disallow ".." in the path components provided by the user (after tilde expansion)
        # This is a surface-level check before full resolution.
        # Path.resolve() will handle ".." correctly to produce a canonical path,
        # but this check adds a layer of explicitness against using ".." components.
        if ".." in expanded_path.parts: # Check parts of the potentially relative path
            raise ValueError(f"Invalid path: {path_str} contains parent directory references")

        # Resolve the path to an absolute path, simplifying "." and ".." (if any slipped through or are legitimate)
        # and resolving symbolic links.
        resolved_path = expanded_path.resolve()
        return str(resolved_path)
    except (TypeError, ValueError) as e: # Catch Path construction errors or our ".." ValueError
        raise ValueError(f"Invalid path: \"{path_str}\". Error: {e}") from e
    except Exception as e: # Catch other unexpected errors during path operations
        raise ValueError(f"Error normalizing path: \"{path_str}\". Details: {e}") from e

# --------------------------------------------------------------------------------
# 5. Conversation state
# --------------------------------------------------------------------------------
conversation_history = [
    {"role": "system", "content": system_PROMPT}
]

# --------------------------------------------------------------------------------
# 6. LLM API interaction with streaming
# --------------------------------------------------------------------------------

RISKY_TOOLS = {"create_file", "create_multiple_files", "edit_file", "connect_remote_mcp_sse"}

def execute_function_call_dict(tool_call_dict) -> str:
    """Execute a function call from a dictionary format and return the result as a string."""
    function_name = "unknown_function" # Default if parsing fails early
    try:
        function_name = tool_call_dict["function"]["name"]
        arguments = json.loads(tool_call_dict["function"]["arguments"])
        
        if function_name == "read_file":
            file_path = arguments["file_path"]
            normalized_path = normalize_path(file_path)
            content = read_local_file(normalized_path)
            return f"Content of file '{normalized_path}':\n\n{content}"
            
        elif function_name == "read_multiple_files":
            file_paths = arguments["file_paths"]
            results = []
            for file_path in file_paths:
                try:
                    normalized_path = normalize_path(file_path)
                    content = read_local_file(normalized_path)
                    results.append(f"Content of file '{normalized_path}':\n\n{content}")
                except OSError as e:
                    results.append(f"Error reading '{file_path}': {e}")
            return "\n\n" + "="*50 + "\n\n".join(results)
            
        elif function_name == "create_file":
            file_path = arguments["file_path"]
            content = arguments["content"]
            create_file(file_path, content)
            return f"Successfully created file '{file_path}'"
            
        elif function_name == "create_multiple_files":
            files_to_create_data = arguments.get("files", [])
            successful_paths = []
            success_messages_for_result = [] # For "File ... created." part in return
            first_error_detail_for_return = None 

            for file_info_data in files_to_create_data:
                path = file_info_data.get("path", "unknown_path")
                content = file_info_data.get("content", "")
                try:
                    # In the test TestHelperFunctions.test_execute_function_call_dict,
                    # de.create_file is mocked (as mock_create).
                    # The mock's side_effect is [None, Exception("Create multiple sub-error")]
                    # So, the first call to de.create_file("f_ok.txt", "c_ok") will not raise.
                    # The second call to de.create_file("f_err.txt", "c_err") will raise Exception.
                    create_file(path, content)  # Changed de.create_file to create_file
                    successful_paths.append(path)
                    success_messages_for_result.append(f"File {path} created.")
                except Exception as e_create:
                    # This console print is asserted by the test for the failing file.
                    console.print(f"[red]Error creating file {path}: {str(e_create)}[/red]")
                    if not first_error_detail_for_return:
                        first_error_detail_for_return = f"Error during create_multiple_files: {str(e_create)}"
                    # Test implies processing continues for other files if any.
            
            if first_error_detail_for_return:
                return "\n".join(success_messages_for_result) + "\n" + first_error_detail_for_return if success_messages_for_result else first_error_detail_for_return
            
            return f"Successfully created {len(successful_paths)} files: {', '.join(successful_paths)}"
            
        elif function_name == "edit_file":
            file_path = arguments["file_path"]
            original_snippet = arguments["original_snippet"]
            new_snippet = arguments["new_snippet"]
            
            # Ensure file is in context first
            if not ensure_file_in_context(file_path):
                return f"Error: Could not read file '{file_path}' for editing"
            
            apply_diff_edit(file_path, original_snippet, new_snippet)
            return f"Successfully edited file '{file_path}'"
        
        elif function_name == "connect_local_mcp_stream":
            endpoint_url = arguments["endpoint_url"]
            timeout_seconds = arguments.get("timeout_seconds", 30) # Default from schema
            max_data_chars = arguments.get("max_data_chars", 10000) # Default from schema
            
            # Basic validation before calling the handler, complementing handler's own validation
            try:
                parsed_url = httpx.URL(endpoint_url)
                if not (parsed_url.host.lower() in ("localhost", "127.0.0.1") and parsed_url.scheme.lower() in ("http", "https")):
                     return f"Error: For connect_local_mcp_stream, endpoint_url must be for localhost (http or https). Provided: {endpoint_url}"
            except Exception as e_val:
                return f"Error validating local MCP stream URL '{endpoint_url}' before execution: {str(e_val)}"
            
            return _handle_local_mcp_stream(endpoint_url, timeout_seconds, max_data_chars)

        elif function_name == "connect_remote_mcp_sse":
            endpoint_url = arguments["endpoint_url"]
            max_events = arguments.get("max_events", 10) # Default from schema
            listen_timeout_seconds = arguments.get("listen_timeout_seconds", 60) # Default from schema

            # Basic validation before calling the handler
            try:
                parsed_url = httpx.URL(endpoint_url)
                if not parsed_url.scheme.lower() in ("http", "https"): # Allow http and https
                    return f"Error: For connect_remote_mcp_sse, endpoint_url must be a valid HTTP/HTTPS URL. Provided: {endpoint_url}"
            except Exception as e_val:
                return f"Error validating remote MCP SSE URL '{endpoint_url}' before execution: {str(e_val)}"

            return _handle_remote_mcp_sse(endpoint_url, max_events, listen_timeout_seconds)

            
        else:
            return f"Unknown function: {function_name}"
            
    except Exception as e:
        # This block handles errors from json.loads or any of the specific tool functions
        error_message = f"Error executing {function_name}: {str(e)}"
        console.print(f"[red]{error_message}[/red]") # Print the error to console
        return error_message # Return the error message string

def trim_conversation_history():
    """Trim conversation history to prevent token limit issues while preserving tool call sequences"""
    if len(conversation_history) <= 20:  # Don't trim if conversation is still small
        return
        
    # Always keep the system prompt
    system_msgs = [msg for msg in conversation_history if msg["role"] == "system"]
    other_msgs = [msg for msg in conversation_history if msg["role"] != "system"]
    
    # Keep only the last 15 messages to prevent token overflow
    if len(other_msgs) > 15:
        other_msgs = other_msgs[-15:]
    
    # Rebuild conversation history
    conversation_history.clear()
    conversation_history.extend(system_msgs + other_msgs)

def stream_llm_response(user_message: str):
    """
    Sends the conversation to the LLM using litellm and streams the response.
    Handles regular text responses, reasoning steps, and tool calls.
    """
    # Add the user message to conversation history
    conversation_history.append({"role": "user", "content": user_message})
    
    # Trim conversation history if it's getting too long
    trim_conversation_history()

    try:
        # Create a deep copy of the conversation history for this specific API call
        # This prevents issues in testing where the mock might see a mutated list.
        messages_for_api_call = copy.deepcopy(conversation_history)

        # Get model and API base from environment variables, with defaults
        model_name = os.getenv("LITELLM_MODEL", "deepseek-reasoner")
        api_base_url = os.getenv("LITELLM_API_BASE", "https://api.deepseek.com")

        # API call using litellm
        stream = completion(
            model=model_name,
            messages=messages_for_api_call,
            tools=tools,
            max_tokens=8192,
            api_base=api_base_url,           # Explicitly pass for this call
            stream=True
        )

        console.print("\n[bold bright_blue]🐋 Seeking...[/bold bright_blue]")
        reasoning_started = False
        reasoning_content = ""
        final_content = ""
        tool_calls = []

        for chunk in stream:
            # Handle reasoning content if available
            if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                if not reasoning_started:
                    console.print("\n[bold blue]💭 Reasoning:[/bold blue]")
                    reasoning_started = True
                console.print(chunk.choices[0].delta.reasoning_content, end="")
                reasoning_content += chunk.choices[0].delta.reasoning_content
            elif chunk.choices[0].delta.content:
                if reasoning_started:
                    console.print("\n")  # Add spacing after reasoning
                    console.print("\n[bold bright_blue]🤖 Assistant>[/bold bright_blue] ", end="")
                    reasoning_started = False
                final_content += chunk.choices[0].delta.content
                console.print(chunk.choices[0].delta.content, end="")
            elif chunk.choices[0].delta.tool_calls:
                # Handle tool calls
                for tool_call_delta in chunk.choices[0].delta.tool_calls:
                    if tool_call_delta.index is not None:
                        # Ensure we have enough tool_calls
                        while len(tool_calls) <= tool_call_delta.index:
                            tool_calls.append({
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            })
                        
                        if tool_call_delta.id:
                            tool_calls[tool_call_delta.index]["id"] = tool_call_delta.id
                        if tool_call_delta.function:
                            if tool_call_delta.function.name:
                                tool_calls[tool_call_delta.index]["function"]["name"] += tool_call_delta.function.name
                            if tool_call_delta.function.arguments:
                                tool_calls[tool_call_delta.index]["function"]["arguments"] += tool_call_delta.function.arguments

        console.print()  # New line after streaming

        # Store the assistant's response in conversation history
        assistant_message = {
            "role": "assistant",
            "content": final_content if final_content else None
        }
        
        if tool_calls:
            # Convert our tool_calls format to the expected format
            formatted_tool_calls = []
            for i, tc in enumerate(tool_calls):
                if tc["function"]["name"]:  # Only add if we have a function name
                    # Ensure we have a valid tool call ID
                    tool_id = tc["id"] if tc["id"] else f"call_{i}_{int(time.time() * 1000)}"
                    
                    formatted_tool_calls.append({
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"]
                        }
                    })
            
            if formatted_tool_calls:
                # Important: When there are tool calls, content should be None or empty
                if not final_content:
                    assistant_message["content"] = None
                    
                assistant_message["tool_calls"] = formatted_tool_calls
                conversation_history.append(assistant_message)
                
                # Execute tool calls and add results immediately
                console.print(f"\n[bold bright_cyan]⚡ Executing {len(formatted_tool_calls)} function call(s)...[/bold bright_cyan]")
                
                executed_tool_call_ids_and_results = [] # To store results for history
                all_tool_calls_confirmed_and_successful = True # Track if all operations proceed

                for tool_call in formatted_tool_calls:
                    tool_name = tool_call['function']['name']
                    console.print(f"[bright_blue]→ {tool_name}[/bright_blue]")

                    user_confirmed_or_not_risky = True
                    if tool_name in RISKY_TOOLS:
                        console.print(f"[bold yellow]⚠️ This is a risky operation: {tool_name}[/bold yellow]")
                        # Provide a summary of the operation
                        args = json.loads(tool_call['function']['arguments'])
                        if tool_name == "create_file":
                            console.print(f"   Action: Create/overwrite file '{args.get('file_path')}'")
                            content_summary = args.get('content', '')[:100] + "..." if len(args.get('content', '')) > 100 else args.get('content', '')
                            console.print(Panel(content_summary, title="Content Preview", border_style="yellow", expand=False))
                        elif tool_name == "create_multiple_files":
                            file_paths = [f.get('path', 'unknown') for f in args.get('files', [])]
                            console.print(f"   Action: Create/overwrite multiple files: {', '.join(file_paths)}")
                        elif tool_name == "edit_file":
                            console.print(f"   Action: Edit file '{args.get('file_path')}'")
                            original_snippet_summary = args.get('original_snippet', '')[:70] + "..." if len(args.get('original_snippet', '')) > 70 else args.get('original_snippet', '')
                            new_snippet_summary = args.get('new_snippet', '')[:70] + "..." if len(args.get('new_snippet', '')) > 70 else args.get('new_snippet', '')
                            diff_table = Table(show_header=False, box=None, padding=0)
                            diff_table.add_row("[red]- Original:[/red]", original_snippet_summary)
                            diff_table.add_row("[green]+ New:     [/green]", new_snippet_summary)
                            console.print(diff_table)
                        elif tool_name == "connect_remote_mcp_sse":
                            console.print(f"   Action: Connect to remote SSE endpoint '{args.get('endpoint_url')}'")
                        
                        confirmation = prompt_session.prompt("Proceed with this operation? [Y/n]: ", default="y").strip().lower()
                        if confirmation not in ["y", "yes", ""]:
                            user_confirmed_or_not_risky = False
                            all_tool_calls_confirmed_and_successful = False
                            result = f"User cancelled execution of {tool_name}."
                            console.print(f"[yellow]ℹ️ Operation cancelled by user.[/yellow]")
                        
                    if user_confirmed_or_not_risky:
                        try:
                            result = execute_function_call_dict(tool_call)
                            # Check if the result string itself indicates an error from execute_function_call_dict
                            if isinstance(result, str) and result.lower().startswith("error:"):
                                all_tool_calls_confirmed_and_successful = False
                        except Exception as e_exec:
                            function_name_from_tool_call = tool_call.get("function", {}).get("name", "unknown_tool")
                            error_message_for_console = f"Error executing {function_name_from_tool_call}: {str(e_exec)}"
                            console.print(f"[red]{error_message_for_console}[/red]") # Already printed by execute_function_call_dict
                            result = f"Error: {str(e_exec)}" # This is the content for history
                            all_tool_calls_confirmed_and_successful = False

                    executed_tool_call_ids_and_results.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result
                    })

                # Add all tool results (or cancellations) to conversation history
                for tool_res in executed_tool_call_ids_and_results:
                    conversation_history.append(tool_res) # type: ignore

                # If any critical tool call was cancelled or failed, the LLM needs to know.
                # The tool messages already reflect this. The LLM will see "User cancelled..." or "Error..."
                # and should adjust its follow-up.

                # Get follow-up response after tool execution
                console.print("\n[bold bright_blue]🔄 Processing results...[/bold bright_blue]")
                
                # Use the same model_name and api_base_url for the follow-up
                follow_up_stream = completion(
                    model=model_name,
                    messages=conversation_history, # History now contains tool results/cancellations
                    tools=tools,
                    max_tokens=8192,
                    api_base=api_base_url,
                    stream=True
                )
                
                follow_up_content = ""
                reasoning_started = False
                
                for chunk in follow_up_stream:
                    if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                        if not reasoning_started:
                            console.print("\n[bold blue]💭 Reasoning:[/bold blue]")
                            reasoning_started = True
                        console.print(chunk.choices[0].delta.reasoning_content, end="")
                    elif chunk.choices[0].delta.content:
                        if reasoning_started:
                            console.print("\n")
                            console.print("\n[bold bright_blue]🤖 Assistant>[/bold bright_blue] ", end="")
                            reasoning_started = False
                        follow_up_content += chunk.choices[0].delta.content
                        console.print(chunk.choices[0].delta.content, end="")
                
                console.print()
                
                conversation_history.append({
                    "role": "assistant",
                    "content": follow_up_content
                })
        else:
            # No tool calls, just store the regular response
            conversation_history.append(assistant_message)

        return {"success": True}

    except Exception as e:
        error_msg = f"LLM API error: {str(e)}"
        console.print(f"\n[bold red]❌ {error_msg}[/bold red]")
        return {"error": error_msg}

# --------------------------------------------------------------------------------
# 7. Main interactive loop
# --------------------------------------------------------------------------------

def main():
    # Create a beautiful gradient-style welcome panel
    welcome_text = """[bold bright_blue]🐋 AI Engineer[/bold bright_blue] [bright_cyan]with Function Calling[/bright_cyan]
[dim blue]Powered by DeepSeek-R1 with Chain-of-Thought Reasoning[/dim blue]""" # Note: DeepSeek-R1 is kept as it's a specific model name
    
    console.print(Panel.fit(
        welcome_text,
        border_style="bright_blue",
        padding=(1, 2),
        title="[bold bright_cyan]🤖 AI Code Assistant[/bold bright_cyan]",
        title_align="center"
    ))
    
    # Create an elegant instruction panel
    instructions = """[bold bright_blue]📁 File Operations:[/bold bright_blue]
  • [bright_cyan]/add path/to/file[/bright_cyan] - Include a single file in conversation
  • [bright_cyan]/add path/to/folder[/bright_cyan] - Include all files in a folder
  • [dim]The AI can automatically read and create files using function calls[/dim]

[bold bright_blue]🎯 Commands:[/bold bright_blue]
  • [bright_cyan]exit[/bright_cyan] or [bright_cyan]quit[/bright_cyan] - End the session
  • Just ask naturally - the AI will handle file operations automatically!"""
    
    console.print(Panel(
        instructions,
        border_style="blue",
        padding=(1, 2),
        title="[bold blue]💡 How to Use[/bold blue]",
        title_align="left"
    ))
    console.print()

    while True:
        try:
            user_input = prompt_session.prompt("🔵 You> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold yellow]👋 Exiting gracefully...[/bold yellow]")
            sys.exit(0) # Explicitly exit

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            console.print("[bold bright_blue]👋 Goodbye! Happy coding![/bold bright_blue]")
            sys.exit(0) # Explicitly exit

        if try_handle_add_command(user_input):
            continue

        response_data = stream_llm_response(user_input)
        
        if response_data.get("error"):
            # stream_llm_response already prints its own detailed API error.
            # This print in main should match the test TestMainLoop.test_main_loop_llm_error
            # stream_llm_response already prints its own detailed API error,
            # so we can skip printing it again here to avoid redundancy.
            pass


    console.print("[bold blue]✨ Session finished. Thank you for using AI Engineer![/bold blue]")
    sys.exit(0) # Ensure exit at the end of main too

if __name__ == "__main__":
    main()
