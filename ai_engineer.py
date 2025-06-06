#!/usr/bin/env python3
"""
AI Engineer: An AI-powered coding assistant.

This script provides an interactive terminal interface for code development,
leveraging AI's reasoning models for intelligent file operations,
code analysis, and development assistance via natural conversation and function calling.
"""

import os
import json
from pathlib import Path
import sys # Keep sys for exit
from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.panel import Panel
from rich.table import Table # For risky tool confirmation
from prompt_toolkit import PromptSession
import time # For tool call IDs
import subprocess # For /shell command
import copy # For deepcopy
import httpx # For new MCP tools
# Removed: import tomllib # For reading TOML config (Python 3.11+) - now in config_utils.py

# Import modules from src/
from src.config_utils import (
    load_configuration as load_app_configuration, get_config_value,
    SUPPORTED_SET_PARAMS, MAX_FILES_TO_PROCESS_IN_DIR, MAX_FILE_SIZE_BYTES,
    RUNTIME_OVERRIDES,
    get_model_context_window # Import the helper function
)
from src.file_utils import (
    normalize_path, is_binary_file, read_local_file as util_read_local_file, # Renamed read_local_file to util_read_local_file
    create_file as util_create_file, apply_diff_edit as util_apply_diff_edit # Import file utility functions
)
from src.tool_defs import tools, RISKY_TOOLS
from src.prompts import system_PROMPT
from prompt_toolkit.styles import Style as PromptStyle # Keep PromptStyle import
# Import litellm
from litellm import completion, token_counter # Added token_counter
from rich.markdown import Markdown as RichMarkdown # Import Rich's Markdown

# Initialize Rich console and prompt session
console = Console()
prompt_session = PromptSession(
    style=PromptStyle.from_dict({
        'prompt': '#0066ff bold',  # Bright blue prompt
        'completion-menu.completion': 'bg:#1e3a8a fg:#ffffff',
        'completion-menu.completion.current': 'bg:#3b82f6 fg:#ffffff bold',
    })
)


# Removed: def load_configuration(): ...
# Removed: load_configuration() # Load configurations at startup

# Load configurations at startup using the imported function
load_app_configuration(console)

class FileToCreate(BaseModel):
    path: str
    content: str

    model_config = ConfigDict(extra='ignore', frozen=True)

class FileToEdit(BaseModel):
    path: str
    original_snippet: str
    new_snippet: str

    model_config = ConfigDict(extra='ignore', frozen=True)


# Removed: system_PROMPT = dedent(...)
# This is now imported from prompts.py


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
        if parsed_url.scheme.lower() not in ("http", "https"):
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



def try_handle_add_command(user_input: str) -> bool:
    command_name_lower = "/add"
    prefix_with_space = command_name_lower + " "
    stripped_input_lower = user_input.strip().lower()

    if stripped_input_lower == command_name_lower or stripped_input_lower.startswith(prefix_with_space):
        path_to_add = ""
        if stripped_input_lower.startswith(prefix_with_space):
            path_to_add = user_input.strip()[len(prefix_with_space):].strip()

        if not path_to_add:
            console.print("[yellow]Usage: /add <file_path_or_folder_path>[/yellow]")
            console.print("[yellow]  Example: /add src/my_file.py[/yellow]")
            console.print("[yellow]  Example: /add ./my_project_folder[/yellow]")
            return True
        try:
            # Use imported normalize_path
            normalized_path = normalize_path(path_to_add)
            if os.path.isdir(normalized_path):
                # Handle entire directory
                add_directory_to_conversation(normalized_path)
            else:
                # Handle a single file as before
                # Use imported util_read_local_file
                content = util_read_local_file(normalized_path)
                conversation_history.append({ # Add to global history
                    "role": "system",
                    "content": f"Content of file '{normalized_path}':\n\n{content}"
                })
                console.print(f"[bold blue]✓[/bold blue] Added file '[bright_cyan]{normalized_path}[/bright_cyan]' to conversation.\n")
        except ValueError as e: # Catch errors from normalize_path
             console.print(f"[bold red]✗[/bold red] Could not add path '[bright_cyan]{path_to_add}[/bright_cyan]': {e}\n")
        except OSError as e:
            console.print(f"[bold red]✗[/bold red] Could not add path '[bright_cyan]{path_to_add}[/bright_cyan]': {e}\n")
        return True
    return False


def try_handle_set_command(user_input: str) -> bool:
    """
    Handles the /set command to change configuration parameters at runtime.
    """
    prefix = "/set "
    command_name_lower = "/set"
    stripped_input_lower = user_input.strip().lower()

    if stripped_input_lower == command_name_lower or stripped_input_lower.startswith(prefix):
        command_body = ""
        if stripped_input_lower.startswith(prefix):
            command_body = user_input.strip()[len(prefix):].strip()

        if not command_body:
            # Use imported SUPPORTED_SET_PARAMS
            console.print("[yellow]Available parameters to set:[/yellow]")
            for p_name, p_config in SUPPORTED_SET_PARAMS.items():
                console.print(f"  [bright_cyan]{p_name}[/bright_cyan]: {p_config['description']}")
                if "allowed_values" in p_config:
                    console.print(f"    Allowed: {', '.join(p_config['allowed_values'])}")
            console.print("\n[yellow]Usage: /set <parameter> <value>[/yellow]")
            console.print("[yellow]  Example: /set model gpt-4o[/yellow]")
            return True

        command_parts = command_body.split(maxsplit=1)
        if len(command_parts) < 2:
            console.print("[yellow]Usage: /set <parameter> <value>[/yellow]")
            return True

        param_name, value = command_parts[0].lower(), command_parts[1]

        # Use imported SUPPORTED_SET_PARAMS
        if param_name not in SUPPORTED_SET_PARAMS:
            console.print(f"[red]Error: Unknown parameter '{param_name}'. Supported parameters: {', '.join(SUPPORTED_SET_PARAMS.keys())}[/red]")
            return True

        param_config = SUPPORTED_SET_PARAMS[param_name]

        if "allowed_values" in param_config and value.lower() not in param_config["allowed_values"]:
            # Use imported SUPPORTED_SET_PARAMS
            console.print(f"[red]Error: Invalid value '{value}' for '{param_name}'. Allowed values: {', '.join(param_config['allowed_values'])}[/red]")
            return True

        # Type conversion and validation based on parameter
        if param_name == "max_tokens":
            try:
                int_value = int(value)
                if int_value <= 0:
                    raise ValueError("max_tokens must be a positive integer.")
                value = int_value # Store as int
            except ValueError:
                console.print(f"[red]Error: Invalid value '{value}' for 'max_tokens'. Must be a positive integer.[/red]")
                return True
        elif param_name == "temperature":
            try:
                float_value = float(value)
                if not (0.0 <= float_value <= 2.0): # Temperature is typically between 0.0 and 2.0
                    raise ValueError("Temperature must be a float between 0.0 and 2.0.")
                value = float_value # Store as float
            except ValueError as e:
                console.print(f"[red]Error: Invalid value '{value}' for 'temperature'. Must be a float between 0.0 and 2.0. Details: {e}[/red]")
                return True
        # For other parameters (model, api_base, reasoning_style, reasoning_effort, reply_effort),
        # the value can be stored as a string directly.

        # Use imported RUNTIME_OVERRIDES
        RUNTIME_OVERRIDES[param_name] = value
        console.print(f"[green]✓ Parameter '{param_name}' set to '{value}' for the current session.[/green]")
        return True
    return False

def try_handle_help_command(user_input: str) -> bool:
    """
    Handles the /help command to display the help markdown file.
    """
    prefix = "/help"
    if user_input.strip().lower() == prefix:
        help_file_path = Path(__file__).parent / "ai_engineer_help.md"
        try:
            # Use imported util_read_local_file
            help_content = util_read_local_file(str(help_file_path))
            console.print(Panel(
                RichMarkdown(help_content), # Use Rich's Markdown object
                title="[bold blue]📚 AI Engineer Help[/bold blue]",
                title_align="left",
                border_style="blue"
            ))
        except FileNotFoundError:
            console.print(f"[red]Error: Help file not found at '{help_file_path}'[/red]")
        except OSError as e:
            console.print(f"[red]Error reading help file: {e}[/red]")
        # Help command does not add to conversation history
        return True
    return False

def try_handle_shell_command(user_input: str) -> bool:
    """
    Handles the /shell command to execute a shell command and add output to history.
    """
    prefix = "/shell "
    command_name_lower = "/shell"
    stripped_input_lower = user_input.strip().lower()

    if stripped_input_lower == command_name_lower or stripped_input_lower.startswith(prefix):
        command_body = ""
        if stripped_input_lower.startswith(prefix):
            command_body = user_input.strip()[len(prefix):].strip()

        if not command_body:
            console.print("[yellow]Usage: /shell <command and arguments>[/yellow]")
            console.print("[yellow]  Example: /shell ls -l[/yellow]")
            console.print("[bold yellow]⚠️ Warning: Executing arbitrary shell commands can be risky.[/bold yellow]")
            return True

        console.print(f"[bold bright_blue]🐚 Executing shell command: '{command_body}'[/bold bright_blue]")
        console.print("[dim]Output:[/dim]")

        try:
            # Execute the command
            # Use shell=True for simplicity as requested, but note security risks
            # capture_output=True captures stdout and stderr
            # text=True decodes stdout/stderr as text
            result = subprocess.run(command_body, shell=True, capture_output=True, text=True, check=False)

            output = result.stdout.strip()
            error_output = result.stderr.strip()
            return_code = result.returncode

            # Print output to console in real-time (or after execution for subprocess.run)
            if output:
                console.print(output)
            if error_output:
                console.print(f"[red]Stderr:[/red]\n{error_output}")
            if return_code != 0:
                 console.print(f"[red]Command exited with non-zero status code: {return_code}[/red]")

            # Add output to conversation history
            history_content = f"Shell command executed: '{command_body}'\n\n"
            if output:
                history_content += f"Stdout:\n```\n{output}\n```\n"
            if error_output:
                 history_content += f"Stderr:\n```\n{error_output}\n```\n"
            if return_code != 0:
                 history_content += f"Return Code: {return_code}\n"

            conversation_history.append({
                "role": "system",
                "content": history_content.strip()
            })
            console.print("[bold blue]✓[/bold blue] Shell output added to conversation history.\n")

        except FileNotFoundError:
            # This happens if shell=False and the command itself is not found
             console.print(f"[red]Error: Command not found: '{command_body.split()[0]}'[/red]")
             conversation_history.append({
                "role": "system",
                "content": f"Error: Shell command not found: '{command_body.split()[0]}'"
            })
        except Exception as e:
            console.print(f"[red]An error occurred during shell execution: {e}[/red]")
            conversation_history.append({
                "role": "system",
                "content": f"Error executing shell command '{command_body}': {e}"
            })

        return True
    return False

def try_handle_context_command(user_input: str) -> bool:
    """
    Handles /context commands (save, load, list, summarize).
    """
    prefix = "/context "
    # Ensure we are checking the stripped input for the prefix
    stripped_input_lower = user_input.strip().lower()
    if stripped_input_lower.startswith(prefix) or stripped_input_lower == "/context":
        # Get command body relative to the actual prefix used (e.g. "/context " or "/context")
        if stripped_input_lower.startswith(prefix): # Handles "/context <subcommand>"
            command_body = user_input.strip()[len(prefix):].strip()
        else: # Handles just "/context"
            command_body = ""

        parts = command_body.split(maxsplit=1)
        sub_command = parts[0].lower() if parts else ""
        arg = parts[1] if len(parts) > 1 else ""

        if sub_command == "save":
            if not arg:
                console.print("[yellow]Usage: /context save <name>[/yellow]")
                return True
            save_context(arg)
            return True

        elif sub_command == "load":
            if not arg:
                console.print("[yellow]Usage: /context load <name>[/yellow]")
                return True
            load_context(arg)
            return True

        elif sub_command == "list":
            list_contexts(arg if arg else ".") # List in current dir if no path given
            return True

        elif sub_command == "summarize":
            summarize_context()
            return True

        else: # Handles empty sub_command (just "/context") or unknown sub_command
            console.print("[yellow]Usage: /context <save|load|list|summarize> [name/path][/yellow]")
            console.print("[yellow]  save <name>     - Save current context to a file.[/yellow]")
            console.print("[yellow]  load <name>     - Load context from a file.[/yellow]")
            console.print("[yellow]  list [path]     - List saved contexts in a directory.[/yellow]")
            console.print("[yellow]  summarize       - Summarize current context using the LLM.[/yellow]")
            return True
    return False

def try_handle_session_command(user_input: str) -> bool:
    """Handles /session commands by delegating to the /context command handler."""
    stripped_input = user_input.strip()
    if stripped_input.lower().startswith("/session"):
        # Construct the equivalent /context command
        # /session -> /context
        # /session foo -> /context foo
        # len("/session") is 8
        arguments_part = stripped_input[len("/session"):] # Takes arguments after "/session" (e.g., " save foo" or "")
        context_equivalent_input = "/context" + arguments_part
        
        console.print(f"[dim]Executing '{stripped_input}' as '{context_equivalent_input.strip()}'[/dim]")
        return try_handle_context_command(context_equivalent_input)
    return False

def save_context(name: str):
    """Saves the current conversation history to a JSON file."""
    file_name = f"context_{name}.json"
    try:
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(conversation_history, f, indent=2)
        console.print(f"[bold blue]✓[/bold blue] Context saved to '[bright_cyan]{file_name}[/bright_cyan]'\n")
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Failed to save context to '{file_name}': {e}\n")

def load_context(name: str):
    """Loads conversation history from a JSON file."""
    file_name = f"context_{name}.json"
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            loaded_history = json.load(f)

        # Validate loaded history structure (basic check)
        if not isinstance(loaded_history, list) or not all(isinstance(msg, dict) and "role" in msg for msg in loaded_history):
             raise ValueError("Invalid context file format.")

        # Keep the initial system prompt, replace the rest
        global conversation_history
        initial_system_prompt = conversation_history[0] if conversation_history and conversation_history[0]["role"] == "system" else {"role": "system", "content": system_PROMPT}
        conversation_history = [initial_system_prompt] + [msg for msg in loaded_history if msg["role"] != "system"]

        console.print(f"[bold blue]✓[/bold blue] Context loaded from '[bright_cyan]{file_name}[/bright_cyan]'\n")
    except FileNotFoundError:
        console.print(f"[bold red]✗[/bold red] Context file not found: '[bright_cyan]{file_name}[/bright_cyan]'\n")
    except json.JSONDecodeError:
        console.print(f"[bold red]✗[/bold red] Failed to parse JSON from context file: '[bright_cyan]{file_name}[/bright_cyan]'. File might be corrupted.\n")
    except ValueError as e:
         console.print(f"[bold red]✗[/bold red] Invalid context file format for '[bright_cyan]{file_name}[/bright_cyan]': {e}\n")
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Failed to load context from '{file_name}': {e}\n")

def list_contexts(path: str):
    """Lists potential context files in the specified directory."""
    try:
        # Use imported normalize_path
        normalized_path_str = normalize_path(path)
        target_dir = Path(normalized_path_str)

        if not target_dir.is_dir():
            console.print(f"[bold red]✗[/bold red] Path is not a directory: '[bright_cyan]{path}[/bright_cyan]'\n")
            return

        console.print(f"[bold bright_blue]📚 Saved Contexts in '[bright_cyan]{target_dir}[/bright_cyan]':[/bold bright_blue]")
        found_files = list(target_dir.glob("context_*.json"))

        if not found_files:
            console.print("  [dim]No context files found.[/dim]\n")
            return

        for f in found_files:
            console.print(f"  [bright_cyan]{f.name}[/bright_cyan]")
        console.print() # Final newline

    except ValueError as e: # From normalize_path
        console.print(f"[bold red]✗[/bold red] Invalid path '[bright_cyan]{path}[/bright_cyan]': {e}\n")
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Failed to list contexts in '{path}': {e}\n")

def summarize_context():
    """Summarizes the current conversation history using the LLM and replaces the history."""
    global conversation_history

    if len(conversation_history) <= 1: # Only system prompt
        console.print("[yellow]No conversation history to summarize.[/yellow]\n")
        return

    console.print("[bold bright_blue]✨ Summarizing conversation history...[/bold bright_blue]")

    # Create a temporary history for the summary request
    # Keep the original system prompt, add a user message asking for summary
    summary_messages = [
        conversation_history[0], # Original system prompt
        {"role": "user", "content": "Please provide a concise summary of our conversation so far. Focus on the key topics discussed, decisions made, and actions taken (like file operations). This summary will replace the detailed history."}
    ]

    # Add the rest of the history for the LLM to read
    summary_messages.extend(conversation_history[1:])

    try:
        # Use imported get_config_value for model/api_base
        model_name = get_config_value("model", "gpt-4o") # Use a capable model for summary
        api_base_url = get_config_value("api_base", None)

        # Call LLM for summary (non-streaming for simplicity here)
        # Use a lower temperature for a more focused summary
        response = completion(
            model=model_name,
            messages=summary_messages,
            temperature=0.3,
            max_tokens=1024, # Limit summary length
            api_base=api_base_url,
            stream=False # Don't stream the summary response
        )

        summary_content = response.choices[0].message.content

        if summary_content:
            console.print("\n[bold blue]Summary:[/bold blue]")
            console.print(Panel(summary_content, border_style="blue"))

            # Replace history with system prompt + summary
            initial_system_prompt = conversation_history[0] if conversation_history and conversation_history[0]["role"] == "system" else {"role": "system", "content": system_PROMPT}
            conversation_history = [
                initial_system_prompt,
                {"role": "system", "content": f"Conversation Summary:\n\n{summary_content}"}
            ]
            console.print("[bold blue]✓[/bold blue] Conversation history replaced with summary.\n")
        else:
            console.print("[yellow]LLM returned an empty summary.[/yellow]\n")

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Failed to summarize context: {e}\n")
        # History remains unchanged on error


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

        for root, dirs, files in os.walk(directory_path): # Use imported MAX_FILES_TO_PROCESS_IN_DIR
            if total_files_processed >= MAX_FILES_TO_PROCESS_IN_DIR:
                console.print(f"[bold yellow]⚠[/bold yellow] Reached maximum file limit ({MAX_FILES_TO_PROCESS_IN_DIR})")
                break

            status.update(f"[bold bright_blue]🔍 Scanning {root}...[/bold bright_blue]")
            # Skip hidden directories and excluded directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in excluded_files]

            for file in files: # Use imported MAX_FILES_TO_PROCESS_IN_DIR
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
                    if is_binary_file(full_path):
                        skipped_files.append(full_path)
                        continue

                    # Use imported normalize_path
                    normalized_path = normalize_path(full_path)
                    # Use imported util_read_local_file
                    content = util_read_local_file(normalized_path)
                    conversation_history.append({
                        "role": "system",
                        "content": f"Content of file '{normalized_path}':\n\n{content}"
                    })
                    added_files.append(normalized_path)
                    total_files_processed += 1

                except OSError: # Catch read errors
                    skipped_files.append(str(full_path)) # Ensure it's a string
                except ValueError as e: # Catch errors from normalize_path
                     skipped_files.append(f"{full_path} (Invalid path: {e})")


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
        console.print() # Final newline after directory scan summary

def ensure_file_in_context(file_path: str) -> bool:
    try:
        # Use imported normalize_path
        normalized_path = normalize_path(file_path)
        # Use imported util_read_local_file
        content = util_read_local_file(normalized_path)
        file_marker = f"Content of file '{normalized_path}'"
        # Check if the file content is already in the history (by looking for the marker)
        # Also ensure the message has a 'content' key before checking
        if not any(file_marker in msg["content"] for msg in conversation_history if msg.get("content")):
            conversation_history.append({
                "role": "system",
                "content": f"{file_marker}:\n\n{content}"
            })
        return True
    except (OSError, ValueError) as e: # Catch OSError from read or ValueError from normalize
        console.print(f"[bold red]✗[/bold red] Could not read file '[bright_cyan]{file_path}[/bright_cyan]' for editing context: {e}")
        return False

# --------------------------------------------------------------------------------
# 5. Conversation state
# --------------------------------------------------------------------------------
# system_PROMPT is now imported
conversation_history = [
    {"role": "system", "content": system_PROMPT}
]


# --------------------------------------------------------------------------------
# 6. LLM API interaction with streaming
# --------------------------------------------------------------------------------


def execute_function_call_dict(tool_call_dict) -> str:
    """Execute a function call from a dictionary format and return the result as a string."""
    function_name = "unknown_function" # Default if parsing fails early
    try:
        function_name = tool_call_dict["function"]["name"]
        arguments = json.loads(tool_call_dict["function"]["arguments"])

        if function_name == "read_file":
            file_path = arguments["file_path"]
            normalized_path = normalize_path(file_path)
            content = util_read_local_file(normalized_path)
            return f"Content of file '{normalized_path}':\n\n{content}"

        elif function_name == "read_multiple_files":
            file_paths = arguments["file_paths"]
            results = []
            # Use imported normalize_path and util_read_local_file
            for file_path in file_paths:
                try:
                    normalized_path = normalize_path(file_path)
                    content = util_read_local_file(normalized_path)
                    results.append(f"Content of file '{normalized_path}':\n\n{content}")
                except (OSError, ValueError) as e: # Catch read errors or normalize errors
                    results.append(f"Error reading '{file_path}': {e}")
            return "\n\n" + "="*50 + "\n\n".join(results)

        elif function_name == "create_file":
            file_path = arguments["file_path"]
            content = arguments["content"]
            util_create_file(file_path, content, console, MAX_FILE_SIZE_BYTES)
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
                    # The test mocks the imported util_create_file.
                    # Call the imported function directly.
                    util_create_file(path, content, console, MAX_FILE_SIZE_BYTES)
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

            util_apply_diff_edit(file_path, original_snippet, new_snippet, console, MAX_FILE_SIZE_BYTES)
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
                if parsed_url.scheme.lower() not in ("http", "https"): # Allow http and https
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
    # Get configuration settings first, including reasoning_effort
    trim_conversation_history()

    try:
        # Create a deep copy of the conversation history *before* this turn's user message.
        # The user_message for this turn will be added to this copy, potentially augmented.
        messages_for_api_call = copy.deepcopy(conversation_history)

        default_max_tokens_val = 8192
        default_reasoning_effort_val = "medium"
        default_reply_effort_val = "medium"
        default_temperature_val = 0.7 # Common default for temperature

        # Removed local get_config_value definition.
        # Use the imported get_config_value function.
        # It already uses CONFIG_FROM_TOML, RUNTIME_OVERRIDES, SUPPORTED_SET_PARAMS which are imported globals.
        # Use imported constants for defaults where applicable
        model_name = get_config_value("model", "gpt-4o") # Use a reasonable default if not configured
        api_base_url = get_config_value("api_base", None) # Let litellm handle default if not configured
        reasoning_style = str(get_config_value("reasoning_style", "full")).lower()

        max_tokens_raw = get_config_value("max_tokens", default_max_tokens_val)
        try:
            max_tokens = int(max_tokens_raw)
            if max_tokens <= 0:
                max_tokens = default_max_tokens_val
        except (ValueError, TypeError):
            max_tokens = default_max_tokens_val

        temperature_raw = get_config_value("temperature", default_temperature_val)
        try:
            temperature = float(temperature_raw)
            # Use imported get_config_value
            # Optional: Add range validation here if desired, e.g., if not (0.0 <= temperature <= 2.0):
        except (ValueError, TypeError):
            console.print(f"[yellow]Warning: Invalid temperature value '{temperature_raw}'. Using default {default_temperature_val}.[/yellow]")
            temperature = default_temperature_val

        reasoning_effort_setting = str(get_config_value("reasoning_effort", default_reasoning_effort_val)).lower()
        reply_effort_setting = str(get_config_value("reply_effort", default_reply_effort_val)).lower()

        # Prepare the user's message for this turn, augmenting it with effort control instructions.
        # The system prompt (in messages_for_api_call[0]) already defines *what* these settings mean.
        # Here, we provide the *current values* for this specific turn.
        effort_instructions = (
            f"\n\n[System Instructions For This Turn Only]:\n"
            f"- Current `reasoning_effort`: {reasoning_effort_setting}\n"
            f"- Current `reply_effort`: {reply_effort_setting}\n"
            f"Please adhere to these specific effort levels for your reasoning and reply in this turn."
        )
        augmented_user_message_content = user_message + effort_instructions

        messages_for_api_call.append({
            "role": "user",
            "content": augmented_user_message_content
        })

        console.print("\n[bold bright_blue]🐋 Seeking...[/bold bright_blue]")
        reasoning_content_accumulated = "" # To store full reasoning if needed later
        final_content = ""
        tool_calls = []

        # Determine if we should print the "Reasoning:" header at all
        # For "silent", we don't print it. For "compact", we print it once. For "full", we print it once.
        reasoning_started_printed = False # Track if "💭 Reasoning:" has been printed


        # API call using litellm
        stream = completion(
            model=model_name,
            messages=messages_for_api_call, # Use the augmented messages
            tools=tools, 
            max_tokens=max_tokens, 
            api_base=api_base_url,
            temperature=temperature,
            stream=True
        )

        for chunk in stream:
            # Handle reasoning content if available
            if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                reasoning_chunk_content = chunk.choices[0].delta.reasoning_content
                reasoning_content_accumulated += reasoning_chunk_content

                if reasoning_style == "full":
                    if not reasoning_started_printed:
                        console.print("\n[bold blue]💭 Reasoning:[/bold blue]")
                        reasoning_started_printed = True
                    console.print(reasoning_chunk_content, end="")
                elif reasoning_style == "compact":
                    if not reasoning_started_printed:
                        console.print("\n[bold blue]💭 Reasoning...[/bold blue]", end="") # Print header once
                        reasoning_started_printed = True
                    console.print(".", end="") # Print a dot for progress
                # If style is "silent", do nothing here for reasoning_content

            elif chunk.choices[0].delta.content:
                if reasoning_started_printed and reasoning_style != "full": # Add newline if dots or compact header was printed
                    console.print() # Newline after dots or compact header before assistant content
                    reasoning_started_printed = False # Reset for next potential reasoning block in follow-up

                if not final_content: # First content chunk
                    console.print("\n\n[bold bright_blue]🤖 Assistant>[/bold bright_blue] ", end="")

                final_content += chunk.choices[0].delta.content
                console.print(chunk.choices[0].delta.content, end="")

            elif chunk.choices[0].delta.tool_calls:
                if reasoning_started_printed and reasoning_style != "full": # Newline if dots were printed
                    console.print()
                    reasoning_started_printed = False

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

        if reasoning_started_printed and reasoning_style == "compact" and not final_content and not tool_calls:
            # If only reasoning dots were printed and no content/tool_calls followed, add a newline.
            console.print()

        console.print()  # New line after streaming is complete

        # Add the original (non-augmented) user message to the global conversation history
        conversation_history.append({"role": "user", "content": user_message})

        # Store the assistant's response in conversation history
        assistant_message = {
            "role": "assistant",
            "content": final_content if final_content else None # Ensure None if empty
        }
        # Add reasoning_content to the assistant message if it was captured (for full history, not for display)
        # This is useful if we ever want to inspect the full reasoning later, regardless of display style.
        if reasoning_content_accumulated:
            assistant_message["reasoning_content_full"] = reasoning_content_accumulated # Store under a different key

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
                if not final_content: # If there was no regular content, set it to None
                    assistant_message["content"] = None

                assistant_message["tool_calls"] = formatted_tool_calls
                conversation_history.append(assistant_message)

                # Execute tool calls and add results immediately
                console.print(f"\n[bold bright_cyan]⚡ Executing {len(formatted_tool_calls)} function call(s)...[/bold bright_cyan]")

                executed_tool_call_ids_and_results = [] # To store results for history

                for tool_call in formatted_tool_calls:
                    tool_name = tool_call['function']['name']
                    console.print(f"[bright_blue]→ {tool_name}[/bright_blue]")

                    user_confirmed_or_not_risky = True
                    if tool_name in RISKY_TOOLS:
                        console.print(f"[bold yellow]⚠️ This is a risky operation: {tool_name}[/bold yellow]")
                        try:
                            args = json.loads(tool_call['function']['arguments'])
                        except json.JSONDecodeError:
                            console.print("[red]Error: Could not parse tool arguments.[/red]")
                            executed_tool_call_ids_and_results.append({
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": f"Error: Could not parse arguments for {tool_name}"
                            })
                            continue 

                        if tool_name == "create_file":
                            console.print(f"   Action: Create/overwrite file '{args.get('file_path')}'")
                            content_summary = args.get('content', '')[:100] + "..." if len(args.get('content', '')) > 100 else args.get('content', '')
                            console.print(Panel(content_summary, title="Content Preview", border_style="yellow", expand=False))
                        elif tool_name == "create_multiple_files":
                            files_to_create_preview = args.get('files', [])
                            if files_to_create_preview:
                                file_paths = [f.get('path', 'unknown') for f in files_to_create_preview]
                                console.print(f"   Action: Create/overwrite {len(file_paths)} files: {', '.join(file_paths[:5])}{'...' if len(file_paths) > 5 else ''}")
                            else:
                                console.print("   Action: Create multiple files (none specified).")
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
                            console.print("[yellow]ℹ️ Operation cancelled by user.[/yellow]")
                            result = "User cancelled execution of this tool call." 
                        
                    if user_confirmed_or_not_risky:
                        try:
                            result = execute_function_call_dict(tool_call) 
                            if isinstance(result, str) and result.lower().startswith("error:"):
                                pass 
                        except Exception as e_exec:
                            console.print(f"[red]Unexpected error during tool execution: {str(e_exec)}[/red]")
                            result = f"Error: Unexpected error during tool execution: {str(e_exec)}"

                    executed_tool_call_ids_and_results.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result
                    })

                for tool_res in executed_tool_call_ids_and_results:
                    conversation_history.append(tool_res) # type: ignore

                console.print("\n[bold bright_blue]🔄 Processing results...[/bold bright_blue]")

                follow_up_stream = completion(
                    model=model_name,
                    messages=conversation_history, 
                    tools=tools, 
                    max_tokens=max_tokens, 
                    api_base=api_base_url,
                    stream=True
                )

                follow_up_content = ""
                reasoning_started_printed_follow_up = False 
                reasoning_content_accumulated_follow_up = ""

                for chunk in follow_up_stream:
                    if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                        reasoning_chunk_content_follow_up = chunk.choices[0].delta.reasoning_content
                        reasoning_content_accumulated_follow_up += reasoning_chunk_content_follow_up
                        if reasoning_style == "full":
                            if not reasoning_started_printed_follow_up:
                                console.print("\n[bold blue]💭 Reasoning:[/bold blue]")
                                reasoning_started_printed_follow_up = True
                            console.print(reasoning_chunk_content_follow_up, end="")
                        elif reasoning_style == "compact":
                            if not reasoning_started_printed_follow_up:
                                console.print("\n[bold blue]💭 Reasoning...[/bold blue]", end="")
                                reasoning_started_printed_follow_up = True
                            console.print(".", end="")
                        
                    elif chunk.choices[0].delta.content:
                        if reasoning_started_printed_follow_up and reasoning_style != "full":
                            console.print() 
                            reasoning_started_printed_follow_up = False

                        if not follow_up_content: 
                             console.print("\n\n[bold bright_blue]🤖 Assistant>[/bold bright_blue] ", end="")

                        follow_up_content += chunk.choices[0].delta.content
                        console.print(chunk.choices[0].delta.content, end="")

                if reasoning_started_printed_follow_up and reasoning_style == "compact" and not follow_up_content:
                    console.print() 

                console.print() 

                assistant_follow_up_message = {
                    "role": "assistant",
                    "content": follow_up_content
                }
                if reasoning_content_accumulated_follow_up: 
                    assistant_follow_up_message["reasoning_content_full"] = reasoning_content_accumulated_follow_up
                conversation_history.append(assistant_follow_up_message)
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

def clear_screen():
    """Clears the terminal screen."""
    # For Windows
    if os.name == 'nt':
        _ = os.system('cls')
    # For Mac and Linux (os.name is 'posix')
    else:
        _ = os.system('clear')

def main():
    clear_screen()
    # Get the current model name to display in the welcome panel
    # Use the imported get_config_value function
    current_model_name = get_config_value("model", "gpt-4o") # Default if not found

    # Create an elegant instruction panel listing key commands
    instructions = f"""[bold bright_blue]🎯 Commands:[/bold bright_blue]
  • [bright_cyan]/exit[/bright_cyan] or [bright_cyan]/quit[/bright_cyan] - End the session.
  • [bright_cyan]/help[/bright_cyan] - Display detailed help.

Model: [bold magenta]{current_model_name}[/bold magenta]

[bold white]👥 Just ask naturally, like you are communicating with a colleague.[/bold white]"""

    console.print(Panel(
        instructions,
        border_style="blue",
        padding=(1, 2),
        title="[bold blue]🤖 AI Code Assistant[/bold blue]",
        title_align="left"
    ))
    console.print()

    while True:
        try:
            prompt_prefix = ""
            # Calculate context usage for the prompt
            # Ensure conversation_history is accessible here (it's global)
            # and get_config_value, get_model_context_window are imported
            if conversation_history: # Only calculate if there's history
                try:
                    current_model_name = get_config_value("model", "gpt-4o") # Get current model
                    # Get window size and status if default was used due to no match
                    context_window_size, used_default_window_due_to_no_match = get_model_context_window(current_model_name, return_match_status=True)

                    # Make sure conversation_history is not empty and model_name is valid for token_counter
                    if conversation_history and current_model_name:
                        # The first message is the system prompt, which is always there.
                        tokens_used = token_counter(model=current_model_name, messages=conversation_history)

                        if context_window_size > 0: # Avoid division by zero
                            percentage_used = (tokens_used / context_window_size) * 100
                            default_note = ""
                            # Add a note if the default window was used because the model name wasn't specifically found
                            if used_default_window_due_to_no_match:
                                default_note = " (default window)"
                            prompt_prefix = f"[Ctx: {percentage_used:.0f}%{default_note}] "
                        else: # Should not happen if get_model_context_window returns a default
                            prompt_prefix = f"[Ctx: {tokens_used} toks] "
                except Exception as e:
                    # Silently fail or print a dim message if token calculation fails
                    # For example, if litellm.token_counter doesn't support the model
                    # console.print(f"[dim]Could not calculate token usage: {e}[/dim]")
                    pass # Keep prompt clean if there's an error


            user_input = prompt_session.prompt(f"{prompt_prefix}🔵 You> ").strip()

        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold yellow]👋 Exiting gracefully...[/bold yellow]")
            sys.exit(0) # Explicitly exit

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "/exit", "/quit"]:
            console.print("[bold bright_blue]👋 Goodbye! Happy coding![/bold bright_blue]")
            sys.exit(0) # Explicitly exit

        # Check for and handle commands
        if try_handle_add_command(user_input):
            continue
        if try_handle_set_command(user_input):
            continue
        if try_handle_help_command(user_input):
            continue
        if try_handle_shell_command(user_input):
            continue
        if try_handle_session_command(user_input): # New: session handler
            continue
        if try_handle_context_command(user_input):
            continue

        # Use imported stream_llm_response
        response_data = stream_llm_response(user_input) # user_input is the raw message

        if response_data.get("error"):
            # stream_llm_response already prints its own detailed API error.
            pass


    console.print("[bold blue]✨ Session finished. Thank you for using AI Engineer![/bold blue]")
    sys.exit(0) # Ensure exit at the end of main too


if __name__ == "__main__":
    main()
