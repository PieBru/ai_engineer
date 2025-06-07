#!/usr/bin/env python3
"""
AI Engineer: An AI-powered coding assistant.

This script provides an interactive terminal interface for code development,
leveraging AI's reasoning models for intelligent file operations,
code analysis, and development assistance via natural conversation and function calling.
"""

# Import default constants from config_utils first
from src.config_utils import (
    DEFAULT_LITELLM_MODEL,
    DEFAULT_LITELLM_API_BASE,
    DEFAULT_LITELLM_MAX_TOKENS,
    DEFAULT_LITELLM_MODEL_ROUTING,
    DEFAULT_LITELLM_MODEL_TOOLS,
    DEFAULT_LITELLM_MODEL_CODING,
    DEFAULT_LITELLM_MODEL_KNOWLEDGE,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_REASONING_STYLE
)

# Now, define module-level configurations using these defaults
import os
LITELLM_MODEL = os.getenv("LITELLM_MODEL", DEFAULT_LITELLM_MODEL)
LITELLM_MODEL_DEFAULT = LITELLM_MODEL # Alias for clarity, LITELLM_MODEL is the default
LITELLM_MODEL_ROUTING = os.getenv("LITELLM_MODEL_ROUTING", DEFAULT_LITELLM_MODEL_ROUTING)
LITELLM_MODEL_TOOLS = os.getenv("LITELLM_MODEL_TOOLS", DEFAULT_LITELLM_MODEL_TOOLS)
LITELLM_MODEL_CODING = os.getenv("LITELLM_MODEL_CODING", DEFAULT_LITELLM_MODEL_CODING)
LITELLM_MODEL_KNOWLEDGE = os.getenv("LITELLM_MODEL_KNOWLEDGE", DEFAULT_LITELLM_MODEL_KNOWLEDGE)

LITELLM_API_BASE = os.getenv("LITELLM_API_BASE", DEFAULT_LITELLM_API_BASE)
LITELLM_MAX_TOKENS = int(os.getenv("LITELLM_MAX_TOKENS", DEFAULT_LITELLM_MAX_TOKENS))
REASONING_EFFORT = os.getenv("REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
REASONING_STYLE = os.getenv("REASONING_STYLE", DEFAULT_REASONING_STYLE)
import json
from pathlib import Path
import sys # Keep sys for exit
from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.panel import Panel
from rich.table import Table # For risky tool confirmation
from prompt_toolkit import PromptSession
import time # For tool call IDs and test inference timing
import subprocess # For /shell command
import copy # For deepcopy
import argparse # For command-line argument handling
from textwrap import dedent # For /prompt command meta-prompts
import httpx # For new MCP tools
import fnmatch # For wildcard matching in test_inference

# Import modules from src/
from src.config_utils import (
    load_configuration as load_app_configuration, get_config_value,
    SUPPORTED_SET_PARAMS, MAX_FILES_TO_PROCESS_IN_DIR, MAX_FILE_SIZE_BYTES, MODEL_CONTEXT_WINDOWS, SHOW_TEST_INFERENCE_NOTES_ERRORS_COLUMN,
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
import litellm # Import the base module to set options
import logging # For configuring logger levels
from typing import Dict # For type hinting

# Define the application version
__version__ = "0.2.0" # Example version

# Initialize Rich console and prompt session
console = Console()
prompt_session = PromptSession(
    style=PromptStyle.from_dict({
        'prompt': '#0066ff bold',  # Bright blue prompt
        'completion-menu.completion': 'bg:#1e3a8a fg:#ffffff',
        'completion-menu.completion.current': 'bg:#3b82f6 fg:#ffffff bold',
    })
)

# Global state for timestamp display
SHOW_TIMESTAMP_IN_PROMPT = False

# Load configurations at startup using the imported function
load_app_configuration(console)

# --- Suppress LiteLLM informational messages ---
# Suppress the "Give Feedback / Get Help" banner
litellm.suppress_debug_info = True
# Suppress "LiteLLM.Info" messages by setting logger level to WARNING
logging.getLogger("litellm").setLevel(logging.WARNING)
# --- End LiteLLM message suppression ---

class FileToCreate(BaseModel):
    path: str
    content: str
    model_config = ConfigDict(extra='ignore', frozen=True)

class FileToEdit(BaseModel):
    path: str
    original_snippet: str
    new_snippet: str
    model_config = ConfigDict(extra='ignore', frozen=True)

# --------------------------------------------------------------------------------
# 4. Helper functions
# --------------------------------------------------------------------------------

def _handle_local_mcp_stream(endpoint_url: str, timeout_seconds: int, max_data_chars: int) -> str:
    """
    Connects to a local MCP server endpoint that provides a streaming response.
    Reads the stream and returns the aggregated data.
    """
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
                response.raise_for_status()
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
            e.response.read()
            error_detail += f" - {e.response.text[:200]}"
        except httpx.ResponseNotRead:
            error_detail += " - (Error response body could not be read for details)"
        except Exception:
            error_detail += " - (Failed to retrieve error response body details)"
        return f"Error connecting to local MCP stream '{endpoint_url}': {error_detail}"
    except httpx.RequestError as e:
        return f"Error connecting to local MCP stream '{endpoint_url}': {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred with local MCP stream '{endpoint_url}': {str(e)}"

def _handle_remote_mcp_sse(endpoint_url: str, max_events: int, listen_timeout_seconds: int) -> str:
    """
    Connects to a remote MCP server endpoint using Server-Sent Events (SSE).
    Listens for events and returns a summary of received events.
    """
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
                    if not line:
                        if current_event_data:
                            events_received.append("".join(current_event_data))
                            current_event_data = []
                            if len(events_received) >= max_events:
                                break
                    elif line.startswith("data:"):
                        current_event_data.append(line[5:].strip() + "\n")

                if current_event_data and len(events_received) < max_events:
                    events_received.append("".join(current_event_data).strip())

        summary = f"Received {len(events_received)} SSE event(s) from '{endpoint_url}'.\n"
        summary += "Last few events:\n" + "\n---\n".join(events_received[-5:])
        return summary
    except httpx.HTTPStatusError as e:
        error_detail = f"HTTP {e.response.status_code}"
        try:
            e.response.read()
            error_detail += f" - {e.response.text[:200]}"
        except httpx.ResponseNotRead:
            error_detail += " - (Error response body could not be read for details)"
        except Exception:
            error_detail += " - (Failed to retrieve error response body details)"
        return f"Error connecting to remote MCP SSE '{endpoint_url}': {error_detail}"
    except httpx.RequestError as e:
        return f"Error connecting to remote MCP SSE '{endpoint_url}': {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred with remote MCP SSE '{endpoint_url}': {str(e)}"

def try_handle_add_command(user_input: str) -> bool:
    command_prefix = "/add"
    stripped_input = user_input.strip()
    if not (stripped_input.lower() == command_prefix or stripped_input.lower().startswith(command_prefix + " ")):
        return False
    prefix_with_space = command_prefix + " "
    path_to_add = ""
    if stripped_input.lower().startswith(prefix_with_space):
        path_to_add = stripped_input[len(prefix_with_space):].strip()
    elif stripped_input.lower() == command_prefix:
        path_to_add = ""
    if not path_to_add:
        console.print("[yellow]Usage: /add <file_path_or_folder_path>[/yellow]")
        console.print("[yellow]  Example: /add src/my_file.py[/yellow]")
        console.print("[yellow]  Example: /add ./my_project_folder[/yellow]")
        return True
    try:
        normalized_path = normalize_path(path_to_add)
        if os.path.isdir(normalized_path):
            add_directory_to_conversation(normalized_path)
        else:
            content = util_read_local_file(normalized_path)
            conversation_history.append({
                "role": "system",
                "content": f"Content of file '{normalized_path}':\n\n{content}"
            })
            console.print(f"[bold blue]✓[/bold blue] Added file '[bright_cyan]{normalized_path}[/bright_cyan]' to conversation.\n")
    except ValueError as e:
            console.print(f"[bold red]✗[/bold red] Could not add path '[bright_cyan]{path_to_add}[/bright_cyan]': {e}\n")
    except OSError as e:
        console.print(f"[bold red]✗[/bold red] Could not add path '[bright_cyan]{path_to_add}[/bright_cyan]': {e}\n")
    return True

def try_handle_set_command(user_input: str) -> bool:
    command_prefix = "/set"
    stripped_input = user_input.strip()
    if not (stripped_input.lower() == command_prefix or stripped_input.lower().startswith(command_prefix + " ")):
        return False
    prefix_with_space = command_prefix + " "
    command_body = ""
    if stripped_input.lower().startswith(prefix_with_space):
        command_body = stripped_input[len(prefix_with_space):].strip()
    if not command_body:
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
    if param_name == "system_prompt":
        file_path = value
        try:
            normalized_path = normalize_path(file_path)
            new_prompt_content = util_read_local_file(normalized_path)
            if conversation_history and conversation_history[0]["role"] == "system":
                conversation_history[0]["content"] = new_prompt_content
            else:
                conversation_history.insert(0, {"role": "system", "content": new_prompt_content})
            RUNTIME_OVERRIDES[param_name] = new_prompt_content
            console.print(f"[green]✓ System prompt updated from file '[bright_cyan]{normalized_path}[/bright_cyan]'.[/green]")
        except FileNotFoundError:
            console.print(f"[red]Error: File not found at '[bright_cyan]{file_path}[/bright_cyan]'. System prompt not changed.[/red]")
        except (OSError, ValueError) as e:
            console.print(f"[red]Error reading or normalizing file '[bright_cyan]{file_path}[/bright_cyan]': {e}. System prompt not changed.[/red]")
        return True
    elif param_name in SUPPORTED_SET_PARAMS:
        param_config = SUPPORTED_SET_PARAMS[param_name]
        if "allowed_values" in param_config and value.lower() not in param_config["allowed_values"]:
            console.print(f"[red]Error: Invalid value '{value}' for '{param_name}'. Allowed values: {', '.join(param_config.get('allowed_values', []))}[/red]")
            return True
        if param_name == "max_tokens":
            try:
                int_value = int(value)
                if int_value <= 0:
                    raise ValueError("max_tokens must be a positive integer.")
                value = int_value
            except ValueError:
                console.print(f"[red]Error: Invalid value '{value}' for 'max_tokens'. Must be a positive integer.[/red]")
                return True
        elif param_name == "temperature":
            try:
                float_value = float(value)
                if not (0.0 <= float_value <= 2.0):
                    raise ValueError("Temperature must be a float between 0.0 and 2.0.")
                value = float_value
            except ValueError as e:
                console.print(f"[red]Error: Invalid value '{value}' for 'temperature'. Must be a float between 0.0 and 2.0. Details: {e}[/red]")
                return True
        RUNTIME_OVERRIDES[param_name] = value
        console.print(f"[green]✓ Parameter '{param_name}' set to '{value}' for the current session.[/green]")
        return True
    else:
        console.print(f"[red]Error: Unknown parameter '{param_name}'. Supported parameters: {', '.join(SUPPORTED_SET_PARAMS.keys())}[/red]")
        return True

def try_handle_help_command(user_input: str) -> bool:
    command_prefix = "/help"
    if user_input.strip().lower() == command_prefix:
        help_file_path = Path(__file__).parent / "ai_engineer_help.md"
        try:
            help_content = util_read_local_file(str(help_file_path))
            console.print(Panel(
                RichMarkdown(help_content),
                title="[bold blue]📚 AI Engineer Help[/bold blue]",
                title_align="left",
                border_style="blue"
            ))
        except FileNotFoundError:
            console.print(f"[red]Error: Help file not found at '{help_file_path}'[/red]")
        except OSError as e:
            console.print(f"[red]Error reading help file: {e}[/red]")
        return True
    return False

def try_handle_shell_command(user_input: str) -> bool:
    command_prefix = "/shell"
    stripped_input = user_input.strip()
    if not (stripped_input.lower() == command_prefix or stripped_input.lower().startswith(command_prefix + " ")):
        return False
    prefix_with_space = command_prefix + " "
    command_body = ""
    if stripped_input.lower().startswith(prefix_with_space):
        command_body = stripped_input[len(prefix_with_space):].strip()
    if not command_body:
            console.print("[yellow]Usage: /shell <command and arguments>[/yellow]")
            console.print("[yellow]  Example: /shell ls -l[/yellow]")
            console.print("[bold yellow]⚠️ Warning: Executing arbitrary shell commands can be risky.[/bold yellow]")
            return True
    console.print(f"[bold bright_blue]🐚 Executing shell command: '{command_body}'[/bold bright_blue]")
    console.print("[dim]Output:[/dim]")
    try:
        result = subprocess.run(command_body, shell=True, capture_output=True, text=True, check=False)
        output = result.stdout.strip()
        error_output = result.stderr.strip()
        return_code = result.returncode
        if output:
            console.print(output)
        if error_output:
            console.print(f"[red]Stderr:[/red]\n{error_output}")
        if return_code != 0:
            console.print(f"[red]Command exited with non-zero status code: {return_code}[/red]")
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

def try_handle_rules_command(user_input: str) -> bool:
    command_prefix = "/rules"
    stripped_input = user_input.strip()
    if not (stripped_input.lower() == command_prefix or stripped_input.lower().startswith(command_prefix + " ")):
        return False
    command_body = stripped_input[len(command_prefix):].strip()
    parts = command_body.split(maxsplit=1)
    sub_command = parts[0].lower() if parts else ""
    arg = parts[1] if len(parts) > 1 else ""
    if sub_command == "show":
        console.print("\n[bold blue]📚 Current System Prompt (Rules):[/bold blue]")
        if conversation_history and conversation_history[0]["role"] == "system":
            console.print(Panel(
                RichMarkdown(conversation_history[0]["content"]),
                title="[bold blue]System Prompt[/bold blue]",
                title_align="left",
                border_style="blue"
            ))
        else:
            console.print("[yellow]No system prompt found in history.[/yellow]")
        return True
    elif sub_command == "list":
        rules_dir = Path("./.aie_rules/")
        console.print(f"\n[bold blue]📚 Rules files in '[bright_cyan]{rules_dir}[/bright_cyan]':[/bold blue]")
        try:
            files = sorted([f.name for f in rules_dir.iterdir() if f.is_file()])
            if files:
                for file_name in files:
                    console.print(f"  - {file_name}")
            else:
                console.print("[yellow]No rule files found in this directory.[/yellow]")
        except FileNotFoundError:
            console.print(f"[red]Error: Directory '[bright_cyan]{rules_dir}[/bright_cyan]' not found.[/red]")
        except Exception as e:
            console.print(f"[red]Error listing files in '[bright_cyan]{rules_dir}[/bright_cyan]': {e}[/red]")
        return True
    elif sub_command == "add":
        if not arg:
            console.print("[yellow]Usage: /rules add <rule-file>[/yellow]")
            console.print("[yellow]  Example: /rules add ./new_guidelines.md[/yellow]")
            return True
        try:
            normalized_path = normalize_path(arg)
            rule_content = util_read_local_file(normalized_path)
            conversation_history[0]["content"] += f"\n\n## Additional Rules from {normalized_path}:\n\n{rule_content}"
            console.print(f"[green]✓ Added rules from '[bright_cyan]{normalized_path}[/bright_cyan]' to the system prompt for this session.[/green]")
        except (FileNotFoundError, OSError, ValueError) as e:
            console.print(f"[bold red]✗[/bold red] Could not add rules from '[bright_cyan]{arg}[/bright_cyan]': {e}[/bold red]")
        return True
    elif sub_command == "reset":
        if conversation_history and conversation_history[0]["role"] == "system":
            conversation_history[0]["content"] = ""
            RUNTIME_OVERRIDES.pop("system_prompt", None)
            console.print("[green]✓ System prompt emptied.[/green]")
        else:
            console.print("[yellow]Warning: Could not find system prompt in history to reset.[/yellow]")
            conversation_history.insert(0, {"role": "system", "content": system_PROMPT})
        default_rules_dir = Path("./.aie_rules/")
        default_rules_file_name = "_default.md"
        default_rules_path = default_rules_dir / default_rules_file_name
        default_rules_path_str = str(default_rules_path)
        confirmation = prompt_session.prompt(
            f"Load default rules from '[bright_cyan]{default_rules_path_str}[/bright_cyan]'? [Y/n]: ",
            default="y"
        ).strip().lower()
        if confirmation in ["y", "yes", ""]:
            try:
                normalized_path_str_val = "" # Initialize to handle potential NameError
                try:
                    normalized_path_str_val = normalize_path(default_rules_path_str)
                    rule_content = util_read_local_file(normalized_path_str_val)
                    conversation_history[0]["content"] += f"\n\n## Additional Rules from {normalized_path_str_val}:\n\n{rule_content}"
                    console.print(f"[green]✓ Added default rules from '[bright_cyan]{normalized_path_str_val}[/bright_cyan]' to the system prompt.[/green]")
                except FileNotFoundError:
                    console.print(f"[yellow]⚠ Default rules file '[bright_cyan]{normalized_path_str_val or default_rules_path_str}[/bright_cyan]' not found. No default rules loaded.[/yellow]")
            except (OSError, ValueError) as e:
                console.print(f"[bold red]✗[/bold red] Could not load default rules from '[bright_cyan]{default_rules_path_str}[/bright_cyan]': {e}[/bold red]")
        else:
            console.print("[yellow]ℹ️ Default rules not loaded.[/yellow]")
        return True
    else:
        console.print("[yellow]Usage: /rules <show|list|add|reset> [arguments][/yellow]")
        console.print("[yellow]  show                - Display the current system prompt (rules).[/yellow]")
        console.print("[yellow]  list                - List available rule files in ./.aie_rules/.[/yellow]")
        console.print("[yellow]  add <rule-file>     - Add rules from a file to the system prompt.[/yellow]")
        console.print("[yellow]  reset               - Reset system prompt to default, optionally load ./.aie_rules/_default.md.[/yellow]")
        return True

def try_handle_context_command(user_input: str) -> bool:
    command_prefix = "/context"
    stripped_input = user_input.strip()
    if not (stripped_input.lower() == command_prefix or stripped_input.lower().startswith(command_prefix + " ")):
        return False
    prefix_with_space = command_prefix + " "
    command_body = ""
    if stripped_input.lower().startswith(prefix_with_space):
        command_body = stripped_input[len(prefix_with_space):].strip()
    elif stripped_input.lower() == command_prefix:
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
        list_contexts(arg if arg else ".")
        return True
    elif sub_command == "summarize":
        summarize_context()
        return True
    else:
        console.print("[yellow]Usage: /context <save|load|list|summarize> [name/path][/yellow]")
        console.print("[yellow]  save <name>     - Save current context to a file.[/yellow]")
        console.print("[yellow]  load <name>     - Load context from a file.[/yellow]")
        console.print("[yellow]  list [path]     - List saved contexts in a directory.[/yellow]")
        console.print("[yellow]  summarize       - Summarize current context using the LLM.[/yellow]")
        return True

def try_handle_session_command(user_input: str) -> bool:
    stripped_input = user_input.strip()
    if stripped_input.lower().startswith("/session"):
        arguments_part = stripped_input[len("/session"):]
        context_equivalent_input = "/context" + arguments_part
        console.print(f"[dim]Executing '{stripped_input}' as '{context_equivalent_input.strip()}'[/dim]")
        return try_handle_context_command(context_equivalent_input)
    return False

def save_context(name: str):
    file_name = f"context_{name}.json"
    try:
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(conversation_history, f, indent=2)
        console.print(f"[bold blue]✓[/bold blue] Context saved to '[bright_cyan]{file_name}[/bright_cyan]'\n")
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Failed to save context to '{file_name}': {e}\n")

def load_context(name: str):
    file_name = f"context_{name}.json"
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            loaded_history = json.load(f)
        if not isinstance(loaded_history, list) or not all(isinstance(msg, dict) and "role" in msg for msg in loaded_history):
             raise ValueError("Invalid context file format.")
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
    try:
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
        console.print()
    except ValueError as e:
        console.print(f"[bold red]✗[/bold red] Invalid path '[bright_cyan]{path}[/bright_cyan]': {e}\n")
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Failed to list contexts in '{path}': {e}\n")

def summarize_context():
    global conversation_history
    if len(conversation_history) <= 1:
        console.print("[yellow]No conversation history to summarize.[/yellow]\n")
        return
    console.print("[bold bright_blue]✨ Summarizing conversation history...[/bold bright_blue]")
    summary_messages = [
        conversation_history[0],
        {"role": "user", "content": "Please provide a concise summary of our conversation so far. Focus on the key topics discussed, decisions made, and actions taken (like file operations). This summary will replace the detailed history."}
    ]
    summary_messages.extend(conversation_history[1:])
    try:
        model_name = get_config_value("model", "gpt-4o")
        api_base_url = get_config_value("api_base", None)
        response = completion(
            model=model_name,
            messages=summary_messages,
            temperature=0.3,
            max_tokens=1024,
            api_base=api_base_url,
            stream=False
        )
        summary_content = response.choices[0].message.content
        if summary_content:
            console.print("\n[bold blue]Summary:[/bold blue]")
            console.print(Panel(summary_content, border_style="blue"))
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

def _call_llm_for_prompt_generation(user_text: str, mode: str) -> str:
    console.print(f"[bold bright_blue]⚙️ Processing text for prompt {mode}ing...[/bold bright_blue]")
    if mode == "refine":
        meta_system_prompt = dedent("""\
            You are a prompt engineering assistant. Your task is to refine the following user-provided text into an optimized prompt suitable for an AI coding assistant like AI Engineer. The refined prompt should be clear, concise, and actionable, guiding the AI to provide the best possible coding assistance.
            The output should be ONLY the refined prompt text, without any preamble or explanation.
            """)
        user_query = f"Refine this text into a prompt for an AI coding assistant:\n\n---\n{user_text}\n---"
    elif mode == "detail":
        meta_system_prompt = dedent("""\
            You are a prompt engineering assistant. Your task is to expand the following user-provided text into a more detailed and comprehensive prompt suitable for an AI coding assistant like AI Engineer. The detailed prompt should elaborate on the user's initial idea, adding necessary context, specifying desired outcomes, and anticipating potential ambiguities to guide the AI effectively.
            The output should be ONLY the detailed prompt text, without any preamble or explanation.
            """)
        user_query = f"Expand this text into a detailed prompt for an AI coding assistant:\n\n---\n{user_text}\n---"
    else:
        return "Error: Invalid mode for prompt generation."
    messages = [
        {"role": "system", "content": meta_system_prompt},
        {"role": "user", "content": user_query}
    ]
    try:
        model_name = get_config_value("model", "gpt-4o")
        api_base_url = get_config_value("api_base", None)
        max_tokens_val = get_config_value("max_tokens", 2048)
        response = completion(
            model=model_name,
            messages=messages,
            temperature=0.5,
            max_tokens=max_tokens_val,
            api_base=api_base_url,
            stream=False
        )
        generated_prompt = response.choices[0].message.content.strip()
        return generated_prompt
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Failed to generate prompt: {e}\n")
        return ""

def try_handle_prompt_command(user_input: str) -> bool:
    prefix = "/prompt "
    command_name_lower = "/prompt"
    stripped_input = user_input.strip()
    if not (stripped_input.lower() == command_name_lower or stripped_input.lower().startswith(prefix)):
        return False
    command_body = ""
    if stripped_input.lower().startswith(prefix):
        command_body = stripped_input[len(prefix):].strip()
    parts = command_body.split(maxsplit=1)
    sub_command = parts[0].lower() if parts else ""
    text_to_process = parts[1] if len(parts) > 1 else ""
    if sub_command in ["refine", "detail"]:
        if not text_to_process:
            console.print(f"[yellow]Usage: /prompt {sub_command} <text_to_{sub_command}>[/yellow]")
            return True
        generated_prompt = _call_llm_for_prompt_generation(text_to_process, sub_command)
        if generated_prompt:
            console.print(f"\n[bold blue]✨ Generated Prompt ({sub_command.capitalize()}d):[/bold blue]")
            console.print(Panel(generated_prompt, border_style="green", expand=True, title_align="left"))
        return True
    console.print("[yellow]Usage: /prompt <refine|detail> <text>[/yellow]")
    console.print("[yellow]  refine <text>  - Optimizes <text> into a clearer and more effective prompt for AI Engineer.[/yellow]")
    console.print("[yellow]  detail <text>  - Expands <text> into a more comprehensive and detailed prompt for AI Engineer.[/yellow]")
    return True

def add_directory_to_conversation(directory_path: str):
    with console.status("[bold bright_blue]🔍 Scanning directory...[/bold bright_blue]") as status:
        excluded_files = {
            ".DS_Store", "Thumbs.db", ".gitignore", ".python-version",
            "uv.lock", ".uv", "uvenv", ".uvenv", ".venv", "venv",
            "__pycache__", ".pytest_cache", ".coverage", ".mypy_cache",
            "node_modules", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
            ".next", ".nuxt", "dist", "build", ".cache", ".parcel-cache",
            ".turbo", ".vercel", ".output", ".contentlayer",
            "out", "coverage", ".nyc_output", "storybook-static",
            ".env", ".env.local", ".env.development", ".env.production",
            ".git", ".svn", ".hg", "CVS"
        }
        excluded_extensions = {
            ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp", ".avif",
            ".mp4", ".webm", ".mov", ".mp3", ".wav", ".ogg",
            ".zip", ".tar", ".gz", ".7z", ".rar",
            ".exe", ".dll", ".so", ".dylib", ".bin",
            ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
            ".pyc", ".pyo", ".pyd", ".egg", ".whl",
            ".uv", ".uvenv",
            ".db", ".sqlite", ".sqlite3", ".log",
            ".idea", ".vscode",
            ".map", ".chunk.js", ".chunk.css",
            ".min.js", ".min.css", ".bundle.js", ".bundle.css",
            ".cache", ".tmp", ".temp",
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
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in excluded_files]
            for file in files:
                if total_files_processed >= MAX_FILES_TO_PROCESS_IN_DIR:
                    break
                if file.startswith('.') or file in excluded_files:
                    skipped_files.append(str(Path(root) / file))
                    continue
                _, ext = os.path.splitext(file)
                if ext.lower() in excluded_extensions:
                    skipped_files.append(os.path.join(root, file))
                    continue
                full_path = str(Path(root) / file)
                try:
                    if os.path.getsize(full_path) > MAX_FILE_SIZE_BYTES:
                        skipped_files.append(f"{full_path} (exceeds size limit)")
                        continue
                    if is_binary_file(full_path):
                        skipped_files.append(full_path)
                        continue
                    normalized_path = normalize_path(full_path)
                    content = util_read_local_file(normalized_path)
                    conversation_history.append({
                        "role": "system",
                        "content": f"Content of file '{normalized_path}':\n\n{content}"
                    })
                    added_files.append(normalized_path)
                    total_files_processed += 1
                except OSError:
                    skipped_files.append(str(full_path))
                except ValueError as e:
                     skipped_files.append(f"{full_path} (Invalid path: {e})")
        console.print(f"[bold blue]✓[/bold blue] Added folder '[bright_cyan]{directory_path}[/bright_cyan]' to conversation.")
        if added_files:
            console.print(f"\n[bold bright_blue]📁 Added files:[/bold bright_blue] [dim]({len(added_files)} of {total_files_processed})[/dim]")
            for f_path in added_files:
                console.print(f"  [bright_cyan]📄 {f_path}[/bright_cyan]")
        if skipped_files:
            console.print(f"\n[bold yellow]⏭ Skipped files:[/bold yellow] [dim]({len(skipped_files)})[/dim]")
            for f_path in skipped_files[:10]:
                console.print(f"  [yellow dim]⚠ {f_path}[/yellow dim]")
            if len(skipped_files) > 10:
                console.print(f"  [dim]... and {len(skipped_files) - 10} more[/dim]")
        console.print()

def ensure_file_in_context(file_path: str) -> bool:
    try:
        normalized_path = normalize_path(file_path)
        content = util_read_local_file(normalized_path)
        file_marker = f"Content of file '{normalized_path}'"
        if not any(file_marker in msg["content"] for msg in conversation_history if msg.get("content")):
            conversation_history.append({
                "role": "system",
                "content": f"{file_marker}:\n\n{content}"
            })
        return True
    except (OSError, ValueError) as e:
        console.print(f"[bold red]✗[/bold red] Could not read file '[bright_cyan]{file_path}[/bright_cyan]' for editing context: {e}")
        return False

# --------------------------------------------------------------------------------
# 5. Conversation state
# --------------------------------------------------------------------------------
conversation_history = [
    {"role": "system", "content": system_PROMPT}
]

# --------------------------------------------------------------------------------
# 6. LLM API interaction with streaming
# --------------------------------------------------------------------------------

def execute_function_call_dict(tool_call_dict) -> str:
    function_name = "unknown_function"
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
            for file_path in file_paths:
                try:
                    normalized_path = normalize_path(file_path)
                    content = util_read_local_file(normalized_path)
                    results.append(f"Content of file '{normalized_path}':\n\n{content}")
                except (OSError, ValueError) as e:
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
            success_messages_for_result = []
            first_error_detail_for_return = None
            for file_info_data in files_to_create_data:
                path = file_info_data.get("path", "unknown_path")
                content = file_info_data.get("content", "")
                try:
                    util_create_file(path, content, console, MAX_FILE_SIZE_BYTES)
                    successful_paths.append(path)
                    success_messages_for_result.append(f"File {path} created.")
                except Exception as e_create:
                    console.print(f"[red]Error creating file {path}: {str(e_create)}[/red]")
                    if not first_error_detail_for_return:
                        first_error_detail_for_return = f"Error during create_multiple_files: {str(e_create)}"
            if first_error_detail_for_return:
                return "\n".join(success_messages_for_result) + "\n" + first_error_detail_for_return if success_messages_for_result else first_error_detail_for_return
            return f"Successfully created {len(successful_paths)} files: {', '.join(successful_paths)}"
        elif function_name == "edit_file":
            file_path = arguments["file_path"]
            original_snippet = arguments["original_snippet"]
            new_snippet = arguments["new_snippet"]
            if not ensure_file_in_context(file_path):
                return f"Error: Could not read file '{file_path}' for editing"
            util_apply_diff_edit(file_path, original_snippet, new_snippet, console, MAX_FILE_SIZE_BYTES)
            return f"Successfully edited file '{file_path}'"
        elif function_name == "connect_local_mcp_stream":
            endpoint_url = arguments["endpoint_url"]
            timeout_seconds = arguments.get("timeout_seconds", 30)
            max_data_chars = arguments.get("max_data_chars", 10000)
            try:
                parsed_url = httpx.URL(endpoint_url)
                if not (parsed_url.host.lower() in ("localhost", "127.0.0.1") and parsed_url.scheme.lower() in ("http", "https")):
                     return f"Error: For connect_local_mcp_stream, endpoint_url must be for localhost (http or https). Provided: {endpoint_url}"
            except Exception as e_val:
                return f"Error validating local MCP stream URL '{endpoint_url}' before execution: {str(e_val)}"
            return _handle_local_mcp_stream(endpoint_url, timeout_seconds, max_data_chars)
        elif function_name == "connect_remote_mcp_sse":
            endpoint_url = arguments["endpoint_url"]
            max_events = arguments.get("max_events", 10)
            listen_timeout_seconds = arguments.get("listen_timeout_seconds", 60)
            try:
                parsed_url = httpx.URL(endpoint_url)
                if parsed_url.scheme.lower() not in ("http", "https"):
                    return f"Error: For connect_remote_mcp_sse, endpoint_url must be a valid HTTP/HTTPS URL. Provided: {endpoint_url}"
            except Exception as e_val:
                return f"Error validating remote MCP SSE URL '{endpoint_url}' before execution: {str(e_val)}"
            return _handle_remote_mcp_sse(endpoint_url, max_events, listen_timeout_seconds)
        else:
            return f"Unknown function: {function_name}"
    except Exception as e:
        error_message = f"Error executing {function_name}: {str(e)}"
        console.print(f"[red]{error_message}[/red]")
        return error_message

def trim_conversation_history():
    if len(conversation_history) <= 20:
        return
    system_msgs = [msg for msg in conversation_history if msg["role"] == "system"]
    other_msgs = [msg for msg in conversation_history if msg["role"] != "system"]
    if len(other_msgs) > 15:
        other_msgs = other_msgs[-15:]
    conversation_history.clear()
    conversation_history.extend(system_msgs + other_msgs)

def stream_llm_response(user_message: str):
    trim_conversation_history()
    try:
        messages_for_api_call = copy.deepcopy(conversation_history)
        default_reply_effort_val = "medium"
        default_temperature_val = 0.7
        model_name = get_config_value("model", DEFAULT_LITELLM_MODEL)
        api_base_url = get_config_value("api_base", DEFAULT_LITELLM_API_BASE)
        reasoning_style = str(get_config_value("reasoning_style", DEFAULT_REASONING_STYLE)).lower()
        max_tokens_raw = get_config_value("max_tokens", DEFAULT_LITELLM_MAX_TOKENS)
        try:
            max_tokens = int(max_tokens_raw)
            if max_tokens <= 0:
                max_tokens = DEFAULT_LITELLM_MAX_TOKENS
        except (ValueError, TypeError):
            max_tokens = DEFAULT_LITELLM_MAX_TOKENS
        temperature_raw = get_config_value("temperature", default_temperature_val)
        try:
            temperature = float(temperature_raw)
        except (ValueError, TypeError):
            console.print(f"[yellow]Warning: Invalid temperature value '{temperature_raw}'. Using default {default_temperature_val}.[/yellow]")
            temperature = default_temperature_val
        reasoning_effort_setting = str(get_config_value("reasoning_effort", DEFAULT_REASONING_EFFORT)).lower()
        reply_effort_setting = str(get_config_value("reply_effort", default_reply_effort_val)).lower()
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
        reasoning_content_accumulated = ""
        final_content = ""
        tool_calls = []
        reasoning_started_printed = False
        stream = completion(
            model=model_name,
            messages=messages_for_api_call,
            tools=tools,
            max_tokens=max_tokens,
            api_base=api_base_url,
            temperature=temperature,
            stream=True
        )
        for chunk in stream:
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
                        console.print("\n[bold blue]💭 Reasoning...[/bold blue]", end="")
                        reasoning_started_printed = True
                    console.print(".", end="")
            elif chunk.choices[0].delta.content:
                if reasoning_started_printed and reasoning_style != "full":
                    console.print()
                    reasoning_started_printed = False
                if not final_content:
                    console.print("\n\n[bold bright_blue]🤖 Assistant>[/bold bright_blue] ", end="")
                final_content += chunk.choices[0].delta.content
                console.print(chunk.choices[0].delta.content, end="")
            elif chunk.choices[0].delta.tool_calls:
                if reasoning_started_printed and reasoning_style != "full":
                    console.print()
                    reasoning_started_printed = False
                for tool_call_delta in chunk.choices[0].delta.tool_calls:
                    if tool_call_delta.index is not None:
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
            console.print()
        console.print()
        conversation_history.append({"role": "user", "content": user_message})
        assistant_message = {
            "role": "assistant",
            "content": final_content if final_content else None
        }
        if reasoning_content_accumulated:
            assistant_message["reasoning_content_full"] = reasoning_content_accumulated
        if tool_calls:
            formatted_tool_calls = []
            for i, tc in enumerate(tool_calls):
                if tc["function"]["name"]:
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
                if not final_content:
                    assistant_message["content"] = None
                assistant_message["tool_calls"] = formatted_tool_calls
                conversation_history.append(assistant_message)
                console.print(f"\n[bold bright_cyan]⚡ Executing {len(formatted_tool_calls)} function call(s)...[/bold bright_cyan]")
                executed_tool_call_ids_and_results = []
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
            conversation_history.append(assistant_message)
        return {"success": True}
    except Exception as e:
        error_msg = f"LLM API error: {str(e)}"
        console.print(f"\n[bold red]❌ {error_msg}[/bold red]")
        return {"error": error_msg}

# --------------------------------------------------------------------------------
# 7. Test & Main interactive loop
# --------------------------------------------------------------------------------

def _summarize_error_message(error_message: str, summary_model_name: str, api_base: str) -> str:
    """Uses an LLM to summarize an error message concisely."""
    if not error_message:
        return ""
    
    # Limit the error message length to avoid excessive token usage for summarization
    max_error_len_for_summary = 1000
    truncated_error_message = error_message[:max_error_len_for_summary]
    if len(error_message) > max_error_len_for_summary:
        truncated_error_message += "..."

    prompt = f"Summarize the following technical error message very concisely (e.g., in 5-10 words or a short phrase). Focus on the core issue:\n\n---\n{truncated_error_message}\n---\n\nConcise Summary:"
    
    try:
        response = completion(
            model=summary_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=50, # Allow short summary
            api_base=api_base,
            stream=False
        )
        summary = response.choices[0].message.content.strip()
        return f"Summary: {summary}" if summary else error_message # Fallback to original if summary is empty
    except Exception as e:
        # If summarization fails, return the original (potentially truncated) error
        # console.print(f"[dim]Error summarizing error message: {e}[/dim]")
        return truncated_error_message # Or just error_message if you prefer full original on summary failure

def _test_single_model_capabilities(model_label: str, model_name_to_test: str, api_base_to_test: str, expect_tools: bool) -> Dict:
    """Helper function to test capabilities of a single model."""
    results = {
        "label": model_label,
        "name": model_name_to_test,
        "available": "N",
        "tool_support": "N/A",
        "context_kb": "N/A",
        "inference_time_s": "N/A",
        "error_details": ""
    }
    if not model_name_to_test:
        results["error_details"] = "Model name not configured."
        results["available"] = "Skipped"
        return results

    console.print(f"\n[bold blue]🧪 Testing Model: [cyan]{model_label} ({model_name_to_test})[/cyan]...[/bold blue]")
    temperature_for_test = 0.1
    start_time = time.time()

    test_messages = [{"role": "user", "content": "Test: Respond with 'ok'."}]
    api_key_name_hint = "API key"
    if api_base_to_test:
        if "openai" in api_base_to_test.lower() or "gpt-" in model_name_to_test.lower():
            api_key_name_hint = "OPENAI_API_KEY"
        elif "deepseek" in api_base_to_test.lower() or "deepseek" in model_name_to_test.lower():
            api_key_name_hint = "DEEPSEEK_API_KEY"
        elif "anthropic" in api_base_to_test.lower() or "claude" in model_name_to_test.lower():
            api_key_name_hint = "ANTHROPIC_API_KEY"
        elif "openrouter" in api_base_to_test.lower():
            api_key_name_hint = "OPENROUTER_API_KEY"

    api_base_for_call = api_base_to_test # Start with the global/passed base
    pre_call_notes = [] # List to store notes to print before the "Attempting..." line

    # Heuristic to decide if we should override the global api_base for this specific model test.
    # This allows testing direct provider models when the global LITELLM_API_BASE is set
    # to a different provider or an aggregator that might not proxy this specific model correctly.
    if model_name_to_test and api_base_for_call: # Only adjust if a global base is actually set
        provider_from_model_name = model_name_to_test.split('/')[0].lower()
        
        is_ollama_model = provider_from_model_name.startswith("ollama") # Covers "ollama" and "ollama_chat"

        # Known direct provider domains that LiteLLM has good defaults for.
        # Keys are provider prefixes (lowercase), values are parts of their default domain.
        known_direct_providers_domains = {
            "openai": "api.openai.com",
            "anthropic": "api.anthropic.com",
            "deepseek": "api.deepseek.com", # Native DeepSeek API
            "google": "googleapis.com",    # For Gemini models (Vertex AI or AI Studio)
            "cohere": "api.cohere.ai",
            "cerebras": "api.cerebras.com" # LiteLLM's default for Cerebras
        }

        if is_ollama_model:
            # If it's an Ollama model, and the global api_base is NOT an Ollama-like URL (typically http://localhost...),
            # then set api_base_for_call to None to let LiteLLM use OLLAMA_API_BASE env var or its default.
            # Check if the global base is NOT http or NOT localhost/127.0.0.1
            if not api_base_for_call.lower().startswith("http://") or \
               not ("localhost" in api_base_for_call.lower() or "127.0.0.1" in api_base_for_call.lower()):
                api_base_for_call = None
                # pre_call_notes.append(f"[dim]  (Note: Testing Ollama model '{model_name_to_test}' with LiteLLM's default Ollama endpoint resolution, not global '{api_base_to_test}')[/dim]")
        elif provider_from_model_name in known_direct_providers_domains and \
             known_direct_providers_domains[provider_from_model_name] not in api_base_for_call.lower():
            if provider_from_model_name == "google":
                api_base_for_call = "https://generativelanguage.googleapis.com"
                pre_call_notes.append(f"[dim]  (Note: Testing '{model_name_to_test}' with explicit Google API base: '{api_base_for_call}')[/dim]")
            else:
                # For other direct providers, let LiteLLM use its default by setting api_base to None
                api_base_for_call = None

    if "gemini" in model_name_to_test.lower() or "google" in model_name_to_test.lower() : # Broaden check for Gemini
        pre_call_notes.append(f"[bold yellow blink]DEBUG Gemini Test Params:[/bold yellow blink] model='{model_name_to_test}', api_base_for_call='{api_base_for_call}'")

    # Prepare keyword arguments for litellm.completion
    completion_kwargs = {}
    if model_name_to_test and "google" in model_name_to_test.split('/')[0].lower() and \
       api_base_for_call == "https://generativelanguage.googleapis.com":
        completion_kwargs["proxies"] = None 
        pre_call_notes.append(f"[dim]  (Note: Explicitly disabling proxies for direct Google API call to '{model_name_to_test}')[/dim]")

    # Print all collected notes before the "Attempting..." line
    for note in pre_call_notes:
        console.print(note)

    console.print("[yellow]  1. Attempting basic API call...[/yellow]", end="")

    try:
        response = completion(
            model=model_name_to_test,
            messages=test_messages,
            api_base=api_base_for_call, # Use the potentially adjusted API base
            temperature=temperature_for_test,
            max_tokens=20,
            timeout=30,
            **completion_kwargs # Pass additional kwargs like proxies
        )
        response_content = response.choices[0].message.content
        console.print(f"[green] ✓ OK[/green] (LLM: \"{response_content.strip()}\")")
        results["available"] = "Y"
        results["inference_time_s"] = f"{time.time() - start_time:.2f}"

        console.print("[yellow]  2. Testing token counting & context...[/yellow]", end="")
        try:
            context_size, _ = get_model_context_window(model_name_to_test, return_match_status=True)
            results["context_kb"] = f"{context_size // 1000}k"
            console.print(f"[green] ✓ OK[/green] (Context: {results['context_kb']})")
        except Exception as e_tc:
            console.print(f"[yellow] ⚠️ Failed[/yellow] ({e_tc})")
            results["context_kb"] = "Error"

        console.print("[yellow]  3. Testing tool calling capability...[/yellow]", end="")
        if expect_tools:
            dummy_tool_for_test = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_dummy_data_for_test",
                        "description": "A dummy function to test tool calling. Retrieves dummy data.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "data_type": {
                                    "type": "string",
                                    "description": "The type of dummy data to retrieve, e.g., 'text'."
                                }
                            },
                            "required": ["data_type"]
                        }
                    }
                }
            ]
            tool_test_messages = [{"role": "user", "content": "Can you use a tool to get dummy text data for a test?"}]
            try:
                tool_response = completion(
                    model=model_name_to_test,
                    messages=tool_test_messages,
                    tools=dummy_tool_for_test,
                    api_base=api_base_for_call, # Use the potentially adjusted API base
                    temperature=temperature_for_test,
                    max_tokens=150,
                    timeout=30,
                    **completion_kwargs # Pass additional kwargs like proxies
                )
                if tool_response.choices[0].message.tool_calls and len(tool_response.choices[0].message.tool_calls) > 0:
                    console.print(f"[green] ✓ Yes[/green] (Called: '{tool_response.choices[0].message.tool_calls[0].function.name}')")
                    results["tool_support"] = "Y"
                elif tool_response.choices[0].message.content:
                    console.print(f"[yellow] ⚠️ No[/yellow] (Responded with text)")
                    results["tool_support"] = "N"
                else:
                    console.print(f"[yellow] ⚠️ Inconclusive[/yellow]")
                    results["tool_support"] = "N"
            except Exception as e_tool_call:
                console.print(f"[red] ❌ Error[/red] ({e_tool_call})")
                results["tool_support"] = "Error"
                if not results["error_details"]: results["error_details"] = f"Tool call test: {e_tool_call}"
        else:
            console.print(" [dim]N/A[/dim]")
            results["tool_support"] = "N/A"
    except httpx.ConnectError as e:
        console.print(f"[red] ❌ Connection Error[/red] ({api_base_to_test or 'Default LiteLLM endpoint'})")
        results["available"] = "N"
        results["error_details"] = f"Connection Error: {e}"
    except httpx.TimeoutException as e:
        console.print(f"[red] ❌ Timeout[/red]")
        results["available"] = "N"
        results["error_details"] = f"Timeout: {e}"
    except Exception as e:
        error_str = str(e)
        # Display a truncated version of the error directly on the CLI
        console.print(f"[red] ❌ API Error[/red]: {error_str[:250]}{'...' if len(error_str) > 250 else ''}")
        results["available"] = "N"
        results["error_details"] = error_str
        if "authentication" in error_str.lower() or "api key" in error_str.lower() or "401" in error_str:
            results["error_details"] += f" (Hint: Check {api_key_name_hint})"
        elif "model_not_found" in error_str.lower() or ("404" in error_str and "Model" in error_str):
            results["error_details"] += f" (Hint: Model '{model_name_to_test}' not found or misspelled at '{api_base_to_test}')"
    if results["available"] == "N" and results["inference_time_s"] == "N/A":
         results["inference_time_s"] = f"{time.time() - start_time:.2f}"
    return results

def test_inference_endpoint(specific_model_name: str = None):
    """Tests all configured inference endpoints and capabilities, then exits."""
    if specific_model_name:
        console.print(f"[bold blue]🧪 Testing Specific Model: [cyan]{specific_model_name}[/cyan]...[/bold blue]")
    else:
        console.print("[bold blue]🧪 Testing All Configured Inference Endpoints & Capabilities...[/bold blue]")
        console.print("[dim]Note: This test covers models from MODEL_CONTEXT_WINDOWS and explicitly configured role-based models.[/dim]")
    
    api_base_url = get_config_value("api_base", DEFAULT_LITELLM_API_BASE)
    role_based_models_config = [
        {"label": "DEFAULT",   "name_var": LITELLM_MODEL_DEFAULT,   "expect_tools": True},
        {"label": "ROUTING",   "name_var": LITELLM_MODEL_ROUTING,   "expect_tools": False}, # Usually no tools
        {"label": "TOOLS",     "name_var": LITELLM_MODEL_TOOLS,     "expect_tools": True},
        {"label": "CODING",    "name_var": LITELLM_MODEL_CODING,    "expect_tools": False}, # Usually no general tools
        {"label": "KNOWLEDGE", "name_var": LITELLM_MODEL_KNOWLEDGE, "expect_tools": False}, # Usually no tools
    ]
    model_details = {}
    default_model_available = False
    all_results = []
    overall_success = True
    
    # First, gather all known models from MODEL_CONTEXT_WINDOWS and roles
    all_known_models_set = set(MODEL_CONTEXT_WINDOWS.keys())
    for config in role_based_models_config:
        model_name_val = config["name_var"]
        if model_name_val:
            all_known_models_set.add(model_name_val)

    models_to_test_set = set()
    if specific_model_name:
        if "*" in specific_model_name or "?" in specific_model_name: # Wildcard detected
            console.print(f"[dim]Filtering models with wildcard: '{specific_model_name}'[/dim]")
            for model_n in all_known_models_set:
                if fnmatch.fnmatch(model_n, specific_model_name):
                    models_to_test_set.add(model_n)
            if not models_to_test_set:
                console.print(f"[yellow]Warning: No models matched the wildcard pattern '{specific_model_name}'.[/yellow]")
        else: # Specific model name without wildcard
            models_to_test_set = {specific_model_name}
            if specific_model_name not in all_known_models_set:
                # This warning is useful if a user types a specific name not in our known lists.
                # We'll still attempt to test it directly.
                console.print(f"[yellow]Warning: Model '{specific_model_name}' is not in the known model lists (MODEL_CONTEXT_WINDOWS or configured roles). Testing it directly.[/yellow]")
    else: # Test all known models
        models_to_test_set = all_known_models_set

    # Populate model_details for roles, regardless of single or all test mode
    model_details = {} # Re-initialize here to ensure it's fresh based on models_to_test_set
    for config in role_based_models_config:
        model_name_val = config["name_var"] # Use a different variable name to avoid confusion
        if model_name_val: # Check if the model name from config is not None or empty
            if model_name_val not in model_details:
                model_details[model_name_val] = {"roles": [], "expect_tools": config["expect_tools"]}
            model_details[model_name_val]["roles"].append(config["label"])
            if config["expect_tools"]: # If any role for this model expects tools, set it to true for the test
                model_details[model_name_val]["expect_tools"] = True
    
    if not models_to_test_set:
        console.print("[yellow]Warning: No models specified or found to test.[/yellow]")
        sys.exit(0)

    for model_name_iter in sorted(list(models_to_test_set)): # Use a different loop variable
        if not model_name_iter:
            all_results.append({
                "label": "Invalid/Empty", "name": "Not Configured", "available": "N/A",
                "tool_support": "N/A", "context_kb": "N/A", "inference_time_s": "N/A", "error_details": "Empty model name encountered."
            })
            continue

        current_model_details = model_details.get(model_name_iter)
        display_label = model_name_iter
        expect_tools_for_this_model = True # Default for models only from MODEL_CONTEXT_WINDOWS

        if current_model_details:
            display_label = f"{model_name_iter} ({', '.join(current_model_details['roles'])})"
            expect_tools_for_this_model = current_model_details['expect_tools']
        elif model_name_iter in MODEL_CONTEXT_WINDOWS and not specific_model_name: # Only add (Context Map) if not specifically testing
            display_label = f"{model_name_iter} (Context Map)"
        
        api_base_for_model = api_base_url
        result = _test_single_model_capabilities(
            model_label=display_label,
            model_name_to_test=model_name_iter,
            api_base_to_test=api_base_for_model,
            expect_tools=expect_tools_for_this_model
        )
        all_results.append(result)
        if result["available"] != "Y":
            overall_success = False
        if model_name_iter == LITELLM_MODEL_DEFAULT and result["available"] == "Y": # Check if the default model is one of the tested and available
            default_model_available = True
            
    # Summarize errors if DEFAULT model is available
    if SHOW_TEST_INFERENCE_NOTES_ERRORS_COLUMN: # Only summarize if the column will be shown
        if default_model_available:
            console.print(f"\n[dim]Default model ([cyan]{LITELLM_MODEL_DEFAULT}[/cyan]) is available. Attempting to summarize error messages for other models...[/dim]")
            for res in all_results:
                # Ensure error_details is a string before attempting to summarize
                # This handles cases where it might be None or another type if a model was skipped early.
                if not isinstance(res.get("error_details"), str):
                    res["error_details"] = str(res.get("error_details", "")) # Convert to string or empty string

                # Only summarize for other models that failed and have error details
                if res["name"] != LITELLM_MODEL_DEFAULT and res["available"] == "N" and res["error_details"]:
                    original_error = res["error_details"]
                    res["error_details"] = _summarize_error_message(original_error, LITELLM_MODEL_DEFAULT, api_base_url)


    console.print("\n\n[bold green]📊 Inference Test Summary[/bold green]")
    summary_table = Table(title="Model Capabilities Test Results", show_lines=True)
    summary_table.add_column("Model / Role", style="cyan", no_wrap=True, max_width=50)
    summary_table.add_column("Model Name (Actual)", style="magenta", max_width=40)
    summary_table.add_column("Available", justify="center")
    summary_table.add_column("Tool Support", justify="center")
    summary_table.add_column("Context", justify="right")
    summary_table.add_column("Time (s)", justify="right")
    if SHOW_TEST_INFERENCE_NOTES_ERRORS_COLUMN:
        notes_errors_column_width = console.width // 3 if console.width > 120 else 60
        summary_table.add_column("Notes/Errors", style="dim", overflow="fold", max_width=notes_errors_column_width)

    for res in all_results:
        available_style = "green" if res["available"] == "Y" else "red" if res["available"] == "N" else "yellow"
        tool_style = "green" if res["tool_support"] == "Y" else "red" if res["tool_support"] == "N" else "dim"
            
        row_data = [res["label"], res["name"], f"[{available_style}]{res['available']}[/{available_style}]", f"[{tool_style}]{res['tool_support']}[/{tool_style}]", res["context_kb"], res["inference_time_s"]]
        if SHOW_TEST_INFERENCE_NOTES_ERRORS_COLUMN:
            row_data.append(res["error_details"])
        summary_table.add_row(*row_data)

    # Calculate and add Totals/Summary Stats row
    total_models_tested = len([res for res in all_results if res["available"] != "N/A" and res["name"] != "Not Configured"])
    total_available_y = sum(1 for res in all_results if res["available"] == "Y")
    total_available_n = sum(1 for res in all_results if res["available"] == "N") 
    total_tool_support_y = sum(1 for res in all_results if res["tool_support"] == "Y")
    
    # Context stats
    total_context_known = sum(1 for res in all_results if res["context_kb"] != "N/A" and res["context_kb"] != "Error")
    total_context_unknown_or_error = sum(1 for res in all_results if res["context_kb"] == "N/A" or res["context_kb"] == "Error")

    summary_table.add_section() # Adds a visual separator line
    overall_stats_row_data = [
        "[bold]Overall Stats[/bold]",
        f"[dim]{total_models_tested} models tested[/dim]",
        f"[bold green]{total_available_y}Y[/bold green] / [bold red]{total_available_n}N[/bold red]", 
        f"[bold green]{total_tool_support_y}Y[/bold green]", 
        f"[bold green]{total_context_known}✓[/bold green] / [bold red]{total_context_unknown_or_error}?[/bold red]", 
        "N/A", 
    ]
    if SHOW_TEST_INFERENCE_NOTES_ERRORS_COLUMN:
        overall_stats_row_data.append("") 

    summary_table.add_row(*overall_stats_row_data)
    console.print(summary_table)
    if overall_success:
        console.print("\n[bold green]✅ All actively tested models passed basic availability checks.[/bold green]")
    else:
        console.print("\n[bold red]❌ Some models failed availability checks. Please review the errors above and your .env configuration.[/bold red]")
    
    if any(r['available'] == 'N' for r in all_results if r['name'] != "Not Configured"):
        sys.exit(1)
    else:
        sys.exit(0)

def clear_screen():
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')

def main():
    parser = argparse.ArgumentParser(
        description="AI Engineer: An AI-powered coding assistant.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )
    parser.add_argument(
        '--script',
        metavar='SCRIPT_PATH',
        type=str,
        help='Path to a script file to execute on startup.'
    )
    parser.add_argument(
        '--noconfirm',
        action='store_true',
        help='Skip confirmation prompts when using --script.'
    )
    parser.add_argument(
        '--time',
        action='store_true',
        help='Enable timestamp display in the user prompt.'
    )
    parser.add_argument(
        '--test-inference',
        metavar='MODEL_NAME',
        type=str,
        nargs='?', # Makes the argument optional
        const='__TEST_ALL_MODELS__', # Special value if flag is present without an argument
        help='Test capabilities. If MODEL_NAME is provided (wildcards * and ? supported), tests matching models. Otherwise, tests all known/configured models.'
    )
    args = parser.parse_args()
    clear_screen()
    if args.test_inference is not None: # If the flag was used
        if args.test_inference == '__TEST_ALL_MODELS__':
            test_inference_endpoint(specific_model_name=None) # Test all
        else:
            test_inference_endpoint(specific_model_name=args.test_inference) # Test specific model

    current_model_name_for_display = get_config_value("model", LITELLM_MODEL_DEFAULT)
    context_window_size, used_default = get_model_context_window(current_model_name_for_display, return_match_status=True)
    context_window_display = f"{context_window_size // 1000}k tokens"
    if used_default:
        context_window_display += " (default)"
    instructions = f"""🧠 Default Model: [bold magenta]{current_model_name_for_display}[/bold magenta] ([dim]{context_window_display}[/dim])
   Routing: [dim]{LITELLM_MODEL_ROUTING or 'Not Set'}[/dim] | Tools: [dim]{LITELLM_MODEL_TOOLS or 'Not Set'}[/dim]
   Coding: [dim]{LITELLM_MODEL_CODING or 'Not Set'}[/dim] | Knowledge: [dim]{LITELLM_MODEL_KNOWLEDGE or 'Not Set'}[/dim]

[bold bright_blue]🎯 Commands:[/bold bright_blue]
  • [bright_cyan]/exit[/bright_cyan] or [bright_cyan]/quit[/bright_cyan] - End the session.
  • [bright_cyan]/help[/bright_cyan] - Display detailed help.

[bold white]👥 Just ask naturally, like you are communicating with a colleague.[/bold white]"""
    console.print(Panel(
        instructions,
        border_style="blue",
        padding=(1, 2),
        title="[bold blue]🤖 AI Code Assistant (Multi-Model)[/bold blue]",
        title_align="left"
    ))
    console.print()
    global SHOW_TIMESTAMP_IN_PROMPT
    if args.time:
        SHOW_TIMESTAMP_IN_PROMPT = True
        console.print("[green]✓ Timestamp display in prompt enabled via --time flag.[/green]")
    if args.script:
        if not args.noconfirm:
            confirmation = prompt_session.prompt(
                f"Execute script '[bright_cyan]{args.script}[/bright_cyan]'? [y/N]: ",
                default="n"
            ).strip().lower()
            if confirmation not in ["y", "yes"]:
                console.print("[yellow]Script execution cancelled by user.[/yellow]")
            else:
                console.print(f"[bold green]Executing script: {args.script}[/bold green]")
                script_command_str = f"/script {args.script}"
                try_handle_script_command(script_command_str, is_startup_script=True)
        else:
            script_command_str = f"/script {args.script}"
            try_handle_script_command(script_command_str, is_startup_script=True)
    while True:
        try:
            prompt_prefix = ""
            if conversation_history:
                try:
                    active_model_for_prompt_context = get_config_value("model", LITELLM_MODEL_DEFAULT)
                    context_window_size, used_default_window_due_to_no_match = get_model_context_window(active_model_for_prompt_context, return_match_status=True)
                    if conversation_history and active_model_for_prompt_context:
                        tokens_used = token_counter(model=active_model_for_prompt_context, messages=conversation_history)
                        if context_window_size > 0:
                            percentage_used = (tokens_used / context_window_size) * 100
                            default_note = ""
                            if used_default_window_due_to_no_match:
                                default_note = " (default window)"
                            prompt_prefix = f"[Ctx: {percentage_used:.0f}%{default_note}] "
                        else:
                            prompt_prefix = f"[Ctx: {tokens_used} toks] "
                except Exception:
                    pass
            if SHOW_TIMESTAMP_IN_PROMPT:
                prompt_prefix += f"{time.strftime('%H:%M:%S')} "
            user_input = prompt_session.prompt(f"{prompt_prefix}🔵 You> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold yellow]👋 Exiting gracefully...[/bold yellow]")
            sys.exit(0)
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit", "/exit", "/quit"]:
            console.print("[bold bright_blue]👋 Goodbye! Happy coding![/bold bright_blue]")
            sys.exit(0)
        if try_handle_add_command(user_input):
            continue
        if try_handle_set_command(user_input):
            continue
        if try_handle_help_command(user_input):
            continue
        if try_handle_shell_command(user_input):
            continue
        if try_handle_session_command(user_input):
            continue
        if try_handle_rules_command(user_input):
            continue
        if try_handle_context_command(user_input):
            continue
        if try_handle_prompt_command(user_input):
            continue
        if try_handle_script_command(user_input):
            continue
        if try_handle_ask_command(user_input):
            continue
        if try_handle_time_command(user_input):
            continue
        if try_handle_test_command(user_input):
            continue
        target_model_name = get_config_value("model", LITELLM_MODEL_DEFAULT)
        response_data = stream_llm_response(user_input)
        if response_data.get("error"):
            pass
    console.print("[bold blue]✨ Session finished. Thank you for using AI Engineer![/bold blue]")
    sys.exit(0)

def execute_script_line(line: str):
    console.print(f"\n[bold bright_magenta]📜 Script> {line}[/bold bright_magenta]")
    if try_handle_add_command(line): return
    if try_handle_set_command(line): return
    if try_handle_help_command(line): return
    if try_handle_shell_command(line): return
    if try_handle_session_command(line): return
    if try_handle_rules_command(line): return
    if try_handle_context_command(line): return
    if try_handle_prompt_command(line): return
    if try_handle_test_command(line): return
    stream_llm_response(line)

def try_handle_script_command(user_input: str, is_startup_script: bool = False) -> bool:
    command_prefix = "/script"
    stripped_input = user_input.strip()
    if not (stripped_input.lower() == command_prefix or stripped_input.lower().startswith(command_prefix + " ") or (is_startup_script and stripped_input.lower() == command_prefix)):
        return False
    prefix_with_space = command_prefix + " "
    script_path_arg = ""
    if stripped_input.lower().startswith(prefix_with_space):
        script_path_arg = stripped_input[len(prefix_with_space):].strip()
    elif is_startup_script and stripped_input.lower() == command_prefix:
        console.print("[yellow]Usage: --script <script_path>[/yellow]")
        return True
    if not script_path_arg:
        console.print("[yellow]Usage: /script <script_path>[/yellow]")
        console.print("[yellow]  Example: /script ./my_setup_script.aiescript[/yellow]")
        console.print("[yellow]  The script file contains AI Engineer commands, one per line. Lines starting with '#' are comments.[/yellow]")
        return True
    try:
        normalized_script_path = normalize_path(script_path_arg)
        console.print(f"[bold blue]🚀 Executing script: [bright_cyan]{normalized_script_path}[/bright_cyan][/bold blue]")
        with open(normalized_script_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                stripped_line = line.strip()
                if not stripped_line or stripped_line.startswith("#"):
                    continue
                execute_script_line(stripped_line)
        console.print(f"[bold green]✅ Script execution finished: [bright_cyan]{normalized_script_path}[/bright_cyan]\n")
    except FileNotFoundError:
        console.print(f"[bold red]✗ Error: Script file not found at '[bright_cyan]{script_path_arg}[/bright_cyan]'[/bold red]")
    except ValueError as e:
        console.print(f"[bold red]✗ Error: Invalid script path '[bright_cyan]{script_path_arg}[/bright_cyan]': {e}[/bold red]")
    except Exception as e:
        console.print(f"[bold red]✗ Error during script execution from '{script_path_arg}': {e}[/bold red]")
    return True

def try_handle_ask_command(user_input: str) -> bool:
    command_prefix = "/ask"
    stripped_input = user_input.strip()
    if not (stripped_input.lower() == command_prefix or stripped_input.lower().startswith(command_prefix + " ")):
        return False
    prefix_with_space = command_prefix + " "
    text_to_ask = ""
    if stripped_input.lower().startswith(prefix_with_space):
        text_to_ask = stripped_input[len(prefix_with_space):].strip()
    elif stripped_input.lower() == command_prefix:
        console.print("[yellow]Usage: /ask <text>[/yellow]")
        console.print("[yellow]  Example: /ask What is the capital of France?[/yellow]")
        return True
    stream_llm_response(text_to_ask)
    return True

def try_handle_time_command(user_input: str) -> bool:
    global SHOW_TIMESTAMP_IN_PROMPT
    command_name_lower = "/time"
    if user_input.strip().lower() == command_name_lower:
        SHOW_TIMESTAMP_IN_PROMPT = not SHOW_TIMESTAMP_IN_PROMPT
        if SHOW_TIMESTAMP_IN_PROMPT:
            console.print("[green]✓ Timestamp display in prompt: ON[/green]")
        else:
            console.print("[yellow]✓ Timestamp display in prompt: OFF[/yellow]")
        return True
    return False

def try_handle_test_command(user_input: str) -> bool:
    command_prefix = "/test"
    stripped_input = user_input.strip()
    if not (stripped_input.lower() == command_prefix or stripped_input.lower().startswith(command_prefix + " ")):
        return False
    prefix_with_space = command_prefix + " "
    command_body = ""
    if stripped_input.lower().startswith(prefix_with_space):
        command_body = stripped_input[len(prefix_with_space):].strip()
    parts = command_body.split(maxsplit=1)
    sub_command = parts[0].lower() if parts else ""
    if sub_command == "inference":
        model_to_test = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
        if model_to_test:
            console.print(f"[dim]Testing specific model: {model_to_test}[/dim]")
        test_inference_endpoint(specific_model_name=model_to_test) # Pass None to test all, or model_name
        return True
    elif sub_command == "all":
        console.print("[bold blue]Running all available tests...[/bold blue]")
        test_inference_endpoint(specific_model_name=None) # "/test all" implies testing all configured/known models
        return True
    else:
        console.print("[yellow]Usage: /test <subcommand> [arguments][/yellow]")
        console.print("[yellow]  all         - Run all available tests (currently runs 'inference' for all known/configured models).[/yellow]")
        console.print("[yellow]  inference [model_pattern] - Test capabilities. If model_pattern (wildcards * and ? supported) is provided, tests matching models. Otherwise, tests all known/configured models.[/yellow]")
        return True

if __name__ == "__main__":
    main()
