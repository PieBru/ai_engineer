#!/usr/bin/env python3
"""
Software Engineer AI Assistant: An AI-powered coding assistant.

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
    DEFAULT_LM_STUDIO_API_BASE, # Import for checking
    DEFAULT_LITELLM_MODEL_KNOWLEDGE,
    DEFAULT_LITELLM_MODEL_PLANNER,
    DEFAULT_LITELLM_MODEL_TASK_MANAGER,
    DEFAULT_LITELLM_MODEL_RULE_ENHANCER,
    DEFAULT_LITELLM_MODEL_PROMPT_ENHANCER,
    DEFAULT_LITELLM_MODEL_WORKFLOW_MANAGER,
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
LITELLM_MODEL_PLANNER = os.getenv("LITELLM_MODEL_PLANNER", DEFAULT_LITELLM_MODEL_PLANNER)
LITELLM_MODEL_TASK_MANAGER = os.getenv("LITELLM_MODEL_TASK_MANAGER", DEFAULT_LITELLM_MODEL_TASK_MANAGER)
LITELLM_MODEL_RULE_ENHANCER = os.getenv("LITELLM_MODEL_RULE_ENHANCER", DEFAULT_LITELLM_MODEL_RULE_ENHANCER)
LITELLM_MODEL_PROMPT_ENHANCER = os.getenv("LITELLM_MODEL_PROMPT_ENHANCER", DEFAULT_LITELLM_MODEL_PROMPT_ENHANCER)
LITELLM_MODEL_WORKFLOW_MANAGER = os.getenv("LITELLM_MODEL_WORKFLOW_MANAGER", DEFAULT_LITELLM_MODEL_WORKFLOW_MANAGER)

LITELLM_API_BASE = os.getenv("LITELLM_API_BASE", DEFAULT_LITELLM_API_BASE)
LITELLM_MAX_TOKENS = int(os.getenv("LITELLM_MAX_TOKENS", DEFAULT_LITELLM_MAX_TOKENS))
REASONING_EFFORT = os.getenv("REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
REASONING_STYLE = os.getenv("REASONING_STYLE", DEFAULT_REASONING_STYLE)
import json # Needed for context save/load, and risky tool preview
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
    SUPPORTED_SET_PARAMS, MAX_FILES_TO_PROCESS_IN_DIR, MAX_FILE_SIZE_BYTES, 
    MODEL_CONFIGURATIONS, SHOW_TEST_INFERENCE_NOTES_ERRORS_COLUMN, # Use MODEL_CONFIGURATIONS
    RUNTIME_OVERRIDES,
    get_model_test_expectations as get_full_model_expectations, # Renamed for clarity
    get_model_test_expectations, # Use new helper
    get_model_context_window # Keep for general use, test uses expectations
)
from src.file_utils import (
    normalize_path, is_binary_file, read_local_file as util_read_local_file,
    create_file as util_create_file, apply_diff_edit as util_apply_diff_edit
)
from src.tool_defs import tools, RISKY_TOOLS
from src.prompts import system_PROMPT, RichMarkdown # Moved RichMarkdown import here
from src.data_models import FileToCreate, FileToEdit # Import new models
from src.network_utils import handle_local_mcp_stream, handle_remote_mcp_sse # Import network utils
# Import new modules for refactored logic
from src.file_context_utils import (
    add_directory_to_conversation,
    ensure_file_in_context
)
from src.llm_interaction import stream_llm_response # Removed execute_function_call_dict, trim_conversation_history
from prompt_toolkit.styles import Style as PromptStyle
from litellm import completion, token_counter
import litellm 
import logging # type: ignore
from typing import Dict, Any, Optional

__version__ = "0.2.1" # Updated version

console = Console()
prompt_session = PromptSession(
    style=PromptStyle.from_dict({
        'prompt': '#0066ff bold',
        'completion-menu.completion': 'bg:#1e3a8a fg:#ffffff',
        'completion-menu.completion.current': 'bg:#3b82f6 fg:#ffffff bold',
    })
)

SHOW_TIMESTAMP_IN_PROMPT = False
load_app_configuration(console)

litellm.suppress_debug_info = True
logging.getLogger("litellm").setLevel(logging.WARNING)

# ... (Keep _handle_local_mcp_stream, _handle_remote_mcp_sse, and all try_handle_* command functions as they are) ...
# ... (No changes to these helper functions from the previous full file content)

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
        console.print("[yellow]  Example: /add ./my_project_folder[/yellow]") # Keep this line
        return True
    try:
        normalized_path = normalize_path(path_to_add)
        if os.path.isdir(normalized_path):
            add_directory_to_conversation(normalized_path)
        else:
            content = util_read_local_file(normalized_path)
            conversation_history.append({ # type: ignore
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
    stripped_input = user_input.strip() # Keep original case for argument processing

    if not stripped_input.lower().startswith(command_prefix.lower()): # Case-insensitive command check
        return False

    # Extract argument part, keeping its original form for now
    # e.g., "/help foo bar " -> arg_text = "foo bar"
    # e.g., "/help" -> arg_text = ""
    arg_text = stripped_input[len(command_prefix):].strip()

    help_file_basename = "help" # Default base name for the .md file (e.g., "help.md")

    if arg_text:
        # Sanitize argument: keep alphanumeric, '_', '-', '.'
        # Example: "/task" -> "task", "my topic" -> "mytopic", "valid_file-1.0" -> "valid_file-1.0"
        sanitized_arg_basename = "".join(c for c in arg_text if c.isalnum() or c in ['_', '-', '.'])
        if sanitized_arg_basename: # If something remains after sanitization
            help_file_basename = sanitized_arg_basename
        else:
            # If arg_text was provided but sanitized_arg_basename is empty (e.g., arg_text was "///" or "$%^")
            console.print(f"[yellow]Warning: Invalid characters in help topic '{arg_text}'. Showing default help page.[/yellow]")
            # help_file_basename remains "help" (the default)
    
    help_file_name_md = f"{help_file_basename}.md"
    default_help_file_name_md = "help.md"

    script_dir = Path(__file__).resolve().parent
    script_help_dir = script_dir / "help"
    
    # Path for current working directory's help folder
    cwd_help_dir = Path("./help").resolve() 

    # Paths to try for the requested/default help topic
    requested_script_path = script_help_dir / help_file_name_md
    requested_cwd_path = cwd_help_dir / help_file_name_md

    # Paths for the main/default help.md file
    default_script_path = script_help_dir / default_help_file_name_md
    default_cwd_path = cwd_help_dir / default_help_file_name_md

    help_content = None
    loaded_from_cwd = False
    is_default_fallback = False

    # Attempt 1: Load requested file from script directory
    try:
        help_content = util_read_local_file(str(requested_script_path))
    except FileNotFoundError:
        # Attempt 2: Load requested file from CWD
        try:
            help_content = util_read_local_file(str(requested_cwd_path))
            loaded_from_cwd = True
            console.print(f"[dim]Info: Help file '{help_file_name_md}' loaded from CWD ('{requested_cwd_path}').[/dim]")
        except FileNotFoundError:
            # Requested file not found in either location
            if help_file_basename.lower() != "help": # If it was a specific topic, try default help
                console.print(f"[red]Error: Help topic '{arg_text}' (file '{help_file_name_md}') not found in script directory ('{requested_script_path}') or CWD ('{requested_cwd_path}'). Attempting to show default help page.[/red]")
                is_default_fallback = True
                # Attempt 3: Load default help.md from script directory
                try:
                    help_content = util_read_local_file(str(default_script_path))
                    loaded_from_cwd = False # Reset as this is from script dir
                except FileNotFoundError:
                    # Attempt 4: Load default help.md from CWD
                    try:
                        help_content = util_read_local_file(str(default_cwd_path))
                        loaded_from_cwd = True
                        console.print(f"[dim]Info: Default help file '{default_help_file_name_md}' loaded from CWD ('{default_cwd_path}').[/dim]")
                    except FileNotFoundError:
                        console.print(f"[red]Error: Default help file ('{default_help_file_name_md}') also not found in script directory ('{default_script_path}') or CWD ('{default_cwd_path}').[/red]")
                        return True # Command handled, error shown
                    except OSError as e_default_cwd:
                        console.print(f"[red]Error reading default help file from CWD '{default_cwd_path}': {e_default_cwd}[/red]")
                        return True
                except OSError as e_default_script:
                    console.print(f"[red]Error reading default help file from script dir '{default_script_path}': {e_default_script}[/red]")
                    return True
            else: # Main help.md itself was not found
                console.print(f"[red]Error: Main help file ('{help_file_name_md}') not found in script directory ('{requested_script_path}') or CWD ('{requested_cwd_path}').[/red]")
                return True # Command handled, error shown
        except OSError as e_cwd:
            console.print(f"[red]Error reading help file from CWD '{requested_cwd_path}': {e_cwd}[/red]")
            return True
    except OSError as e_script:
        console.print(f"[red]Error reading help file from script dir '{requested_script_path}': {e_script}[/red]")
        return True

    if help_content:
        # Determine panel title
        title_base_name_for_display = help_file_basename
        if is_default_fallback: # If we are showing help.md because the original topic wasn't found
            title_base_name_for_display = "Default" 

        title_location_suffix = " (from CWD)" if loaded_from_cwd else ""
        
        if help_file_basename.lower() == "help" and not is_default_fallback:
            # User asked for /help, and we found help.md (not as a fallback for another topic)
            panel_title = f"[bold blue]📚 Software Engineer AI Assistant Help{title_location_suffix}[/bold blue]"
        else:
            # User asked for /help <topic> and we found <topic>.md
            # OR user asked for /help <topic>, <topic>.md not found, so we are showing help.md (Default)
            panel_title = f"[bold blue]📚 Software Engineer AI Assistant Help ({title_base_name_for_display}){title_location_suffix}[/bold blue]"
            
        console.print(Panel(
            RichMarkdown(help_content),
            title=panel_title,
            title_align="left",
            border_style="blue"
        ))
    
    return True # Command was handled (or attempted)

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
            console.print(f"[red]Command exited with non-zero status code: {return_code}[/red]") # Keep this line
        history_content = f"Shell command executed: '{command_body}'\n\n"
        if output:
            history_content += f"Stdout:\n```\n{output}\n```\n" # Keep this line
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
            You are a prompt engineering assistant. Your task is to refine the following user-provided text into an optimized prompt suitable for an AI coding assistant like Software Engineer AI Assistant. The refined prompt should be clear, concise, and actionable, guiding the AI to provide the best possible coding assistance.
            The output should be ONLY the refined prompt text, without any preamble or explanation.
            """)
        user_query = f"Refine this text into a prompt for an AI coding assistant:\n\n---\n{user_text}\n---"
    elif mode == "detail":
        meta_system_prompt = dedent("""\
            You are a prompt engineering assistant. Your task is to expand the following user-provided text into a more detailed and comprehensive prompt suitable for an AI coding assistant like Software Engineer AI Assistant. The detailed prompt should elaborate on the user's initial idea, adding necessary context, specifying desired outcomes, and anticipating potential ambiguities to guide the AI effectively.
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
    console.print("[yellow]  refine <text>  - Optimizes <text> into a clearer and more effective prompt for Software Engineer AI Assistant.[/yellow]")
    console.print("[yellow]  detail <text>  - Expands <text> into a more comprehensive and detailed prompt for Software Engineer AI Assistant.[/yellow]")
    return True

# --------------------------------------------------------------------------------
# 5. Conversation state
# --------------------------------------------------------------------------------
conversation_history = [
    {"role": "system", "content": system_PROMPT}
]

# --------------------------------------------------------------------------------
# 6. LLM API interaction with streaming
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# 7. Test & Main interactive loop
# --------------------------------------------------------------------------------

def _summarize_error_message(error_message: str, summary_model_name: str, api_base: str) -> str:
    """Uses an LLM to summarize an error message concisely."""
    if not error_message:
        return ""
    
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
            max_tokens=50,
            api_base=api_base,
            stream=False
        )
        summary = response.choices[0].message.content.strip()
        return f"Summary: {summary}" if summary else error_message
    except Exception:
        return truncated_error_message

def _perform_api_call_for_test(model_name_to_test: str, messages: list, api_base_for_call: Optional[str], temperature: float, max_tokens_val: int, timeout_val: int, completion_kwargs: dict, tools_list: Optional[list] = None) -> Any:
    """Helper to make a single API call and handle exceptions."""
    call_args = {
        "model": model_name_to_test,
        "messages": messages,
        "api_base": api_base_for_call,
        "temperature": temperature,
        "max_tokens": max_tokens_val,
        "timeout": timeout_val,
        **completion_kwargs
    }

    # Add dummy API key for LM Studio models.
    # api_base_for_call is the resolved API base (model-specific or global)
    # model_name_to_test is the name of the model being called.
    if model_name_to_test.startswith("lm_studio/"):
        call_args["api_key"] = "dummy"

    if tools_list:
        call_args["tools"] = tools_list
    return completion(**call_args)


def _test_single_model_capabilities(model_label: str, model_name_to_test: str, api_base_to_test: str, role_expect_tools: bool) -> Dict[str, Any]:
    """Helper function to test capabilities of a single model."""
    
    model_expectations = get_full_model_expectations(model_name_to_test) # Use renamed import
    expected_tool_support = model_expectations.get("supports_tools", role_expect_tools) # Prioritize model-specific, fallback to role
    expected_thinking_model = model_expectations.get("is_thinking_model", False)
    # thinking_type = model_expectations.get("thinking_type") # For future use # Corrected divisor for KB
    context_window_kb_str = f"{model_expectations.get('context_window', 0) // 1024}k"


    results: Dict[str, Any] = {
        "label": model_label,
        "name": model_name_to_test,
        "available": "N",
        "tool_support": "N/A",
        "is_thinking_model": "N",
        "context_kb": context_window_kb_str if model_expectations.get('context_window', 0) > 0 else "N/A",
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
    api_key_name_hint = "API key" # Default hint

    if api_base_to_test:
        if "openai" in api_base_to_test.lower() or "gpt-" in model_name_to_test.lower(): api_key_name_hint = "OPENAI_API_KEY"
        elif "deepseek" in api_base_to_test.lower() or "deepseek" in model_name_to_test.lower(): api_key_name_hint = "DEEPSEEK_API_KEY"
        elif "anthropic" in api_base_to_test.lower() or "claude" in model_name_to_test.lower(): api_key_name_hint = "ANTHROPIC_API_KEY"
        elif "openrouter" in api_base_to_test.lower(): api_key_name_hint = "OPENROUTER_API_KEY"
    
    # Determine the final API base for this specific model test
    api_base_for_call = model_expectations.get("api_base") 
    if api_base_for_call is None: # If model config didn't specify one (or it was explicitly None)
        api_base_for_call = api_base_to_test # Fallback to the global/passed-in API base

    pre_call_notes = []
    completion_kwargs: Dict[str, Any] = {} # For proxies, etc.

    if model_name_to_test and api_base_for_call:
        provider_from_model_name = model_name_to_test.split('/')[0].lower()
        is_ollama_model = provider_from_model_name.startswith("ollama")
        known_direct_providers_domains = {
            "openai": "api.openai.com", "anthropic": "api.anthropic.com",
            "deepseek": "api.deepseek.com", "google": "googleapis.com",
            "cohere": "api.cohere.ai", "cerebras": "api.cerebras.com"
        }

        # This logic might override a specifically configured api_base from MODEL_CONFIGURATIONS
        # if it doesn't match the expected domain for a known provider.
        # This could be problematic if a user is proxying a known provider through a custom domain.
        # For now, we'll keep it, but it's a point of attention.
        # If api_base_for_call was already set (e.g. from MODEL_CONFIGURATIONS), we should be careful.
        # The current `api_base_for_call` is already resolved. This block might re-resolve it.
        # Let's assume `api_base_for_call` is the one to use, and this block is for providers
        # that LiteLLM handles internally without an explicit api_base.
        if api_base_for_call: # Only apply this logic if we have an api_base to check
            if is_ollama_model:
                if not api_base_for_call.lower().startswith("http://") or \
                   not ("localhost" in api_base_for_call.lower() or "127.0.0.1" in api_base_for_call.lower()):
                    # This case implies a misconfiguration if an ollama model is not pointed to a local http endpoint.
                    # Or, it's a remote Ollama, which is fine. Let's not nullify api_base_for_call here.
                    pass # pre_call_notes.append(f"[yellow]Warning: Ollama model '{model_name_to_test}' with non-standard base '{api_base_for_call}'[/yellow]")
            elif provider_from_model_name in known_direct_providers_domains and \
                 known_direct_providers_domains[provider_from_model_name] not in api_base_for_call.lower():
                # If the model is from a known cloud provider, but the api_base_for_call
                # (which could be from MODEL_CONFIGURATIONS or global default) doesn't match the provider's domain,
                # it might be a proxy. LiteLLM usually handles direct cloud providers without an api_base.
                # Setting api_base_for_call to None here lets LiteLLM use its internal defaults for these providers.
                # This is only done if api_base_for_call was NOT specifically set for this model in MODEL_CONFIGURATIONS.
                if model_expectations.get("api_base") is None: # Only override if not model-specific
                    if provider_from_model_name == "google": # Special handling for Google
                        api_base_for_call = "https://generativelanguage.googleapis.com" # Force direct
                        pre_call_notes.append(f"[dim]  (Note: Forcing direct Google API base for '{model_name_to_test}')[/dim]")
                        completion_kwargs["proxies"] = None # Disable proxies for direct Google
                        pre_call_notes.append(f"[dim]  (Note: Disabling proxies for direct Google API call)[/dim]")
                    else:
                         api_base_for_call = None # Let LiteLLM handle (e.g. OpenAI, Anthropic)
                         pre_call_notes.append(f"[dim]  (Note: Letting LiteLLM manage API base for '{model_name_to_test}' with provider '{provider_from_model_name}')[/dim]")
    
    if "gemini" in model_name_to_test.lower() or "google" in model_name_to_test.lower():
         pre_call_notes.append(f"[bold yellow blink]DEBUG Gemini Test Params:[/bold yellow blink] model='{model_name_to_test}', api_base_for_call='{api_base_for_call}'")

    for note in pre_call_notes:
        console.print(note)
    
    console.print("[white]  1. Attempting basic API call...[/white]", end="")
    
    # --- Basic API Call & Thinking Test ---
    observed_thinking_final = False # Initialize before loop
    api_call_error_message = None
    for attempt in range(2): # Max 2 attempts for basic call / thinking
        try:
            response = _perform_api_call_for_test(model_name_to_test, test_messages, api_base_for_call, temperature_for_test, 20, 30, completion_kwargs)
            response_content = response.choices[0].message.content
            current_attempt_thinking = response_content.strip().lower().startswith("<think>")
            
            if attempt == 0: # First attempt
                console.print(f"[green] ✓ OK[/green] (LLM: \"{response_content.strip()[:60]}{'...' if len(response_content.strip()) > 60 else ''}\")")
                results["available"] = "Y"
                results["inference_time_s"] = f"{time.time() - start_time:.2f}"
                api_call_error_message = None # Clear previous error if any

            observed_thinking_final = current_attempt_thinking # Update with current attempt's observation
            
            if expected_thinking_model == observed_thinking_final: # Expectation met
                if attempt > 0: console.print(f"[green] ✓ Retry OK[/green] (Thinking: {observed_thinking_final})")
                break # Exit retry loop
            elif attempt == 0: # Mismatch on first attempt, prepare to retry
                console.print(f"[yellow] ⚠️ Thinking mismatch (Expected: {expected_thinking_model}, Got: {observed_thinking_final}). Retrying basic call...[/yellow]", end="")
            # If mismatch on second attempt, loop will end, error logged after loop.

        except Exception as e:
            api_call_error_message = str(e)
            if attempt == 0:
                console.print(f"[red] ❌ API Error[/red]: {api_call_error_message[:150]}{'...' if len(api_call_error_message) > 150 else ''}")
                results["error_details"] += f"Basic API call failed (1st attempt): {api_call_error_message}\n"
            else: # Error on retry
                console.print(f"[red] ❌ Retry API Error[/red]: {api_call_error_message[:150]}{'...' if len(api_call_error_message) > 150 else ''}")
                results["error_details"] += f"Basic API call failed (2nd attempt): {api_call_error_message}\n"
            if "authentication" in api_call_error_message.lower() or "api key" in api_call_error_message.lower() or "401" in api_call_error_message:
                results["error_details"] += f" (Hint: Check {api_key_name_hint})\n"
            elif "model_not_found" in api_call_error_message.lower() or ("404" in api_call_error_message and "Model" in api_call_error_message):
                results["error_details"] += f" (Hint: Model '{model_name_to_test}' not found or misspelled at '{api_base_to_test}')\n"
            
            if attempt == 0 and not (expected_thinking_model == observed_thinking_final): # If first attempt failed and thinking was also a concern
                pass # Retry will happen
            else: # If error on retry, or error on first try and thinking wasn't the issue to retry for
                break # Exit retry loop, error is logged

    results["is_thinking_model"] = "Y" if observed_thinking_final else "N"
    if api_call_error_message and results["available"] == "N": # If still not available after retries
        if results["inference_time_s"] == "N/A": results["inference_time_s"] = f"{time.time() - start_time:.2f}"
        return results # Cannot proceed if basic API call failed

    if expected_thinking_model != observed_thinking_final:
        mismatch_note = f"Thinking model expectation mismatch: Expected '{expected_thinking_model}', Got '{observed_thinking_final}' after retries."
        results["error_details"] += mismatch_note + "\n"
        console.print(f"[yellow]  {mismatch_note}[/yellow]")


    # --- Token Counting & Context (uses pre-defined expectation) ---
    console.print("[white]  2. Testing token counting & context...[/white]", end="")
    if results["context_kb"] != "N/A": # If context was defined
        console.print(f"[green] ✓ OK[/green] (Context: {results['context_kb']})")
    else:
        console.print(f"[yellow] ⚠️ Not Defined[/yellow]")


    # --- Tool Calling Capability Test --- # Changed to white
    console.print("[white]  3. Testing tool calling capability...[/white]", end="")
    if expected_tool_support:
        observed_tool_support_final = "N" # Default to N if all attempts fail
        tool_call_error_message = None
        dummy_tool_for_test = [{"type": "function", "function": {"name": "get_dummy_data_for_test", "description": "A dummy function.", "parameters": {"type": "object", "properties": {"data_type": {"type": "string"}}, "required": ["data_type"]}}}]
        tool_test_messages = [{"role": "user", "content": "Use a tool to get dummy text data."}]

        for attempt in range(2): # Max 2 attempts for tool call
            try:
                tool_response = _perform_api_call_for_test(model_name_to_test, tool_test_messages, api_base_for_call, temperature_for_test, 150, 30, completion_kwargs, tools_list=dummy_tool_for_test)
                tool_call_error_message = None # Clear previous error
                if tool_response.choices[0].message.tool_calls and len(tool_response.choices[0].message.tool_calls) > 0:
                    observed_tool_support_final = "Y"
                    if attempt == 0: console.print(f"[green] ✓ Yes[/green] (Called: '{tool_response.choices[0].message.tool_calls[0].function.name}')")
                    else: console.print(f"[green] ✓ Retry OK[/green] (Tool Support: Yes)")
                    break # Success, exit retry loop
                elif tool_response.choices[0].message.content: # Responded with text, no tool call
                    observed_tool_support_final = "N"
                    if attempt == 0: console.print(f"[yellow] ⚠️ No[/yellow] (Responded with text)")
                    # If mismatch on first attempt, will retry. If still mismatch on second, error logged after loop.
                else: # Inconclusive
                    observed_tool_support_final = "N"
                    if attempt == 0: console.print(f"[yellow] ⚠️ Inconclusive[/yellow]")

                if observed_tool_support_final == "Y": break # Should be caught by tool_calls check above, but for safety
                if attempt == 0: # Mismatch on first attempt, prepare to retry
                     console.print(f"[yellow] Retrying tool call test...[/yellow]", end="")

            except Exception as e_tool_call:
                tool_call_error_message = str(e_tool_call)
                observed_tool_support_final = "Error"
                if attempt == 0:
                    console.print(f"[red] ❌ Error[/red] ({tool_call_error_message[:100]})")
                    results["error_details"] += f"Tool call test failed (1st attempt): {tool_call_error_message}\n"
                    console.print(f"[yellow] Retrying tool call test...[/yellow]", end="")
                else: # Error on retry
                    console.print(f"[red] ❌ Retry Error[/red] ({tool_call_error_message[:100]})")
                    results["error_details"] += f"Tool call test failed (2nd attempt): {tool_call_error_message}\n"
                    break # Exit retry loop, error is logged
        
        results["tool_support"] = observed_tool_support_final
        if expected_tool_support and observed_tool_support_final != "Y":
            mismatch_note = f"Tool support expectation mismatch: Expected 'Y', Got '{observed_tool_support_final}' after retries."
            results["error_details"] += mismatch_note + "\n"
            console.print(f"[yellow]  {mismatch_note}[/yellow]")
        elif not expected_tool_support and observed_tool_support_final == "Y":
            mismatch_note = f"Tool support expectation mismatch: Expected 'N', Got 'Y'."
            results["error_details"] += mismatch_note + "\n" # More of a note
            console.print(f"[dim]  {mismatch_note}[/dim]")
            
    else: # Not expected to support tools
        console.print(" [dim]No (Test Disabled in `config_utils.py`)[/dim]") # Indicate 'No' tool support, as expected
        results["tool_support"] = "N" # Set to 'N' for consistency in table (will show as red 'N')

    if results["available"] == "N" and results["inference_time_s"] == "N/A": # Ensure time is recorded if initial call failed
         results["inference_time_s"] = f"{time.time() - start_time:.2f}"
    return results


def test_inference_endpoint(specific_model_name: str = None):
    """Tests all configured inference endpoints and capabilities, then exits."""
    if specific_model_name:
        console.print(f"[bold blue]🧪 Testing Specific Model: [cyan]{specific_model_name}[/cyan]...[/bold blue]")
    else:
        console.print("[bold blue]🧪 Testing All Configured Inference Endpoints & Capabilities...[/bold blue]")
        console.print("[dim]Note: This test covers models from MODEL_CONFIGURATIONS and explicitly configured role-based models.[/dim]")
    
    api_base_url = get_config_value("api_base", DEFAULT_LITELLM_API_BASE)
    role_based_models_config = [
        {"label": "DEFAULT",   "name_var": LITELLM_MODEL_DEFAULT,   "expect_tools": True},
        {"label": "ROUTING",   "name_var": LITELLM_MODEL_ROUTING,   "expect_tools": False},
        {"label": "TOOLS",     "name_var": LITELLM_MODEL_TOOLS,     "expect_tools": True},
        {"label": "CODING",    "name_var": LITELLM_MODEL_CODING,    "expect_tools": False},
        {"label": "KNOWLEDGE", "name_var": LITELLM_MODEL_KNOWLEDGE, "expect_tools": False},
        {"label": "PLANNER",   "name_var": LITELLM_MODEL_PLANNER,   "expect_tools": False},
        {"label": "TASK_MGR",  "name_var": LITELLM_MODEL_TASK_MANAGER, "expect_tools": False},
        {"label": "RULE_ENH",  "name_var": LITELLM_MODEL_RULE_ENHANCER, "expect_tools": False},
        {"label": "PROMPT_ENH","name_var": LITELLM_MODEL_PROMPT_ENHANCER, "expect_tools": False},
        {"label": "WORKFLOW_MGR","name_var": LITELLM_MODEL_WORKFLOW_MANAGER, "expect_tools": True}, # Workflow manager might need tools
    ]
    
    default_model_available = False
    all_results = []
    overall_success = True
    
    all_known_models_from_config = set(MODEL_CONFIGURATIONS.keys())
    for config_item in role_based_models_config: # Use config_item to avoid conflict
        model_name_val = config_item["name_var"]
        if model_name_val:
            all_known_models_from_config.add(model_name_val)

    models_to_test_set = set()
    if specific_model_name:
        if "*" in specific_model_name or "?" in specific_model_name:
            console.print(f"[dim]Filtering models with wildcard: '{specific_model_name}'[/dim]")
            for model_n in all_known_models_from_config:
                if fnmatch.fnmatch(model_n, specific_model_name):
                    models_to_test_set.add(model_n)
            if not models_to_test_set:
                console.print(f"[yellow]Warning: No models matched the wildcard pattern '{specific_model_name}'.[/yellow]")
        else:
            models_to_test_set = {specific_model_name}
            if specific_model_name not in all_known_models_from_config:
                console.print(f"[yellow]Warning: Model '{specific_model_name}' is not in MODEL_CONFIGURATIONS or configured roles. Testing it directly with default expectations.[/yellow]")
    else:
        models_to_test_set = all_known_models_from_config

    # Populate model_details for roles, this helps in labeling if a model serves a specific role
    # This is separate from the primary model_expectations fetched from MODEL_CONFIGURATIONS
    role_details_map = {} 
    for config_item in role_based_models_config:
        model_name_val = config_item["name_var"]
        if model_name_val:
            if model_name_val not in role_details_map:
                role_details_map[model_name_val] = {"roles": [], "role_expect_tools": config_item["expect_tools"]}
            role_details_map[model_name_val]["roles"].append(config_item["label"])
            # If any role expects tools, the role_expect_tools for that model name becomes True
            if config_item["expect_tools"]:
                 role_details_map[model_name_val]["role_expect_tools"] = True
    
    if not models_to_test_set:
        console.print("[yellow]Warning: No models specified or found to test.[/yellow]")
        sys.exit(0)

    for model_name_iter in sorted(list(models_to_test_set)):
        if not model_name_iter:
            all_results.append({
                "label": "Invalid/Empty", "name": "Not Configured", "available": "N/A",
                "tool_support": "N/A", "is_thinking_model": "N/A", "context_kb": "N/A", 
                "inference_time_s": "N/A", "error_details": "Empty model name encountered."
            })
            continue

        current_role_details = role_details_map.get(model_name_iter)
        display_label = model_name_iter
        # Default role_expect_tools to False if not found in role_details_map;
        # _test_single_model_capabilities will prioritize MODEL_CONFIGURATIONS
        role_tool_expectation = current_role_details["role_expect_tools"] if current_role_details else False


        if current_role_details:
            display_label = f"{model_name_iter} ({', '.join(current_role_details['roles'])})"
        elif model_name_iter in MODEL_CONFIGURATIONS and not specific_model_name:
             display_label = f"{model_name_iter} (Configured)" # Changed from (Context Map)
        
        api_base_for_model = api_base_url # Global API base
        result = _test_single_model_capabilities(
            model_label=display_label,
            model_name_to_test=model_name_iter,
            api_base_to_test=api_base_for_model,
            role_expect_tools=role_tool_expectation 
        )
        all_results.append(result)
        if result["available"] != "Y":
            overall_success = False
        if model_name_iter == LITELLM_MODEL_DEFAULT and result["available"] == "Y":
            default_model_available = True
            
    if SHOW_TEST_INFERENCE_NOTES_ERRORS_COLUMN:
        if default_model_available:
            console.print(f"\n[dim]Default model ([cyan]{LITELLM_MODEL_DEFAULT}[/cyan]) is available. Attempting to summarize error messages for other models...[/dim]")
            for res in all_results:
                if not isinstance(res.get("error_details"), str):
                    res["error_details"] = str(res.get("error_details", ""))
                if res["name"] != LITELLM_MODEL_DEFAULT and res["available"] == "N" and res["error_details"]:
                    original_error = res["error_details"]
                    res["error_details"] = _summarize_error_message(original_error, LITELLM_MODEL_DEFAULT, api_base_url)

    console.print("\n\n[bold green]📊 Inference Test Summary[/bold green]")
    summary_table = Table(title="Model Capabilities Test Results", show_lines=True)
    summary_table.add_column("Model / Role", style="cyan", no_wrap=True, max_width=50)
    summary_table.add_column("Model Name (Actual)", style="magenta", max_width=40)
    summary_table.add_column("Available", justify="center")
    summary_table.add_column("Tool Support", justify="center")
    summary_table.add_column("Thinking", justify="center")
    summary_table.add_column("Context", justify="right")
    summary_table.add_column("Time (s)", justify="right")
    if SHOW_TEST_INFERENCE_NOTES_ERRORS_COLUMN:
        notes_errors_column_width = console.width // 3 if console.width > 120 else 60
        summary_table.add_column("Notes/Errors", style="dim", overflow="fold", max_width=notes_errors_column_width)

    for res in all_results:
        available_style = "green" if res["available"] == "Y" else "red" if res["available"] == "N" else "yellow"
        tool_style = "green" if res.get("tool_support") == "Y" else "red" if res.get("tool_support") == "N" else "dim"
        thinking_style = "green" if res.get("is_thinking_model") == "Y" else "dim"
            
        row_data = [
            res["label"], 
            res["name"], 
            f"[{available_style}]{res['available']}[/{available_style}]", 
            f"[{tool_style}]{res.get('tool_support', 'N/A')}[/{tool_style}]",
            f"[{thinking_style}]{res.get('is_thinking_model', 'N')}[/{thinking_style}]",
            res.get("context_kb", "N/A"), 
            res.get("inference_time_s", "N/A")]
        if SHOW_TEST_INFERENCE_NOTES_ERRORS_COLUMN:
            row_data.append(res.get("error_details", ""))
        summary_table.add_row(*row_data)

    total_models_tested = len([res for res in all_results if res["available"] != "N/A" and res["name"] != "Not Configured"])
    total_available_y = sum(1 for res in all_results if res["available"] == "Y")
    total_available_n = sum(1 for res in all_results if res["available"] == "N") 
    total_tool_support_y = sum(1 for res in all_results if res.get("tool_support") == "Y")
    total_thinking_y = sum(1 for res in all_results if res.get("is_thinking_model") == "Y")
    
    total_context_known = sum(1 for res in all_results if res.get("context_kb", "N/A") not in ["N/A", "Error"])
    total_context_unknown_or_error = sum(1 for res in all_results if res.get("context_kb", "N/A") in ["N/A", "Error"])

    summary_table.add_section()

    # Build available string
    available_parts = []
    if total_available_y > 0:
        available_parts.append(f"[bold green]{total_available_y}Y[/bold green]")
    if total_available_n > 0:
        available_parts.append(f"[bold red]{total_available_n}N[/bold red]")
    available_str = " / ".join(available_parts) if available_parts else ""

    tool_support_str = f"[bold green]{total_tool_support_y}Y[/bold green]" if total_tool_support_y > 0 else ""
    thinking_str = f"[bold green]{total_thinking_y}Y[/bold green]" if total_thinking_y > 0 else ""

    # Build context string
    context_parts = []
    if total_context_known > 0:
        context_parts.append(f"[bold green]{total_context_known}✓[/bold green]")
    if total_context_unknown_or_error > 0:
        context_parts.append(f"[bold red]{total_context_unknown_or_error}?[/bold red]")
    context_str = " / ".join(context_parts) if context_parts else ""

    overall_stats_row_data = [
        "[bold]Overall Stats[/bold]",
        f"[dim]{total_models_tested} models tested[/dim]",
        available_str,
        tool_support_str,
        thinking_str,
        context_str,
        "", # Changed from "N/A" to empty string for Time (s) in overall stats
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
        description="Software Engineer AI Assistant: An AI-powered coding assistant.",
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
        nargs='?', 
        const='__TEST_ALL_MODELS__', 
        help='Test capabilities. If MODEL_NAME is provided (wildcards * and ? supported), tests matching models. Otherwise, tests all known/configured models.'
    )
    args = parser.parse_args()
    clear_screen()
    if args.test_inference is not None: 
        if args.test_inference == '__TEST_ALL_MODELS__':
            test_inference_endpoint(specific_model_name=None) 
        else:
            test_inference_endpoint(specific_model_name=args.test_inference) 

    current_model_name_for_display = get_config_value("model", LITELLM_MODEL_DEFAULT)
    # Use get_model_context_window for display, as it's simpler for this purpose
    context_window_size_display, used_default_display = get_model_context_window(current_model_name_for_display, return_match_status=True)
    context_window_display_str = f"{context_window_size_display // 1000}k tokens"
    current_working_directory = os.getcwd()
    if used_default_display:
        context_window_display_str += " (default)"
        
    instructions = f"""🧠 Default Model: [bold magenta]{current_model_name_for_display}[/bold magenta] ([dim]{context_window_display_str}[/dim])
   Routing: [dim]{LITELLM_MODEL_ROUTING or 'Not Set'}[/dim] | Tools: [dim]{LITELLM_MODEL_TOOLS or 'Not Set'}[/dim]
   Coding: [dim]{LITELLM_MODEL_CODING or 'Not Set'}[/dim] | Knowledge: [dim]{LITELLM_MODEL_KNOWLEDGE or 'Not Set'}[/dim]
   📁 Current Directory: [bold green]{current_working_directory}[/bold green]
   Planner: [dim]{LITELLM_MODEL_PLANNER or 'Not Set'}[/dim] | Task Mgr: [dim]{LITELLM_MODEL_TASK_MANAGER or 'Not Set'}[/dim]
   Rule Enh: [dim]{LITELLM_MODEL_RULE_ENHANCER or 'Not Set'}[/dim]
   Prompt Enh: [dim]{LITELLM_MODEL_PROMPT_ENHANCER or 'Not Set'}[/dim]
   Workflow Mgr: [dim]{LITELLM_MODEL_WORKFLOW_MANAGER or 'Not Set'}[/dim]


[bold bright_blue]▶️ Commands:[/bold bright_blue]
  • [bright_cyan]/exit[/bright_cyan] or [bright_cyan]/quit[/bright_cyan] - End the session.
  • [bright_cyan]/help[/bright_cyan] - Display detailed help.

[bold white]👥 Just ask naturally, like you are communicating with a colleague.[/bold white]"""
    console.print(Panel(
        instructions,
        border_style="blue",
        padding=(1, 2),
        title="[bold blue]🎯 Software Engineer AI Assistant[/bold blue]",
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
                    # Use get_model_context_window for prompt display
                    context_window_size_prompt, used_default_prompt = get_model_context_window(active_model_for_prompt_context, return_match_status=True)
                    if conversation_history and active_model_for_prompt_context:
                        tokens_used = token_counter(model=active_model_for_prompt_context, messages=conversation_history)
                        if context_window_size_prompt > 0:
                            percentage_used = (tokens_used / context_window_size_prompt) * 100
                            default_note = ""
                            if used_default_prompt:
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
        if try_handle_add_command(user_input): continue
        if try_handle_set_command(user_input): continue
        if try_handle_help_command(user_input): continue
        if try_handle_shell_command(user_input): continue
        if try_handle_session_command(user_input): continue
        if try_handle_rules_command(user_input): continue
        if try_handle_context_command(user_input): continue
        if try_handle_prompt_command(user_input): continue
        if try_handle_script_command(user_input): continue
        if try_handle_ask_command(user_input): continue
        if try_handle_time_command(user_input): continue
        if try_handle_test_command(user_input): continue # Keep this line

        # Call with all necessary arguments
        stream_llm_response(user_input, conversation_history, console, prompt_session)
    console.print("[bold blue]✨ Session finished. Thank you for using Software Engineer AI Assistant![/bold blue]")
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
    # Call with all necessary arguments
    stream_llm_response(line, conversation_history, console, prompt_session)

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
        console.print("[yellow]  The script file contains Software Engineer AI Assistant commands, one per line. Lines starting with '#' are comments.[/yellow]")
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
        console.print("[yellow]Usage: /ask <text>[/yellow]\n[yellow]  Example: /ask What is the capital of France?[/yellow]")
        return True
    # Call with all necessary arguments
    stream_llm_response(text_to_ask, conversation_history, console, prompt_session)
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
        if model_to_test: # This was specific_model_name before, now model_to_test
            console.print(f"[dim]Testing specific model pattern: {model_to_test}[/dim]")
        test_inference_endpoint(specific_model_name=model_to_test)
        return True
    elif sub_command == "all":
        console.print("[bold blue]Running all available tests...[/bold blue]")
        test_inference_endpoint(specific_model_name=None)
        return True
    else:
        console.print("[yellow]Usage: /test <subcommand> [arguments][/yellow]")
        console.print("[yellow]  all         - Run all available tests (currently runs 'inference' for all known/configured models).[/yellow]")
        console.print("[yellow]  inference [model_pattern] - Test capabilities. If model_pattern (wildcards * and ? supported) is provided, tests matching models. Otherwise, tests all known/configured models.[/yellow]")
        return True

if __name__ == "__main__":
    main()
