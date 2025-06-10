#!/usr/bin/env python3
 
"""
Software Engineer AI Assistant: An AI-powered coding assistant.

This script provides an interactive terminal interface for code development,
leveraging AI's reasoning models for intelligent file operations,
code analysis, and development assistance via natural conversation and function calling.

Original source: https://github.com/PieBru/ai_engineer
"""

# Perform startup checks (e.g., venv) VERY FIRST.
# This import should be safe as startup_checks is designed to have minimal dependencies
# or handle missing ones (like Rich) for the check itself.
from src import startup_checks
startup_checks.perform_venv_check() # This will exit if not in a venv
import os
import sys
import argparse
import time
from pathlib import Path

# Import application state and new modules
from src.app_state import AppState
from src.config_utils import (
    load_configuration as load_app_configuration, 
    get_config_value,
    get_model_context_window
)
from src import command_handlers
from src import routing_logic
from src import inference_tester
from src import rules_manager # New import for rules system
from src.llm_interaction import stream_llm_response
from src.prompts import RichMarkdown # If RichMarkdown is used directly for help panel

# Import Rich components (Panel moved to ui_display)
from rich.console import Console # For specific stderr prints if needed

# Import the new UI display module
from src.ui_display import display_welcome_panel
# LiteLLM specific imports
from litellm import token_counter
import litellm

__version__ = "0.2.2" # Updated version for refactor

# Note: Module-level model configurations (LITELLM_MODEL, LITELLM_MAX_TOKENS, etc.)
# are now expected to be handled directly by `get_config_value` using the DEFAULT_ constants as fallbacks.

# Suppress LiteLLM debug info
litellm.suppress_debug_info = True
import logging
logging.getLogger("litellm").setLevel(logging.WARNING)


def get_context_usage_prompt_string(app_state: AppState) -> str:
    """
    Generates the context usage string for the prompt (e.g., "[Ctx: 50%] ").
    Returns an empty string if context info cannot be determined.
    """
    prefix = ""
    # Construct messages for token counting, including the system prompt
    messages_for_count = []
    if app_state.system_prompt and app_state.system_prompt.strip():
        messages_for_count.append({"role": "system", "content": app_state.system_prompt})
    
    # Add conversation history only if it's not empty
    if app_state.conversation_history: # Check if there's any history to add
        messages_for_count.extend(app_state.conversation_history)

    if messages_for_count: # Only proceed if there's something to count (system prompt or history)
        try:
            active_model_for_prompt_context = get_config_value("model", app_state.RUNTIME_OVERRIDES, app_state.console)
            context_window_size_prompt, used_default_prompt = get_model_context_window(active_model_for_prompt_context, return_match_status=True)

            if active_model_for_prompt_context: # Ensure model name is available
                tokens_used = token_counter(model=active_model_for_prompt_context, messages=messages_for_count)
                if context_window_size_prompt > 0:
                    percentage_used = (tokens_used / context_window_size_prompt) * 100
                    default_note = " (default window)" if used_default_prompt else ""
                    prefix = f"[Ctx: {percentage_used:.0f}%{default_note}] "
                else:
                    prefix = f"[Ctx: {tokens_used} toks] "
        except Exception as e:
            if app_state.DEBUG_LLM_INTERACTIONS: # Or a new DEBUG_CONTEXT_USAGE flag
                app_state.console.print(f"[dim red]Error calculating context usage: {e}[/dim red]", stderr=True)
            # Silently pass if not debugging, to avoid breaking the prompt line
            # but the error will be logged if debug is on.
            pass
    return prefix

# --- Command Handler Registries ---
MAIN_LOOP_COMMAND_HANDLERS = [
    command_handlers.try_handle_add_command,
    command_handlers.try_handle_show_command, # New handler for /show
    command_handlers.try_handle_set_command,
    command_handlers.try_handle_help_command,
    command_handlers.try_handle_shell_command,
    command_handlers.try_handle_session_command, # Alias for /context
    command_handlers.try_handle_rules_command,
    command_handlers.try_handle_context_command,
    command_handlers.try_handle_prompt_command,
    command_handlers.try_handle_script_command, # For interactive /script
    command_handlers.try_handle_ask_command,
    command_handlers.try_handle_debug_command,
    command_handlers.try_handle_time_command,
    inference_tester.try_handle_test_command,
]

SCRIPT_EXECUTION_COMMAND_HANDLERS = [
    command_handlers.try_handle_add_command,
    command_handlers.try_handle_show_command, # Also useful in scripts
    command_handlers.try_handle_set_command,
    command_handlers.try_handle_help_command,
    command_handlers.try_handle_shell_command,
    command_handlers.try_handle_session_command, # Alias for /context
    command_handlers.try_handle_rules_command,
    command_handlers.try_handle_context_command,
    command_handlers.try_handle_prompt_command,
    inference_tester.try_handle_test_command,
    # Note: /script itself is not typically called from within a script.
    # /debug and /time are interactive toggles, less common/useful in non-interactive scripts.
]

def execute_script_line(line: str, app_state: AppState):
    """Executes a single line from a script file."""
    app_state.console.print(f"\n[bold bright_magenta]📜 Script> {line}[/bold bright_magenta]")
    
    # Try registered command handlers for script execution
    for handler_func in SCRIPT_EXECUTION_COMMAND_HANDLERS:
        if handler_func(line, app_state):
            return # Command was handled

    # If not a known command, treat as a prompt for the LLM
    target_model_for_script_line = get_config_value("model", app_state.RUNTIME_OVERRIDES, app_state.console)
    is_script_command = line.startswith("/") # Re-check, though most /commands are handled above

    if line and not is_script_command:
        app_state.console.print("[dim]↪ Routing query...[/dim]")
        expert_keyword_script = routing_logic.get_routing_expert_keyword(line, app_state)
        target_model_for_script_line = routing_logic.map_expert_to_model(expert_keyword_script, app_state)

    stream_llm_response(
        line,
        app_state,
        target_model_override=target_model_for_script_line
    )

def clear_screen():
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')

def main():
    app_state = AppState()

    # Load .env, config.toml
    load_app_configuration(app_state.console) # This might set some initial RUNTIME_OVERRIDES if config.toml is used

    # Initialize the rules system (ensure dirs, load system rules, build initial prompt)
    # This should happen after config is loaded (in case DEBUG_RULES is set in config)
    rules_manager.initialize_rules_system(app_state)

    parser = argparse.ArgumentParser(
        description="Software Engineer AI Assistant: An AI-powered coding assistant.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('--script', metavar='SCRIPT_PATH', type=str, help='Path to a script file to execute on startup.')
    parser.add_argument('--noconfirm', action='store_true', help='Skip confirmation prompts when using --script.')
    parser.add_argument('--time', action='store_true', help='Enable timestamp display in the user prompt.')
    parser.add_argument(
        '--test-inference', metavar='MODEL_NAME', type=str, nargs='?',
        const='__TEST_ALL_MODELS__',
        help='Test capabilities. If MODEL_NAME (wildcards * and ? supported), tests matching models. Else, tests all.'
    )
    args = parser.parse_args()

    clear_screen()

    if args.test_inference is not None:
        model_to_test = None if args.test_inference == '__TEST_ALL_MODELS__' else args.test_inference
        inference_tester.test_inference_endpoint(app_state, specific_model_name=model_to_test)
        # test_inference_endpoint should handle its own sys.exit()

    display_welcome_panel(app_state)

    if args.time:
        app_state.SHOW_TIMESTAMP_IN_PROMPT = True
        app_state.console.print("[green]✓ Timestamp display in prompt enabled via --time flag.[/green]")

    if args.script:
        # command_handlers.try_handle_script_command will use app_state.prompt_session for confirmation
        command_handlers.try_handle_script_command(f"/script {args.script}", app_state, is_startup_script=True, noconfirm=args.noconfirm)

    while True:
        try:
            context_usage_str = get_context_usage_prompt_string(app_state)
            timestamp_str = ""
            if app_state.SHOW_TIMESTAMP_IN_PROMPT:
                timestamp_str = f"{time.strftime('%H:%M:%S')} "

            user_input = app_state.prompt_session.prompt(f"{timestamp_str}🔵 {context_usage_str}You> ").strip()
        except (EOFError, KeyboardInterrupt):
            app_state.console.print("\n[bold yellow]👋 Exiting gracefully...[/bold yellow]")
            sys.exit(0)

        if not user_input:
            continue
        
        if user_input.lower() in ["exit", "quit", "/exit", "/quit"]:
            app_state.console.print("[bold bright_blue]👋 Goodbye! Happy coding![/bold bright_blue]")
            sys.exit(0)

        # Dispatch to command handlers
        command_handled = False
        for handler_func in MAIN_LOOP_COMMAND_HANDLERS:
            if handler_func(user_input, app_state):
                command_handled = True
                break
        
        if command_handled:
            continue

        # If input started with '/' but was not handled by any command, it's an unknown command
        if user_input.startswith("/"):
            app_state.console.print(f"[yellow]Unknown command: '{user_input.split()[0]}'. Type '/help' for a list of commands.[/yellow]")
            continue


        # --- Routing Step for non-commands ---
        target_model_for_this_turn = get_config_value("model", app_state.RUNTIME_OVERRIDES, app_state.console)

        app_state.console.print("[dim]↪ Routing query...[/dim]")

        import re # Moved import here as it's only used in this block
        greeting_pattern_for_routing_bypass = r"^\s*(hello|hi|hey|good\s+(morning|afternoon|evening)|how\s+are\s+you|how's\s+it\s+going|what's\s+up|sup)[\s!\.,\?]*\s*$"
        is_simple_greeting_for_bypass = bool(re.match(greeting_pattern_for_routing_bypass, user_input.strip(), re.IGNORECASE))

        if is_simple_greeting_for_bypass:
            if app_state.DEBUG_LLM_INTERACTIONS:
                app_state.console.print("[dim]ROUTER DEBUG: Simple greeting detected, bypassing LLM router, using DEFAULT expert.[/dim]", stderr=True)
            expert_keyword = "DEFAULT"
            target_model_for_this_turn = routing_logic.map_expert_to_model(expert_keyword, app_state)
            app_state.console.print(f"[dim]  ➔ Routed to: {expert_keyword} (Bypassed for greeting)[/dim]")
        else:
            expert_keyword = routing_logic.get_routing_expert_keyword(user_input, app_state)
            target_model_for_this_turn = routing_logic.map_expert_to_model(expert_keyword, app_state)

        # --- End Routing Step ---

        stream_llm_response(
            user_input,
            app_state,
            target_model_override=target_model_for_this_turn
        )

    app_state.console.print("[bold blue]✨ Session finished. Thank you for using Software Engineer AI Assistant![/bold blue]")
    sys.exit(0)

if __name__ == "__main__":
    main()
