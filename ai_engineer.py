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
import re
from pathlib import Path

# Import default constants from config_utils first
from src.config_utils import (
    DEFAULT_LITELLM_MODEL,
    DEFAULT_LITELLM_MAX_TOKENS,
    DEFAULT_LITELLM_MODEL_ROUTING,
    DEFAULT_LITELLM_MODEL_TOOLS,
    DEFAULT_LITELLM_MODEL_CODING,
    DEFAULT_LITELLM_MODEL_SUMMARIZE,
    DEFAULT_LITELLM_MODEL_KNOWLEDGE,
    DEFAULT_LITELLM_MODEL_PLANNER,
    DEFAULT_LITELLM_MODEL_TASK_MANAGER,
    DEFAULT_LITELLM_MODEL_RULE_ENHANCER,
    DEFAULT_LITELLM_MODEL_PROMPT_ENHANCER,
    DEFAULT_LITELLM_MODEL_WORKFLOW_MANAGER,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_REASONING_STYLE
)

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
from src.llm_interaction import stream_llm_response
from src.prompts import RichMarkdown # If RichMarkdown is used directly for help panel

# Import Rich components if used directly in this file (e.g., for welcome panel)
from rich.panel import Panel
from rich.console import Console # For specific stderr prints if needed

# LiteLLM specific imports
from litellm import token_counter
import litellm

__version__ = "0.2.2" # Updated version for refactor

# Module-level model configurations (loaded from env or defaults)
# These are used by get_config_value, which now takes runtime_overrides from AppState
LITELLM_MODEL = os.getenv("LITELLM_MODEL", DEFAULT_LITELLM_MODEL)
LITELLM_MODEL_DEFAULT = LITELLM_MODEL
LITELLM_MODEL_ROUTING = os.getenv("LITELLM_MODEL_ROUTING", DEFAULT_LITELLM_MODEL_ROUTING)
LITELLM_MODEL_TOOLS = os.getenv("LITELLM_MODEL_TOOLS", DEFAULT_LITELLM_MODEL_TOOLS)
LITELLM_MODEL_CODING = os.getenv("LITELLM_MODEL_CODING", DEFAULT_LITELLM_MODEL_CODING)
LITELLM_MODEL_SUMMARIZE = os.getenv("LITELLM_MODEL_SUMMARIZE", DEFAULT_LITELLM_MODEL_SUMMARIZE)
LITELLM_MODEL_KNOWLEDGE = os.getenv("LITELLM_MODEL_KNOWLEDGE", DEFAULT_LITELLM_MODEL_KNOWLEDGE)
LITELLM_MODEL_PLANNER = os.getenv("LITELLM_MODEL_PLANNER", DEFAULT_LITELLM_MODEL_PLANNER)
LITELLM_MODEL_TASK_MANAGER = os.getenv("LITELLM_MODEL_TASK_MANAGER", DEFAULT_LITELLM_MODEL_TASK_MANAGER)
LITELLM_MODEL_RULE_ENHANCER = os.getenv("LITELLM_MODEL_RULE_ENHANCER", DEFAULT_LITELLM_MODEL_RULE_ENHANCER)
LITELLM_MODEL_PROMPT_ENHANCER = os.getenv("LITELLM_MODEL_PROMPT_ENHANCER", DEFAULT_LITELLM_MODEL_PROMPT_ENHANCER)
LITELLM_MODEL_WORKFLOW_MANAGER = os.getenv("LITELLM_MODEL_WORKFLOW_MANAGER", DEFAULT_LITELLM_MODEL_WORKFLOW_MANAGER)

LITELLM_MAX_TOKENS = int(os.getenv("LITELLM_MAX_TOKENS", DEFAULT_LITELLM_MAX_TOKENS))
REASONING_EFFORT = os.getenv("REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
REASONING_STYLE = os.getenv("REASONING_STYLE", DEFAULT_REASONING_STYLE)

# Suppress LiteLLM debug info
litellm.suppress_debug_info = True
import logging
logging.getLogger("litellm").setLevel(logging.WARNING)


def display_welcome_panel(app_state: AppState):
    """Displays the welcome panel."""
    current_model_name_for_display = get_config_value("model", LITELLM_MODEL_DEFAULT, app_state.RUNTIME_OVERRIDES, app_state.console)
    context_window_size_display, used_default_display = get_model_context_window(current_model_name_for_display, return_match_status=True)
    context_window_display_str = f"Context:{context_window_size_display // 1024}k tokens"
    current_working_directory = os.getcwd()
    if used_default_display:
        context_window_display_str += " (default)"
        
    instructions = f"""  📁 [bold bright_blue]Current Directory: [/bold bright_blue][bold green]{current_working_directory}[/bold green]

  🧠 [bold bright_blue]Default Model: [/bold bright_blue][bold magenta]{current_model_name_for_display}[/bold magenta] ([dim]{context_window_display_str}[/dim])
     Routing: [dim]{get_config_value("model_routing", LITELLM_MODEL_ROUTING, app_state.RUNTIME_OVERRIDES) or 'Not Set'}[/dim] | Tools: [dim]{get_config_value("model_tools", LITELLM_MODEL_TOOLS, app_state.RUNTIME_OVERRIDES) or 'Not Set'}[/dim]
     Coding: [dim]{get_config_value("model_coding", LITELLM_MODEL_CODING, app_state.RUNTIME_OVERRIDES) or 'Not Set'}[/dim] | Knowledge: [dim]{get_config_value("model_knowledge", LITELLM_MODEL_KNOWLEDGE, app_state.RUNTIME_OVERRIDES) or 'Not Set'}[/dim]
     Summarize: [dim]{get_config_value("model_summarize", LITELLM_MODEL_SUMMARIZE, app_state.RUNTIME_OVERRIDES) or 'Not Set'}[/dim] | Planner: [dim]{get_config_value("model_planner", LITELLM_MODEL_PLANNER, app_state.RUNTIME_OVERRIDES) or 'Not Set'}[/dim]
     Task Mgr: [dim]{get_config_value("model_task_manager", LITELLM_MODEL_TASK_MANAGER, app_state.RUNTIME_OVERRIDES) or 'Not Set'}[/dim] | Rule Enh: [dim]{get_config_value("model_rule_enhancer", LITELLM_MODEL_RULE_ENHANCER, app_state.RUNTIME_OVERRIDES) or 'Not Set'}[/dim]
     Prompt Enh: [dim]{get_config_value("model_prompt_enhancer", LITELLM_MODEL_PROMPT_ENHANCER, app_state.RUNTIME_OVERRIDES) or 'Not Set'}[/dim] | Workflow Mgr: [dim]{get_config_value("model_workflow_manager", LITELLM_MODEL_WORKFLOW_MANAGER, app_state.RUNTIME_OVERRIDES) or 'Not Set'}[/dim]

  ❓ [bold bright_blue]/help[/bold bright_blue] - Documentation runtime entry point.

  🌐 [bold bright_blue]Online project resources:[/bold bright_blue]
     • Official Github Repository: https://github.com/PieBru/ai_engineer

  👥 [bold white]Just ask naturally, like you are explaining to a Software Engineer.[/bold white]"""

    app_state.console.print(Panel(
        instructions,
        border_style="blue",
        padding=(1, 2),
        title="[bold blue]🎯 Welcome to your Software Engineer AI Assistant[/bold blue]",
        title_align="left"
    ))
    app_state.console.print()

def get_context_usage_prompt_string(app_state: AppState) -> str:
    """
    Generates the context usage string for the prompt (e.g., "[Ctx: 50%] ").
    Returns an empty string if context info cannot be determined.
    """
    prefix = ""
    if app_state.conversation_history:
        try:
            active_model_for_prompt_context = get_config_value("model", LITELLM_MODEL_DEFAULT, app_state.RUNTIME_OVERRIDES, app_state.console)
            context_window_size_prompt, used_default_prompt = get_model_context_window(active_model_for_prompt_context, return_match_status=True)
            
            if app_state.conversation_history and active_model_for_prompt_context:
                tokens_used = token_counter(model=active_model_for_prompt_context, messages=app_state.conversation_history)
                if context_window_size_prompt > 0:
                    percentage_used = (tokens_used / context_window_size_prompt) * 100
                    default_note = " (default window)" if used_default_prompt else ""
                    prefix = f"[Ctx: {percentage_used:.0f}%{default_note}] "
                else:
                    prefix = f"[Ctx: {tokens_used} toks] "
        except Exception:
            # If token counting fails, don't break the prompt
            pass 
    return prefix

def execute_script_line(line: str, app_state: AppState):
    """Executes a single line from a script file."""
    app_state.console.print(f"\n[bold bright_magenta]📜 Script> {line}[/bold bright_magenta]")
    
    # Try command handlers first
    if command_handlers.try_handle_add_command(line, app_state): return
    if command_handlers.try_handle_set_command(line, app_state): return
    if command_handlers.try_handle_help_command(line, app_state): return
    if command_handlers.try_handle_shell_command(line, app_state): return
    if command_handlers.try_handle_session_command(line, app_state): return # Alias for /context
    if command_handlers.try_handle_rules_command(line, app_state): return
    if command_handlers.try_handle_context_command(line, app_state): return
    if command_handlers.try_handle_prompt_command(line, app_state): return
    if inference_tester.try_handle_test_command(line, app_state): return
    # Note: /debug and /time are interactive toggles, less common in scripts but could be supported
    # if command_handlers.try_handle_debug_command(line, app_state): return
    # if command_handlers.try_handle_time_command(line, app_state): return

    # If not a known command, treat as a prompt for the LLM
    target_model_for_script_line = get_config_value("model", LITELLM_MODEL_DEFAULT, app_state.RUNTIME_OVERRIDES, app_state.console)
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
        if command_handlers.try_handle_add_command(user_input, app_state): command_handled = True
        elif command_handlers.try_handle_set_command(user_input, app_state): command_handled = True
        elif command_handlers.try_handle_help_command(user_input, app_state): command_handled = True
        elif command_handlers.try_handle_shell_command(user_input, app_state): command_handled = True
        elif command_handlers.try_handle_session_command(user_input, app_state): command_handled = True
        elif command_handlers.try_handle_rules_command(user_input, app_state): command_handled = True
        elif command_handlers.try_handle_context_command(user_input, app_state): command_handled = True
        elif command_handlers.try_handle_prompt_command(user_input, app_state): command_handled = True
        elif command_handlers.try_handle_script_command(user_input, app_state): command_handled = True # For interactive /script
        elif command_handlers.try_handle_ask_command(user_input, app_state): command_handled = True
        elif command_handlers.try_handle_debug_command(user_input, app_state): command_handled = True
        elif command_handlers.try_handle_time_command(user_input, app_state): command_handled = True
        elif inference_tester.try_handle_test_command(user_input, app_state): command_handled = True
        
        if command_handled:
            continue

        # If input started with '/' but was not handled by any command, it's an unknown command
        if user_input.startswith("/"):
            app_state.console.print(f"[yellow]Unknown command: '{user_input.split()[0]}'. Type '/help' for a list of commands.[/yellow]")
            continue


        # --- Routing Step for non-commands ---
        target_model_for_this_turn = get_config_value("model", LITELLM_MODEL_DEFAULT, app_state.RUNTIME_OVERRIDES, app_state.console)
        
        app_state.console.print("[dim]↪ Routing query...[/dim]")

        greeting_pattern_for_routing_bypass = r"^\s*(hello|hi|hey|good\s+(morning|afternoon|evening)|how\s+are\s+you|how's\s+it\s+going|what's\s+up|sup)[\s!\.,\?]*\s*$"
        is_simple_greeting_for_bypass = bool(re.match(greeting_pattern_for_routing_bypass, user_input.strip(), re.IGNORECASE))

        if is_simple_greeting_for_bypass:
            if app_state.DEBUG_LLM_INTERACTIONS:
                Console(stderr=True).print("[dim]ROUTER DEBUG: Simple greeting detected, bypassing LLM router, using DEFAULT expert.[/dim]")
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
