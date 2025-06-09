# In a new script_runner.py or within command_handlers.py
# from src import command_handlers # If in a separate file
# from src import routing_logic, stream_llm_response, inference_tester, get_config_value, LITELLM_MODEL_DEFAULT # etc.

def process_script_line(line: str, app_state: 'AppState', is_recursive_script_call: bool = False):
    """
    Processes a single line of input, similar to the main loop in ai_engineer.py.
    The is_recursive_script_call flag is to prevent /script from calling /script infinitely.
    """
    app_state.console.print(f"\n[bold bright_magenta]📜 Script> {line}[/bold bright_magenta]") # Or adjust prefix

    # Try command handlers first
    # Note: Be careful with commands that might be problematic in scripts or cause loops.
    if command_handlers.try_handle_add_command(line, app_state): return
    if command_handlers.try_handle_set_command(line, app_state): return
    # ... other safe commands ...
    if command_handlers.try_handle_rules_command(line, app_state): return
    # ...
    if command_handlers.try_handle_ask_command(line, app_state): return

    # Prevent /script from calling /script from within a script to avoid loops, unless designed carefully.
    if line.lower().startswith("/script") and is_recursive_script_call:
        app_state.console.print("[yellow]Recursive /script call detected and skipped within script execution.[/yellow]")
        return

    # If not a known command, treat as a prompt for the LLM
    # (This part is simplified from ai_engineer.py's main loop)
    from src.routing_logic import get_routing_expert_keyword, map_expert_to_model # Local import
    from src.llm_interaction import stream_llm_response # Local import
    from src.config_utils import get_config_value, DEFAULT_LITELLM_MODEL # Local import

    target_model_for_script_line = get_config_value("model", DEFAULT_LITELLM_MODEL, app_state.RUNTIME_OVERRIDES, app_state.console)
    
    if line and not line.startswith("/"): # Assuming non-command lines are prompts
        app_state.console.print("[dim]↪ Routing query (from script)...[/dim]")
        expert_keyword_script = get_routing_expert_keyword(line, app_state)
        target_model_for_script_line = map_expert_to_model(expert_keyword_script, app_state)
    
    stream_llm_response(
        line,
        app_state,
        target_model_override=target_model_for_script_line
    )
