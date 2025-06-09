# /home/piero/Piero/AI/AI-Engineer/src/commands/debug_command.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.app_state import AppState

def try_handle_debug_command(user_input: str, app_state: 'AppState') -> bool:
    command_prefix = "/debug"
    stripped_input = user_input.strip().lower()

    if not stripped_input.startswith(command_prefix):
        return False

    parts = stripped_input.split()
    if len(parts) == 1 and parts[0] == command_prefix:
        app_state.console.print(f"[yellow]Usage: /debug <on|off|rules|prompts|routing|tool_calls>[/yellow]")
        app_state.console.print(f"[dim]Current LLM interaction debug mode: {'ON' if app_state.DEBUG_LLM_INTERACTIONS else 'OFF'}[/dim]")
        app_state.console.print(f"[dim]Current Rules debug mode: {'ON' if app_state.DEBUG_RULES else 'OFF'}[/dim]")
        return True
    
    if len(parts) == 2:
        action = parts[1]
        if action == "on":
            app_state.DEBUG_LLM_INTERACTIONS = True
            app_state.console.print("[green]✓ LLM Interaction Debugging: ON[/green]")
        elif action == "off":
            app_state.DEBUG_LLM_INTERACTIONS = False
            app_state.DEBUG_RULES = False # Turn off sub-flags too
            app_state.console.print("[yellow]✓ LLM Interaction Debugging: OFF[/yellow]")
        elif action == "rules":
            app_state.DEBUG_RULES = not app_state.DEBUG_RULES
            status = "ON" if app_state.DEBUG_RULES else "OFF"
            app_state.console.print(f"[green]✓ Rules Debugging: {status}[/green]")
        else: app_state.console.print(f"[yellow]Unknown /debug action: {action}. Usage: /debug <on|off|rules|...>[/yellow]")
    else: app_state.console.print(f"[yellow]Usage: /debug <on|off|rules|...>[/yellow]")
    return True