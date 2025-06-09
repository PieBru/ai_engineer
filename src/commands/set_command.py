# /home/piero/Piero/AI/AI-Engineer/src/commands/set_command.py
from typing import TYPE_CHECKING

from src.config_utils import SUPPORTED_SET_PARAMS, update_runtime_override

if TYPE_CHECKING:
    from src.app_state import AppState

def try_handle_set_command(user_input: str, app_state: 'AppState') -> bool:
    command_prefix = "/set"
    stripped_input = user_input.strip()

    if not stripped_input.lower().startswith(command_prefix.lower()):
        return False

    args_text = stripped_input[len(command_prefix):].strip()
    
    if not args_text:
        app_state.console.print("[yellow]Usage: /set <parameter_name> <value>[/yellow]")
        app_state.console.print("[dim]Example: /set model ollama_chat/mistral-small[/dim]")
        app_state.console.print("[dim]Type '/help set' for a list of settable parameters.[/dim]")
        return True

    parts = args_text.split(maxsplit=1)
    if len(parts) < 2:
        app_state.console.print("[yellow]Usage: /set <parameter_name> <value>[/yellow]")
        app_state.console.print(f"[dim]You provided: /set {args_text}[/dim]")
        return True

    param_name, param_value = parts[0].lower(), parts[1]

    if param_name not in SUPPORTED_SET_PARAMS:
        app_state.console.print(f"[red]Error: Unknown parameter '{param_name}'. Type '/help set' for options.[/red]")
        return True

    update_runtime_override(param_name, param_value, app_state.RUNTIME_OVERRIDES, app_state.console)
    return True