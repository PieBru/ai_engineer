# /home/piero/Piero/AI/AI-Engineer/src/commands/time_command.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.app_state import AppState

def try_handle_time_command(user_input: str, app_state: 'AppState') -> bool:
    command_prefix = "/time"
    stripped_input = user_input.strip().lower()

    if not stripped_input.startswith(command_prefix):
        return False

    app_state.SHOW_TIMESTAMP_IN_PROMPT = not app_state.SHOW_TIMESTAMP_IN_PROMPT
    status_msg = "[green]✓ Timestamp display in prompt: ON[/green]" if app_state.SHOW_TIMESTAMP_IN_PROMPT else "[yellow]✓ Timestamp display in prompt: OFF[/yellow]"
    app_state.console.print(status_msg)
    return True