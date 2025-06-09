# /home/piero/Piero/AI/AI-Engineer/src/commands/add_command.py
from typing import TYPE_CHECKING
from pathlib import Path

from src.file_utils import normalize_path
from src.file_context_utils import add_directory_to_conversation, ensure_file_in_context

if TYPE_CHECKING:
    from src.app_state import AppState

def try_handle_add_command(user_input: str, app_state: 'AppState') -> bool:
    command_prefix = "/add"
    stripped_input = user_input.strip()

    if not stripped_input.lower().startswith(command_prefix.lower()):
        return False

    path_arg = stripped_input[len(command_prefix):].strip()

    if not path_arg:
        app_state.console.print("[yellow]Usage: /add <path/to/file_or_folder>[/yellow]")
        return True

    try:
        normalized_path_str = normalize_path(path_arg)
        path_obj = Path(normalized_path_str)

        if path_obj.is_file():
            ensure_file_in_context(str(path_obj), app_state.conversation_history, app_state.console)
        elif path_obj.is_dir():
            add_directory_to_conversation(str(path_obj), app_state.conversation_history, app_state.console)
        else:
            app_state.console.print(f"[red]Error: Path '{path_arg}' is not a valid file or directory.[/red]")
    except ValueError as e: # Catches errors from normalize_path or other path issues
        app_state.console.print(f"[red]Error processing path '{path_arg}': {e}[/red]")
    
    return True # Command was handled, even if there was an error processing the path