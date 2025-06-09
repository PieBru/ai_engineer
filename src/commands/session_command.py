# /home/piero/Piero/AI/AI-Engineer/src/commands/session_command.py
from typing import TYPE_CHECKING
from .context_command import try_handle_context_command # Import from sibling module

if TYPE_CHECKING:
    from src.app_state import AppState

def try_handle_session_command(user_input: str, app_state: 'AppState') -> bool:
    if user_input.lower().strip().startswith("/session"):
        return try_handle_context_command(user_input, app_state) # Delegate
    return False