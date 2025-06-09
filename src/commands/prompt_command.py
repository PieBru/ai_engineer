# /home/piero/Piero/AI/AI-Engineer/src/commands/prompt_command.py
from typing import TYPE_CHECKING
from src import rules_manager # Needs rules_manager for show_active_rules_command

if TYPE_CHECKING:
    from src.app_state import AppState

def try_handle_prompt_command(user_input: str, app_state: 'AppState') -> bool:
    if user_input.lower().strip() == "/prompt":
        rules_manager.show_active_rules_command(app_state) # Alias to /rules show
        return True
    return False