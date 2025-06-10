# src/command_handlers.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.app_state import AppState

# Import individual command handlers
from src.commands.add_command import try_handle_add_command
from src.commands.show_command import try_handle_show_command # Import the new handler
from src.commands.set_command import try_handle_set_command
from src.commands.help_command import try_handle_help_command
from src.commands.shell_command import try_handle_shell_command
from src.commands.session_command import try_handle_session_command
from src.commands.rules_command import try_handle_rules_command
from src.commands.context_command import try_handle_context_command
from src.commands.prompt_command import try_handle_prompt_command
from src.commands.script_command import try_handle_script_command
from src.commands.ask_command import try_handle_ask_command
from src.commands.debug_command import try_handle_debug_command
from src.commands.time_command import try_handle_time_command

# Add other try_handle_... functions as you create/move them.