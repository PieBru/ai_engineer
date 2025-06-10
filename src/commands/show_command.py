from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.app_state import AppState

from src.ui_display import display_welcome_panel # Import the moved function

def try_handle_show_command(user_input: str, app_state: 'AppState') -> bool:
    """
    Handles the /show command.
    Currently supports:
        /show welcome - Displays the welcome panel with current stats.
    """
    parts = user_input.lower().strip().split()
    if not parts or parts[0] != "/show":
        return False

    if len(parts) == 2 and parts[1] == "welcome":
        display_welcome_panel(app_state)
        return True
    elif len(parts) == 1: # Just "/show"
        app_state.console.print("[yellow]Usage: /show welcome[/yellow]")
        return True
    else:
        app_state.console.print(f"[yellow]Unknown argument for /show: '{parts[1]}'. Try '/show welcome'.[/yellow]")
        return True