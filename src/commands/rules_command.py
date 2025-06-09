# /home/piero/Piero/AI/AI-Engineer/src/commands/rules_command.py
from typing import TYPE_CHECKING

from src import rules_manager
from src.prompts import RichMarkdown # For display_rules_help_summary

if TYPE_CHECKING:
    from src.app_state import AppState

def display_rules_help_summary(app_state: 'AppState'):
    """Displays a brief help summary for /rules, pointing to /help rules."""
    app_state.console.print(RichMarkdown(
        "**`/rules <subcommand> [arguments]`** - Manage AI's guiding rules.\n"
        "Subcommands: `show`, `list [all|enabled|disabled]`, `enable <pattern>`, `disable <pattern>`, `reset`.\n"
        "Type `/help rules` for detailed documentation."
    ))

def try_handle_rules_command(user_input: str, app_state: 'AppState') -> bool:
    if user_input.lower().startswith("/rules"):
        parts = user_input.split()

        if len(parts) == 1: # Just "/rules"
            display_rules_help_summary(app_state)
            return True

        subcommand = parts[1].lower()

        if subcommand == "show":
            rules_manager.show_active_rules_command(app_state)
        elif subcommand == "list":
            list_filter = parts[2].lower() if len(parts) > 2 else "enabled"
            if list_filter not in ["enabled", "disabled", "all"]:
                app_state.console.print(f"[yellow]Invalid filter for '/rules list'. Use 'enabled', 'disabled', or 'all'.[/yellow]")
            else:
                rules_manager.list_rules_command(app_state, list_filter)
        elif subcommand == "enable":
            rule_pattern = " ".join(parts[2:]) if len(parts) > 2 else ""
            if not rule_pattern: app_state.console.print("[yellow]Usage: /rules enable <rule_pattern>[/yellow]")
            else: rules_manager.enable_rules_command(app_state, rule_pattern)
        elif subcommand == "disable":
            rule_pattern = " ".join(parts[2:]) if len(parts) > 2 else ""
            if not rule_pattern: app_state.console.print("[yellow]Usage: /rules disable <rule_pattern>[/yellow]")
            else: rules_manager.disable_rules_command(app_state, rule_pattern)
        elif subcommand == "reset":
            rules_manager.reset_rules_command(app_state)
        else:
            app_state.console.print(f"[yellow]Unknown /rules subcommand: '{subcommand}'. Type '/rules' or '/help rules' for usage.[/yellow]")
        return True
    return False