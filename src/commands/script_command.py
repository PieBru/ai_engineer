# /home/piero/Piero/AI/AI-Engineer/src/commands/script_command.py
from typing import TYPE_CHECKING
from pathlib import Path

from src.file_utils import normalize_path
# from src.script_runner import process_script_line # This will be needed once script_runner is finalized

if TYPE_CHECKING:
    from src.app_state import AppState

def try_handle_script_command(user_input: str, app_state: 'AppState', is_startup_script: bool = False, noconfirm: bool = False) -> bool:
    command_prefix = "/script"
    stripped_input = user_input.strip()

    if not stripped_input.lower().startswith(command_prefix.lower()):
        return False

    script_path_arg = stripped_input[len(command_prefix):].strip()

    if not script_path_arg:
        if not is_startup_script:
            app_state.console.print("[yellow]Usage: /script <path/to/script_file.txt>[/yellow]")
        return True

    try:
        normalized_script_path_str = normalize_path(script_path_arg)
        script_file = Path(normalized_script_path_str)

        if not script_file.is_file():
            app_state.console.print(f"[red]Error: Script file not found at '{normalized_script_path_str}'[/red]")
            return True

        app_state.console.print(f"[bold green]Executing script: {normalized_script_path_str}[/bold green]")
        with open(script_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"): continue
                app_state.console.print(f"[dim]Script Line {i+1}: {line} (Execution logic to be fully implemented here via script_runner.process_script_line)[/dim]")
                # process_script_line(line, app_state, is_recursive_script_call=True) # Call the actual processor
        app_state.console.print(f"[bold green]Finished executing script: {normalized_script_path_str}[/bold green]")
    except ValueError as e: app_state.console.print(f"[red]Error processing script path '{script_path_arg}': {e}[/red]")
    except OSError as e: app_state.console.print(f"[red]Error reading script file '{script_path_arg}': {e}[/red]")
    except Exception as e: app_state.console.print(f"[red]An unexpected error occurred during script execution: {e}[/red]")
    
    return True