# /home/piero/Piero/AI/AI-Engineer/src/commands/shell_command.py
from typing import TYPE_CHECKING
import subprocess
from pathlib import Path

if TYPE_CHECKING:
    from src.app_state import AppState

def try_handle_shell_command(user_input: str, app_state: 'AppState') -> bool:
    command_prefix_shell = "/shell"
    command_prefix_bang = "/!"
    stripped_input = user_input.strip()
    
    shell_command_text = None

    if stripped_input.lower().startswith(command_prefix_shell):
        shell_command_text = stripped_input[len(command_prefix_shell):].strip()
    elif stripped_input.startswith(command_prefix_bang):
        shell_command_text = stripped_input[len(command_prefix_bang):].strip()
    else:
        return False

    if not shell_command_text:
        app_state.console.print("[yellow]Usage: /shell <command_to_execute>  OR  /! <command_to_execute>[/yellow]")
        return True

    app_state.console.print(f"[bold cyan]Executing shell command: '{shell_command_text}'[/bold cyan]")
    try:
        app_state.console.print("[bold yellow]⚠️  Executing with shell=True. Ensure commands are safe.[/bold yellow]")
        process = subprocess.Popen(
            shell_command_text, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=Path.cwd()
        )
        stdout, stderr = process.communicate(timeout=60)

        if process.returncode == 0:
            app_state.console.print("[bold green]Shell Command Output:[/bold green]")
            if stdout: app_state.console.print(stdout.strip())
        else:
            app_state.console.print(f"[bold red]Shell Command Error (Code: {process.returncode}):[/bold red]")
            if stderr: app_state.console.print(stderr.strip(), style="red")
            elif stdout: app_state.console.print(stdout.strip(), style="yellow")
    except subprocess.TimeoutExpired:
        app_state.console.print("[bold red]Shell command timed out after 60 seconds.[/bold red]")
    except Exception as e:
        app_state.console.print(f"[bold red]Error executing shell command: {e}[/bold red]")
    
    return True