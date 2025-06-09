# /home/piero/Piero/AI/AI-Engineer/src/commands/help_command.py
from typing import TYPE_CHECKING
from pathlib import Path

from rich.panel import Panel

from src.file_utils import read_local_file as util_read_local_file
from src.prompts import RichMarkdown

if TYPE_CHECKING:
    from src.app_state import AppState

def try_handle_help_command(user_input: str, app_state: 'AppState') -> bool:
    command_prefix = "/help"
    stripped_input = user_input.strip()

    if not stripped_input.lower().startswith(command_prefix.lower()):
        return False

    arg_text = stripped_input[len(command_prefix):].strip()
    help_file_basename = "help" # Default

    if arg_text:
        sanitized_arg_basename = "".join(c for c in arg_text if c.isalnum() or c in ['_', '-', '.'])
        if sanitized_arg_basename:
            help_file_basename = sanitized_arg_basename
        else:
            app_state.console.print(f"[yellow]Warning: Invalid characters in help topic '{arg_text}'. Showing default help page.[/yellow]")
    
    help_file_name_md = f"{help_file_basename}.md"
    default_help_file_name_md = "help.md"

    project_root_dir = Path(__file__).resolve().parent.parent.parent 
    script_help_dir = project_root_dir / "help"
    cwd_help_dir = Path("./help").resolve()

    requested_script_path = script_help_dir / help_file_name_md
    requested_cwd_path = cwd_help_dir / help_file_name_md
    default_script_path = script_help_dir / default_help_file_name_md
    default_cwd_path = cwd_help_dir / default_help_file_name_md

    help_content = None
    loaded_from_cwd = False
    is_default_fallback = False
    loaded_file_path_for_title = requested_script_path

    try:
        help_content = util_read_local_file(str(requested_script_path))
        loaded_file_path_for_title = requested_script_path
    except FileNotFoundError:
        try:
            help_content = util_read_local_file(str(requested_cwd_path))
            loaded_from_cwd = True
            loaded_file_path_for_title = requested_cwd_path
            app_state.console.print(f"[dim]Info: Help file '{help_file_name_md}' loaded from CWD ('{requested_cwd_path}').[/dim]")
        except FileNotFoundError:
            if help_file_basename.lower() != "help":
                app_state.console.print(f"[red]Error: Help topic '{arg_text}' (file '{help_file_name_md}') not found. Attempting default help.[/red]")
                is_default_fallback = True
                try:
                    help_content = util_read_local_file(str(default_script_path))
                    loaded_file_path_for_title = default_script_path
                except FileNotFoundError:
                    try:
                        help_content = util_read_local_file(str(default_cwd_path))
                        loaded_from_cwd = True
                        loaded_file_path_for_title = default_cwd_path
                        app_state.console.print(f"[dim]Info: Default help file '{default_help_file_name_md}' loaded from CWD ('{default_cwd_path}').[/dim]")
                    except FileNotFoundError:
                        app_state.console.print(f"[red]Error: Default help file ('{default_help_file_name_md}') also not found.[/red]")
                        return True 
                    except OSError as e: app_state.console.print(f"[red]Error reading default help from CWD: {e}[/red]"); return True
                except OSError as e: app_state.console.print(f"[red]Error reading default help from script dir: {e}[/red]"); return True
            else:
                app_state.console.print(f"[red]Error: Main help file ('{help_file_name_md}') not found.[/red]")
                return True
        except OSError as e: app_state.console.print(f"[red]Error reading help from CWD: {e}[/red]"); return True
    except OSError as e: app_state.console.print(f"[red]Error reading help from script dir: {e}[/red]"); return True

    if help_content:
        title_base_name_for_display = Path(loaded_file_path_for_title).stem
        if is_default_fallback and help_file_basename.lower() != "help":
             title_base_name_for_display = f"Default (requested: {help_file_basename})"
        elif is_default_fallback:
             title_base_name_for_display = "Default"

        title_location_suffix = " (from CWD)" if loaded_from_cwd else ""
        panel_title = f"[bold blue]📚 Software Engineer AI Assistant Help ({title_base_name_for_display}){title_location_suffix}[/bold blue]"
            
        app_state.console.print(Panel(
            RichMarkdown(help_content), title=panel_title, title_align="left", border_style="blue"
        ))
    
    return True