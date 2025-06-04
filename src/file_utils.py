# /run/media/piero/NVMe-4TB/Piero/AI/AI-Engineer/file_utils.py
import os
from pathlib import Path
from typing import List
from rich.panel import Panel # For apply_diff_edit

# MAX_FILE_SIZE_BYTES will be imported from config_utils by the calling code
# and passed to functions like create_file.

def normalize_path(path_str: str) -> str:
    """Return a canonical, absolute version of the path with security checks."""
    try:
        if not path_str:
            raise ValueError("Path cannot be empty.")
        expanded_path = Path(path_str).expanduser()
        if ".." in expanded_path.parts:
            raise ValueError(f"Invalid path: {path_str} contains parent directory references")
        resolved_path = expanded_path.resolve()
        return str(resolved_path)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid path: \"{path_str}\". Error: {e}") from e
    except Exception as e:
        raise ValueError(f"Error normalizing path: \"{path_str}\". Details: {e}") from e

def is_binary_file(file_path: str, peek_size: int = 1024) -> bool:
    """Checks if a file is likely binary by looking for null bytes."""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(peek_size)
        if b'\0' in chunk:
            return True
        return False
    except Exception:
        return True # Err on the side of caution

def read_local_file(file_path: str) -> str:
    """Return the text content of a local file.
    Raises FileNotFoundError or OSError on issues.
    """
    # Normalization should be done by the caller if needed before passing here,
    # or this function could call normalize_path itself.
    # For this refactor, assuming caller (e.g., execute_function_call_dict) handles normalization.
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise
    except OSError:
        raise

def create_file(path: str, content: str, console_obj, max_file_size_bytes: int):
    """Create (or overwrite) a file at 'path' with the given 'content'."""
    if path.lstrip().startswith('~'):
        raise ValueError("Home directory references not allowed for create_file directly; normalize path first.")

    try:
        # Assuming path is already normalized by the caller if it came from user input.
        # If this function is called internally with potentially non-normalized paths,
        # normalization should happen here or be guaranteed by the caller.
        # For simplicity in this refactor, we'll assume `path` is ready to be used or
        # `normalize_path` has been called by the orchestrating function.
        # Let's add normalization here for safety if called directly.
        normalized_file_path_str = normalize_path(path)
    except ValueError as e:
        console_obj.print(f"[bold red]✗[/bold red] Could not create file. Invalid path: '[bright_cyan]{path}[/bright_cyan]'. Error: {e}")
        raise ValueError(f"Invalid path for create_file: {path}. Details: {e}") from e

    normalized_file_path_obj = Path(normalized_file_path_str)

    if len(content) > max_file_size_bytes:
        err_msg = f"File content exceeds {max_file_size_bytes // (1024*1024)}MB size limit"
        console_obj.print(f"[bold red]✗[/bold red] {err_msg}")
        raise ValueError(err_msg)
    
    try:
        normalized_file_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(normalized_file_path_obj, "w", encoding="utf-8") as f:
            f.write(content)
        console_obj.print(f"[bold blue]✓[/bold blue] Created/updated file at '[bright_cyan]{normalized_file_path_obj}[/bright_cyan]'")
    except OSError as e:
        err_msg = f"Failed to write file '{normalized_file_path_obj}': {e}"
        console_obj.print(f"[bold red]✗[/bold red] {err_msg}")
        raise OSError(err_msg) from e

def apply_diff_edit(path: str, original_snippet: str, new_snippet: str, console_obj, max_file_size_bytes: int):
    """Reads the file at 'path', replaces 'original_snippet' with 'new_snippet', then overwrites."""
    content = ""
    normalized_path_str = "" # Initialize
    try:
        normalized_path_str = normalize_path(path) # Normalize path first
        content = read_local_file(normalized_path_str)
        
        occurrences = content.count(original_snippet)
        if occurrences == 0:
            raise ValueError("Original snippet not found")
        if occurrences > 1:
            console_obj.print(f"[bold yellow]⚠ Multiple matches ({occurrences}) found - requiring line numbers for safety. Snippet found in '{normalized_path_str}'[/bold yellow]")
            console_obj.print("[dim]Use format:\n--- original.py (lines X-Y)\n+++ modified.py[/dim]")
            raise ValueError(f"Ambiguous edit: {occurrences} matches")
        
        updated_content = content.replace(original_snippet, new_snippet, 1)
        
        create_file(normalized_path_str, updated_content, console_obj, max_file_size_bytes)

        console_obj.print(f"[bold blue]✓[/bold blue] Applied diff edit to '[bright_cyan]{normalized_path_str}[/bright_cyan]'")

    except FileNotFoundError:
        msg = f"File not found for diff editing: '{normalized_path_str or path}'"
        console_obj.print(f"[bold red]✗[/bold red] {msg}")
        raise FileNotFoundError(msg) from None
    except ValueError as e:
        err_msg = str(e)
        console_obj.print(f"[bold yellow]⚠[/bold yellow] {err_msg} in '[bright_cyan]{normalized_path_str or path}[/bright_cyan]'. No changes made.")
        if "Original snippet not found" not in err_msg and "Ambiguous edit" not in err_msg and content:
            console_obj.print("\n[bold blue]Expected snippet:[/bold blue]")
            console_obj.print(Panel(original_snippet, title="Expected", border_style="blue", title_align="left"))
            console_obj.print("\n[bold blue]Actual file content:[/bold blue]")
            console_obj.print(Panel(content, title="Actual", border_style="yellow", title_align="left"))
        raise ValueError(f"Failed to apply diff to '{normalized_path_str or path}': {err_msg}") from e
    except OSError as e:
        err_msg = str(e)
        console_obj.print(f"[bold red]✗[/bold red] OS error during diff edit for '{normalized_path_str or path}': {err_msg}")
        raise OSError(f"OS error during diff edit for '{normalized_path_str or path}': {err_msg}") from e
