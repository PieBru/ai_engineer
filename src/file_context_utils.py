# /home/piero/Piero/AI/AI-Engineer/src/file_context_utils.py
import os
from pathlib import Path
from rich.console import Console

from src.file_utils import normalize_path, is_binary_file, read_local_file as util_read_local_file
from src.config_utils import MAX_FILES_TO_PROCESS_IN_DIR, MAX_FILE_SIZE_BYTES

def add_directory_to_conversation(
    directory_path: str,
    conversation_history: list,
    console: Console
):
    """
    Scans a directory and adds the content of eligible files to the conversation history.
    """
    with console.status("[bold bright_blue]🔍 Scanning directory...[/bold bright_blue]") as status:
        excluded_files = {
            ".DS_Store", "Thumbs.db", ".gitignore", ".python-version",
            "uv.lock", ".uv", "uvenv", ".uvenv", ".venv", "venv",
            "__pycache__", ".pytest_cache", ".coverage", ".mypy_cache",
            "node_modules", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
            ".next", ".nuxt", "dist", "build", ".cache", ".parcel-cache",
            ".turbo", ".vercel", ".output", ".contentlayer",
            "out", "coverage", ".nyc_output", "storybook-static",
            ".env", ".env.local", ".env.development", ".env.production",
            ".git", ".svn", ".hg", "CVS"
        }
        excluded_extensions = {
            ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp", ".avif",
            ".mp4", ".webm", ".mov", ".mp3", ".wav", ".ogg",
            ".zip", ".tar", ".gz", ".7z", ".rar",
            ".exe", ".dll", ".so", ".dylib", ".bin",
            ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
            ".pyc", ".pyo", ".pyd", ".egg", ".whl",
            ".uv", ".uvenv",
            ".db", ".sqlite", ".sqlite3", ".log",
            ".idea", ".vscode",
            ".map", ".chunk.js", ".chunk.css",
            ".min.js", ".min.css", ".bundle.js", ".bundle.css",
            ".cache", ".tmp", ".temp",
            ".ttf", ".otf", ".woff", ".woff2", ".eot"
        }
        skipped_files = []
        added_files = []
        total_files_processed = 0
        for root, dirs, files in os.walk(directory_path):
            if total_files_processed >= MAX_FILES_TO_PROCESS_IN_DIR:
                console.print(f"[bold yellow]⚠[/bold yellow] Reached maximum file limit ({MAX_FILES_TO_PROCESS_IN_DIR})")
                break
            status.update(f"[bold bright_blue]🔍 Scanning {root}...[/bold bright_blue]")
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in excluded_files]
            for file in files:
                if total_files_processed >= MAX_FILES_TO_PROCESS_IN_DIR:
                    break
                if file.startswith('.') or file in excluded_files:
                    skipped_files.append(str(Path(root) / file))
                    continue
                _, ext = os.path.splitext(file)
                if ext.lower() in excluded_extensions:
                    skipped_files.append(os.path.join(root, file))
                    continue
                full_path = str(Path(root) / file)
                try:
                    if os.path.getsize(full_path) > MAX_FILE_SIZE_BYTES:
                        skipped_files.append(f"{full_path} (exceeds size limit)")
                        continue
                    if is_binary_file(full_path):
                        skipped_files.append(full_path)
                        continue
                    normalized_path = normalize_path(full_path)
                    content = util_read_local_file(normalized_path)
                    conversation_history.append({
                        "role": "system",
                        "content": f"Content of file '{normalized_path}':\n\n{content}"
                    })
                    added_files.append(normalized_path)
                    total_files_processed += 1
                except OSError:
                    skipped_files.append(str(full_path))
                except ValueError as e:
                     skipped_files.append(f"{full_path} (Invalid path: {e})")
        console.print(f"[bold blue]✓[/bold blue] Added folder '[bright_cyan]{directory_path}[/bright_cyan]' to conversation.")
        if added_files:
            console.print(f"\n[bold bright_blue]📁 Added files:[/bold bright_blue] [dim]({len(added_files)} of {total_files_processed})[/dim]")
            for f_path in added_files:
                console.print(f"  [bright_cyan]📄 {f_path}[/bright_cyan]")
        if skipped_files:
            console.print(f"\n[bold yellow]⏭ Skipped files:[/bold yellow] [dim]({len(skipped_files)})[/dim]")
            for f_path in skipped_files[:10]:
                console.print(f"  [yellow dim]⚠ {f_path}[/yellow dim]")
            if len(skipped_files) > 10:
                console.print(f"  [dim]... and {len(skipped_files) - 10} more[/dim]")
        console.print()

def ensure_file_in_context(
    file_path: str,
    conversation_history: list,
    console: Console
) -> bool:
    """
    Ensures that the content of the given file_path is present in the conversation_history.
    If not, it reads the file and adds its content.
    Returns True if the file content is in context (or successfully added), False otherwise.
    """
    try:
        normalized_path = normalize_path(file_path)
        content = util_read_local_file(normalized_path)
        file_marker = f"Content of file '{normalized_path}'"
        if not any(file_marker in msg["content"] for msg in conversation_history if msg.get("content")):
            conversation_history.append({
                "role": "system",
                "content": f"{file_marker}:\n\n{content}"
            })
        return True
    except (OSError, ValueError) as e:
        console.print(f"[bold red]✗[/bold red] Could not read file '[bright_cyan]{file_path}[/bright_cyan]' for editing context: {e}")
        return False