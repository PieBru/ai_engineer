# src/startup_checks.py
import os
import sys
from pathlib import Path

# Try to import Rich Console for nicer output, but fall back to plain print if not available.
# This is important because if we're not in a venv, Rich might not be installed.
try:
    from rich.console import Console
    from rich.panel import Panel
except ImportError:
    Console = None
    Panel = None

def perform_venv_check():
    """
    Checks if the application is running in an active virtual environment.
    If not, prints an error message and exits.
    """
    if os.getenv("VIRTUAL_ENV") is None:
        # Not in an active venv. Check if a common venv directory exists.
        venv_dir_found = False
        activation_suggestion = ""
        common_venv_names = [".venv", "venv"]

        for venv_name in common_venv_names:
            potential_venv_path = Path(venv_name)
            if potential_venv_path.is_dir():
                venv_dir_found = True
                if sys.platform == "win32":
                    activate_script_path = potential_venv_path / "Scripts" / "activate.bat"
                    activate_command = f"{venv_name}\\Scripts\\activate"
                    if not activate_script_path.exists(): # Check for PowerShell variant
                        activate_script_path_ps = potential_venv_path / "Scripts" / "Activate.ps1"
                        if activate_script_path_ps.exists():
                             activate_command = f".\\{venv_name}\\Scripts\\Activate.ps1" # PowerShell
                else: # Linux/macOS
                    activate_script_path = potential_venv_path / "bin" / "activate"
                    activate_command = f"source {venv_name}/bin/activate"
                
                activation_suggestion = (
                    f"It looks like a virtual environment '{venv_name}' might exist in the current directory.\n"
                    f"Try activating it with:\n  {activate_command}"
                )
                break

        if venv_dir_found and activation_suggestion:
            message = (
                "ERROR: Not running in an active Python virtual environment.\n\n"
                f"{activation_suggestion}\n\n"
                "Once activated, ensure dependencies are installed (if not already):\n"
                "  uv pip install -r requirements.txt -U --link-mode=copy\n\n"
                "If this is not the correct environment, or if you need to create a new one:\n"
                "Example (Linux/macOS):\n"
                "  python3 -m venv <new_venv_name>\n"
                "  source <new_venv_name>/bin/activate\n"
                "Example (Windows):\n"
                "  python -m venv <new_venv_name>\n"
                "  <new_venv_name>\\Scripts\\activate\n\n"
                "Exiting."
            )
        else:
            message = (
                "ERROR: Not running in a Python virtual environment.\n\n"
                "This application relies on specific dependencies managed within a virtual environment.\n"
                "Please activate your virtual environment or create one and install dependencies.\n\n"
                "Example (Linux/macOS):\n"
                "  python3 -m venv .venv\n"
                "  source .venv/bin/activate\n"
                "  uv pip install -r requirements.txt -U --link-mode=copy\n\n"
                "Example (Windows):\n"
                "  python -m venv .venv\n"
                "  .venv\\Scripts\\activate\n"
                "  pip install -r requirements.txt\n\n"
                "Exiting."
            )

        if Console and Panel:
            console = Console(stderr=True)
            console.print(Panel(message, title="[bold red]Virtual Environment Check Failed[/bold red]", border_style="red", expand=False))
        else:
            sys.stderr.write("------------------------------------------------------------\n")
            sys.stderr.write("           Virtual Environment Check Failed\n")
            sys.stderr.write("------------------------------------------------------------\n")
            sys.stderr.write(message + "\n")
        sys.exit(1)