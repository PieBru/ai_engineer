# /home/piero/Piero/AI/AI-Engineer/src/commands/context_command.py
from typing import TYPE_CHECKING
from rich.panel import Panel
from rich.text import Text

if TYPE_CHECKING:
    from src.app_state import AppState

def try_handle_context_command(user_input: str, app_state: 'AppState') -> bool:
    # This function handles both /context and /session (via try_handle_session_command)
    if not (user_input.lower().strip().startswith("/context") or user_input.lower().strip().startswith("/session")):
        return False

    if not app_state.conversation_history:
        app_state.console.print("[yellow]Conversation history is empty.[/yellow]")
        return True

    app_state.console.print(Panel(
        title="[bold blue]💬 Conversation Context[/bold blue]",
        border_style="blue", padding=(1,1)
    ))

    for i, msg in enumerate(app_state.conversation_history):
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls")

        role_color = "green" if role == "User" else "cyan" if role == "Assistant" else "magenta" if role == "Tool" else "yellow"
        app_state.console.print(f"╭─ [bold {role_color}]{role}[/bold {role_color}] ({i+1}/{len(app_state.conversation_history)})")
        if content: app_state.console.print(f"│  {content[:200]}{'...' if content and len(content) > 200 else ''}")
        if tool_calls: app_state.console.print(f"│  [dim]Tool Calls: {len(tool_calls)} call(s)[/dim]")
        app_state.console.print(f"╰─")
    app_state.console.print()
    return True