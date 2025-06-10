import os

# Import Rich components
from rich.panel import Panel

# Import application state and config utilities
from src.app_state import AppState
from src.config_utils import (
    get_config_value,
    get_model_context_window
)

def display_welcome_panel(app_state: AppState):
    """Displays the welcome panel."""
    current_model_name_for_display = get_config_value("model", app_state.RUNTIME_OVERRIDES, app_state.console)
    context_window_size_display, used_default_display = get_model_context_window(current_model_name_for_display, return_match_status=True)
    context_window_display_str = f"Context:{context_window_size_display // 1024}k tokens"
    current_working_directory = os.getcwd()
    if used_default_display:
        context_window_display_str += " (default)"

    instructions = f"""  📁 [bold bright_blue]Current Directory: [/bold bright_blue][bold green]{current_working_directory}[/bold green]

  🧠 [bold bright_blue]Default Model: [/bold bright_blue][bold magenta]{current_model_name_for_display}[/bold magenta] ([dim]{context_window_display_str}[/dim])
     Routing: [dim]{get_config_value("model_routing", app_state.RUNTIME_OVERRIDES, app_state.console) or 'Not Set'}[/dim] | Tools: [dim]{get_config_value("model_tools", app_state.RUNTIME_OVERRIDES, app_state.console) or 'Not Set'}[/dim]
     Coding: [dim]{get_config_value("model_coding", app_state.RUNTIME_OVERRIDES, app_state.console) or 'Not Set'}[/dim] | Knowledge: [dim]{get_config_value("model_knowledge", app_state.RUNTIME_OVERRIDES, app_state.console) or 'Not Set'}[/dim]
     Summarize: [dim]{get_config_value("model_summarize", app_state.RUNTIME_OVERRIDES, app_state.console) or 'Not Set'}[/dim] | Planner: [dim]{get_config_value("model_planner", app_state.RUNTIME_OVERRIDES, app_state.console) or 'Not Set'}[/dim]
     Task Mgr: [dim]{get_config_value("model_task_manager", app_state.RUNTIME_OVERRIDES, app_state.console) or 'Not Set'}[/dim] | Rule Enh: [dim]{get_config_value("model_rule_enhancer", app_state.RUNTIME_OVERRIDES, app_state.console) or 'Not Set'}[/dim]
     Prompt Enh: [dim]{get_config_value("model_prompt_enhancer", app_state.RUNTIME_OVERRIDES, app_state.console) or 'Not Set'}[/dim] | Workflow Mgr: [dim]{get_config_value("model_workflow_manager", app_state.RUNTIME_OVERRIDES, app_state.console) or 'Not Set'}[/dim]

  ❓ [bold bright_blue]/help[/bold bright_blue] - Documentation runtime entry point.

  🌐 [bold bright_blue]Online project resources:[/bold bright_blue]
     • Official Github Repository: https://github.com/PieBru/ai_engineer

  👥 [bold white]Just ask naturally, like you are explaining to a Software Engineer.[/bold white]"""

    app_state.console.print(Panel(
        instructions,
        border_style="blue",
        padding=(1, 2),
        title="[bold blue]🎯 Welcome to your Software Engineer AI Assistant[/bold blue]",
        title_align="left"
    ))
    app_state.console.print()