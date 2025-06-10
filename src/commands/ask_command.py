# /home/piero/Piero/AI/AI-Engineer/src/commands/ask_command.py
from typing import TYPE_CHECKING

from src.config_utils import get_config_value
from src.llm_interaction import stream_llm_response

if TYPE_CHECKING:
    from src.app_state import AppState

def try_handle_ask_command(user_input: str, app_state: 'AppState') -> bool:
    command_prefix = "/ask"
    if user_input.lower().startswith(command_prefix.lower()):
        prompt_content = user_input[len(command_prefix):].strip()
        if not prompt_content:
            app_state.console.print("[yellow]Usage: /ask <your question for the default LLM>[/yellow]")
        else:
            target_model = get_config_value("model", app_state.RUNTIME_OVERRIDES, app_state.console)
            app_state.console.print(f"[dim]Directly asking default model ({target_model})...[/dim]")
            stream_llm_response(
                prompt_content,
                app_state,
                target_model_override=target_model
            )
        return True
    return False