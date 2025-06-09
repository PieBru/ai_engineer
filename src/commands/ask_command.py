# /home/piero/Piero/AI/AI-Engineer/src/commands/ask_command.py
from typing import TYPE_CHECKING

from src.llm_interaction import stream_llm_response
from src.config_utils import get_config_value, DEFAULT_LITELLM_MODEL

if TYPE_CHECKING:
    from src.app_state import AppState

def try_handle_ask_command(user_input: str, app_state: 'AppState') -> bool:
    command_prefix = "/ask"
    if user_input.lower().startswith(command_prefix.lower()):
        prompt_text = user_input[len(command_prefix):].strip()
        if not prompt_text:
            app_state.console.print("[yellow]Usage: /ask <text_to_send_to_llm>[/yellow]")
        else:
            app_state.console.print(f"[dim]💬 Sending direct prompt to LLM: '{prompt_text[:50]}...'[/dim]")
            default_model = get_config_value("model", DEFAULT_LITELLM_MODEL, app_state.RUNTIME_OVERRIDES, app_state.console)
            stream_llm_response(prompt_text, app_state, target_model_override=default_model)
        return True
    return False