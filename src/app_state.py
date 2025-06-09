# src/app_state.py
import os
from rich.console import Console
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style as PromptStyle
from typing import List, Dict, Any


class AppState:
    def __init__(self):
        self.console = Console()
        self.prompt_session = PromptSession(
            style=PromptStyle.from_dict({
                'prompt': '#0066ff bold',
                'completion-menu.completion': 'bg:#1e3a8a fg:#ffffff',
                'completion-menu.completion.current': 'bg:#3b82f6 fg:#ffffff bold',
            })
        )
        # Conversation history will store user/assistant/tool turns.
        # The system prompt is managed by self.system_prompt and prepended by llm_interaction.
        self.conversation_history: List[Dict[str, Any]] = []
        self.DEBUG_LLM_INTERACTIONS: bool = False
        self.SHOW_TIMESTAMP_IN_PROMPT: bool = False
        self.RUNTIME_OVERRIDES: Dict[str, Any] = {}
        self.DEBUG_RULES = os.getenv("AIE_DEBUG_RULES", "false").lower() == "true"
        self.system_prompt = "" # Will be populated by rules_manager.initialize_rules_system