# src/app_state.py
from rich.console import Console
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style as PromptStyle
from typing import List, Dict, Any

from src.prompts import system_PROMPT

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
        self.conversation_history: List[Dict[str, Any]] = [
            {"role": "system", "content": system_PROMPT}
        ]
        self.DEBUG_LLM_INTERACTIONS: bool = False
        self.SHOW_TIMESTAMP_IN_PROMPT: bool = False
        self.RUNTIME_OVERRIDES: Dict[str, Any] = {}
        # Potentially add other shared states if they become numerous