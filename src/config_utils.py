# /run/media/piero/NVMe-4TB/Piero/AI/AI-Engineer/config_utils.py
import os
from pathlib import Path
import tomllib # For reading TOML config (Python 3.11+)
from typing import Dict, Any
from dotenv import load_dotenv

# Global dictionary to hold configurations loaded from config.toml
CONFIG_FROM_TOML: Dict[str, Any] = {}
RUNTIME_OVERRIDES: Dict[str, Any] = {}

# Define module-level constants for limits
MAX_FILES_TO_PROCESS_IN_DIR = 1000
MAX_FILE_SIZE_BYTES = 5_000_000  # 5MB

SUPPORTED_SET_PARAMS = {
    "model": {
        "env_var": "LITELLM_MODEL",
        "toml_section": "litellm",
        "toml_key": "model",
        "default_value_key": "default_model",
        "description": "The language model to use (e.g., 'gpt-4o', 'deepseek-reasoner')."
    },
    "api_base": {
        "env_var": "LITELLM_API_BASE",
        "toml_section": "litellm",
        "toml_key": "api_base",
        "default_value_key": "default_api_base",
        "description": "The API base URL for the LLM provider."
    },
    "reasoning_style": {
        "env_var": "REASONING_STYLE",
        "toml_section": "ui",
        "toml_key": "reasoning_style",
        "default_value_key": "default_reasoning_style",
        "allowed_values": ["full", "compact", "silent"],
        "description": "Controls the display of AI's reasoning process: 'full' (stream all reasoning), 'compact' (show progress indicator), or 'silent' (no reasoning output during generation)."
    },
    "max_tokens": {
        "env_var": "LITELLM_MAX_TOKENS",
        "toml_section": "litellm",
        "toml_key": "max_tokens",
        "default_value_key": "default_max_tokens",
        "description": "Maximum number of tokens for the LLM response (e.g., 4096)."
    },
    "reasoning_effort": {
        "env_var": "REASONING_EFFORT",
        "toml_section": "litellm",
        "toml_key": "reasoning_effort",
        "default_value_key": "default_reasoning_effort",
        "allowed_values": ["low", "medium", "high"],
        "description": "Controls the AI's internal 'thinking' phase: 'low' (minimal thinking, direct answer), 'medium' (standard thinking depth), 'high' (deep, detailed thinking process, may use more tokens/time)."
    },
    "reply_effort": {
        "env_var": "REPLY_EFFORT",
        "toml_section": "ui",
        "toml_key": "reply_effort",
        "default_value_key": "default_reply_effort",
        "allowed_values": ["low", "medium", "high"],
        "description": "Controls the verbosity of the AI's final reply: 'low' (synthetic, concise summary), 'medium' (standard detail, default), 'high' (detailed and comprehensive report/explanation)."
        },
    "temperature": {
        "env_var": "LITELLM_TEMPERATURE",
        "toml_section": "litellm",
        "toml_key": "temperature",
        "default_value_key": "default_temperature",
        "description": "Controls the randomness/creativity of the response (0.0 to 2.0, lower is more deterministic)."
    }
}

def load_configuration(console_obj):
    """
    Loads configuration with the following precedence:
    1. Environment variables (highest)
    2. .env file
    3. config.toml file
    4. Hardcoded defaults (lowest)
    
    Updates the global CONFIG_FROM_TOML dictionary.
    """
    global CONFIG_FROM_TOML

    # Load .env file into environment variables
    load_dotenv()

    # Load config.toml
    config_file_path = Path("config.toml")
    if config_file_path.exists():
        try:
            with open(config_file_path, "rb") as f:
                CONFIG_FROM_TOML = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            console_obj.print(f"[yellow]Warning: Could not parse config.toml: {e}. Using defaults and environment variables.[/yellow]")
        except Exception as e: # pylint: disable=broad-except
            console_obj.print(f"[yellow]Warning: Error loading config.toml: {e}. Using defaults and environment variables.[/yellow]")

def get_config_value(param_name: str, default_value: Any, console_obj=None) -> Any:
    """
    Retrieves a configuration value based on precedence:
    1. Runtime overrides
    2. Environment variables
    3. TOML configuration
    4. Provided default value
    """
    p_config = SUPPORTED_SET_PARAMS[param_name]
    runtime_val = RUNTIME_OVERRIDES.get(param_name)
    if runtime_val is not None:
        if param_name == "max_tokens": return int(runtime_val) if isinstance(runtime_val, str) and runtime_val.isdigit() else runtime_val
        if param_name == "temperature": return float(runtime_val) if isinstance(runtime_val, str) else runtime_val
        return runtime_val
    
    env_val = os.getenv(p_config["env_var"])
    if env_val is not None:
        # Convert env var string to appropriate type if necessary
        if param_name == "max_tokens": return int(env_val) if env_val.isdigit() else default_value
        if param_name == "temperature":
            try: return float(env_val)
            except ValueError: return default_value
        # For reasoning_style, reasoning_effort, reply_effort, env_val is fine as string if it matches allowed_values
        if "allowed_values" in p_config and env_val.lower() in p_config["allowed_values"]:
            return env_val.lower()
        elif "allowed_values" not in p_config: # For model, api_base
             return env_val
        # If env_val is set but not allowed for style/effort, it will fall through to TOML/default
        # This could be logged as a warning if desired.

    toml_section_name = p_config.get("toml_section")
    toml_key_name = p_config.get("toml_key")
    if toml_section_name and toml_key_name:
        toml_val = CONFIG_FROM_TOML.get(toml_section_name, {}).get(toml_key_name)
        if toml_val is not None:
            return toml_val
    
    return default_value
