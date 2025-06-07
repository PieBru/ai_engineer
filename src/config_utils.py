# /run/media/piero/NVMe-4TB/Piero/AI/AI-Engineer/config_utils.py
import os
from pathlib import Path
from typing import Dict, Any, Union, Tuple, Optional
from dotenv import load_dotenv

# Default LiteLLM configuration values - LITELLM_MODEL serves as LITELLM_MODEL_DEFAULT
DEFAULT_LITELLM_MODEL = "deepseek/deepseek-chat"
DEFAULT_LITELLM_API_BASE = "https://api.deepseek.com/v1"
DEFAULT_LITELLM_MAX_TOKENS = 8192  # For models which context window size is unknown

# Defaults for specialized models. Users should override these via .env for optimal use.
DEFAULT_LITELLM_MODEL_ROUTING = "ollama_chat/gemma3:4b-it-qat" # Often a smaller, faster model
DEFAULT_LITELLM_MODEL_TOOLS = "deepseek/deepseek-chat" # By default, tools model is same as default
DEFAULT_LITELLM_MODEL_CODING = "deepseek/deepseek-coder" # Specialized coding model, also "deepseek/deepseek-reasoner" (SOTA 2025, slower, more expensive)
DEFAULT_LITELLM_MODEL_KNOWLEDGE = "ollama_chat/gemma3:27b-it-qat" # For general knowledge, summarization

# Default UI and Reasoning configuration values
DEFAULT_REASONING_EFFORT = "medium"  # Possible values: "low", "medium", "high"
DEFAULT_REASONING_STYLE = "full"     # Possible values: "full", "compact", "silent"

# Global dictionary to hold runtime overrides set by /set command
RUNTIME_OVERRIDES: Dict[str, Any] = {}

# Configuration for the --test-inference summary table
SHOW_TEST_INFERENCE_NOTES_ERRORS_COLUMN = False # Set to False to hide the "Notes/Errors" column

# Define module-level constants for limits
MAX_FILES_TO_PROCESS_IN_DIR = 1000
MAX_FILE_SIZE_BYTES = 5_000_000  # 5MB

DEFAULT_MODEL_TEST_EXPECTATIONS: Dict[str, Any] = {
    "context_window": 16384,  # Fallback if a model is not in the map
    "supports_tools": False,  # Default expectation for tool support
    "is_thinking_model": False, # Default expectation for <think> prefix
    "thinking_type": None     # E.g., "qwen3", "deepseek", "mistral"
}

# Model-specific configurations including context window and test expectations
# LiteLLM documentation here: https://docs.litellm.ai/docs/providers
MODEL_CONFIGURATIONS: Dict[str, Dict[str, Any]] = {
    # Example structure:
    # "model_name": {
    #     "context_window": 128000,
    #     "supports_tools": True,
    #     "is_thinking_model": True,
    #     "thinking_type": "deepseek"
    # },

    "deepseek/deepseek-chat": {
        "context_window": 128000,
        "supports_tools": True, # DeepSeek models generally support tools
        "is_thinking_model": False,
    },
    "deepseek/deepseek-reasoner": {
        "context_window": 128000,
        "supports_tools": True,
        "is_thinking_model": False, # Reasons internally, does not show <think>
    },
    "deepseek/deepseek-coder": {
        "context_window": 128000,
        "supports_tools": True,
        "is_thinking_model": False,
    },
    
    "ollama_chat/qwen3:1.7b": { # Used as DEFAULT_LITELLM_MODEL_ROUTING
        "context_window": 32768, # Qwen models often have large context
        "supports_tools": True, # Smaller models might not always be tuned for tools
        "is_thinking_model": True, # Qwen models often use <think> or similar
        "thinking_type": "qwen"  # Use <think> ... </think>
    },
    "ollama_chat/gemma3:27b-it-qat": { # Used as DEFAULT_LITELLM_MODEL_KNOWLEDGE
        "context_window": 128000,
        "supports_tools": False,
        "is_thinking_model": False
    },
    "ollama_chat/qwq:latest": {
        "context_window": 131072,
        "supports_tools": True,
        "is_thinking_model": True,
        "thinking_type": "qwen"
    },
    "ollama_chat/deepcoder:14b-preview-q8_0": {
        "context_window": 16384,
        "supports_tools": True,
        "is_thinking_model": True,
        "thinking_type": "qwen"
    },

    # Add other models here with their specific configurations
    # OpenRouter Examples (FIXME: Update with actuals and expectations)
    # "openrouter/deepseek/deepseek-coder": {"context_window": 128000, "supports_tools": True, "is_thinking_model": True, "thinking_type": "deepseek"},
    # "openrouter/meta-llama/llama-3-8b-instruct": {"context_window": 8192, "supports_tools": True},

    # Cerebras Examples (FIXME: Update with actuals and expectations)
    # "cerebras/llama-3.3-70b": {"context_window": 32768, "supports_tools": True},
    # "cerebras/llama-4-scout-17b-16e-instruct": {"context_window": 32768, "supports_tools": True},

    # Gemini Examples (FIXME: Update with actuals and expectations)
    # "gemini/gemini-1.5-pro-latest": {"context_window": 1048576, "supports_tools": True},
}

SUPPORTED_SET_PARAMS = {
    "model": {
        "env_var": "LITELLM_MODEL",
        "description": "The default language model for general interaction (e.g., 'gpt-4o', 'deepseek-reasoner')."
    },
    "model_routing": {
        "env_var": "LITELLM_MODEL_ROUTING",
        "description": "The model used for routing requests to specialized models."
    },
    "model_tools": {
        "env_var": "LITELLM_MODEL_TOOLS",
        "description": "The model used for tasks requiring tool usage and orchestration."
    },
    "model_coding": {
        "env_var": "LITELLM_MODEL_CODING",
        "description": "The model specialized for code generation, analysis, and debugging."
    },
    "model_knowledge": {
        "env_var": "LITELLM_MODEL_KNOWLEDGE",
        "description": "The language model to use (e.g., 'gpt-4o', 'deepseek-reasoner')." # This description seems duplicated
    },
    "api_base": {
        "env_var": "LITELLM_API_BASE",
        "description": "The primary API base URL. Assumed to be used by all models unless a model-specific API base is configured (not yet supported)."
    },
    "max_tokens": {
        "env_var": "LITELLM_MAX_TOKENS",
        "description": "Maximum number of tokens for the LLM response (e.g., 4096)." # Corrected description
    },
    "reasoning_style": {
        "env_var": "REASONING_STYLE",
        "allowed_values": ["full", "compact", "silent"],
        "description": "Controls the display of AI's reasoning process: 'full' (stream all reasoning), 'compact' (show progress indicator), or 'silent' (no reasoning output during generation)."
    },
    "reasoning_effort": {
        "env_var": "REASONING_EFFORT",
        "allowed_values": ["low", "medium", "high"],
        "description": "Controls the AI's internal 'thinking' phase: 'low' (minimal thinking, direct answer), 'medium' (standard thinking depth), 'high' (deep, detailed thinking process, may use more tokens/time)."
    },
    "reply_effort": {
        "env_var": "REPLY_EFFORT",
        "allowed_values": ["low", "medium", "high"],
        "description": "Controls the verbosity of the AI's final reply: 'low' (synthetic, concise summary), 'medium' (standard detail, default), 'high' (detailed and comprehensive report/explanation)."
        },
    "temperature": {
        "env_var": "LITELLM_TEMPERATURE",
        "description": "Controls the randomness/creativity of the response (0.0 to 2.0, lower is more deterministic)."
    },
    "system_prompt": {
        "description": "Path to a file whose content will replace the current system prompt."
    }
}

def load_configuration(console_obj):
    """
    Loads .env file into environment variables.
    """
    load_dotenv()

def get_model_test_expectations(model_name: str) -> Dict[str, Any]:
    """
    Retrieves the full configuration dictionary for a given model name for testing purposes.
    It tries to find an exact match or a prefix match in MODEL_CONFIGURATIONS.
    Returns a dictionary with all expected fields, using defaults if not found.
    """
    if not model_name:
        return DEFAULT_MODEL_TEST_EXPECTATIONS.copy()

    # Exact match
    if model_name in MODEL_CONFIGURATIONS:
        # Merge with defaults to ensure all keys are present
        config = DEFAULT_MODEL_TEST_EXPECTATIONS.copy()
        config.update(MODEL_CONFIGURATIONS[model_name])
        return config

    # Prefix match (e.g., "gpt-4o-mini" should match "gpt-4o")
    # Sort keys by length descending to match more specific prefixes first
    sorted_model_keys = sorted(MODEL_CONFIGURATIONS.keys(), key=len, reverse=True)
    for prefix in sorted_model_keys:
        if model_name.startswith(prefix):
            config = DEFAULT_MODEL_TEST_EXPECTATIONS.copy()
            config.update(MODEL_CONFIGURATIONS[prefix])
            # If the original model name had a more specific context window in the old map,
            # we might want to preserve that. For now, prefix match takes all from matched entry.
            return config
            
    return DEFAULT_MODEL_TEST_EXPECTATIONS.copy()


def get_model_context_window(model_name: str, return_match_status: bool = False) -> Union[int, Tuple[int, bool]]:
    """
    Retrieves the context window size for a given model name.
    If return_match_status is True, returns a tuple (window_size, used_default_due_to_no_match).
    'used_default_due_to_no_match' is True if the default context window was returned because
    the model_name did not match any specific entry.
    """
    expectations = get_model_test_expectations(model_name)
    context_window = expectations.get("context_window", DEFAULT_MODEL_TEST_EXPECTATIONS["context_window"])
    
    used_default_val = True # Assume default unless a specific entry was found
    if model_name and model_name in MODEL_CONFIGURATIONS:
        used_default_val = False
    else: # Check prefix match
        sorted_model_keys = sorted(MODEL_CONFIGURATIONS.keys(), key=len, reverse=True)
        for prefix in sorted_model_keys:
            if model_name and model_name.startswith(prefix):
                used_default_val = False
                break
    
    if return_match_status:
        return context_window, used_default_val
    return context_window


def get_config_value(param_name: str, default_value: Any, console_obj=None) -> Any:
    """
    Retrieves a configuration value based on precedence:
    1. Runtime overrides
    2. Environment variables
    4. Provided default value
    """
    # Ensure param_name is valid before proceeding
    if param_name not in SUPPORTED_SET_PARAMS:
        # This case should ideally not be reached if callers use valid param_names
        if console_obj:
            console_obj.print(f"[yellow]Warning: Attempted to get unknown config param '{param_name}'. Using default.[/yellow]")
        return default_value
        
    p_config = SUPPORTED_SET_PARAMS[param_name]
    runtime_val = RUNTIME_OVERRIDES.get(param_name)
    if runtime_val is not None:
        if param_name == "max_tokens": return int(runtime_val) if isinstance(runtime_val, str) and runtime_val.isdigit() else runtime_val
        if param_name == "temperature": return float(runtime_val) if isinstance(runtime_val, (str, int, float)) else runtime_val # Allow int for temp
        return runtime_val
    
    env_var_name = p_config.get("env_var")
    env_val = None
    if env_var_name:
        env_val = os.getenv(env_var_name)

    if env_val is not None:
        if param_name == "max_tokens": return int(env_val) if env_val.isdigit() else default_value
        if param_name == "temperature":
            try: return float(env_val)
            except ValueError: return default_value
        if "allowed_values" in p_config and env_val.lower() in p_config["allowed_values"]:
            return env_val.lower()
        elif "allowed_values" not in p_config: 
             return env_val
    
    return default_value

# For backward compatibility with MODEL_CONTEXT_WINDOWS direct access if any part of the code still uses it.
# However, new logic should prefer get_model_test_expectations or get_model_context_window.
MODEL_CONTEXT_WINDOWS: Dict[str, int] = {
    k: v.get("context_window", DEFAULT_MODEL_TEST_EXPECTATIONS["context_window"]) 
    for k, v in MODEL_CONFIGURATIONS.items()
}
MODEL_CONTEXT_WINDOWS.update({ # Add defaults for role-based models if not explicitly in MODEL_CONFIGURATIONS
    DEFAULT_LITELLM_MODEL_ROUTING: get_model_test_expectations(DEFAULT_LITELLM_MODEL_ROUTING)["context_window"],
    DEFAULT_LITELLM_MODEL_TOOLS: get_model_test_expectations(DEFAULT_LITELLM_MODEL_TOOLS)["context_window"],
    DEFAULT_LITELLM_MODEL_CODING: get_model_test_expectations(DEFAULT_LITELLM_MODEL_CODING)["context_window"],
    DEFAULT_LITELLM_MODEL_KNOWLEDGE: get_model_test_expectations(DEFAULT_LITELLM_MODEL_KNOWLEDGE)["context_window"],
})

