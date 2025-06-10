# /run/media/piero/NVMe-4TB/Piero/AI/AI-Engineer/config_utils.py
import os
from pathlib import Path
from typing import Dict, Any, Union, Tuple, Optional
import toml # Added for TOML parsing
from dotenv import load_dotenv

# --- Ultimate Fallback Defaults ---
# These are used if config.toml is missing or a key is not found,
# and no environment variable or runtime override is set.
ULTIMATE_DEFAULTS = {
    "api_base_ollama": "http://localhost:11434",
    "api_base_lm_studio": "http://localhost:1234/v1",
    "model": "ollama_chat/mistral-small",
    "model_routing": "ollama_chat/gemma3:1b-it-qat",
    "model_tools": "ollama_chat/mistral-small", # Placeholder, will use 'model' if not in TOML
    "model_coding": "ollama_chat/devstral",
    "model_knowledge": "ollama_chat/mistral-small", # Placeholder
    "model_summarize": "ollama_chat/mistral-small", # Placeholder
    "model_planner": "ollama_chat/mistral-small", # Placeholder
    "model_task_manager": "ollama_chat/mistral-small", # Placeholder
    "model_rule_enhancer": "ollama_chat/mistral-small", # Placeholder
    "model_prompt_enhancer": "ollama_chat/mistral-small", # Placeholder
    "model_workflow_manager": "ollama_chat/mistral-small", # Placeholder
    "max_tokens": 32768,
    "max_tokens_routing": 150,
    "reasoning_effort": "medium",
    "reasoning_style": "full",
    "temperature": 0.6, # Added a default for temperature
}

# This dictionary will hold configurations loaded from config.toml
_CONFIG_FROM_TOML: Dict[str, Any] = {}

# Python-defined constants (could also be moved to TOML if desired later)
DEFAULT_REASONING_EFFORT_PY = "medium"  # Possible values: "low", "medium", "high"
DEFAULT_REASONING_STYLE_PY = "full"     # Possible values: "full", "compact", "silent"

# Configuration for the --test-inference summary table
SHOW_TEST_INFERENCE_NOTES_ERRORS_COLUMN = False # Set to False to hide the "Notes/Errors" column

# Define module-level constants for limits
MAX_FILES_TO_PROCESS_IN_DIR = 1000
MAX_FILE_SIZE_BYTES = 5_000_000  # 5MB

DEFAULT_MODEL_TEST_EXPECTATIONS: Dict[str, Any] = {
    "context_window": 16384,  # Fallback if a model is not in the map
    "supports_tools": False,  # Default expectation for tool support
    "is_thinking_model": False, # Default expectation for <think> prefix
    "thinking_type": None,    # E.g., "qwen3", "deepseek", "mistral"
    "api_base": None          # Default API base for a model, None means use global or provider default
}

# Model-specific configurations including context window and test expectations
# LiteLLM documentation here: https://docs.litellm.ai/docs/providers
MODEL_CONFIGURATIONS: Dict[str, Dict[str, Any]] = {
    # Example structure:
    # "model_name": {
    #     "context_window": 128000,
    #     "supports_tools": True,
    #     "is_thinking_model": True,
    #     "thinking_type": "qwen",
    #     "api_base": None # Example: "http://my-specific-ollama:11434" or None to use default
    #     "api_base_provider_key": "ollama" # New way to link to TOML api_bases

    # },

    # To do
    # "ollama_chat/mistral-small3.1:latest"
    # "ollama_chat/codestral:22b-v0.1-q5_K_S"
    # "openrouter/meta-llama/llama-4-scout:free"
    # "openrouter/meta-llama/llama-4-maverick:free"
    # "ollama_chat/phi4:14b-q8_0"
    # "ollama_chat/phi4-mini:3.8b-q8_0"
    # "ollama_chat/phi4-reasoning:latest"
    # "ollama_chat/THUDM_GLM-4-32B-0414-Q5_K_M.gguf:latest"
    # "cerebras/llama-4-scout-17b-16e-instruct"
    # "gemini/gemini-2.0-flash": 128000,  # Assuming a large context for a Flash model
    # "gemini/gemini-2.0-pro": 256000,    # Assuming a very large context for a Pro model
    # "gemini/gemini-2.5-flash": 128000,  # Assuming similar to 2.0 Flash or potentially larger
    # "gemini/gemini-2.5-pro": 1048576,   # Assuming it matches or aims for Gemini 1.5 Pro's scale

    "ollama_chat/mistral-small": {  # https://ollama.com/library/mistral-small:24b
        "context_window": 32768,
        "supports_tools": True,
        "is_thinking_model": False,
        "api_base_provider_key": "ollama"
    },
    "ollama_chat/devstral": {  # https://ollama.com/library/devstral
        "context_window": 131072,
        "supports_tools": False,
        "is_thinking_model": False,
        "api_base_provider_key": "ollama"
    },
    "ollama_chat/deepcoder:14b-preview-q8_0": {  # https://www.together.ai/blog/deepcoder
        "context_window": 65536,
        "supports_tools": True,
        "is_thinking_model": True,
        "thinking_type": "qwen",
        "api_base_provider_key": "ollama"
    },
   "ollama_chat/qwen2.5-coder:3b": {
        "context_window": 32768,
        "supports_tools": False,
        "is_thinking_model": False,
        "api_base_provider_key": "ollama"
    },
   "ollama_chat/qwen2.5-coder:7b": {
        "context_window": 131072,
        "supports_tools": False,
        "is_thinking_model": False,
        "api_base_provider_key": "ollama"
    },
   "ollama_chat/qwen2.5-coder:14b": {
        "context_window": 131072,
        "supports_tools": False,
        "is_thinking_model": False,
        "api_base_provider_key": "ollama"
    },
   "ollama_chat/qwen2.5-coder:32b": {  # https://qwenlm.github.io/blog/qwen2.5-coder-family/
        "context_window": 131072,
        "supports_tools": False,
        "is_thinking_model": False,
        "api_base_provider_key": "ollama"
    },
    "ollama_chat/qwen3:0.6b": {
        "context_window": 40000,
        "supports_tools": False,
        "is_thinking_model": True,
        "thinking_type": "qwen",  # Use <think> ... </think>
        "api_base_provider_key": "ollama"
    },
    "ollama_chat/qwen3:1.7b": {
        "context_window": 40000,
        "supports_tools": True,
        "is_thinking_model": True,
        "thinking_type": "qwen",  # Use <think> ... </think>
        "api_base_provider_key": "ollama"
    },
    "ollama_chat/qwen3:4b": {
        "context_window": 40000,
        "supports_tools": True,
        "is_thinking_model": True,
        "thinking_type": "qwen",  # Use <think> ... </think>
        "api_base_provider_key": "ollama"
    },
    "ollama_chat/qwen3:8b": {
        "context_window": 40000,
        "supports_tools": True,
        "is_thinking_model": True,
        "thinking_type": "qwen",  # Use <think> ... </think>
        "api_base_provider_key": "ollama"
    },
    "ollama_chat/qwen3:14b": {
        "context_window": 40000,
        "supports_tools": True,
        "is_thinking_model": True,
        "thinking_type": "qwen",  # Use <think> ... </think>
        "api_base_provider_key": "ollama"
    },
    "ollama_chat/qwen3:30b": {
        "context_window": 40000,
        "supports_tools": False,
        "is_thinking_model": True,
        "thinking_type": "qwen",  # Use <think> ... </think>
        "api_base_provider_key": "ollama"
    },
    "ollama_chat/qwen3:32b": {
        "context_window": 40000,
        "supports_tools": True,
        "is_thinking_model": True,
        "thinking_type": "qwen",  # Use <think> ... </think>
        "api_base_provider_key": "ollama"
    },
    "ollama_chat/gemma3:1b-it-qat": {
        "context_window": 128000,
        "supports_tools": False,
        "is_thinking_model": False,
        "api_base_provider_key": "ollama"
    },
    "ollama_chat/gemma3:4b-it-qat": {
        "context_window": 128000,
        "supports_tools": False,
        "is_thinking_model": False,
        "api_base_provider_key": "ollama"
    },
    "ollama_chat/gemma3:12b-it-qat": {
        "context_window": 128000,
        "supports_tools": False,
        "is_thinking_model": False,
        "api_base_provider_key": "ollama"
    },
    "ollama_chat/gemma3:27b-it-qat": {
        "context_window": 128000,
        "supports_tools": False,
        "is_thinking_model": False,
        "api_base_provider_key": "ollama"
    },
    "ollama_chat/qwq": {
        "context_window": 131072,
        "supports_tools": True,
        "is_thinking_model": True,
        "thinking_type": "qwen",
        "api_base_provider_key": "ollama"
    },

    "lm_studio/deepseek-r1-0528-qwen3-8b@q8_0": {
        "context_window": 131072,
        "supports_tools": False,
        "is_thinking_model": True,
        "thinking_type": "qwen",
        "api_base_provider_key": "lm_studio"
    },

    #"deepseek/deepseek-chat": {
    #    "context_window": 128000,
    #    "supports_tools": True,
    #    "is_thinking_model": False,
    #},
    #"deepseek/deepseek-reasoner": {
    #    "context_window": 128000,
    #    "supports_tools": True,
    #    "is_thinking_model": False, # Reasons internally, does not show <think>
    #},
    #"deepseek/deepseek-coder": {
    #    "context_window": 128000,
    #    "supports_tools": True,
    #    "is_thinking_model": False,
    #    "api_base": None # Uses LiteLLM's default for deepseek
    #},

    # Add other models here with their specific configurations
    # OpenRouter Examples (FIXME: Update with actuals and expectations)
    # "openrouter/deepseek/deepseek-coder": {"context_window": 128000, "supports_tools": True, "is_thinking_model": True, "thinking_type": "deepseek"},
    # "openrouter/meta-llama/llama-3-8b-instruct": {"context_window": 8192, "supports_tools": True},
    # For OpenRouter, api_base is typically None as LiteLLM handles it.

    # Cerebras Examples (FIXME: Update with actuals and expectations)
    # "cerebras/llama-3.3-70b": {"context_window": 32768, "supports_tools": True},
    # "cerebras/llama-4-scout-17b-16e-instruct": {"context_window": 32768, "supports_tools": True},
    # For Cerebras, api_base is typically None.

    # Gemini Examples (FIXME: Update with actuals and expectations)
    # "gemini/gemini-1.5-pro-latest": {"context_window": 1048576, "supports_tools": True},
    # For Gemini, api_base is typically None.
}

SUPPORTED_SET_PARAMS = {
    "model": {
        "env_var": "LITELLM_MODEL",
        "description": "The default language model for general interaction (e.g., 'ollama_chat/devstral', 'deepseek-reasoner')."
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
        "description": "The language model to use (e.g., 'ollama_chat/devstral', 'deepseek-reasoner')." # This description seems duplicated
    },
    "model_summarize": {
        "env_var": "LITELLM_MODEL_SUMMARIZE",
        "description": "The model specialized for summarization tasks."
    },
    "model_planner": {
        "env_var": "LITELLM_MODEL_PLANNER",
        "description": "The model used for planning complex tasks and outlining steps."
    },
    "model_task_manager": {
        "env_var": "LITELLM_MODEL_TASK_MANAGER",
        "description": "The model used for breaking down planned tasks into smaller, manageable sub-tasks."
    },
    "model_rule_enhancer": {
        "env_var": "LITELLM_MODEL_RULE_ENHANCER",
        "description": "The model used for analyzing and enhancing system rules or existing prompts."
    },
    "model_prompt_enhancer": {
        "env_var": "LITELLM_MODEL_PROMPT_ENHANCER",
        "description": "The model used for refining and detailing user-provided prompts for better AI interaction."
    },
    "model_workflow_manager": {
        "env_var": "LITELLM_MODEL_WORKFLOW_MANAGER",
        "description": "The model used for managing and orchestrating multi-step workflows or agentic sequences."
    },
    "api_base": {
        "env_var": "LITELLM_API_BASE",
        "description": "The primary API base URL. Assumed to be used by all models unless a model-specific API base is configured (not yet supported)."
    },
    "max_tokens": {
        "env_var": "LITELLM_MAX_TOKENS",
        "description": "Maximum number of tokens for the LLM response (e.g., 4096)." # Corrected description
    },
    "max_tokens_routing": {
        "env_var": "LITELLM_MAX_TOKENS_ROUTING",
        "description": "Maximum number of tokens for the routing LLM response (e.g., 150)."
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

def update_runtime_override(param_name: str, value: Any, runtime_overrides: Dict[str, Any], console_obj=None):
    """
    Updates a runtime override for a given parameter.
    Validates against SUPPORTED_SET_PARAMS.
    """
    param_name_lower = param_name.lower()
    if param_name_lower not in SUPPORTED_SET_PARAMS:
        if console_obj:
            console_obj.print(f"[red]Error: Unknown parameter '{param_name}'. Cannot set override.[/red]")
        return

    config_details = SUPPORTED_SET_PARAMS[param_name_lower]
    allowed_values = config_details.get("allowed_values")

    # Type conversion and validation
    if param_name_lower in ["max_tokens", "max_tokens_routing"]:
        try:
            value = int(value)
            if value <= 0:
                raise ValueError("Max tokens must be a positive integer.")
        except ValueError:
            if console_obj:
                console_obj.print(f"[red]Error: Invalid value '{value}' for {param_name_lower}. Must be a positive integer.[/red]")
            return
    elif param_name_lower == "temperature":
        try:
            value = float(value)
            if not (0.0 <= value <= 2.0): # Common range for temperature
                raise ValueError("Temperature must be between 0.0 and 2.0.")
        except ValueError:
            if console_obj:
                console_obj.print(f"[red]Error: Invalid value '{value}' for {param_name_lower}. Must be a number (e.g., 0.7).[/red]")
            return

    if allowed_values and str(value).lower() not in allowed_values:
        if console_obj:
            console_obj.print(f"[red]Error: Invalid value '{value}' for {param_name_lower}. Allowed values: {', '.join(allowed_values)}[/red]")
        return

    runtime_overrides[param_name_lower] = value
    if console_obj:
        console_obj.print(f"[green]✓ Runtime override set: {param_name_lower} = {value}[/green]")

def remove_runtime_override(param_name: str, runtime_overrides: Dict[str, Any], console_obj=None):
    """Removes a runtime override."""
    if param_name.lower() in runtime_overrides:
        del runtime_overrides[param_name.lower()]
        if console_obj: console_obj.print(f"[yellow]✓ Runtime override removed for: {param_name.lower()}[/yellow]")
    elif console_obj: console_obj.print(f"[dim]No runtime override found for '{param_name.lower()}' to remove.[/dim]")

def list_runtime_overrides(runtime_overrides: Dict[str, Any], console_obj):
    """Lists current runtime overrides."""
    if not runtime_overrides:
        console_obj.print("[dim]No active runtime overrides.[/dim]")
        return
    console_obj.print("[bold blue]Active Runtime Overrides:[/bold blue]")
    for key, value in runtime_overrides.items():
        console_obj.print(f"  - {key}: {value}")

def load_configuration(console_obj):
    """
    Loads .env file into environment variables and config.toml into _CONFIG_FROM_TOML.
    Then, updates MODEL_CONFIGURATIONS with API bases from TOML.
    """
    load_dotenv()
    global _CONFIG_FROM_TOML, MODEL_CONFIGURATIONS

    try:
        toml_config_path = Path("config.toml")
        if toml_config_path.exists():
            loaded_toml = toml.load(toml_config_path)
            
            # Flatten TOML structure into _CONFIG_FROM_TOML for easier access
            # e.g., models.default becomes "model" (matching SUPPORTED_SET_PARAMS keys)
            # e.g., api_bases.ollama becomes "api_base_ollama"
            if "api_bases" in loaded_toml and isinstance(loaded_toml["api_bases"], dict):
                for key, value in loaded_toml["api_bases"].items():
                    _CONFIG_FROM_TOML[f"api_base_{key}"] = value
            
            if "models" in loaded_toml and isinstance(loaded_toml["models"], dict):
                for key, value in loaded_toml["models"].items():
                    # Map TOML model keys to SUPPORTED_SET_PARAMS keys
                    # 'default' -> 'model', 'routing' -> 'model_routing', etc.
                    param_key = f"model_{key}" if key != "default" else "model"
                    _CONFIG_FROM_TOML[param_key] = value

            if "tokens" in loaded_toml and isinstance(loaded_toml["tokens"], dict):
                _CONFIG_FROM_TOML["max_tokens"] = loaded_toml["tokens"].get("default_max")
                _CONFIG_FROM_TOML["max_tokens_routing"] = loaded_toml["tokens"].get("routing_max")

            if "reasoning" in loaded_toml and isinstance(loaded_toml["reasoning"], dict):
                _CONFIG_FROM_TOML["reasoning_effort"] = loaded_toml["reasoning"].get("effort")
                _CONFIG_FROM_TOML["reasoning_style"] = loaded_toml["reasoning"].get("style")

            # Dynamically update api_base in MODEL_CONFIGURATIONS
            for model_name, model_cfg in MODEL_CONFIGURATIONS.items():
                provider_key = model_cfg.get("api_base_provider_key")
                if provider_key:
                    model_cfg["api_base"] = _CONFIG_FROM_TOML.get(f"api_base_{provider_key}", ULTIMATE_DEFAULTS.get(f"api_base_{provider_key}"))
    except Exception as e:
        if console_obj:
            console_obj.print(f"[yellow]Warning: Could not load or parse config.toml: {e}. Using internal defaults.[/yellow]")

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

    # Prefix match (e.g., "ollama_chat/devstral-mini" should match "ollama_chat/devstral")
    # Sort keys by length descending to match more specific prefixes first
    sorted_model_keys = sorted(MODEL_CONFIGURATIONS.keys(), key=len, reverse=True)
    for prefix in sorted_model_keys:
        if model_name.startswith(prefix):
            config = DEFAULT_MODEL_TEST_EXPECTATIONS.copy()
            config.update(MODEL_CONFIGURATIONS[prefix])
            # Ensure api_base from the specific prefix match is preserved if it was set
            if MODEL_CONFIGURATIONS[prefix].get("api_base") is not None:
                config["api_base"] = MODEL_CONFIGURATIONS[prefix]["api_base"]
            return config
            
    # If no match, return a copy of defaults
    config = DEFAULT_MODEL_TEST_EXPECTATIONS.copy()
    # Apply provider-default API base if model_name gives a hint and api_base is still None
    # This logic is now mostly handled by MODEL_CONFIGURATIONS' api_base_provider_key and load_configuration
    if config.get("api_base") is None and config.get("api_base_provider_key") is None:
        if model_name.startswith("ollama_chat/"):
            config["api_base"] = _CONFIG_FROM_TOML.get("api_base_ollama", ULTIMATE_DEFAULTS["api_base_ollama"])
        elif model_name.startswith("lm_studio/"):
            config["api_base"] = _CONFIG_FROM_TOML.get("api_base_lm_studio", ULTIMATE_DEFAULTS["api_base_lm_studio"])
    return config


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


def get_config_value(param_name: str, runtime_overrides: Dict[str, Any], console_obj=None) -> Any:
    """
    Retrieves a configuration value based on precedence:
    1. Runtime overrides
    2. Environment variables
    3. Values from config.toml (_CONFIG_FROM_TOML)
    4. Ultimate hardcoded defaults (ULTIMATE_DEFAULTS)
    """
    if param_name not in SUPPORTED_SET_PARAMS:
        if console_obj:
            console_obj.print(f"[yellow]Warning: Attempted to get unknown config param '{param_name}'. Using default.[/yellow]")
        return ULTIMATE_DEFAULTS.get(param_name) # Or raise error
        
    p_config = SUPPORTED_SET_PARAMS[param_name]
    runtime_val = runtime_overrides.get(param_name)

    if runtime_val is not None:
        if param_name == "max_tokens": return int(runtime_val) if isinstance(runtime_val, str) and runtime_val.isdigit() else runtime_val
        if param_name == "max_tokens_routing": return int(runtime_val) if isinstance(runtime_val, str) and runtime_val.isdigit() else runtime_val
        if param_name == "temperature": return float(runtime_val) if isinstance(runtime_val, (str, int, float)) else runtime_val # Allow int for temp
        return runtime_val

    env_var_name = p_config.get("env_var")
    env_val = None
    if env_var_name:
        env_val = os.getenv(env_var_name)

    if env_val is not None:
        # Use ultimate default as fallback for type conversion errors from env var
        ultimate_fallback = _CONFIG_FROM_TOML.get(param_name, ULTIMATE_DEFAULTS.get(param_name))
        if param_name == "max_tokens": return int(env_val) if env_val.isdigit() else ultimate_fallback
        if param_name == "max_tokens_routing": return int(env_val) if env_val.isdigit() else ultimate_fallback
        if param_name == "temperature":
            try: return float(env_val)
            except ValueError: return ultimate_fallback
        if "allowed_values" in p_config and env_val.lower() in p_config["allowed_values"]:
            return env_val.lower()
        elif "allowed_values" not in p_config: 
             return env_val

    # Fallback to TOML config, then to ultimate defaults
    # For specialized models, if not in TOML, use the 'model' (default model) value
    if param_name.startswith("model_") and param_name != "model_routing": # e.g. model_coding, model_tools
        toml_val = _CONFIG_FROM_TOML.get(param_name)
        if toml_val is not None:
            return toml_val
        # Fallback to the general 'model' value from TOML or ultimate defaults
        return _CONFIG_FROM_TOML.get("model", ULTIMATE_DEFAULTS.get("model"))

    return _CONFIG_FROM_TOML.get(param_name, ULTIMATE_DEFAULTS.get(param_name))
