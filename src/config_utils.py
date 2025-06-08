# /run/media/piero/NVMe-4TB/Piero/AI/AI-Engineer/config_utils.py
import os
from pathlib import Path
from typing import Dict, Any, Union, Tuple, Optional
from dotenv import load_dotenv

# Default LiteLLM configuration values - LITELLM_MODEL serves as LITELLM_MODEL_DEFAULT
DEFAULT_OLLAMA_API_BASE = "http://localhost:11434"
DEFAULT_LM_STUDIO_API_BASE = "http://localhost:1234/v1"
DEFAULT_LITELLM_MAX_TOKENS = 4096  # Example, adjust as needed
DEFAULT_LITELLM_MAX_TOKENS = 32768  # For models which context window size is unknown

# Defaults for specialized models. Users should override these via .env for optimal use.
DEFAULT_LITELLM_MODEL =                  "ollama_chat/mistral-small"
DEFAULT_LITELLM_MODEL_ROUTING =          "ollama_chat/qwen3:4b" # Often a smaller, faster model
DEFAULT_LITELLM_MODEL_TOOLS =            DEFAULT_LITELLM_MODEL # A good tool calls handler
DEFAULT_LITELLM_MODEL_CODING =           "ollama_chat/devstral" # Specialized coding model, also "deepseek/deepseek-reasoner" (SOTA 2025, slower, more expensive)
DEFAULT_LITELLM_MODEL_KNOWLEDGE =        DEFAULT_LITELLM_MODEL # For general knowledge
DEFAULT_LITELLM_MODEL_SUMMARIZE =        DEFAULT_LITELLM_MODEL # For text summarization
DEFAULT_LITELLM_MODEL_PLANNER =          DEFAULT_LITELLM_MODEL # For planning complex tasks
DEFAULT_LITELLM_MODEL_TASK_MANAGER =     DEFAULT_LITELLM_MODEL # For breaking down tasks
DEFAULT_LITELLM_MODEL_RULE_ENHANCER =    DEFAULT_LITELLM_MODEL # For refining rules/prompts
DEFAULT_LITELLM_MODEL_PROMPT_ENHANCER =  DEFAULT_LITELLM_MODEL # For enhancing user prompts
DEFAULT_LITELLM_MODEL_WORKFLOW_MANAGER = DEFAULT_LITELLM_MODEL # For managing multi-step workflows

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
        "api_base": DEFAULT_OLLAMA_API_BASE
    },
    "ollama_chat/devstral": {  # https://ollama.com/library/devstral
        "context_window": 131072,
        "supports_tools": False,
        "is_thinking_model": False,
        "api_base": DEFAULT_OLLAMA_API_BASE
    },
    "ollama_chat/deepcoder:14b-preview-q8_0": {  # https://www.together.ai/blog/deepcoder
        "context_window": 65536,
        "supports_tools": True,
        "is_thinking_model": True,
        "thinking_type": "qwen",
        "api_base": DEFAULT_OLLAMA_API_BASE
    },
   "ollama_chat/qwen2.5-coder:3b": {
        "context_window": 32768,
        "supports_tools": False,
        "is_thinking_model": False,
        "api_base": DEFAULT_OLLAMA_API_BASE
    },
   "ollama_chat/qwen2.5-coder:7b": {
        "context_window": 131072,
        "supports_tools": False,
        "is_thinking_model": False,
        "api_base": DEFAULT_OLLAMA_API_BASE
    },
   "ollama_chat/qwen2.5-coder:14b": {
        "context_window": 131072,
        "supports_tools": False,
        "is_thinking_model": False,
        "api_base": DEFAULT_OLLAMA_API_BASE
    },
   "ollama_chat/qwen2.5-coder:32b": {  # https://qwenlm.github.io/blog/qwen2.5-coder-family/
        "context_window": 131072,
        "supports_tools": False,
        "is_thinking_model": False,
        "api_base": DEFAULT_OLLAMA_API_BASE
    },
    "ollama_chat/qwen3:0.6b": {
        "context_window": 40000,
        "supports_tools": False,
        "is_thinking_model": True,
        "thinking_type": "qwen",  # Use <think> ... </think>
        "api_base": DEFAULT_OLLAMA_API_BASE
    },
    "ollama_chat/qwen3:1.7b": {
        "context_window": 40000,
        "supports_tools": True,
        "is_thinking_model": True,
        "thinking_type": "qwen",  # Use <think> ... </think>
        "api_base": DEFAULT_OLLAMA_API_BASE
    },
    "ollama_chat/qwen3:4b": {
        "context_window": 40000,
        "supports_tools": True,
        "is_thinking_model": True,
        "thinking_type": "qwen",  # Use <think> ... </think>
        "api_base": DEFAULT_OLLAMA_API_BASE
    },
    "ollama_chat/qwen3:8b": {
        "context_window": 40000,
        "supports_tools": True,
        "is_thinking_model": True,
        "thinking_type": "qwen",  # Use <think> ... </think>
        "api_base": DEFAULT_OLLAMA_API_BASE
    },
    "ollama_chat/qwen3:14b": {
        "context_window": 40000,
        "supports_tools": True,
        "is_thinking_model": True,
        "thinking_type": "qwen",  # Use <think> ... </think>
        "api_base": DEFAULT_OLLAMA_API_BASE
    },
    "ollama_chat/qwen3:30b": {
        "context_window": 40000,
        "supports_tools": False,
        "is_thinking_model": True,
        "thinking_type": "qwen",  # Use <think> ... </think>
        "api_base": DEFAULT_OLLAMA_API_BASE
    },
    "ollama_chat/qwen3:32b": {
        "context_window": 40000,
        "supports_tools": True,
        "is_thinking_model": True,
        "thinking_type": "qwen",  # Use <think> ... </think>
        "api_base": DEFAULT_OLLAMA_API_BASE
    },
    "ollama_chat/gemma3:1b-it-qat": {
        "context_window": 128000,
        "supports_tools": False,
        "is_thinking_model": False,
        "api_base": DEFAULT_OLLAMA_API_BASE
    },
    "ollama_chat/gemma3:4b-it-qat": {
        "context_window": 128000,
        "supports_tools": False,
        "is_thinking_model": False,
        "api_base": DEFAULT_OLLAMA_API_BASE
    },
    "ollama_chat/gemma3:12b-it-qat": {
        "context_window": 128000,
        "supports_tools": False,
        "is_thinking_model": False,
        "api_base": DEFAULT_OLLAMA_API_BASE
    },
    "ollama_chat/gemma3:27b-it-qat": {
        "context_window": 128000,
        "supports_tools": False,
        "is_thinking_model": False,
        "api_base": DEFAULT_OLLAMA_API_BASE
    },
    "ollama_chat/qwq": {
        "context_window": 131072,
        "supports_tools": True,
        "is_thinking_model": True,
        "thinking_type": "qwen",
        "api_base": DEFAULT_OLLAMA_API_BASE # Or "http://specific-ollama-server:11434"
    },

    "lm_studio/deepseek-r1-0528-qwen3-8b@q8_0": {
        "context_window": 131072,
        "supports_tools": False,
        "is_thinking_model": True,
        "thinking_type": "qwen",
        "api_base": DEFAULT_LM_STUDIO_API_BASE
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
    if config.get("api_base") is None:
        if model_name.startswith("ollama_chat/"):
            config["api_base"] = DEFAULT_OLLAMA_API_BASE
        elif model_name.startswith("lm_studio/"):
            config["api_base"] = DEFAULT_LM_STUDIO_API_BASE
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
    DEFAULT_LITELLM_MODEL_SUMMARIZE: get_model_test_expectations(DEFAULT_LITELLM_MODEL_SUMMARIZE)["context_window"],
    DEFAULT_LITELLM_MODEL_PLANNER: get_model_test_expectations(DEFAULT_LITELLM_MODEL_PLANNER)["context_window"],
    DEFAULT_LITELLM_MODEL_TASK_MANAGER: get_model_test_expectations(DEFAULT_LITELLM_MODEL_TASK_MANAGER)["context_window"],
    DEFAULT_LITELLM_MODEL_RULE_ENHANCER: get_model_test_expectations(DEFAULT_LITELLM_MODEL_RULE_ENHANCER)["context_window"],
    DEFAULT_LITELLM_MODEL_PROMPT_ENHANCER: get_model_test_expectations(DEFAULT_LITELLM_MODEL_PROMPT_ENHANCER)["context_window"],
    DEFAULT_LITELLM_MODEL_WORKFLOW_MANAGER: get_model_test_expectations(DEFAULT_LITELLM_MODEL_WORKFLOW_MANAGER)["context_window"],
})
