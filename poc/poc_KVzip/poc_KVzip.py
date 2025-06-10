# KVzip PoC by piebru at gmail
# License: MIT
# FIXME see below: 5. Path Forward

'''
# **References**
- Paper: https://arxiv.org/abs/2505.23416
- Blog: https://janghyun1230.github.io/kvzip/
- Repo: https://github.com/snu-mllab/KVzip

** Quick Setup **

1. **venv**
deactivate  # Make sure we don't have an active venv
cd poc
uv venv
source .venv/bin/activate
which python && python -c "import sys; print(sys.executable)"  # Should be both the same.

2. **Install KVzip**
git clone https://github.com/snu-mllab/KVzip

cd KVzip
git pull
uv pip install -r requirements.txt -U --link-mode=copy
uv pip install triton -U --link-mode=copy
uv pip install flash-attn==2.7.4.post1 --no-build-isolation --link-mode=copy
make i  # If you use uv, ignore the pip error
uv pip install -e . --link-mode=copy
cd ..

3. **Initialize PoC**
uv pip install litellm prompt_toolkit -U --link-mode=copy
export KVZIP_MODEL="Qwen/Qwen2.5-7B-Instruct-1M"
export LITELLM_MODEL="ollama_chat/qwen2.5:7b" 
# Make sure you are in the AI-Engineer/poc directory
# and your venv is active (source .venv/bin/activate)
.venv/bin/python poc_KVzip.py

4. Test PoC
export TRANSFORMERS_OFFLINE="1"
.venv/bin/python poc_KVzip.py --help

5. Path Forward
5.1 Continue with Fallback: The current error handling in poc_KVzip.py is the best approach for now.
5.2 Report to LiteLLM (Optional but Recommended): Since the issue persists with the latest LiteLLM version, consider creating a minimal reproducible example (MRE) and reporting it to the LiteLLM project on GitHub. An MRE would involve:
  - A very simple CustomLLM class (simpler than KVzipLiteLLMProvider, perhaps one that just returns "hello").
  - A small script that tries to register an instance of this simple custom LLM and calls litellm.completion with it. This would help them diagnose if there's a general bug in how they handle CustomLLM instances during registration or invocation.
5.3 Monitor LiteLLM Updates: Keep an eye on future LiteLLM releases, as a fix might be included if this is a recognized bug.

For now, your PoC is structured to gracefully handle this LiteLLM limitation. The focus of poc_KVzip.py remains on demonstrating KVzip's capabilities, and when LiteLLM's custom provider support is fully stable for your use case, the KVzip provider can be seamlessly enabled.
'''

import os
import sys
from pathlib import Path
import argparse
from typing import Optional, Union, Dict, Any

import litellm
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

# --- Add src to sys.path to import config_utils ---
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent # Assumes poc_KVzip.py is in AI-Engineer/poc/
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from config_utils import get_model_context_window, get_config_value, DEFAULT_LITELLM_MODEL
except ImportError as e:
    print(f"Error importing from config_utils: {e}\n"
          "Please ensure config_utils.py is in the src directory (e.g., AI-Engineer/src/) and accessible.")
    # Allow to continue if only chat is used without stats, but stats will fail.
    get_model_context_window = None # Placeholder

# --- Import the custom KVzip LiteLLM Provider ---
try:
    from kvzip_litellm_provider import KVzipLiteLLMProvider
except ImportError as e:
    print(f"Error importing KVzipLiteLLMProvider: {e}\n"
          "Please ensure kvzip_litellm_provider.py is in the same directory or accessible in PYTHONPATH.")
    KVzipLiteLLMProvider = None # Allow script to continue if provider is missing, chat will fail

# Attempt to import ModelKVzip. Provide guidance if not found.
try:
    from KVzip.model import ModelKVzip # This assumes KVzip is in PYTHONPATH or installed
    from KVzip.attention.kvcache import EvictCache, RetainCache # For type hinting and potential use
except ImportError as e:
    print("Error: KVzip's ModelKVzip not found.")
    print(f"Original ImportError: {e}") # Print the actual ImportError
    print("You might need to clone the KVzip repository and add its path to PYTHONPATH, e.g.:")
    print("  git clone https://github.com/snu-mllab/KVzip.git")
    print("  export PYTHONPATH=$PYTHONPATH:$(pwd)/KVzip  (adjust path as needed)")
    ModelKVzip = None # Ensure it's None so the script can potentially proceed if only LiteLLM part is used
    EvictCache = None
    RetainCache = None
    # sys.exit(1) # Don't exit here if KVzipLiteLLMProvider might still work or if only fallback is used
except Exception as e:
    print(f"Error importing KVzip components: {e}")
    print("Please check your KVzip installation and dependencies (like torch, transformers).")
    # sys.exit(1) # Don't exit here

# --- Configuration ---
# Model for KVzip to wrap and perform caching operations on
KVZIP_MODEL_NAME = os.getenv("KVZIP_MODEL", "Qwen/Qwen2.5-7B-Instruct-1M")

# Alias for our custom KVzip provider in LiteLLM
LITELLM_KVZIP_PROVIDER_ALIAS = "kvzip_custom_provider" 

# Fallback LiteLLM model if KVzip provider fails or is not used for some commands
LITELLM_FALLBACK_CHAT_MODEL = os.getenv("LITELLM_FALLBACK_MODEL", DEFAULT_LITELLM_MODEL if DEFAULT_LITELLM_MODEL else "ollama_chat/qwen2.5:7b")

KVZIP_PRUNE_RATIO = float(os.getenv("KVZIP_PRUNE_RATIO", 0.3))
KVZIP_PRUNE_LEVEL = os.getenv("KVZIP_PRUNE_LEVEL", "pair")

# --- Global State ---
kvzip_litellm_provider_instance: Optional[KVzipLiteLLMProvider] = None # Instance of our custom provider
kvzip_provider_registered_successfully = False # Flag to track if registration succeeded
current_context_text = ""   # Stores the raw text of the loaded document(s)

# For temporary stats display, these are separate from the provider's internal state
_active_kvzip_cache_for_stats: Optional[Union[EvictCache, RetainCache]] = None
_active_kvzip_compression_details_for_stats: Optional[Dict[str, Any]] = None


def initialize_kvzip_provider():
    global kvzip_litellm_provider_instance, kvzip_provider_registered_successfully
    if not KVzipLiteLLMProvider:
        print("KVzipLiteLLMProvider class not available. Cannot initialize.")
        return

    print(f"Initializing KVzipLiteLLMProvider with underlying HF model: {KVZIP_MODEL_NAME}...")
    temp_instance = None # Temporary holder for the instance
    try:
        temp_instance = KVzipLiteLLMProvider(
            hf_model_name=KVZIP_MODEL_NAME,
            prune_ratio=KVZIP_PRUNE_RATIO,
            prune_level=KVZIP_PRUNE_LEVEL
            # Add kvzip_init_args if needed, e.g., {'kv_type': 'evict'}
        )
        # Instance created successfully. Now try to register it.
        litellm.register_model({
            LITELLM_KVZIP_PROVIDER_ALIAS: temp_instance
        })
        kvzip_litellm_provider_instance = temp_instance # Assign to global on full success
        kvzip_provider_registered_successfully = True
        print(f"KVzipLiteLLMProvider initialized and registered with LiteLLM as '{LITELLM_KVZIP_PROVIDER_ALIAS}'.")

    except AttributeError as e:
        if temp_instance is not None and "object has no attribute 'items'" in str(e):
            # This is the specific registration error.
            # In LiteLLM versions exhibiting this, direct instance usage also tends to fail (e.g., with a '.split()' error).
            print(f"[ERROR] KVzipLiteLLMProvider instance was created successfully for '{KVZIP_MODEL_NAME}'.")
            print(f"        However, LiteLLM failed to register it due to an internal error: {e}")
            print(f"        This LiteLLM version appears to have issues with both registering custom models")
            print(f"        and using them directly as instances (which can lead to further errors like '... object has no attribute 'split'').")
            print(f"        Therefore, the KVzip provider will be DISABLED for this session to prevent further errors.")
            print(f"        HIGHLY RECOMMENDED: Update LiteLLM ('python -m pip install -U litellm') to resolve these issues.")
            print(f"        Chat will use fallback model: {LITELLM_FALLBACK_CHAT_MODEL}")
            kvzip_litellm_provider_instance = None # Disable the provider instance
            kvzip_provider_registered_successfully = False # Mark registration as failed
        else:
            # Different AttributeError, or instance creation failed before registration attempt
            print(f"Error during KVzipLiteLLMProvider initialization or an unexpected AttributeError: {e}")
            if temp_instance is None:
                print("  (This error likely occurred during the instantiation of KVzipLiteLLMProvider.)")
            # For any other AttributeError during registration, or if instance creation failed,
            # disable the provider to be safe.
            kvzip_litellm_provider_instance = None
            kvzip_provider_registered_successfully = False
    except Exception as e:
        print(f"General error initializing or registering KVzipLiteLLMProvider: {e}")
        print("Please ensure the KVzip library and its dependencies are correctly installed,")
        print(f"and the model '{KVZIP_MODEL_NAME}' is accessible.")
        kvzip_litellm_provider_instance = None
        kvzip_provider_registered_successfully = False

def handle_add_command(filepath_str):
    global current_context_text
    try:
        filepath = Path(filepath_str)
        if not filepath.is_absolute():
            filepath = Path.cwd() / filepath_str
        
        if not filepath.exists() or not filepath.is_file():
            print(f"Error: File not found at {filepath}")
            return

        print(f"Reading document: {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            doc_content = f.read()
        
        current_context_text = doc_content # Store raw text
        print(f"Document '{filepath.name}' loaded ({len(doc_content)} chars).")
        
        if kvzip_litellm_provider_instance:
            print("This content will be passed as system context to the KVzipLiteLLMProvider on the next query.")
            print("KVzipLiteLLMProvider is active. Context will be processed by it during the next chat completion.")
            # The provider handles prefill/prune on-demand when completion is called with new context.
        else:
            print("Warning: KVzipLiteLLMProvider is not active. Chat will use fallback model or fail if no fallback.")
            print("The loaded document content will be sent as context to the fallback LiteLLM model.")

    except Exception as e:
        print(f"Error reading or processing document: {e}")


def _ensure_kvzip_initialized_for_stats(
        sample_context_text: str,
        compression_ratio_target: float,
        compression_level: str
    ) -> None:
    """
    Initializes a temporary KVzip instance with sample data for stats display.
    This is for demonstration if stats are called "cold" or if the active provider's
    internal state isn't directly queried.
    """
    global _active_kvzip_cache_for_stats, _active_kvzip_compression_details_for_stats

    if not ModelKVzip: # Guard if KVzip couldn't be imported
        print("[Stats Display] KVzip library (ModelKVzip) not available. Cannot generate temporary stats.")
        return

    if not KVZIP_MODEL_NAME:
        print(f"[Stats Display] Warning: KVZIP_MODEL environment variable not set or empty. Cannot show KVzip stats.")
        _active_kvzip_cache_for_stats = None
        return

    try:
        print(f"[Stats Display] Initializing temporary KVzip with model '{KVZIP_MODEL_NAME}' and sample context...")
        # For stats display, 'retain' type might be used if ModelKVzip supports it directly.
        # Using default kv_type for this temporary instance.
        temp_model_kv_for_stats = ModelKVzip(KVZIP_MODEL_NAME) # kv_type="retain" if supported and desired
        
        max_prefill_chars = 5000 
        context_for_prefill = sample_context_text[:max_prefill_chars]
        
        print(f"[Stats Display] Prefilling temporary KVzip with {len(context_for_prefill)} chars of sample context...")
        temp_cache = temp_model_kv_for_stats.prefill(
            context_for_prefill, 
            do_score=(compression_level != "head"), # KVzip's prefill args
            load_score=(compression_level == "head")
        )
        print("[Stats Display] Prefill complete for temporary cache.")

        _active_kvzip_cache_for_stats = temp_cache
        _active_kvzip_compression_details_for_stats = {
            "ratio_target": 1.0, # Default if no pruning
            "ratio_achieved": 1.0,
            "level": "N/A (No pruning for initial stats display)",
            "pruned": False
        }

        if temp_cache and compression_ratio_target < 1.0 and hasattr(temp_cache, 'prune'):
            print(f"[Stats Display] Pruning temporary KV cache (ratio: {compression_ratio_target}, level: '{compression_level}')...")
            _, real_ratio = temp_cache.prune(ratio=compression_ratio_target, level=compression_level)
            _active_kvzip_compression_details_for_stats.update({
                "ratio_target": compression_ratio_target,
                "ratio_achieved": real_ratio,
                "level": compression_level,
                "pruned": True
            })
            print("[Stats Display] Pruning complete for temporary cache.")
        
    except Exception as e:
        print(f"[Stats Display] Error: Failed to initialize or prefill KVzip for temporary stats: {e}")
        _active_kvzip_cache_for_stats = None
        _active_kvzip_compression_details_for_stats = None


def display_context_stats(
    kvzip_enabled_by_app: bool = True, 
    sample_context_for_kvzip: Optional[str] = None,
    kvzip_compression_ratio_target_for_stats: float = 0.3,
    kvzip_compression_level_for_stats: str = "pair"
    ):
    global _active_kvzip_cache_for_stats, _active_kvzip_compression_details_for_stats, current_context_text
    global kvzip_litellm_provider_instance # Access the provider instance

    print("\nContext Statistics:")
    print("-------------------")
    
    # Determine which LiteLLM model is effectively in use for chat
    if kvzip_litellm_provider_instance:
        llm_model_name_for_chat_display = LITELLM_KVZIP_PROVIDER_ALIAS
        underlying_hf_model_for_kvzip = kvzip_litellm_provider_instance.hf_model_name
    else:
        llm_model_name_for_chat_display = LITELLM_FALLBACK_CHAT_MODEL
        underlying_hf_model_for_kvzip = KVZIP_MODEL_NAME # The one we attempted to load

    if get_model_context_window:
        llm_context_window, used_default_cfg = get_model_context_window(llm_model_name_for_chat_display, return_match_status=True)
        print(f"Effective LiteLLM Chat Model: {llm_model_name_for_chat_display}")
        if kvzip_litellm_provider_instance:
            print(f"  (KVzip Provider wrapping: {underlying_hf_model_for_kvzip})")
        print(f"LiteLLM Max Context Window: {llm_context_window} tokens" + (" (default config)" if used_default_cfg else ""))
    else:
        print(f"Effective LiteLLM Chat Model: {llm_model_name_for_chat_display} (Context window info unavailable - config_utils not loaded)")
    print("")

    # KVzip status for stats display
    # kvzip_enabled_by_app is passed in, reflects CLI arg or auto-detection
    # ModelKVzip check is for the ability to create a temporary stats cache
    can_show_temp_kvzip_stats = ModelKVzip is not None and KVZIP_MODEL_NAME
    
    if not kvzip_enabled_by_app:
        print("KVzip Status: Disabled by user for stats display")
    elif kvzip_litellm_provider_instance:
        print(f"KVzip Provider Status: ACTIVE (Model: {kvzip_litellm_provider_instance.hf_model_name})")
        print(f"  Prune Ratio Target: {kvzip_litellm_provider_instance.prune_ratio}, Level: {kvzip_litellm_provider_instance.prune_level}")
        if kvzip_litellm_provider_instance.active_kv_cache:
            # TODO: Enhance KVzipLiteLLMProvider to expose stats from its active_kv_cache
            # For now, we can only say it has one.
            internal_cache = kvzip_litellm_provider_instance.active_kv_cache
            print(f"  Provider has an active internal KV cache.")
            if hasattr(internal_cache, 'ctx_len'):
                 print(f"    Managed Context Segment (tokens): {internal_cache.ctx_len}")
            if hasattr(internal_cache, '_seen_tokens'):
                 print(f"    Total Prefilled Tokens: {internal_cache._seen_tokens}")
            if hasattr(internal_cache, '_mem'):
                 print(f"    Current Cache Size (RAM): {internal_cache._mem():.3f} GB")
            # Pruning details are managed by the provider; could be exposed similarly.
        else:
            print("  Provider is active but has no internal KV cache currently (e.g., no context loaded via chat yet).")
        # Optionally, still show temporary stats if sample context is provided
        if sample_context_for_kvzip and can_show_temp_kvzip_stats:
            print("\n  --- Temporary KVzip Stats (from sample context) ---")
            _ensure_kvzip_initialized_for_stats(
                sample_context_for_kvzip,
                kvzip_compression_ratio_target_for_stats,
                kvzip_compression_level_for_stats
            )
            # Display logic for _active_kvzip_cache_for_stats (see below)
    elif can_show_temp_kvzip_stats: # Provider not active, but we can make a temp one for stats
        print(f"KVzip Provider Status: INACTIVE (or failed to initialize)")
        print(f"Attempting to show stats for KVZIP_MODEL: {KVZIP_MODEL_NAME} using temporary sample context.")
        if sample_context_for_kvzip:
            _ensure_kvzip_initialized_for_stats(
                sample_context_for_kvzip,
                kvzip_compression_ratio_target_for_stats,
                kvzip_compression_level_for_stats
            )
        else:
            print("  No sample context provided for temporary KVzip stats.")
    else: # KVzip not enabled by app AND cannot show temp stats
         print("KVzip Status: Disabled or Not Configured (KVzip library or model name missing for stats).")


    # Display stats from the temporary cache if it was created
    if _active_kvzip_cache_for_stats:
        if not kvzip_litellm_provider_instance or not kvzip_litellm_provider_instance.active_kv_cache : # Avoid redundancy if provider already showed its stats
            print(f"\n  --- Details from Temporary KVzip Cache ({KVZIP_MODEL_NAME}) ---")
        
        print(f"    KVzip-Managed Context Segment (tokens): {_active_kvzip_cache_for_stats.ctx_len if hasattr(_active_kvzip_cache_for_stats, 'ctx_len') else 'N/A'}")
        print(f"    Total Prefilled Tokens (KVzip cache): {_active_kvzip_cache_for_stats._seen_tokens if hasattr(_active_kvzip_cache_for_stats, '_seen_tokens') else 'N/A'}")
        print(f"    Current KV Cache Size (RAM): {_active_kvzip_cache_for_stats._mem() if hasattr(_active_kvzip_cache_for_stats, '_mem') else 'N/A'} GB")
        if _active_kvzip_compression_details_for_stats and _active_kvzip_compression_details_for_stats.get("pruned"):
            print(f"      Target Compression Ratio: {_active_kvzip_compression_details_for_stats['ratio_target']:.2f}")
            print(f"      Achieved Compression Ratio: {_active_kvzip_compression_details_for_stats['ratio_achieved']:.2f}")
            print(f"      Compression Level: {_active_kvzip_compression_details_for_stats['level']}")
        else:
            print("      Compression: N/A (Temporary cache not pruned or full context retained)")
    elif kvzip_enabled_by_app and not kvzip_litellm_provider_instance and can_show_temp_kvzip_stats and not sample_context_for_kvzip:
        print(f"  KVzip Model for stats: {KVZIP_MODEL_NAME}")
        print("  KVzip Status: Enabled for stats, but no sample context provided to generate temporary cache details.")


    print("")
    uncompressed_size_info = f"{len(current_context_text)} chars" if current_context_text else "No document loaded"
    print(f"Uncompressed Full Context Size (current document for chat): {uncompressed_size_info}")
    print("-------------------\n")


def main_chat_loop():
    global current_context_text, kvzip_litellm_provider_instance, kvzip_provider_registered_successfully
    
    history_file = Path.home() / ".kvzip_chat_history"
    session = PromptSession(history=FileHistory(str(history_file)))
    
    print("\n--- KVzip & LiteLLM CLI Chat PoC ---")
    if kvzip_litellm_provider_instance:
        # This block will only be reached if initialization and registration were successful,
        # or if a future version of this script handles other non-fatal registration errors differently.
        # Given the current changes, if .items() error occurs, kvzip_litellm_provider_instance will be None.
        print(f"Using KVzipLiteLLMProvider (Model: {kvzip_litellm_provider_instance.hf_model_name})")
        print(f"  KVzip Prune Ratio Target: {kvzip_litellm_provider_instance.prune_ratio}, Level: {kvzip_litellm_provider_instance.prune_level}")
        print(f"  Registered with LiteLLM alias: '{LITELLM_KVZIP_PROVIDER_ALIAS}'")
    else:
        print(f"Warning: KVzipLiteLLMProvider failed to initialize or is not available.")
        print(f"Chat will attempt to use fallback LiteLLM model: {LITELLM_FALLBACK_CHAT_MODEL}")
        print(f"  (Intended KVZIP_MODEL was: {KVZIP_MODEL_NAME})")

    print("------------------------------------")
    print("Commands:")
    print("  /add <filepath>  - Load a document into context.")
    print("  /context         - Show current raw context (first 200 chars).")
    print("  /quit or /exit   - Exit the chat.")
    print("------------------------------------")

    while True:
        try:
            user_input = session.prompt("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["/quit", "/exit"]:
                print("Exiting chat.")
                break
            
            if user_input.startswith("/add "):
                filepath_str = user_input[len("/add "):].strip()
                if filepath_str:
                    handle_add_command(filepath_str)
                else:
                    print("Usage: /add <filepath>")
                continue
            
            if user_input.lower() == "/context":
                if current_context_text:
                    print(f"Current Raw Context (first 200 chars):\n---\n{current_context_text[:200]}...\n---")
                else:
                    print("System: No document loaded as context.")
                continue

            # --- Prepare messages for LiteLLM ---
            messages = []
            active_llm_model_for_chat = LITELLM_FALLBACK_CHAT_MODEL 
            current_model_display_name = LITELLM_FALLBACK_CHAT_MODEL

            if kvzip_litellm_provider_instance:
                if kvzip_provider_registered_successfully:
                    active_llm_model_for_chat = LITELLM_KVZIP_PROVIDER_ALIAS
                    current_model_display_name = LITELLM_KVZIP_PROVIDER_ALIAS
                else:
                    # Registration failed, but instance is available. Use instance directly.
                    active_llm_model_for_chat = kvzip_litellm_provider_instance
                    current_model_display_name = f"KVzipDirectInstance({kvzip_litellm_provider_instance.hf_model_name})"
                
                # Context handling for KVzip provider (either via alias or direct instance)
                if current_context_text:
                    messages.append({"role": "system", "content": current_context_text})
                else:
                    messages.append({"role": "system", "content": "You are a helpful assistant."})
            else: # Fallback if provider is not active
                if current_context_text:
                     messages.append({"role": "system", "content": f"You are a helpful assistant. Use the following document as context to answer the user's question. If the answer is not in the document, say so.\n\nDocument Context:\n{current_context_text}"})
                else:
                    # No context for fallback model either
                     messages.append({"role": "system", "content": "You are a helpful assistant."})

            messages.append({"role": "user", "content": user_input})
            
            if not kvzip_litellm_provider_instance and not current_context_text and not user_input.startswith("/"):
                 print("System: No document loaded. Please use '/add <filepath>' to provide context for the chat, or ask a general question.")
                 # continue # Optionally, allow general questions without context to fallback

            print(f"System: Sending to LiteLLM ({current_model_display_name})...")
            
            try:
                # litellm.set_verbose = True # Uncomment for debugging LiteLLM calls
                response = litellm.completion(
                    model=active_llm_model_for_chat,
                    messages=messages
                )
                # litellm.set_verbose = False 
                
                if response.choices and response.choices[0].message and response.choices[0].message.content:
                    ai_response = response.choices[0].message.content
                    print(f"AI ({current_model_display_name}): {ai_response}")
                else:
                    print(f"AI ({current_model_display_name}): Received an empty or unexpected response.")
                    print(f"Full response object: {response}")

            except Exception as e:
                print(f"Error during LiteLLM inference with '{current_model_display_name}': {e}")
                if kvzip_litellm_provider_instance: # If we were attempting to use KVzip in any form
                    print("  This might be an issue with the KVzipLiteLLMProvider, its underlying model, or LiteLLM's handling of the provider.")
                else:
                    print(f"  Ensure your LiteLLM model '{active_llm_model_for_chat}' is configured correctly (e.g., Ollama running, API keys set).")

        except KeyboardInterrupt:
            print("\nExiting chat (KeyboardInterrupt).")
            break
        except EOFError:
            print("\nExiting chat (EOFError).")
            break

if __name__ == "__main__":
    initialize_kvzip_provider()  # Initialize and register our custom provider

    cli_parser = argparse.ArgumentParser(description="KVzip & LiteLLM PoC CLI Chat or Context Stats Display.")
    cli_parser.add_argument(
        "--show-context-stats", 
        action="store_true", 
        help="Display context statistics and exit (does not start chat loop)."
    )
    cli_parser.add_argument(
        "--kvzip-enabled-for-stats", 
        action=argparse.BooleanOptionalAction,
        default=None, 
        help="Force KVzip status for stats display (e.g., --no-kvzip-enabled-for-stats). Default: auto-detect."
    )
    cli_parser.add_argument(
        "--sample-context-file-for-stats", 
        type=str, 
        default=str(project_root / "KVzip" / "data" / "harry_potter4.txt"), 
        help="Path to a sample context file if showing KVzip stats from a cold start."
    )
    cli_parser.add_argument(
        "--kvzip-ratio-for-stats", 
        type=float, 
        default=KVZIP_PRUNE_RATIO, 
        help="Target compression ratio for KVzip stats demo (0.0 to 1.0)."
    )
    cli_parser.add_argument(
        "--kvzip-level-for-stats", 
        type=str, 
        default=KVZIP_PRUNE_LEVEL, 
        choices=['pair', 'head', 'pair-uniform'], 
        help="KVzip compression level for stats demo."
    )
    
    cli_args = cli_parser.parse_args()

    if cli_args.show_context_stats:
        # Determine if KVzip is considered enabled for stats display
        if cli_args.kvzip_enabled_for_stats is None:
            # Auto-detect: provider is active OR we can make a temp ModelKVzip for stats
            provider_is_active = kvzip_litellm_provider_instance is not None
            can_make_temp_kvzip = ModelKVzip is not None and bool(KVZIP_MODEL_NAME)
            kvzip_effectively_enabled_for_stats = provider_is_active or can_make_temp_kvzip
        else:
            kvzip_effectively_enabled_for_stats = cli_args.kvzip_enabled_for_stats
        
        sample_text_content = None
        if kvzip_effectively_enabled_for_stats and cli_args.sample_context_file_for_stats:
            # Load sample text if KVzip is enabled for stats and a file is provided
            # This is primarily for the _ensure_kvzip_initialized_for_stats function
            try:
                with open(cli_args.sample_context_file_for_stats, 'r', encoding='utf-8') as f:
                    sample_text_content = f.read()
                print(f"[Info] Loaded sample context for stats from: {cli_args.sample_context_file_for_stats}")
            except Exception as e:
                print(f"[Warning] Could not load sample context file '{cli_args.sample_context_file_for_stats}': {e}")
        
        display_context_stats(
            kvzip_enabled_by_app=kvzip_effectively_enabled_for_stats,
            sample_context_for_kvzip=sample_text_content,
            kvzip_compression_ratio_target_for_stats=cli_args.kvzip_ratio_for_stats,
            kvzip_compression_level_for_stats=cli_args.kvzip_level_for_stats
        )
    else:
        main_chat_loop()
