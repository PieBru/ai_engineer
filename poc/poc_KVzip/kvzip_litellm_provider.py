# /home/piero/Piero/AI/AI-Engineer/poc/kvzip_litellm_provider.py
import os
import time
import random
import string
from typing import Optional, Dict, Any, List, Union

import torch # KVzip's ModelKVzip uses torch
import litellm
from litellm.llms.custom_llm import CustomLLM # Inherit from this
from litellm import ModelResponse, Choices, Message, Usage

# Attempt to import ModelKVzip. Provide guidance if not found.
try:
    from KVzip.model import ModelKVzip
    from KVzip.attention.kvcache import EvictCache, RetainCache # For type hinting
except ImportError as e:
    print("FATAL ERROR: KVzip's ModelKVzip not found in kvzip_litellm_provider.py.")
    print(f"Original ImportError: {e}")
    print("Please ensure the KVzip repository is cloned and its path is added to PYTHONPATH, e.g.:")
    print("  git clone https://github.com/snu-mllab/KVzip.git")
    print("  export PYTHONPATH=$PYTHONPATH:/path/to/KVzip  (adjust path as needed)")
    print("Or that KVzip is installed in your environment.")
    raise e # Re-raise to prevent use of a non-functional provider

class KVzipLiteLLMProvider(CustomLLM):
    def __init__(self,
                 hf_model_name: str,
                 prune_ratio: float = 0.3,
                 prune_level: str = "pair",
                 kvzip_init_args: Optional[Dict[str, Any]] = None):
        super().__init__() # Call superclass __init__

        print(f"[KVzipProvider] Initializing with Hugging Face model: {hf_model_name}")
        print(f"[KVzipProvider] Prune ratio: {prune_ratio}, Prune level: {prune_level}")

        self.hf_model_name = hf_model_name
        self.prune_ratio = prune_ratio
        self.prune_level = prune_level
        
        _kvzip_args = kvzip_init_args or {}
        try:
            self.kvzip_model_instance = ModelKVzip(hf_model_name, **_kvzip_args)
            print(f"[KVzipProvider] ModelKVzip for '{hf_model_name}' initialized successfully.")
        except Exception as e:
            print(f"[KVzipProvider] FATAL ERROR: Could not initialize ModelKVzip for '{hf_model_name}'. Error: {e}")
            print("[KVzipProvider] Please ensure the model is downloaded or accessible by Hugging Face transformers.")
            print("[KVzipProvider] This model might require significant resources (RAM/GPU).")
            raise  # Re-raise the exception to halt if KVzip model can't load

        self.active_kv_cache: Optional[Union[EvictCache, RetainCache]] = None
        self.current_context_signature: Optional[str] = None # To detect context changes

    def _generate_context_signature(self, context_text: str) -> str:
        # Simple signature, consider a hash for very long contexts if performance is an issue
        return f"len:{len(context_text)}_start:{context_text[:100]}_end:{context_text[-100:]}"

    def _load_context_into_kvzip(self, context_text: str):
        new_signature = self._generate_context_signature(context_text)
        if self.active_kv_cache is not None and self.current_context_signature == new_signature:
            print("[KVzipProvider] Context unchanged, reusing existing KV cache.")
            return

        print(f"[KVzipProvider] New or changed context detected. Prefilling KVzip ({len(context_text)} chars)...")
        self.current_context_signature = new_signature # Update signature before potentially long op

        try:
            # `load_score=False` means scores will be computed by KVzip during prefill.
            new_kv = self.kvzip_model_instance.prefill(context_text, load_score=False)
            print("[KVzipProvider] KVzip prefill processing complete.")

            if hasattr(new_kv, 'prune') and callable(new_kv.prune) and self.prune_ratio < 1.0:
                # KVzip's args.py shows choices for level: ['pair', 'head', 'pair-uniform']
                _, real_ratio = new_kv.prune(ratio=self.prune_ratio, level=self.prune_level)
                print(f"[KVzipProvider] KVzip cache pruned. Target ratio: {self.prune_ratio}, Achieved: {real_ratio:.2f}, Level: {self.prune_level}.")
            else:
                print("[KVzipProvider] KVzip cache not pruned (ratio >= 1.0 or prune not available).")
            
            self.active_kv_cache = new_kv
        except Exception as e:
            print(f"[KVzipProvider] Error during KVzip prefill/prune: {e}")
            self.active_kv_cache = None # Invalidate cache on error
            self.current_context_signature = None
            raise # Re-raise to signal failure to the caller

    # completion() is the primary method LiteLLM calls
    def completion(self,
                   model: str, # The alias used, e.g., "kvzip_custom_qwen"
                   messages: List[Dict[str, str]],
                   stream: bool = False,
                   **kwargs) -> ModelResponse: # Add type hint for return

        if stream:
            # TODO: Implement streaming if needed. For now, raise error.
            # Streaming would involve yielding Delta objects.
            # return self.streaming(model, messages, **kwargs)
            raise NotImplementedError("[KVzipProvider] Streaming is not yet implemented.")

        print(f"[KVzipProvider] Received completion request for model alias: {model}")

        # --- Context Handling ---
        system_context_text: Optional[str] = None
        user_query: Optional[str] = None

        for msg in messages:
            if msg["role"] == "system":
                system_context_text = msg["content"] # Take the last system message as context
        
        for msg in reversed(messages): # Find last user message
            if msg["role"] == "user":
                user_query = msg["content"]
                break
        
        if user_query is None:
            raise ValueError("[KVzipProvider] No user query found in messages.")

        if system_context_text is not None:
            try:
                self._load_context_into_kvzip(system_context_text)
            except Exception as e: 
                raise ValueError(f"[KVzipProvider] Failed to load context into KVzip: {e}")

        if self.active_kv_cache is None and system_context_text is not None: # Context was provided but failed to load
             raise RuntimeError("[KVzipProvider] KVzip cache is not active despite context being provided. Check prefill/prune errors.")
        elif self.active_kv_cache is None and system_context_text is None: # No context provided
            # If ModelKVzip.generate cannot handle kv=None or if it bypasses KVzip benefits,
            # it might be better to raise an error or make behavior explicit.
            # For now, we'll proceed with a warning, assuming ModelKVzip.generate can handle kv=None.
            print("[KVzipProvider] Warning: No system context provided; no KVzip prefill/prune performed for this query. Generation will proceed, potentially without KVzip cache benefits.")
            # Allow generation to proceed if ModelKVzip can handle it without a prefilled `kv` object
            # This might not be the intended use for KVzip's efficiency.



        # --- Generation ---
        print(f"[KVzipProvider] Generating response for query: '{user_query[:100]}...' using KVzip.")
        
        query_ids = self.kvzip_model_instance.apply_template(user_query)
        if not isinstance(query_ids, torch.Tensor) or query_ids.ndim == 0:
             raise ValueError(f"[KVzipProvider] apply_template did not return valid tensor of IDs. Got: {query_ids}")

        # Determine max_new_tokens: LiteLLM sends 'max_tokens'.
        # ModelKVzip.generate likely expects 'max_new_tokens'.
        default_max_new_tokens = 256
        max_new_tokens_val = kwargs.pop("max_tokens", default_max_new_tokens) # Prioritize LiteLLM's standard
        if "max_new_tokens" in kwargs: # Allow explicit override if provided
            max_new_tokens_val = kwargs.pop("max_new_tokens")

        generate_params = {
            "kv": self.active_kv_cache, # Can be None if no context was loaded
            "update_cache": kwargs.pop("kvzip_update_cache", False), 
            "max_new_tokens": max_new_tokens_val,
            "temperature": kwargs.pop("temperature", None),
            "top_p": kwargs.pop("top_p", None),
            "top_k": kwargs.pop("top_k", None),
            # Add other KVzip specific generate params here if needed, popping from kwargs
        }
        hf_specific_kwargs = {k: v for k, v in generate_params.items() if v is not None}
        hf_specific_kwargs.update(kwargs) 

        try:
            generated_text = self.kvzip_model_instance.generate(query_ids, **hf_specific_kwargs)
        except Exception as e:
            print(f"[KVzipProvider] Error during KVzip model generation: {e}")
            raise 

        # --- Response Formatting ---
        _message = Message(content=generated_text, role="assistant")
        _choice = Choices(finish_reason="stop", index=0, message=_message) 

        # Calculate prompt_tokens based on query_ids shape
        # Assuming query_ids can be (seq_len) or (batch_size, seq_len)
        if query_ids.ndim == 1: # (seq_len)
            prompt_tokens = query_ids.shape[0]
        elif query_ids.ndim == 2: # (batch_size, seq_len), assume batch_size=1 for typical chat
            prompt_tokens = query_ids.shape[1]
        else: # Should not happen if validation above is sound
            prompt_tokens = 0 
        completion_tokens = len(self.kvzip_model_instance.tokenizer.encode(generated_text))
        _usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens)

        response_id = "chatcmpl-" + ''.join(random.choices(string.ascii_letters + string.digits, k=29))
        response_created_time = int(time.time())

        return ModelResponse(id=response_id, choices=[_choice], created=response_created_time, model=self.hf_model_name, custom_llm_provider="KVzipLiteLLMProvider", object="chat.completion", usage=_usage)

    def embedding(self, model: str, input: list, **kwargs):
        raise NotImplementedError("[KVzipProvider] Embedding is not implemented.")

    def __call__(self, model: str, messages: List[Dict[str, str]], stream: bool = False, **kwargs):
        return self.completion(model=model, messages=messages, stream=stream, **kwargs)