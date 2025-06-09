# src/routing_logic.py
from typing import TYPE_CHECKING, List, Dict, Any, Optional
import json
import re

from litellm import completion
from rich.console import Console # For stderr debug prints
from rich.json import JSON as RichJSON

from src.config_utils import (
    get_config_value, 
    DEFAULT_LITELLM_MODEL_ROUTING, 
    DEFAULT_LITELLM_MAX_TOKENS_ROUTING, # Import the new default
    get_model_test_expectations
)
from src.prompts import ROUTING_SYSTEM_PROMPT

if TYPE_CHECKING:
    from src.app_state import AppState

VALID_ROUTING_KEYWORDS: List[str] = ["ROUTING_SELF", "TOOLS", "CODING", "KNOWLEDGE", "DEFAULT"]

def get_routing_expert_keyword(user_query: str, app_state: 'AppState') -> str:
    """
    Calls the routing LLM to determine which expert should handle the user_query.
    """
    routing_model_name = get_config_value("model_routing", DEFAULT_LITELLM_MODEL_ROUTING, app_state.RUNTIME_OVERRIDES, app_state.console)
    if not routing_model_name:
        app_state.console.print("[yellow]Warning: Routing model not configured. Defaulting to DEFAULT expert.[/yellow]")
        return "DEFAULT"

    routing_model_expectations = get_model_test_expectations(routing_model_name)
    api_base_from_model_config = routing_model_expectations.get("api_base")
    globally_configured_api_base = get_config_value("api_base", None, app_state.RUNTIME_OVERRIDES, app_state.console)

    routing_api_base: Optional[str]
    if api_base_from_model_config is not None:
        routing_api_base = api_base_from_model_config
    elif globally_configured_api_base is not None:
        routing_api_base = globally_configured_api_base
    else:
        routing_api_base = None

    brief_history_messages = []
    if app_state.conversation_history:
        history_to_consider = [msg for msg in app_state.conversation_history if msg.get("role") != "system"]
        last_few_turns = history_to_consider[-4:] # Last 2 user/assistant turns
        for msg in last_few_turns:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                brief_history_messages.append(f"{role}: {content[:150]}{'...' if len(content) > 150 else ''}")
    history_for_router_prompt = "\n".join(brief_history_messages) if brief_history_messages else "No recent conversation history."

    prompt_for_router = ROUTING_SYSTEM_PROMPT.format(
        user_query=user_query,
        history_snippet=history_for_router_prompt
    )
    
    messages_for_routing = [{"role": "system", "content": prompt_for_router}]
    
    max_tokens_for_routing = get_config_value("max_tokens_routing", DEFAULT_LITELLM_MAX_TOKENS_ROUTING, app_state.RUNTIME_OVERRIDES, app_state.console)
    completion_params_routing: Dict[str, Any] = {
        "model": routing_model_name,
        "messages": messages_for_routing,
        "temperature": 0.0,
        "max_tokens": int(max_tokens_for_routing), # Use configurable value
        "api_base": routing_api_base,
        "stream": False
    }
    if routing_model_name.startswith("lm_studio/"):
        completion_params_routing["api_key"] = "dummy"

    if app_state.DEBUG_LLM_INTERACTIONS:
        Console(stderr=True).print(f"[dim bold red]ROUTER DEBUG: Request Params:[/dim bold red]")
        debug_params_log = completion_params_routing.copy()
        if "messages" in debug_params_log: # Log messages separately
            Console(stderr=True).print(f"[dim bold red]ROUTER DEBUG: Request Messages (detail):[/dim bold red]")
            for i, msg_item in enumerate(debug_params_log["messages"]):
                Console(stderr=True).print(f"[dim red]Message {i}: {json.dumps(msg_item, indent=2, default=str)}[/dim red]")
            del debug_params_log["messages"]
        Console(stderr=True).print(RichJSON(json.dumps(debug_params_log, indent=2, default=str)))

    try:
        response = completion(**completion_params_routing)
        raw_response_content = response.choices[0].message.content
        final_keyword_candidate = raw_response_content

        if app_state.DEBUG_LLM_INTERACTIONS:
            Console(stderr=True).print(f"[dim bold red]ROUTER DEBUG: Raw Response:[/dim bold red]")
            try:
                debug_response_data = {"choices": [{"message": {"content": response.choices[0].message.content}, "finish_reason": response.choices[0].finish_reason}], "model": response.model, "usage": dict(response.usage)}
                Console(stderr=True).print(RichJSON(json.dumps(debug_response_data, indent=2, default=str)))
            except Exception as e_debug:
                Console(stderr=True).print(f"[dim red]ROUTER DEBUG: Error serializing response: {e_debug}[/dim red]")
                Console(stderr=True).print(f"[dim red]{response}[/dim red]")

        is_thinking = routing_model_expectations.get("is_thinking_model", False)
        think_type = routing_model_expectations.get("thinking_type")

        if is_thinking and think_type == "qwen": # Adapted from previous ai_engineer.py logic
            temp_lower_content = raw_response_content.lower()
            start_tag = "<think>"
            end_tag = "</think>"
            actual_start_tag_pos_in_lower = temp_lower_content.find(start_tag)

            if actual_start_tag_pos_in_lower != -1 and \
               (temp_lower_content[:actual_start_tag_pos_in_lower].isspace() or actual_start_tag_pos_in_lower == 0):
                end_tag_pos_in_lower = temp_lower_content.find(end_tag, actual_start_tag_pos_in_lower + len(start_tag))

                if end_tag_pos_in_lower != -1:
                    content_after_first_think_block = raw_response_content[end_tag_pos_in_lower + len(end_tag):]
                    found_keyword_in_suffix = None
                    stripped_upper_suffix = content_after_first_think_block.strip().upper()
                    sorted_valid_keywords = sorted(VALID_ROUTING_KEYWORDS, key=len, reverse=True)

                    for valid_kw in sorted_valid_keywords:
                        if stripped_upper_suffix.startswith(valid_kw):
                            if len(stripped_upper_suffix) == len(valid_kw) or \
                               not stripped_upper_suffix[len(valid_kw)].isalnum():
                                found_keyword_in_suffix = valid_kw
                                break
                    
                    if found_keyword_in_suffix:
                        final_keyword_candidate = found_keyword_in_suffix
                        app_state.console.print(f"[dim]   (Thought block stripped, using keyword: '{final_keyword_candidate}')[/dim]")
                    else:
                        app_state.console.print(f"[yellow]Warning: Routing LLM (thinking model) provided a closed thought block, but no valid keyword found right after. Raw suffix: '{content_after_first_think_block.strip()[:70]}...'. Defaulting.[/yellow]")
                        final_keyword_candidate = "DEFAULT"
                else: # Unclosed <think>
                    app_state.console.print(f"[dim]   (Qwen model: <think> found, no </think>.)[/dim]")
                    content_after_open_think_tag = raw_response_content[actual_start_tag_pos_in_lower + len(start_tag):]
                    found_keyword_in_unclosed_think = None
                    stripped_upper_unclosed_suffix = content_after_open_think_tag.strip().upper()
                    best_match_pos = -1
                    sorted_valid_keywords_for_unclosed = sorted(VALID_ROUTING_KEYWORDS, key=len, reverse=True)

                    for valid_kw in sorted_valid_keywords_for_unclosed:
                        for match in re.finditer(r'\b' + re.escape(valid_kw) + r'\b', stripped_upper_unclosed_suffix):
                            if match.start() > best_match_pos:
                                best_match_pos = match.start()
                                found_keyword_in_unclosed_think = valid_kw
                    
                    if found_keyword_in_unclosed_think:
                        final_keyword_candidate = found_keyword_in_unclosed_think
                        app_state.console.print(f"[dim]   (Keyword '{final_keyword_candidate}' found within unclosed <think> block.)[/dim]")
                    else:
                        greeting_pattern = r"^(hello|hi|hey|good\s+(morning|afternoon|evening)|how\s+are\s+you|how's\s+it\s+going|what's\s+up|sup)[\s!\.,\?]*$"
                        is_simple_greeting = bool(re.match(greeting_pattern, user_query.strip(), re.IGNORECASE))
                        if is_simple_greeting:
                            app_state.console.print(f"[dim]   (Info: Routing model produced an unclosed thought for a greeting. Defaulting to DEFAULT. Raw: '{raw_response_content[:70].strip()}...')")
                        else:
                            app_state.console.print(f"[yellow]Warning: Routing LLM (thinking model) started with '{start_tag}' but no closing '{end_tag}', and no keyword found within. Raw: '{raw_response_content[:100].strip()}...'. Defaulting.[/yellow]")
                        final_keyword_candidate = "DEFAULT"

        keyword = final_keyword_candidate.strip().upper()
        app_state.console.print(f"[dim]  ➔ Routed to: {keyword}[/dim]")

        if keyword not in VALID_ROUTING_KEYWORDS:
            app_state.console.print(f"[yellow]Warning: Routing LLM returned unknown keyword '{keyword}'. Defaulting to DEFAULT.[/yellow]")
            return "DEFAULT"
        return keyword
    except Exception as e:
        app_state.console.print(f"[red]Error during routing: {e}. Defaulting to DEFAULT expert.[/red]")
        if app_state.DEBUG_LLM_INTERACTIONS:
            Console(stderr=True).print(f"[dim red]ROUTER DEBUG: Exception: {e}[/dim red]")
        return "DEFAULT"

def map_expert_to_model(expert_keyword: str, app_state: 'AppState') -> str:
    """
    Maps the expert keyword to the corresponding model name.
    """
    from ai_engineer import ( # Local import to access model constants from the main module
        DEFAULT_LITELLM_MODEL_TOOLS, DEFAULT_LITELLM_MODEL_CODING,
        DEFAULT_LITELLM_MODEL_KNOWLEDGE, DEFAULT_LITELLM_MODEL
    )
    if expert_keyword == "TOOLS": return get_config_value("model_tools", DEFAULT_LITELLM_MODEL_TOOLS, app_state.RUNTIME_OVERRIDES, app_state.console)
    if expert_keyword == "CODING": return get_config_value("model_coding", DEFAULT_LITELLM_MODEL_CODING, app_state.RUNTIME_OVERRIDES, app_state.console)
    if expert_keyword == "KNOWLEDGE": return get_config_value("model_knowledge", DEFAULT_LITELLM_MODEL_KNOWLEDGE, app_state.RUNTIME_OVERRIDES, app_state.console)
    return get_config_value("model", DEFAULT_LITELLM_MODEL, app_state.RUNTIME_OVERRIDES, app_state.console)