# src/llm_interaction.py
import copy
import time
import json
import httpx # Used by network_utils, but good to have if direct calls were ever needed

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from prompt_toolkit import PromptSession

from litellm import completion

from src.tool_defs import tools, RISKY_TOOLS
from src.config_utils import (
    get_config_value, DEFAULT_LITELLM_MODEL, DEFAULT_LM_STUDIO_API_BASE,
    get_model_test_expectations, # Import for getting model-specific API base
    DEFAULT_REASONING_STYLE, DEFAULT_LITELLM_MAX_TOKENS, DEFAULT_REASONING_EFFORT,
    MAX_FILE_SIZE_BYTES
)
from src.file_utils import (
    normalize_path,
    read_local_file as util_read_local_file,
    create_file as util_create_file,
    apply_diff_edit as util_apply_diff_edit
)
from src.network_utils import handle_local_mcp_stream, handle_remote_mcp_sse
from src.file_context_utils import ensure_file_in_context


def execute_function_call_dict(
    tool_call_dict: dict,
    console: Console,
    conversation_history: list
) -> str:
    """
    Executes a function call specified by the LLM and returns its string result.
    """
    function_name = "unknown_function"
    try:
        function_name = tool_call_dict["function"]["name"]
        arguments = json.loads(tool_call_dict["function"]["arguments"])

        if function_name == "read_file":
            file_path = arguments["file_path"]
            normalized_path = normalize_path(file_path)
            content = util_read_local_file(normalized_path)
            return f"Content of file '{normalized_path}':\n\n{content}"
        elif function_name == "read_multiple_files":
            file_paths = arguments["file_paths"]
            results = []
            for file_path in file_paths:
                try:
                    normalized_path = normalize_path(file_path)
                    content = util_read_local_file(normalized_path)
                    results.append(f"Content of file '{normalized_path}':\n\n{content}")
                except (OSError, ValueError) as e:
                    results.append(f"Error reading '{file_path}': {e}")
            return "\n\n" + "="*50 + "\n\n".join(results)
        elif function_name == "create_file":
            file_path = arguments["file_path"]
            content = arguments["content"]
            util_create_file(file_path, content, console, MAX_FILE_SIZE_BYTES)
            return f"Successfully created file '{file_path}'"
        elif function_name == "create_multiple_files":
            files_to_create_data = arguments.get("files", [])
            successful_paths = []
            success_messages_for_result = []
            first_error_detail_for_return = None
            for file_info_data in files_to_create_data:
                path = file_info_data.get("path", "unknown_path")
                content = file_info_data.get("content", "")
                try:
                    util_create_file(path, content, console, MAX_FILE_SIZE_BYTES)
                    successful_paths.append(path)
                    success_messages_for_result.append(f"File {path} created.")
                except Exception as e_create:
                    console.print(f"[red]Error creating file {path}: {str(e_create)}[/red]")
                    if not first_error_detail_for_return:
                        first_error_detail_for_return = f"Error during create_multiple_files: {str(e_create)}"
            if first_error_detail_for_return:
                return "\n".join(success_messages_for_result) + "\n" + first_error_detail_for_return if success_messages_for_result else first_error_detail_for_return
            return f"Successfully created {len(successful_paths)} files: {', '.join(successful_paths)}"
        elif function_name == "edit_file":
            file_path = arguments["file_path"]
            original_snippet = arguments["original_snippet"]
            new_snippet = arguments["new_snippet"]
            if not ensure_file_in_context(file_path, conversation_history, console):
                return f"Error: Could not read file '{file_path}' for editing"
            util_apply_diff_edit(file_path, original_snippet, new_snippet, console, MAX_FILE_SIZE_BYTES)
            return f"Successfully edited file '{file_path}'"
        elif function_name == "connect_local_mcp_stream":
            endpoint_url = arguments["endpoint_url"]
            timeout_seconds = arguments.get("timeout_seconds", 30)
            max_data_chars = arguments.get("max_data_chars", 10000)
            try:
                parsed_url = httpx.URL(endpoint_url)
                if not (parsed_url.host.lower() in ("localhost", "127.0.0.1") and parsed_url.scheme.lower() in ("http", "https")):
                     return f"Error: For connect_local_mcp_stream, endpoint_url must be for localhost (http or https). Provided: {endpoint_url}"
            except Exception as e_val: # Catch other potential parsing errors
                return f"Error validating local MCP stream URL '{endpoint_url}' before execution: {str(e_val)}"
            return handle_local_mcp_stream(endpoint_url, timeout_seconds, max_data_chars)
        elif function_name == "connect_remote_mcp_sse":
            endpoint_url = arguments["endpoint_url"]
            max_events = arguments.get("max_events", 10)
            listen_timeout_seconds = arguments.get("listen_timeout_seconds", 60)
            try:
                parsed_url = httpx.URL(endpoint_url)
                if parsed_url.scheme.lower() not in ("http", "https"):
                    return f"Error: For connect_remote_mcp_sse, endpoint_url must be a valid HTTP/HTTPS URL. Provided: {endpoint_url}"
            except Exception as e_val: # Catch other potential parsing errors
                return f"Error validating remote MCP SSE URL '{endpoint_url}' before execution: {str(e_val)}"
            return handle_remote_mcp_sse(endpoint_url, max_events, listen_timeout_seconds)
        else:
            return f"Unknown function: {function_name}"
    except Exception as e:
        error_message = f"Error executing {function_name}: {str(e)}"
        console.print(f"[red]{error_message}[/red]")
        return error_message

def trim_conversation_history(conversation_history: list):
    """
    Trims the conversation history if it exceeds a certain length,
    prioritizing system messages and recent non-system messages.
    """
    if len(conversation_history) <= 20: # Max length before trimming
        return

    system_msgs = [msg for msg in conversation_history if msg["role"] == "system"]
    other_msgs = [msg for msg in conversation_history if msg["role"] != "system"]

    # Keep the last 15 non-system messages
    if len(other_msgs) > 15:
        other_msgs = other_msgs[-15:]

    conversation_history.clear()
    conversation_history.extend(system_msgs + other_msgs)


def stream_llm_response(
    user_message: str,
    conversation_history: list,
    console: Console,
    prompt_session: PromptSession
):
    """
    Sends the user message and conversation history to the LLM and streams the response.
    Handles reasoning, final content, and tool calls.
    Returns a dictionary indicating success or error.
    """
    trim_conversation_history(conversation_history)
    try:
        messages_for_api_call = copy.deepcopy(conversation_history)

        default_reply_effort_val = "medium"
        default_temperature_val = 0.7

        model_name = get_config_value("model", DEFAULT_LITELLM_MODEL)
        
        # Determine the API base for this specific model_name
        # 1. Check model-specific configuration (MODEL_CONFIGURATIONS or ollama/lm_studio defaults via get_model_test_expectations)
        model_expectations = get_model_test_expectations(model_name)
        api_base_from_model_config = model_expectations.get("api_base")

        # 2. Check for a globally configured API base (e.g., LITELLM_API_BASE env var or /set api_base)
        #    If LITELLM_API_BASE env var is not set and no runtime override, this will be None.
        globally_configured_api_base = get_config_value("api_base", None) # Pass None as default to see if it's truly set

        api_base_url: Optional[str]
        if api_base_from_model_config is not None:
            # Priority 1: Model-specific API base from its configuration.
            # This handles explicit api_base in MODEL_CONFIGURATIONS (can be a URL or None),
            # and defaults for "ollama_chat/" or "lm_studio/" prefixes.
            api_base_url = api_base_from_model_config
        elif globally_configured_api_base is not None:
            # Priority 2: A globally configured API base is set (e.g., for a proxy).
            # Use this if the model didn't have its own specific api_base.
            api_base_url = globally_configured_api_base
        else:
            # Priority 3: No model-specific and no global API base configured.
            # This implies a direct provider call using API keys, so api_base should be None.
            api_base_url = None

        reasoning_style = str(get_config_value("reasoning_style", DEFAULT_REASONING_STYLE)).lower()
        max_tokens_raw = get_config_value("max_tokens", DEFAULT_LITELLM_MAX_TOKENS)
        try:
            max_tokens = int(max_tokens_raw)
            if max_tokens <= 0: max_tokens = DEFAULT_LITELLM_MAX_TOKENS # Fallback
        except (ValueError, TypeError):
            max_tokens = DEFAULT_LITELLM_MAX_TOKENS

        temperature_raw = get_config_value("temperature", default_temperature_val)
        try:
            temperature = float(temperature_raw)
        except (ValueError, TypeError):
            console.print(f"[yellow]Warning: Invalid temperature value '{temperature_raw}'. Using default {default_temperature_val}.[/yellow]")
            temperature = default_temperature_val

        reasoning_effort_setting = str(get_config_value("reasoning_effort", DEFAULT_REASONING_EFFORT)).lower()
        reply_effort_setting = str(get_config_value("reply_effort", default_reply_effort_val)).lower()

        effort_instructions = (
            f"\n\n[System Instructions For This Turn Only]:\n"
            f"- Current `reasoning_effort`: {reasoning_effort_setting}\n"
            f"- Current `reply_effort`: {reply_effort_setting}\n"
            f"Please adhere to these specific effort levels for your reasoning and reply in this turn."
        )
        augmented_user_message_content = user_message + effort_instructions

        messages_for_api_call.append({
            "role": "user",
            "content": augmented_user_message_content
        })

        console.print("\n[bold bright_blue]🔍 Seeking...[/bold bright_blue]")

        completion_params = {
            "model": model_name,
            "messages": messages_for_api_call,
            "tools": tools,
            "max_tokens": max_tokens,
            "api_base": api_base_url,
            "temperature": temperature,
            "stream": True
        }

        # Add dummy API key for LM Studio models.
        # api_base_url is the resolved API base (model-specific or global)
        if model_name.startswith("lm_studio/"):
            completion_params["api_key"] = "dummy"

        reasoning_content_accumulated = ""
        final_content = ""
        tool_calls = []
        reasoning_started_printed = False
        stream = completion(**completion_params)

        for chunk in stream:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                reasoning_chunk_content = delta.reasoning_content
                reasoning_content_accumulated += reasoning_chunk_content
                if reasoning_style == "full":
                    if not reasoning_started_printed:
                        console.print("\n[bold blue]💭 Reasoning:[/bold blue]")
                        reasoning_started_printed = True
                    console.print(reasoning_chunk_content, end="")
                elif reasoning_style == "compact":
                    if not reasoning_started_printed:
                        console.print("\n[bold blue]💭 Reasoning...[/bold blue]", end="")
                        reasoning_started_printed = True
                    console.print(".", end="")
            elif delta.content:
                if reasoning_started_printed and reasoning_style != "full":
                    console.print() # Newline after compact reasoning dots
                    reasoning_started_printed = False
                if not final_content: # First part of the actual reply
                    console.print("\n\n[bold bright_blue]🤖 Assistant>[/bold bright_blue] ", end="")
                final_content += delta.content
                console.print(delta.content, end="")
            elif delta.tool_calls:
                if reasoning_started_printed and reasoning_style != "full":
                    console.print()
                    reasoning_started_printed = False
                for tool_call_delta in delta.tool_calls:
                    if tool_call_delta.index is not None: # Ensure index exists
                        while len(tool_calls) <= tool_call_delta.index:
                            tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        if tool_call_delta.id:
                            tool_calls[tool_call_delta.index]["id"] = tool_call_delta.id
                        if tool_call_delta.function:
                            if tool_call_delta.function.name:
                                tool_calls[tool_call_delta.index]["function"]["name"] += tool_call_delta.function.name
                            if tool_call_delta.function.arguments:
                                tool_calls[tool_call_delta.index]["function"]["arguments"] += tool_call_delta.function.arguments

        if reasoning_started_printed and reasoning_style == "compact" and not final_content and not tool_calls:
            console.print() # Ensure newline if only compact reasoning dots were printed

        console.print() # Ensure the next prompt is on a new line

        conversation_history.append({"role": "user", "content": user_message}) # Add original user message
        assistant_message = {"role": "assistant", "content": final_content if final_content else None}
        if reasoning_content_accumulated:
            assistant_message["reasoning_content_full"] = reasoning_content_accumulated

        if tool_calls:
            # Format tool calls correctly
            formatted_tool_calls = []
            for i, tc in enumerate(tool_calls):
                if tc["function"]["name"]: # Ensure the tool call is valid
                    tool_id = tc["id"] if tc["id"] else f"call_{i}_{int(time.time() * 1000)}"
                    formatted_tool_calls.append({
                        "id": tool_id,
                        "type": "function",
                        "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
                    })

            if formatted_tool_calls:
                assistant_message["tool_calls"] = formatted_tool_calls
                conversation_history.append(assistant_message)

                console.print(f"\n[bold bright_cyan]⚡ Executing {len(formatted_tool_calls)} function call(s)...[/bold bright_cyan]")
                executed_tool_call_ids_and_results = []

                for tool_call in formatted_tool_calls:
                    tool_name = tool_call['function']['name']
                    console.print(f"[bright_blue]→ {tool_name}[/bright_blue]")
                    user_confirmed_or_not_risky = True
                    if tool_name in RISKY_TOOLS:
                        console.print(f"[bold yellow]⚠️ This is a risky operation: {tool_name}[/bold yellow]")
                        try:
                            args = json.loads(tool_call['function']['arguments'])
                        except json.JSONDecodeError:
                            console.print("[red]Error: Could not parse tool arguments.[/red]")
                            executed_tool_call_ids_and_results.append({
                                "role": "tool", "tool_call_id": tool_call["id"],
                                "content": f"Error: Could not parse arguments for {tool_name}"
                            })
                            continue

                        # Preview for risky tools
                        if tool_name == "create_file":
                            console.print(f"   Action: Create/overwrite file '{args.get('file_path')}'")
                            content_summary = args.get('content', '')[:100] + "..." if len(args.get('content', '')) > 100 else args.get('content', '')
                            console.print(Panel(content_summary, title="Content Preview", border_style="yellow", expand=False))
                        elif tool_name == "create_multiple_files":
                            files_to_create_preview = args.get('files', [])
                            if files_to_create_preview:
                                file_paths = [f.get('path', 'unknown') for f in files_to_create_preview]
                                console.print(f"   Action: Create/overwrite {len(file_paths)} files: {', '.join(file_paths[:5])}{'...' if len(file_paths) > 5 else ''}")
                            else:
                                console.print("   Action: Create multiple files (none specified).")
                        elif tool_name == "edit_file":
                            console.print(f"   Action: Edit file '{args.get('file_path')}'")
                            original_snippet_summary = args.get('original_snippet', '')[:70] + "..." if len(args.get('original_snippet', '')) > 70 else args.get('original_snippet', '')
                            new_snippet_summary = args.get('new_snippet', '')[:70] + "..." if len(args.get('new_snippet', '')) > 70 else args.get('new_snippet', '')
                            diff_table = Table(show_header=False, box=None, padding=0)
                            diff_table.add_row("[red]- Original:[/red]", original_snippet_summary)
                            diff_table.add_row("[green]+ New:     [/green]", new_snippet_summary)
                            console.print(diff_table)
                        elif tool_name == "connect_remote_mcp_sse":
                             console.print(f"   Action: Connect to remote SSE endpoint '{args.get('endpoint_url')}'")

                        confirmation = prompt_session.prompt("Proceed with this operation? [Y/n]: ", default="y").strip().lower()
                        if confirmation not in ["y", "yes", ""]:
                            user_confirmed_or_not_risky = False
                            console.print("[yellow]ℹ️ Operation cancelled by user.[/yellow]")
                            result = "User cancelled execution of this tool call."

                    if user_confirmed_or_not_risky:
                        try:
                            result = execute_function_call_dict(tool_call, console, conversation_history)
                        except Exception as e_exec:
                            console.print(f"[red]Unexpected error during tool execution: {str(e_exec)}[/red]")
                            result = f"Error: Unexpected error during tool execution: {str(e_exec)}"

                    executed_tool_call_ids_and_results.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result
                    })

                for tool_res in executed_tool_call_ids_and_results:
                    conversation_history.append(tool_res) # type: ignore

                # Get follow-up response from LLM
                console.print("\n[bold bright_blue]🔄 Processing results...[/bold bright_blue]")
                # Recursive call effectively, but LiteLLM handles this by sending history
                return stream_llm_response("Tool execution finished. Please respond to the user based on the results.", conversation_history, console, prompt_session)

        else: # No tool calls, just a regular response
            conversation_history.append(assistant_message)

        return {"success": True}

    except Exception as e:
        error_msg = f"LLM API error: {str(e)}"
        console.print(f"\n[bold red]❌ {error_msg}[/bold red]")
        # Add error to history for context, but as a system message to avoid confusing the LLM
        conversation_history.append({"role": "system", "content": f"Error during LLM call: {error_msg}"})
        return {"error": error_msg}