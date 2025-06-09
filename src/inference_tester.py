# src/inference_tester.py
import time
import json
import fnmatch
from typing import TYPE_CHECKING, Optional, Dict, Any, List

from litellm import completion, ModelResponse, Timeout, APIConnectionError, RateLimitError, ServiceUnavailableError, APIError, ContentPolicyViolationError, InvalidRequestError
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from src.tool_defs import tools # Import the tools definition
from src.config_utils import get_model_test_expectations, MODEL_CONFIGURATIONS, DEFAULT_LITELLM_MODEL, get_config_value, SHOW_TEST_INFERENCE_NOTES_ERRORS_COLUMN
if TYPE_CHECKING:
    from src.app_state import AppState

def try_handle_test_command(user_input: str, app_state: 'AppState') -> bool:
    """
    Placeholder: Handles the /test command.
    This function will be called from command_handlers.py.
    """
    if user_input.lower().startswith("/test"):
        # Actual test logic would be invoked here, potentially parsing args from user_input
        # For now, just acknowledge and call the main test function placeholder
        full_arg_text = user_input.replace("/test", "", 1).strip() # Remove /test prefix, keep case for model name

        model_to_test = None # Default to testing all models
        
        # Check for /test, /test inference, /test all
        if not full_arg_text or full_arg_text.lower() in ["inference", "all"]:
            model_to_test = None # Test all
        # Check for /test inference "model_name"
        elif full_arg_text.lower().startswith("inference "):
            model_name_candidate = full_arg_text[len("inference "):].strip()
            if model_name_candidate: # Ensure there's something after "inference "
                model_to_test = model_name_candidate.strip('"\'') # Remove potential quotes
        else: # /test "model_name"
            model_to_test = full_arg_text.strip('"\'') # Remove potential quotes
        
        test_inference_endpoint(app_state, specific_model_name=model_to_test)
        return True # Indicate the command was handled
    return False

def test_inference_endpoint(app_state: 'AppState', specific_model_name: Optional[str] = None, exit_on_completion: bool = True):
    """
    Tests inference endpoints for specified or all configured models.
    If specific_model_name is provided, it can include wildcards (*, ?).
    """
    app_state.console.print(Panel(
        Text("Initiating LLM inference tests. This may take a few moments...", style="bold yellow"),
        title="[bold blue]🧪 LLM Inference Test[/bold blue]",
        border_style="blue"
    ))

    models_to_test: List[str] = []
    if specific_model_name:
        # Filter MODEL_CONFIGURATIONS keys based on wildcard pattern
        for model_key in MODEL_CONFIGURATIONS.keys():
            if fnmatch.fnmatch(model_key, specific_model_name):
                models_to_test.append(model_key)
        if not models_to_test: # If pattern matches nothing, maybe it's a direct name not in MODEL_CONFIGURATIONS
             models_to_test.append(specific_model_name) # Test it directly
    else:
        # Test all models defined in MODEL_CONFIGURATIONS and the default model
        models_to_test = list(MODEL_CONFIGURATIONS.keys())
        default_model_from_config = get_config_value("model", DEFAULT_LITELLM_MODEL, app_state.RUNTIME_OVERRIDES, app_state.console)
        if default_model_from_config not in models_to_test:
            models_to_test.append(default_model_from_config)

    if not models_to_test:
        app_state.console.print("[bold red]No models found to test.[/bold red]")
        if exit_on_completion:
            import sys
            sys.exit(1)
        return

    results_table = Table(title="Inference Test Results", show_lines=True)
    results_table.add_column("Model Name", style="cyan", no_wrap=True)
    results_table.add_column("Basic Test", style="green") # Renamed from "Status"
    results_table.add_column("Tool Call Test", style="blue")
    results_table.add_column("Time (s)", style="magenta") # Changed from "Response Time (s)"
    results_table.add_column("Output (Snippet)", style="yellow", max_width=50)
    if SHOW_TEST_INFERENCE_NOTES_ERRORS_COLUMN:
        results_table.add_column("Notes/Errors", style="red", overflow="fold")

    test_prompt_messages = [{"role": "user", "content": "Hello, world! Respond with a short greeting."}]

    for model_name_to_test in sorted(list(set(models_to_test))): # Use set to avoid duplicates
        app_state.console.print(f"\n[bold]Testing model: [cyan]{model_name_to_test}[/cyan]...")
        basic_test_status = "[bold red]FAIL[/bold red]"
        response_time_str = "N/A"
        output_snippet = ""
        notes_errors = ""

        model_expectations = get_model_test_expectations(model_name_to_test)
        api_base_from_model_config = model_expectations.get("api_base")
        globally_configured_api_base = get_config_value("api_base", None, app_state.RUNTIME_OVERRIDES, app_state.console)

        api_base_for_call: Optional[str]
        if api_base_from_model_config is not None:
            api_base_for_call = api_base_from_model_config
        elif globally_configured_api_base is not None:
            api_base_for_call = globally_configured_api_base
        else:
            api_base_for_call = None

        completion_params_test: Dict[str, Any] = {
            "model": model_name_to_test,
            "messages": test_prompt_messages,
            "max_tokens": 50,
            "temperature": 0.1,
            "api_base": api_base_for_call
        }
        if model_name_to_test.startswith("lm_studio/"):
            completion_params_test["api_key"] = "dummy" # LM Studio specific

        start_time = time.time()
        try:
            response: ModelResponse = completion(**completion_params_test)
            end_time = time.time()
            response_time = end_time - start_time
            response_time_str = f"{response_time:.2f}"

            if response and response.choices and response.choices[0].message and response.choices[0].message.content:
                basic_test_status = "[bold green]SUCCESS[/bold green]"
                output_snippet = response.choices[0].message.content.strip().replace("\n", " ")[:45] + "..."
            else:
                notes_errors = "Empty or malformed response."
        except (APIConnectionError, Timeout, RateLimitError, ServiceUnavailableError, APIError, ContentPolicyViolationError, InvalidRequestError) as e:
            end_time = time.time() # Capture time even on error
            response_time = end_time - start_time
            response_time_str = f"{response_time:.2f}" # Still record time for basic test attempt
            notes_errors = f"{type(e).__name__}: {str(e)}"
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            response_time_str = f"{response_time:.2f}" # Still record time for basic test attempt
            notes_errors = f"Unexpected Error: {str(e)}"

        # --- Tool Call Test ---
        tool_call_test_status = "[dim]N/A[/dim]" # Default for models not supporting tools or if test skipped
        current_notes_for_tool_test = ""

        if model_expectations.get("supports_tools", False):
            tool_call_test_status = "[bold red]FAIL[/bold red]" # Assume fail for tool-supporting models initially
            tool_test_prompt_messages = [{"role": "user", "content": "Read the content of 'dummy_file_for_tool_test.txt'."}]
            
            completion_params_tool_test: Dict[str, Any] = {
                "model": model_name_to_test,
                "messages": tool_test_prompt_messages,
                "tools": tools, # Pass the imported tools definition
                "max_tokens": 200, # Allow enough tokens for a tool call and brief reasoning
                "temperature": 0.0, # Deterministic for tool call identification
                "api_base": api_base_for_call
            }
            if model_name_to_test.startswith("lm_studio/"):
                completion_params_tool_test["api_key"] = "dummy"

            try:
                # app_state.console.print(f"  [dim]Attempting tool call test for {model_name_to_test}...[/dim]")
                tool_response: ModelResponse = completion(**completion_params_tool_test)
                
                if tool_response.choices and \
                   tool_response.choices[0].message and \
                   tool_response.choices[0].message.tool_calls:
                    actual_tool_calls = tool_response.choices[0].message.tool_calls
                    if actual_tool_calls and actual_tool_calls[0].function.name == "read_file":
                        tool_call_test_status = "[bold green]SUCCESS[/bold green]"
                    else:
                        tool_func_name = actual_tool_calls[0].function.name if actual_tool_calls and actual_tool_calls[0].function else "None"
                        current_notes_for_tool_test = f"Tool Call: Expected 'read_file', got '{tool_func_name}' or unexpected structure."
                else:
                    current_notes_for_tool_test = "Tool Call: No tool_calls object in response message."
            except Exception as e_tool:
                current_notes_for_tool_test = f"Tool Call Error: {type(e_tool).__name__}: {str(e_tool)[:100]}"

        # Combine notes from basic test and tool test
        combined_notes = notes_errors
        if current_notes_for_tool_test:
            if combined_notes:
                combined_notes += f"\n{current_notes_for_tool_test}"
            else:
                combined_notes = current_notes_for_tool_test

        if SHOW_TEST_INFERENCE_NOTES_ERRORS_COLUMN:
            results_table.add_row(model_name_to_test, basic_test_status, tool_call_test_status, response_time_str, output_snippet, combined_notes.strip())
        else:
            results_table.add_row(model_name_to_test, basic_test_status, tool_call_test_status, response_time_str, output_snippet)
            if combined_notes.strip() and (basic_test_status != "[bold green]SUCCESS[/bold green]" or (model_expectations.get("supports_tools") and tool_call_test_status != "[bold green]SUCCESS[/bold green]")):
                app_state.console.print(f"  [red]└─ Notes/Errors: {combined_notes.strip()}[/red]")

    app_state.console.print("\n")
    app_state.console.print(results_table)
    app_state.console.print("\n[bold]Inference testing complete.[/bold]")
    if exit_on_completion:
        import sys
        sys.exit(0)