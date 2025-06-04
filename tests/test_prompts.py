# tests/test_prompts.py
from src.prompts import system_PROMPT

def test_system_prompt_exists_and_is_string():
    """Verify that system_PROMPT exists and is a string."""
    assert system_PROMPT is not None
    assert isinstance(system_PROMPT, str)

def test_system_prompt_is_not_empty():
    """Verify that system_PROMPT is not an empty string."""
    assert len(system_PROMPT.strip()) > 0

def test_system_prompt_mentions_ai_engineer():
    """Verify that the prompt identifies the AI as 'AI Engineer'."""
    assert "AI Engineer" in system_PROMPT

def test_system_prompt_mentions_core_capabilities():
    """Verify that the prompt outlines core capabilities."""
    assert "Core capabilities:" in system_PROMPT
    assert "Conversational Interaction:" in system_PROMPT
    assert "Code Analysis & Discussion:" in system_PROMPT
    assert "File Operations (via function calls):" in system_PROMPT
    assert "Network Operations (via function calls):" in system_PROMPT

def test_system_prompt_greeting_examples():
    """Verify that the prompt includes specific examples for handling greetings."""
    assert "Examples of Handling Simple Greetings (YOU MUST FOLLOW THESE PRECISELY):" in system_PROMPT
    assert "User: hello" in system_PROMPT
    assert "Assistant: Hello! How can I help you today?" in system_PROMPT
    assert "User: hi" in system_PROMPT
    assert "Assistant: Hi there! What can I assist you with?" in system_PROMPT
    assert "PRIORITIZE CONVERSATIONAL RESPONSES FOR SIMPLE INPUTS, ESPECIALLY GREETINGS." in system_PROMPT

def test_system_prompt_file_operations_tools():
    """Verify that the prompt lists expected file operation tools."""
    assert "read_file:" in system_PROMPT
    assert "read_multiple_files:" in system_PROMPT
    assert "create_file:" in system_PROMPT
    assert "create_multiple_files:" in system_PROMPT
    assert "edit_file:" in system_PROMPT

def test_system_prompt_network_operations_tools():
    """Verify that the prompt lists expected network operation tools."""
    assert "connect_local_mcp_stream:" in system_PROMPT
    assert "connect_remote_mcp_sse:" in system_PROMPT

def test_system_prompt_code_citation_format():
    """Verify that the prompt specifies the code citation format."""
    assert "Code Citation Format:" in system_PROMPT
    assert "```startLine:endLine:filepath" in system_PROMPT

def test_system_prompt_effort_control_settings():
    """Verify that the prompt includes instructions for effort control."""
    # Corrected assertion to match the markdown bold formatting in the prompt
    assert "**Effort Control Settings (Instructions for AI)**:" in system_PROMPT
    assert "`reasoning_effort`" in system_PROMPT
    assert "`reply_effort`" in system_PROMPT
    assert "reasoning_effort` defines the depth of your internal thinking process:" in system_PROMPT
    assert "reply_effort` defines the verbosity and detail of your final reply to the user:" in system_PROMPT

def test_system_prompt_general_guidelines_exist():
    """Verify that the prompt has a 'General Guidelines' section."""
    assert "General Guidelines:" in system_PROMPT

def test_system_prompt_emphasizes_no_tools_for_greetings():
    """Verify specific instruction about not using tools for greetings."""
    assert "ABSOLUTELY DO NOT use any tools" in system_PROMPT # for simple greetings
    assert "You MUST NOT attempt file operations or other tool calls for simple greetings." in system_PROMPT

def test_system_prompt_dedent_applied():
    """Verify that the prompt does not start with excessive leading whitespace on lines."""
    lines = system_PROMPT.splitlines()
    # Check the first few lines after the initial dedent line
    if len(lines) > 1:
        # The first line might be empty or just the start of the string
        # Subsequent lines should not have excessive leading space if dedent worked
        for line in lines[1:5]: # Check a few subsequent lines
            if line.strip(): # If the line is not empty
                 assert not line.startswith("    ") # Common indication of dedent not working as expected

