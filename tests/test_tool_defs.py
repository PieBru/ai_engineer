# tests/test_tool_defs.py
from src.tool_defs import RISKY_TOOLS, tools

def test_risky_tools_is_set():
    """Verify that RISKY_TOOLS is a set."""
    assert isinstance(RISKY_TOOLS, set)

def test_risky_tools_contains_expected():
    """Verify that RISKY_TOOLS contains the expected tool names."""
    expected_risky = {"create_file", "create_multiple_files", "edit_file", "connect_remote_mcp_sse"}
    assert RISKY_TOOLS == expected_risky

def test_tools_is_list():
    """Verify that 'tools' is a list."""
    assert isinstance(tools, list)

def test_tools_list_not_empty():
    """Verify that the 'tools' list is not empty."""
    assert len(tools) > 0

def test_each_tool_has_correct_structure():
    """Verify the basic structure of each tool definition in the list."""
    for tool_def in tools:
        assert isinstance(tool_def, dict)
        assert "type" in tool_def
        assert tool_def["type"] == "function"
        assert "function" in tool_def
        assert isinstance(tool_def["function"], dict)

def test_each_function_definition_has_correct_structure():
    """Verify the structure within the 'function' key for each tool."""
    for tool_def in tools:
        func_def = tool_def["function"]
        assert "name" in func_def
        assert isinstance(func_def["name"], str)
        assert func_def["name"]
        assert "description" in func_def
        assert isinstance(func_def["description"], str)
        assert func_def["description"]
        assert "parameters" in func_def
        assert isinstance(func_def["parameters"], dict)

def test_each_parameters_definition_has_correct_structure():
    """Verify the structure within the 'parameters' key for each tool."""
    for tool_def in tools:
        params_def = tool_def["function"]["parameters"]
        assert "type" in params_def
        assert params_def["type"] == "object"
        assert "properties" in params_def
        assert isinstance(params_def["properties"], dict)
        assert "required" in params_def
        assert isinstance(params_def["required"], list)
        # Check that all required properties are listed in properties
        for required_prop in params_def["required"]:
            assert required_prop in params_def["properties"]

def test_all_defined_tools_are_accounted_for():
    """Verify that the names of tools in the list match expectations."""
    expected_tool_names = {
        "read_file",
        "read_multiple_files",
        "create_file",
        "create_multiple_files",
        "edit_file",
        "connect_local_mcp_stream",
        "connect_remote_mcp_sse",
    }
    actual_tool_names = {tool_def["function"]["name"] for tool_def in tools}
    assert actual_tool_names == expected_tool_names

def test_risky_tools_are_subset_of_defined_tools():
    """Verify that all tools listed as RISKY_TOOLS are actually defined in the 'tools' list."""
    actual_tool_names = {tool_def["function"]["name"] for tool_def in tools}
    assert RISKY_TOOLS.issubset(actual_tool_names)

