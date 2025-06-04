"""
Unit tests for the AI Engineer (ai_eng.py) script.
Covers helper functions, command handling, conversation history, tool execution, and LLM interactions.
"""
import pytest
import os
import json
import sys # Import the sys module
from pathlib import Path
from unittest.mock import patch, MagicMock, call, ANY

# Add the project root to sys.path to allow importing deepseek_eng
# Assuming file is now ai_eng.py
sys_path_updated = False
try:
    import ai_eng as de
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import ai_eng as de
    sys_path_updated = True


# Pytest fixtures
@pytest.fixture(autouse=True)
def reset_globals_and_mocks(monkeypatch):
    """Reset conversation history and mock critical global objects before each test."""
    de.conversation_history = [{"role": "system", "content": de.system_PROMPT}]

    # Mock console to prevent actual printing during tests, unless capturing output
    mock_console_instance = MagicMock(spec=de.Console)
    monkeypatch.setattr(de, 'console', mock_console_instance)
    
    # Mock prompt_session
    mock_prompt_session_instance = MagicMock(spec=de.PromptSession)
    monkeypatch.setattr(de, 'prompt_session', mock_prompt_session_instance)

    yield # Test runs here

    # Teardown: Clean up sys.path if it was modified for this test session
    if sys_path_updated and sys.path[0].endswith('ai-engineer'): # Assuming project dir might also be renamed
        sys.path.pop(0)


@pytest.fixture
def mock_env_vars(monkeypatch): # Renamed and expanded
    monkeypatch.setenv("AI_API_KEY", "test_api_key") # Changed DEEPSEEK_API_KEY to AI_API_KEY
    monkeypatch.setenv("LITELLM_MODEL", "test_model_from_env")
    monkeypatch.setenv("LITELLM_API_BASE", "http://test.api.base/v1")
    
# Helper to create mock stream chunks for litellm
def create_mock_litellm_chunk(content=None, reasoning_content=None, tool_calls_delta=None, finish_reason=None):
    mock_chunk = MagicMock()
    mock_choice = MagicMock()
    mock_delta = MagicMock()

    mock_delta.content = content
    # Use setattr for attributes that might not always be present, to mimic hasattr checks
    if reasoning_content is not None:
        setattr(mock_delta, 'reasoning_content', reasoning_content)
    else:
        # Ensure it's not present if None, for hasattr to be False
        if hasattr(mock_delta, 'reasoning_content'):
            delattr(mock_delta, 'reasoning_content')
    
    if tool_calls_delta:
        mock_tool_call_deltas = []
        for tc_delta_data in tool_calls_delta:
            mock_tc_delta = MagicMock()
            mock_tc_delta.index = tc_delta_data.get("index")
            mock_tc_delta.id = tc_delta_data.get("id")
            
            mock_function_delta = MagicMock()
            mock_function_delta.name = tc_delta_data.get("function", {}).get("name")
            mock_function_delta.arguments = tc_delta_data.get("function", {}).get("arguments")
            
            mock_tc_delta.function = mock_function_delta
            mock_tool_call_deltas.append(mock_tc_delta)
        mock_delta.tool_calls = mock_tool_call_deltas
    else:
        mock_delta.tool_calls = None

    mock_choice.delta = mock_delta
    mock_choice.finish_reason = finish_reason
    mock_chunk.choices = [mock_choice]
    return mock_chunk


# --- Tests for Helper Functions ---

class TestHelperFunctions:

    def test_normalize_path(self, tmp_path):
        abs_file = tmp_path / "test.txt"
        abs_file.touch()
        assert de.normalize_path(str(abs_file)) == str(abs_file.resolve())

        relative_file = "test_rel.txt"
        full_relative_path = Path.cwd() / relative_file
        try:
            assert de.normalize_path(relative_file) == str(full_relative_path.resolve())
        finally:
            if full_relative_path.exists(): # Should not exist, but cleanup if test fails mid-way
                 full_relative_path.unlink()


    def test_normalize_path_security(self):
        with pytest.raises(ValueError, match="parent directory references"):
            de.normalize_path("../test.txt")
        # Path.resolve() might remove the tilde, so this check in create_file is more direct
        # For normalize_path itself, Path.resolve() handles tilde expansion.
        # The ValueError for tilde is raised in create_file, not normalize_path directly.
        # So, we test normalize_path's behavior with tilde.
        expanded_home_path = str(Path("~").expanduser() / "test.txt")
        assert de.normalize_path("~/test.txt") == expanded_home_path

        # Test with invalid types that should raise ValueError from Path construction or our checks
        with pytest.raises(ValueError, match="Invalid path:"):
            de.normalize_path(None)
        with pytest.raises(ValueError, match="Invalid path:"):
            de.normalize_path("") # Empty string can be problematic for Path


    def test_read_local_file(self, tmp_path):
        file_path = tmp_path / "test_read.txt"
        file_content = "Hello, World!"
        file_path.write_text(file_content, encoding="utf-8")
        assert de.util_read_local_file(str(file_path)) == file_content

        with pytest.raises(FileNotFoundError):
            de.util_read_local_file(str(tmp_path / "non_existent.txt"))

    def test_create_file(self, tmp_path):
        file_path = tmp_path / "test_create.txt"
        content = "Content to create"
        de.util_create_file(str(file_path), content, de.console, de.MAX_FILE_SIZE_BYTES)
        assert file_path.read_text(encoding="utf-8") == content
 
        # Test overwrite
        new_content = "New content"
        de.util_create_file(str(file_path), new_content, de.console, de.MAX_FILE_SIZE_BYTES)
        assert file_path.read_text(encoding="utf-8") == new_content
 
        # Test subdirectory creation
        sub_dir_file = tmp_path / "subdir" / "test_sub.txt"
        de.util_create_file(str(sub_dir_file), content, de.console, de.MAX_FILE_SIZE_BYTES)
        assert sub_dir_file.read_text(encoding="utf-8") == content
 
        # Test size limit
        large_content = "a" * (5_000_000 + 1)
        with pytest.raises(ValueError, match="File content exceeds 4MB size limit"):
            de.util_create_file(str(tmp_path / "large.txt"), large_content, de.console, de.MAX_FILE_SIZE_BYTES)
 
        # Test tilde path (security check in create_file)
        with pytest.raises(ValueError, match="Home directory references not allowed"):
            de.util_create_file("~/should_not_create.txt", "test", de.console, de.MAX_FILE_SIZE_BYTES)
 
        # Test create_file when normalize_path raises an error (e.g., invalid path components)
        with patch('src.file_utils.normalize_path', side_effect=ValueError("Simulated normalize_path error")):
            with pytest.raises(ValueError, match="Invalid path for create_file: ../invalid.txt. Details: Simulated normalize_path error"): # This is the line 160 from your traceback
                de.util_create_file("../invalid.txt", "content", de.console, de.MAX_FILE_SIZE_BYTES) # Use util_ prefix and add args
        de.console.print.assert_any_call("[bold red]✗[/bold red] Could not create file. Invalid path: '[bright_cyan]../invalid.txt[/bright_cyan]'. Error: Simulated normalize_path error") # ai_eng.console


    def test_is_binary_file(self, tmp_path):
        text_file = tmp_path / "text.txt"
        text_file.write_text("This is a text file.", encoding="utf-8")
        assert not de.is_binary_file(str(text_file))

        binary_file = tmp_path / "binary.bin"
        binary_file.write_bytes(b"This is binary\x00data.")
        assert de.is_binary_file(str(binary_file))

        empty_file = tmp_path / "empty.txt"
        empty_file.touch()
        assert not de.is_binary_file(str(empty_file)) # Empty is not binary by null byte check

    def test_apply_diff_edit(self, tmp_path, capsys):
        file_path = tmp_path / "edit_me.txt"
        original_content = "line1\nline_to_replace\nline3"
        file_path.write_text(original_content, encoding="utf-8")

        # Successful edit
        de.util_apply_diff_edit(str(file_path), "line_to_replace", "new_line_content", de.console, de.MAX_FILE_SIZE_BYTES)
        assert file_path.read_text(encoding="utf-8") == "line1\nnew_line_content\nline3"

        # Snippet not found
        file_path.write_text(original_content, encoding="utf-8") # Reset content
        with pytest.raises(ValueError, match="Original snippet not found"):
             de.util_apply_diff_edit(str(file_path), "non_existent_snippet", "replacement", de.console, de.MAX_FILE_SIZE_BYTES)
        assert file_path.read_text(encoding="utf-8") == original_content # Should not change
 
        # Ambiguous edit
        ambiguous_content = "replace\nsomething\nreplace"
        file_path.write_text(ambiguous_content, encoding="utf-8")
        with pytest.raises(ValueError, match="Ambiguous edit: 2 matches"):
            de.util_apply_diff_edit(str(file_path), "replace", "new_replace", de.console, de.MAX_FILE_SIZE_BYTES)
        assert file_path.read_text(encoding="utf-8") == ambiguous_content # Should not change

        # Check that the warning about multiple matches was printed via the mocked console
        found_multiple_matches_warning = False
        found_format_hint = False
        for print_call in de.console.print.call_args_list:
            call_text = str(print_call[0][0])
            if "Multiple matches (2) found" in call_text:
                found_multiple_matches_warning = True
            if "Use format:" in call_text and "--- original.py (lines X-Y)" in call_text:
                found_format_hint = True
            if found_multiple_matches_warning and found_format_hint:
                break
        assert found_multiple_matches_warning, "Warning for 'Multiple matches (2) found' not printed"
        assert found_format_hint, "Hint for diff format not printed for ambiguous edit"

        # Test FileNotFoundError
        non_existent_file = tmp_path / "does_not_exist.txt"
        with pytest.raises(FileNotFoundError, match=f"File not found for diff editing: '{str(non_existent_file)}'"):
            de.util_apply_diff_edit(str(non_existent_file), "old", "new", de.console, de.MAX_FILE_SIZE_BYTES)
        de.console.print.assert_any_call(f"[bold red]✗[/bold red] File not found for diff editing: '{str(non_existent_file)}'")

        
    def test_ensure_file_in_context(self, tmp_path):
        file_path = tmp_path / "context_file.txt"
        file_content = "File for context"
        file_path.write_text(file_content, encoding="utf-8")
        
        initial_history_len = len(de.conversation_history)
        assert de.ensure_file_in_context(str(file_path))
        assert len(de.conversation_history) == initial_history_len + 1
        assert f"Content of file '{de.normalize_path(str(file_path))}':\n\n{file_content}" in de.conversation_history[-1]["content"]

        # Call again, should not add duplicate
        assert de.ensure_file_in_context(str(file_path))
        assert len(de.conversation_history) == initial_history_len + 1 

        # Non-existent file
        assert not de.ensure_file_in_context(str(tmp_path / "no_such_file.txt"))
        assert len(de.conversation_history) == initial_history_len + 1 # No new system message for error

    def test_trim_conversation_history(self):
        # Short history, no trim
        de.conversation_history = [{"role": "system", "content": "sys"}] + [{"role": "user", "content": f"msg{i}"} for i in range(5)]
        de.trim_conversation_history()
        assert len(de.conversation_history) == 6

        # Long history, trim
        de.conversation_history = [{"role": "system", "content": "sys"}] + [{"role": "user", "content": f"msg{i}"} for i in range(30)]
        de.trim_conversation_history()
        # 1 system message + 15 other messages
        assert len(de.conversation_history) == 1 + 15
        assert de.conversation_history[0]["role"] == "system"
        assert de.conversation_history[1]["content"] == "msg15" # Keeps last 15 of 'other_msgs'

        # Test with only system messages (should not trim beyond system prompt)
        de.conversation_history = [{"role": "system", "content": "sys1"}] + [
            {"role": "system", "content": "sys2"}
        ] * 15 # Make it long
        de.trim_conversation_history()
        assert len(de.conversation_history) > 1 # Should keep all system messages

    @patch('ai_eng.util_read_local_file')
    @patch('ai_eng.util_create_file') 
    @patch('ai_eng.util_apply_diff_edit')
    @patch('ai_eng.ensure_file_in_context', return_value=True) 
    def test_execute_function_call_dict(self, mock_ensure_context, mock_apply_diff, mock_create, mock_read, tmp_path): # mock_read is still patching the old name here, needs update - FIX THIS PATCH
        # read_file
        mock_read.return_value = "file content"
        tool_call = {"function": {"name": "read_file", "arguments": json.dumps({"file_path": "test.txt"})}}
        result = de.execute_function_call_dict(tool_call) # This calls the internal handler which uses util_read_local_file
        mock_read.assert_called_once_with(de.normalize_path("test.txt"))
        assert "Content of file" in result
        assert "file content" in result

        # create_file
        tool_call = {"function": {"name": "create_file", "arguments": json.dumps({"file_path": "new.txt", "content": "new stuff"})}}
        result = de.execute_function_call_dict(tool_call)
        mock_create.assert_called_once_with("new.txt", "new stuff", de.console, de.MAX_FILE_SIZE_BYTES) # Check args
        assert "Successfully created file 'new.txt'" in result

        # edit_file
        tool_call = {"function": {"name": "edit_file", "arguments": json.dumps({
            "file_path": "edit.txt", "original_snippet": "old", "new_snippet": "new"})}}
        result = de.execute_function_call_dict(tool_call)
        mock_ensure_context.assert_called_with("edit.txt")
        mock_apply_diff.assert_called_once_with("edit.txt", "old", "new", de.console, de.MAX_FILE_SIZE_BYTES) # Check args
        assert "Successfully edited file 'edit.txt'" in result
        
        # read_multiple_files
        mock_read.side_effect = ["content1", "content2"]
        tool_call = {"function": {"name": "read_multiple_files", "arguments": json.dumps({"file_paths": ["file1.txt", "file2.txt"]})}}
        result = de.execute_function_call_dict(tool_call)
        assert mock_read.call_count == 3 # 1 from read_file + 2 from here
        assert "Content of file '"+de.normalize_path("file1.txt")+"':\n\ncontent1" in result
        assert "Content of file '"+de.normalize_path("file2.txt")+"':\n\ncontent2" in result

        # create_multiple_files
        files_to_create = [{"path": "f1.txt", "content": "c1"}, {"path": "f2.txt", "content": "c2"}]
        tool_call = {"function": {"name": "create_multiple_files", "arguments": json.dumps({"files": files_to_create})}}
        result = de.execute_function_call_dict(tool_call)
        assert mock_create.call_count == 3 # 1 from create_file + 2 from here
        mock_create.assert_any_call("f1.txt", "c1", de.console, de.MAX_FILE_SIZE_BYTES) # Check args
        mock_create.assert_any_call("f2.txt", "c2", de.console, de.MAX_FILE_SIZE_BYTES) # Check args
        assert "Successfully created 2 files: f1.txt, f2.txt" in result

        # Unknown function
        tool_call = {"function": {"name": "unknown_func", "arguments": "{}"}}
        result = de.execute_function_call_dict(tool_call)
        assert "Unknown function: unknown_func" in result

        # Error in execution
        mock_read.side_effect = Exception("Read error")
        tool_call = {"function": {"name": "read_file", "arguments": json.dumps({"file_path": "error.txt"})}}
        result = de.execute_function_call_dict(tool_call)
        assert "Error executing read_file: Read error" in result

        # Error in json.loads
        mock_read.reset_mock(side_effect=True) # Reset side_effect from previous
        tool_call_bad_json = {"function": {"name": "read_file", "arguments": "this is not json"}}
        # We need to know what function_name will be if arguments parsing fails. It might be undefined.
        # The current implementation will try to use function_name from the input dict.
        result = de.execute_function_call_dict(tool_call_bad_json)
        assert "Error executing read_file: Expecting value: line 1 column 1 (char 0)" in result # Error from json.loads

        # Test generic exception during function execution (covers 463-465)
        mock_create.reset_mock(side_effect=True) # Clear previous side_effect
        mock_create.side_effect = Exception("Generic create error")
        tool_call_generic_error = {"function": {"name": "create_file", "arguments": json.dumps({"file_path": "generic_error.txt", "content": "stuff"})}}
        result = de.execute_function_call_dict(tool_call_generic_error) # This calls the internal handler which uses util_create_file
        assert "Error executing create_file: Generic create error" in result
        de.console.print.assert_any_call("[red]Error executing create_file: Generic create error[/red]")

        # Test exception within create_multiple_files (covers 501-502)
        mock_create.reset_mock(side_effect=True)
        mock_create.side_effect = [None, Exception("Create multiple sub-error")] # First succeeds, second fails
        files_to_create_with_error = [{"path": "f_ok.txt", "content": "c_ok"}, {"path": "f_err.txt", "content": "c_err"}]
        tool_call_cm_error = {"function": {"name": "create_multiple_files", "arguments": json.dumps({"files": files_to_create_with_error})}}
        result = de.execute_function_call_dict(tool_call_cm_error)
        assert "Error during create_multiple_files: Create multiple sub-error" in result
        assert "File f_ok.txt created." in result # Check partial success message
        de.console.print.assert_any_call("[red]Error creating file f_err.txt: Create multiple sub-error[/red]")

        # Test exception within edit_file (covers 535-536)
        mock_apply_diff.reset_mock(side_effect=True)
        mock_apply_diff.side_effect = Exception("Generic edit error")
        mock_ensure_context.reset_mock(return_value=True) # Ensure it doesn't fail before apply_diff
        tool_call_edit_error = {"function": {"name": "edit_file", "arguments": json.dumps({
            "file_path": "edit_error.txt", "original_snippet": "old", "new_snippet": "new"})}} # This calls the internal handler which uses util_apply_diff_edit
        result = de.execute_function_call_dict(tool_call_edit_error)
        assert "Error executing edit_file: Generic edit error" in result
        de.console.print.assert_any_call("[red]Error executing edit_file: Generic edit error[/red]")




class TestCommandHandling:
    @patch('ai_eng.util_read_local_file')
    @patch('ai_eng.add_directory_to_conversation') # This patch is correct
    def test_try_handle_add_command_file(self, mock_add_dir, mock_read_file, tmp_path): # mock_read_file is patching the old name - FIX THIS PATCH
        mock_read_file.return_value = "file content"
        file_to_add = tmp_path / "my_file.txt"
        file_to_add.touch() # Ensure it exists for os.path.isdir

        # Mock os.path.isdir to return False for this path
        with patch('os.path.isdir', return_value=False) as mock_isdir:
            handled = de.try_handle_add_command(f"/add {str(file_to_add)}")
            assert handled
            mock_isdir.assert_called_once_with(de.normalize_path(str(file_to_add)))
            mock_read_file.assert_called_once_with(de.normalize_path(str(file_to_add)))
            assert len(de.conversation_history) == 2 # System prompt + added file
            assert f"Content of file '{de.normalize_path(str(file_to_add))}'" in de.conversation_history[1]["content"]
            mock_add_dir.assert_not_called()

    @patch('ai_eng.add_directory_to_conversation')
    def test_try_handle_add_command_directory(self, mock_add_dir, tmp_path):
        dir_to_add = tmp_path / "my_dir"
        dir_to_add.mkdir()

        with patch('os.path.isdir', return_value=True) as mock_isdir:
            handled = de.try_handle_add_command(f"/add {str(dir_to_add)}")
            assert handled
            mock_isdir.assert_called_once_with(de.normalize_path(str(dir_to_add)))
            mock_add_dir.assert_called_once_with(de.normalize_path(str(dir_to_add)))

    def test_try_handle_add_command_not_add(self):
        assert not de.try_handle_add_command("some other command")

    @patch('os.path.isdir', return_value=False) # Assume it's a file
    @patch('ai_eng.util_read_local_file', side_effect=OSError("Permission denied")) # Update patch target
    def test_try_handle_add_command_file_error(self, mock_read_file, mock_isdir, tmp_path, capsys):
        file_path = tmp_path / "error_file.txt"
        # No need to create file as read_local_file is mocked to raise OSError
        
        handled = de.try_handle_add_command(f"/add {str(file_path)}")
        assert handled
        # Check the string representation of the Rich Text/Markup object passed to the mocked console
        actual_output_str = str(de.console.print.call_args_list[-1][0][0])
        assert f"Could not add path '[bright_cyan]{str(file_path)}[/bright_cyan]': Permission denied" in actual_output_str
        assert len(de.conversation_history) == 1 # Only system prompt
    
    def test_add_directory_to_conversation_max_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(de, 'MAX_FILES_TO_PROCESS_IN_DIR', 2)
        
        # Create a structure that will trigger the warning
        # when os.walk attempts to enter the subdirectory after the limit is met
        # by files in the root of the walked directory.
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.txt").write_text("content2")

        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_text("content3") # This file should not be processed

        de.add_directory_to_conversation(str(tmp_path))

        # Check console output for max files warning
        found_max_files_warning = False
        expected_warning_message = f"Reached maximum file limit ({de.MAX_FILES_TO_PROCESS_IN_DIR})"
        
        for call_args in de.console.print.call_args_list:
            if call_args[0]:
                # call_args[0][0] can be a string or a Rich Text object. Convert to string.
                if expected_warning_message in str(call_args[0][0]):
                    found_max_files_warning = True
                    break
        assert found_max_files_warning, \
            f"Warning '{expected_warning_message}' not printed. Console calls: {de.console.print.call_args_list}"

        # Check conversation history: 
        # Only files from the root of tmp_path should be added
        # System prompt is 1, then MAX_FILES_TO_PROCESS_IN_DIR files are added as system messages
        system_file_messages_count = sum(1 for msg in de.conversation_history if msg["role"] == "system" and "Content of file" in msg["content"])
        assert system_file_messages_count == de.MAX_FILES_TO_PROCESS_IN_DIR

        # Verify that file3.txt from subdir was not added, as the warning should cause a break
        normalized_file3_path = de.normalize_path(str(subdir / "file3.txt"))
        file3_added = any(f"Content of file '{normalized_file3_path}'" in str(msg["content"]) # Convert to string for Rich Text
                          for msg in de.conversation_history if msg["role"] == "system" and "Content of file" in msg["content"])
        assert not file3_added, f"File '{normalized_file3_path}' from subdir should not have been added."

    def test_add_directory_to_conversation(self, tmp_path, capsys):
        # Setup directory structure
        (tmp_path / "file1.py").write_text("python code")
        (tmp_path / "file2.txt").write_text("text content")
        (tmp_path / ".hiddenfile").write_text("hidden")
        (tmp_path / "image.png").write_bytes(b"pngdata") # Binary-like
        (tmp_path / "largefile.txt").write_text("a" * (6_000_000)) # Exceeds size limit
        
        sub_dir = tmp_path / "subdir"
        sub_dir.mkdir()
        (sub_dir / "subfile.js").write_text("javascript code")
        (sub_dir / ".git").mkdir() # Excluded directory name
        (sub_dir / "node_modules").mkdir() # Excluded directory name

        de.add_directory_to_conversation(str(tmp_path))
        
        # Check conversation history
        added_paths = [
            de.normalize_path(str(tmp_path / "file1.py")),
            de.normalize_path(str(tmp_path / "file2.txt")),
            de.normalize_path(str(sub_dir / "subfile.js")),
        ]
        history_contents = [msg["content"] for msg in de.conversation_history if msg["role"] == "system" and "Content of file" in msg["content"]]
        
        assert len(history_contents) == len(added_paths)
        for path_str in added_paths:
            assert any(f"Content of file '{path_str}'" in h_content for h_content in history_contents)

        # Check console output (simplified check)
        # Using the mocked console directly
        printed_output = ""
        for call_args in de.console.print.call_args_list:
            if call_args[0]: # Ensure there are positional arguments
                printed_output += str(call_args[0][0])
        
        assert f"Added folder '[bright_cyan]{str(tmp_path)}[/bright_cyan]'" in printed_output
        assert "file1.py" in printed_output
        assert "file2.txt" in printed_output
        assert str(Path("subdir/subfile.js")) in printed_output # Relative to tmp_path
        assert "Skipped files" in printed_output
        assert ".hiddenfile" in printed_output
        assert "image.png" in printed_output
        assert "largefile.txt (exceeds size limit)" in printed_output
        # .git and node_modules directories themselves should be skipped, not listed as skipped files.
        # The files inside them won't be processed.


class TestStreamLLMResponse:

    @patch('ai_eng.completion')
    @patch('ai_eng.trim_conversation_history')
    def test_simple_text_response(self, mock_trim, mock_litellm_completion, mock_env_vars):
        mock_stream = iter([
            create_mock_litellm_chunk(content="Hello, "),
            create_mock_litellm_chunk(content="world!"),
            create_mock_litellm_chunk(finish_reason="stop")
        ])
        mock_litellm_completion.return_value = mock_stream

        user_message = "Hi there"
        response = de.stream_llm_response(user_message)

        assert response == {"success": True}
        mock_trim.assert_called_once()

        # Calculate the expected augmented user message content
        # These defaults are hardcoded in ai_eng.py stream_llm_response
        default_reasoning_effort_val = "medium"
        default_reply_effort_val = "medium"
        effort_instructions = (
            f"\n\n[System Instructions For This Turn Only]:\n"
            f"- Current `reasoning_effort`: {default_reasoning_effort_val}\n"
            f"- Current `reply_effort`: {default_reply_effort_val}\n"
            f"Please adhere to these specific effort levels for your reasoning and reply in this turn."
        )
        expected_augmented_user_message_content = user_message + effort_instructions

        expected_messages_for_completion = [
            {"role": "system", "content": de.system_PROMPT},
            {"role": "user", "content": expected_augmented_user_message_content}
        ]

        mock_litellm_completion.assert_called_once_with(
            model="test_model_from_env",
            messages=expected_messages_for_completion,
            tools=de.tools,
            max_tokens=8192, # Default from ai_eng.py
            api_base="http://test.api.base/v1",
            temperature=0.7, # Default from ai_eng.py
            stream=True
        )

        # Check conversation history
        assert de.conversation_history[-1] == {"role": "assistant", "content": "Hello, world!"}
        
        # Check console output (via mocked console)
        de.console.print.assert_any_call("\n[bold bright_blue]🐋 Seeking...[/bold bright_blue]")
        de.console.print.assert_any_call("Hello, ", end="")
        de.console.print.assert_any_call("world!", end="")


    @patch('ai_eng.completion')
    def test_response_with_reasoning(self, mock_litellm_completion, mock_env_vars):
        mock_stream = iter([
            create_mock_litellm_chunk(reasoning_content="Thinking... "),
            create_mock_litellm_chunk(reasoning_content="step 1. "),
            create_mock_litellm_chunk(content="Okay, "),
            create_mock_litellm_chunk(content="done."),
            create_mock_litellm_chunk(finish_reason="stop")
        ])
        mock_litellm_completion.return_value = mock_stream

        user_message = "Explain something."
        de.stream_llm_response(user_message)
        
        mock_litellm_completion.assert_called_once_with(
            model="test_model_from_env",
            messages=ANY, # History check is more complex here, focus on other params
            tools=de.tools,
            max_tokens=8192,
            api_base="http://test.api.base/v1",
            temperature=0.7, # Default from ai_eng.py
            stream=True
        )

        # The assistant message should include the accumulated reasoning content
        assert de.conversation_history[-1] == {
            "role": "assistant",
            "content": "Okay, done.",
            "reasoning_content_full": "Thinking... step 1. "}
        # Check console output for reasoning and content
        de.console.print.assert_any_call("\n[bold blue]💭 Reasoning:[/bold blue]")
        de.console.print.assert_any_call("Thinking... ", end="")
        de.console.print.assert_any_call("step 1. ", end="") # This assertion is correct
        de.console.print.assert_any_call("\n\n[bold bright_blue]🤖 Assistant>[/bold bright_blue] ", end="") # Expect two newlines before prompt
        de.console.print.assert_any_call("Okay, ", end="")
        de.console.print.assert_any_call("done.", end="")

    @patch('ai_eng.completion')
    @patch('ai_eng.execute_function_call_dict')
    def test_response_with_tool_calls(self, mock_execute_func, mock_litellm_completion, mock_env_vars):
        # First call: requests a tool
        tool_call_request_stream = iter([
            create_mock_litellm_chunk(tool_calls_delta=[{
                "index": 0, "id": "call_123", "function": {"name": "read_file", "arguments": '{"file_path":'}
            }]),
            create_mock_litellm_chunk(tool_calls_delta=[{
                "index": 0, "function": {"arguments": ' "test.txt"}'}
            }]),
            create_mock_litellm_chunk(finish_reason="tool_calls")
        ])
        
        # Second call: follow-up after tool execution
        follow_up_stream = iter([
            create_mock_litellm_chunk(content="File read. "),
            create_mock_litellm_chunk(reasoning_content="Now I will summarize. "), # Add reasoning to follow-up
            create_mock_litellm_chunk(content="Content is '...'."),
        ])
        
        mock_litellm_completion.side_effect = [tool_call_request_stream, follow_up_stream]
        mock_execute_func.return_value = "Mocked file content for test.txt"

        de.stream_llm_response("Read test.txt")

        # Assertions
        assert mock_litellm_completion.call_count == 2
        
        # Check arguments for the first call to litellm.completion
        first_call_args = mock_litellm_completion.call_args_list[0]
        assert first_call_args[1]['model'] == "test_model_from_env"
        assert first_call_args[1]['api_base'] == "http://test.api.base/v1"
        assert first_call_args[1]['stream'] is True
        assert first_call_args[1]['tools'] == de.tools

        # Check arguments for the second (follow-up) call to litellm.completion
        second_call_args = mock_litellm_completion.call_args_list[1]
        assert second_call_args[1]['model'] == "test_model_from_env"
        assert second_call_args[1]['api_base'] == "http://test.api.base/v1"
        assert second_call_args[1]['stream'] is True
        assert second_call_args[1]['tools'] == de.tools




        mock_execute_func.assert_called_once_with({
            "id": "call_123", 
            "type": "function", 
            "function": {"name": "read_file", "arguments": '{"file_path": "test.txt"}'}
        })

        # Check conversation history
        # 1. System prompt
        # 2. User: "Read test.txt"
        # 3. Assistant: (tool_calls=[...])
        # 4. Tool: (result of read_file)
        # 5. Assistant: "File read. Content is '...'"
        assert len(de.conversation_history) == 5
        assert de.conversation_history[2]["role"] == "assistant"
        assert de.conversation_history[2]["content"] is None # Content should be None if tool_calls are present
        assert len(de.conversation_history[2]["tool_calls"]) == 1
        assert de.conversation_history[2]["tool_calls"][0]["id"] == "call_123"
        assert de.conversation_history[2]["tool_calls"][0]["function"]["name"] == "read_file"
        
        assert de.conversation_history[3]["role"] == "tool"
        assert de.conversation_history[3]["tool_call_id"] == "call_123"
        assert de.conversation_history[3]["content"] == "Mocked file content for test.txt"
        
        assert de.conversation_history[4]["role"] == "assistant"
        assert de.conversation_history[4]["content"] == "File read. Content is '...'."

        de.console.print.assert_any_call("\n[bold bright_cyan]⚡ Executing 1 function call(s)...[/bold bright_cyan]")
        de.console.print.assert_any_call("[bright_blue]→ read_file[/bright_blue]")
        de.console.print.assert_any_call("\n[bold bright_blue]🔄 Processing results...[/bold bright_blue]")
        # Check for reasoning print in follow-up
        reasoning_in_follow_up_printed = any(
            "\n[bold blue]💭 Reasoning:[/bold blue]" in str(call_arg[0][0]) for call_arg in de.console.print.call_args_list if call_arg[0])
        assert reasoning_in_follow_up_printed, "Reasoning in follow-up not printed"


    @patch('ai_eng.completion', side_effect=Exception("API Connection Error"))
    def test_api_error(self, mock_litellm_completion, mock_env_vars):
        response = de.stream_llm_response("Trigger error")
        assert response["error"] == "LLM API error: API Connection Error"
        de.console.print.assert_any_call("\n[bold red]❌ LLM API error: API Connection Error[/bold red]")

    @patch('ai_eng.completion')
    @patch('ai_eng.execute_function_call_dict', side_effect=Exception("Tool execution failed"))
    def test_tool_call_execution_error(self, mock_execute_func, mock_litellm_completion, mock_env_vars):
        # Stream for initial tool call request
        tool_call_request_stream = iter([
            create_mock_litellm_chunk(tool_calls_delta=[{
                "index": 0, "id": "call_err", "function": {"name": "error_tool", "arguments": "{}"}
            }]),
            create_mock_litellm_chunk(finish_reason="tool_calls")
        ])
        # Stream for follow-up (should still happen)
        follow_up_stream = iter([
            create_mock_litellm_chunk(content="Acknowledging tool error.")
        ])
        mock_litellm_completion.side_effect = [tool_call_request_stream, follow_up_stream]

        de.stream_llm_response("Use error_tool")

        mock_execute_func.assert_called_once()
        
        found_tool_error_print = False
        expected_tool_error_msg = "[red]Unexpected error during tool execution: Tool execution failed[/red]" # Corrected expected message
        for call_args in de.console.print.call_args_list:
            if call_args[0] and expected_tool_error_msg == str(call_args[0][0]):
                found_tool_error_print = True
                break
        assert found_tool_error_print, f"Expected console print '{expected_tool_error_msg}' not found. Calls: {de.console.print.call_args_list}"
        
        # Check history: tool message should contain the error
        assert de.conversation_history[-2]["role"] == "tool"
        assert de.conversation_history[-2]["tool_call_id"] == "call_err"
        assert "Tool execution failed" in de.conversation_history[-2]["content"] # Check for the core error message substring
        
        # Follow-up response should still be there
        assert de.conversation_history[-1]["role"] == "assistant"
        assert de.conversation_history[-1]["content"] == "Acknowledging tool error."


class TestMainLoop:
    @patch('ai_eng.stream_llm_response')
    @patch('ai_eng.try_handle_add_command', return_value=False) # Assume no /add commands
    def test_main_loop_exit_quit(self, mock_handle_add, mock_stream_response, monkeypatch):
        # Test 'exit'
        de.prompt_session.prompt = MagicMock(return_value="exit")
        with pytest.raises(SystemExit):
             de.main() # Should break loop and exit
        mock_stream_response.assert_not_called()
        de.console.print.assert_any_call("[bold bright_blue]👋 Goodbye! Happy coding![/bold bright_blue]")

        # Reset mocks for next test case
        de.console.print.reset_mock()
        mock_stream_response.reset_mock()
        de.prompt_session.prompt.reset_mock()
        de.conversation_history = [{"role": "system", "content": de.system_PROMPT}] # Reset history

        # Test 'quit'
        de.prompt_session.prompt = MagicMock(return_value="quit")
        with pytest.raises(SystemExit):
            de.main()
        mock_stream_response.assert_not_called()
        de.console.print.assert_any_call("[bold bright_blue]👋 Goodbye! Happy coding![/bold bright_blue]")

    @patch('ai_eng.stream_llm_response')
    @patch('ai_eng.try_handle_add_command', return_value=False)
    def test_main_loop_eof_keyboard_interrupt(self, mock_handle_add, mock_stream_response, monkeypatch):
        # Test EOFError
        de.prompt_session.prompt = MagicMock(side_effect=EOFError)
        with pytest.raises(SystemExit):
            de.main()
        mock_stream_response.assert_not_called()
        de.console.print.assert_any_call("\n[bold yellow]👋 Exiting gracefully...[/bold yellow]")

        de.console.print.reset_mock()
        mock_stream_response.reset_mock()
        de.prompt_session.prompt.reset_mock()
        de.conversation_history = [{"role": "system", "content": de.system_PROMPT}]

        # Test KeyboardInterrupt
        de.prompt_session.prompt = MagicMock(side_effect=KeyboardInterrupt)
        with pytest.raises(SystemExit):
            de.main()
        mock_stream_response.assert_not_called()
        de.console.print.assert_any_call("\n[bold yellow]👋 Exiting gracefully...[/bold yellow]")

    @patch('ai_eng.stream_llm_response', return_value={"success": True})
    @patch('ai_eng.try_handle_add_command', return_value=False)
    def test_main_loop_empty_and_normal_input(self, mock_handle_add, mock_stream_response, monkeypatch):
        # Simulate empty input, then normal input, then exit
        de.prompt_session.prompt = MagicMock(side_effect=["", "hello", KeyboardInterrupt])
        with pytest.raises(SystemExit): # Due to KeyboardInterrupt
            de.main()
        
        mock_stream_response.assert_called_once_with("hello")
        # Check that empty input "" did not lead to a call to stream_llm_response before "hello"
        assert mock_stream_response.call_args_list == [call("hello")]

    @patch('ai_eng.stream_llm_response', return_value={"error": "Simulated API Error"})
    @patch('ai_eng.try_handle_add_command', return_value=False)
    def test_main_loop_llm_error(self, mock_handle_add, mock_stream_response, monkeypatch):
        # Simulate normal input, then LLM error, then exit
        de.prompt_session.prompt = MagicMock(side_effect=["ask something", KeyboardInterrupt])
        with pytest.raises(SystemExit): # Due to KeyboardInterrupt
            de.main()

        mock_stream_response.assert_called_once_with("ask something")
        # The main loop itself doesn't print the LLM error if stream_llm_response is mocked
        # to return an error dictionary, as stream_llm_response is responsible for that print.
        # We just ensure the loop continued and exited gracefully.
        # The actual printing of the error by stream_llm_response is tested in TestStreamLLMResponse.test_api_error.
        pass
