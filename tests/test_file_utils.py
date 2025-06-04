# tests/test_file_utils.py
import pytest
import os
from pathlib import Path
from rich.panel import Panel # Import Panel
from unittest.mock import MagicMock, patch

# Import the functions from src.file_utils
from src.file_utils import (
    normalize_path,
    is_binary_file,
    read_local_file,
    create_file,
    apply_diff_edit,
)

# Define a mock console object fixture
@pytest.fixture
def mock_console():
    """Fixture for a mock Rich console object."""
    return MagicMock()

# Define a mock max file size for tests
MOCK_MAX_FILE_SIZE_BYTES = 1024 # 1 KB for testing size limits

# --- Tests for normalize_path ---

def test_normalize_path_relative(tmp_path):
    """Test normalizing a relative path."""
    # Create a dummy file to ensure resolve() works
    dummy_file = tmp_path / "subdir" / "test_file.txt"
    dummy_file.parent.mkdir()
    dummy_file.touch()

    os.chdir(tmp_path) # Change directory to tmp_path for relative path testing
    relative_path = "subdir/test_file.txt"
    normalized = normalize_path(relative_path)
    assert Path(normalized).is_absolute()
    assert Path(normalized) == dummy_file.resolve()
    os.chdir(Path(__file__).parent.parent) # Change back

def test_normalize_path_absolute(tmp_path):
    """Test normalizing an absolute path."""
    absolute_path = str(tmp_path / "another_file.txt")
    normalized = normalize_path(absolute_path)
    assert Path(normalized).is_absolute()
    assert Path(normalized) == Path(absolute_path).resolve()

def test_normalize_path_with_tilde(tmp_path):
    """Test normalizing a path with a tilde (~)."""
    # Mock Path.expanduser to return a known path within tmp_path
    with patch("src.file_utils.Path.expanduser") as mock_expanduser:
        mock_expanduser.return_value = tmp_path / "home_dir_file.txt"
        normalized = normalize_path("~/some_file.txt")
        assert Path(normalized).is_absolute()
        assert Path(normalized) == (tmp_path / "home_dir_file.txt").resolve()

def test_normalize_path_with_dots_resolving_within_root(tmp_path):
    """Test normalizing a path with '.' and '..' that stays within the root."""
    dummy_dir = tmp_path / "a" / "b"
    dummy_dir.mkdir(parents=True)
    dummy_file = dummy_dir / "c.txt"
    dummy_file.touch()

    path_with_dots = str(tmp_path / "a" / ".." / "a" / "b" / "." / "c.txt")
    # This path contains ".." and will be caught by the explicit check in normalize_path
    with pytest.raises(ValueError, match="Invalid path: .* contains parent directory references"):
        normalize_path(path_with_dots)

def test_normalize_path_with_dots_escaping_root_raises_valueerror(tmp_path):
    """Test normalizing a path with '..' that attempts to escape the root."""
    path_escaping = str(tmp_path / ".." / "some_file.txt")
    # Path.resolve() should raise an error (often ValueError or OSError depending on OS/context)
    with pytest.raises(ValueError, match="Invalid path: .* contains parent directory references"):
        normalize_path(path_escaping)

def test_normalize_path_empty_raises_valueerror():
    """Test normalizing an empty path string."""
    with pytest.raises(ValueError, match="Invalid path: Path cannot be empty."):
        normalize_path("")

# --- Tests for is_binary_file ---

def test_is_binary_file_text(tmp_path):
    """Test is_binary_file with a text file."""
    text_file = tmp_path / "test.txt"
    text_file.write_text("This is a text file.")
    assert not is_binary_file(str(text_file))

def test_is_binary_file_with_null_byte(tmp_path):
    """Test is_binary_file with a file containing a null byte."""
    binary_file = tmp_path / "binary.bin"
    # Write some text followed by a null byte
    binary_file.write_bytes(b"hello\x00world")
    assert is_binary_file(str(binary_file))

def test_is_binary_file_empty(tmp_path):
    """Test is_binary_file with an empty file."""
    empty_file = tmp_path / "empty.txt"
    empty_file.touch()
    assert not is_binary_file(str(empty_file)) # Empty file is not binary

def test_is_binary_file_non_existent(tmp_path):
    """Test is_binary_file with a non-existent file."""
    non_existent_file = tmp_path / "does_not_exist.bin"
    # The function catches exceptions and returns True for safety
    assert is_binary_file(str(non_existent_file))

# --- Tests for read_local_file ---

def test_read_local_file_success(tmp_path):
    """Test reading an existing text file."""
    test_file = tmp_path / "read_test.txt"
    content = "Line 1\nLine 2\nLine 3"
    test_file.write_text(content, encoding="utf-8")
    read_content = read_local_file(str(test_file))
    assert read_content == content

def test_read_local_file_not_found(tmp_path):
    """Test reading a non-existent file."""
    non_existent_file = tmp_path / "non_existent.txt"
    with pytest.raises(FileNotFoundError):
        read_local_file(str(non_existent_file))

def test_read_local_file_permission_error(tmp_path):
    """Test reading a file with permission issues (mocking)."""
    test_file = tmp_path / "permission_denied.txt"
    test_file.touch() # Create the file

    # Mock the built-in open function to raise OSError
    with patch("builtins.open", side_effect=OSError("Permission denied")):
        with pytest.raises(OSError, match="Permission denied"):
            read_local_file(str(test_file))

def test_read_local_file_encoding_error(tmp_path):
    """Test reading a file with invalid encoding (mocking)."""
    test_file = tmp_path / "bad_encoding.txt"
    # Create a file with non-UTF-8 content (e.g., bytes that are invalid UTF-8)
    # We can simulate this by writing bytes directly
    with open(test_file, "wb") as f:
        f.write(b'\xff\xfe\xfd') # Invalid UTF-8 byte sequence

    # When read_local_file tries to open with encoding="utf-8", it should fail
    with pytest.raises(UnicodeDecodeError):
        read_local_file(str(test_file))

# --- Tests for create_file ---

def test_create_file_new_existing_dir(tmp_path, mock_console):
    """Test creating a new file in an existing directory."""
    file_path = tmp_path / "new_file.txt"
    content = "This is new content."
    create_file(str(file_path), content, mock_console, MOCK_MAX_FILE_SIZE_BYTES)
    assert file_path.exists()
    assert file_path.read_text(encoding="utf-8") == content
    mock_console.print.assert_called_with(f"[bold blue]✓[/bold blue] Created/updated file at '[bright_cyan]{file_path.resolve()}[/bright_cyan]'")

def test_create_file_new_non_existent_dir(tmp_path, mock_console):
    """Test creating a new file in a non-existent directory."""
    file_path = tmp_path / "new_dir" / "another_new_file.txt"
    content = "Content for file in new dir."
    create_file(str(file_path), content, mock_console, MOCK_MAX_FILE_SIZE_BYTES)
    assert file_path.exists()
    assert file_path.read_text(encoding="utf-8") == content
    assert file_path.parent.is_dir() # Ensure parent directory was created
    mock_console.print.assert_called_with(f"[bold blue]✓[/bold blue] Created/updated file at '[bright_cyan]{file_path.resolve()}[/bright_cyan]'")

def test_create_file_overwrite_existing(tmp_path, mock_console):
    """Test overwriting an existing file."""
    file_path = tmp_path / "existing_file.txt"
    initial_content = "Initial content."
    file_path.write_text(initial_content, encoding="utf-8")

    new_content = "Overwritten content."
    create_file(str(file_path), new_content, mock_console, MOCK_MAX_FILE_SIZE_BYTES)
    assert file_path.exists()
    assert file_path.read_text(encoding="utf-8") == new_content
    mock_console.print.assert_called_with(f"[bold blue]✓[/bold blue] Created/updated file at '[bright_cyan]{file_path.resolve()}[/bright_cyan]'")

def test_create_file_exceeds_size_limit(tmp_path, mock_console):
    """Test creating a file with content exceeding the size limit."""
    file_path = tmp_path / "large_file.txt"
    content = "A" * (MOCK_MAX_FILE_SIZE_BYTES + 1) # Content larger than limit

    with pytest.raises(ValueError, match=f"File content exceeds {MOCK_MAX_FILE_SIZE_BYTES // (1024*1024)}MB size limit"):
        create_file(str(file_path), content, mock_console, MOCK_MAX_FILE_SIZE_BYTES)

    assert not file_path.exists() # File should not be created
    mock_console.print.assert_called_with(f"[bold red]✗[/bold red] File content exceeds {MOCK_MAX_FILE_SIZE_BYTES // (1024*1024)}MB size limit")

def test_create_file_invalid_path_from_normalize(tmp_path, mock_console):
    """Test creating a file with an invalid path (caught by normalize_path)."""
    invalid_path_str = str(tmp_path / ".." / "illegal_file.txt")

    # The normalize_path call inside create_file should raise ValueError
    with pytest.raises(ValueError, match="Invalid path for create_file: .* Details: Invalid path: .* contains parent directory references"):
         create_file(invalid_path_str, "content", mock_console, MOCK_MAX_FILE_SIZE_BYTES)

    # Check console output for the specific error message from create_file's catch block
    mock_console.print.assert_called_with(
        f"[bold red]✗[/bold red] Could not create file. Invalid path: '[bright_cyan]{invalid_path_str}[/bright_cyan]'. Error: Invalid path: {invalid_path_str} contains parent directory references"
    )


def test_create_file_permission_error(tmp_path, mock_console):
    """Test creating a file with permission issues (mocking)."""
    # Create a directory that we will make read-only (or simulate permission error)
    protected_dir = tmp_path / "protected_dir"
    protected_dir.mkdir()
    # On some OS/filesystems, making a directory read-only prevents file creation within it.
    # A more reliable way for testing is to mock the open call.

    file_path = protected_dir / "no_write.txt"
    content = "Attempt to write here."

    # Mock the built-in open function to raise OSError
    with patch("builtins.open", side_effect=OSError("Permission denied")):
        with pytest.raises(OSError, match="Failed to write file .*Permission denied"):
            create_file(str(file_path), content, mock_console, MOCK_MAX_FILE_SIZE_BYTES)

    assert not file_path.exists() # File should not be created
    mock_console.print.assert_called_with(
        f"[bold red]✗[/bold red] Failed to write file '{file_path.resolve()}': Permission denied"
    )

def test_create_file_with_tilde_raises_valueerror(tmp_path, mock_console):
    """Test creating a file with a tilde in the path (explicit check)."""
    # This tests the explicit check at the start of create_file
    file_path_str = "~/should_fail.txt"
    with pytest.raises(ValueError, match="Home directory references not allowed for create_file directly; normalize path first."):
        create_file(file_path_str, "content", mock_console, MOCK_MAX_FILE_SIZE_BYTES)
    # No console print is expected from the function before the raise in this specific case

# --- Tests for apply_diff_edit ---

# Mock read_local_file and create_file for apply_diff_edit tests
# This isolates the diff logic from actual file I/O and size checks
@pytest.fixture
def mock_file_io_for_diff():
    """Mocks read_local_file and create_file for apply_diff_edit tests."""
    with patch("src.file_utils.read_local_file") as mock_read, \
         patch("src.file_utils.create_file") as mock_create, \
         patch("src.file_utils.normalize_path") as mock_normalize: # Also mock normalize_path
        # Default behavior for normalize_path is to return the input string as if normalized
        mock_normalize.side_effect = lambda x: x
        yield mock_read, mock_create, mock_normalize

def test_apply_diff_edit_success(mock_file_io_for_diff, mock_console):
    """Test successful application of a diff edit."""
    mock_read, mock_create, mock_normalize = mock_file_io_for_diff
    file_path = "/fake/path/to/file.txt"
    original_content = "Line 1\nSnippet to replace\nLine 3"
    original_snippet = "Snippet to replace"
    new_snippet = "Replacement snippet"
    expected_content = "Line 1\nReplacement snippet\nLine 3"

    mock_read.return_value = original_content

    apply_diff_edit(file_path, original_snippet, new_snippet, mock_console, MOCK_MAX_FILE_SIZE_BYTES)

    mock_normalize.assert_called_once_with(file_path)
    mock_read.assert_called_once_with(file_path) # Called with the (mock) normalized path
    mock_create.assert_called_once_with(file_path, expected_content, mock_console, MOCK_MAX_FILE_SIZE_BYTES)
    mock_console.print.assert_called_with(f"[bold blue]✓[/bold blue] Applied diff edit to '[bright_cyan]{file_path}[/bright_cyan]'")


def test_apply_diff_edit_snippet_not_found(mock_file_io_for_diff, mock_console):
    """Test applying a diff edit when the original snippet is not found."""
    mock_read, mock_create, mock_normalize = mock_file_io_for_diff
    file_path = "/fake/path/to/file.txt"
    original_content = "Line 1\nSome other text\nLine 3"
    original_snippet = "Snippet to replace" # Not in content
    new_snippet = "Replacement snippet"

    mock_read.return_value = original_content

    with pytest.raises(ValueError, match="Failed to apply diff to .*Original snippet not found"):
        apply_diff_edit(file_path, original_snippet, new_snippet, mock_console, MOCK_MAX_FILE_SIZE_BYTES)

    mock_normalize.assert_called_once_with(file_path)
    mock_read.assert_called_once_with(file_path)
    mock_create.assert_not_called() # File should not be created/modified if snippet not found
    mock_console.print.assert_any_call(f"[bold yellow]⚠[/bold yellow] Original snippet not found in '[bright_cyan]{file_path}[/bright_cyan]'. No changes made.")
    # Check for the panels being printed
    mock_console.print.assert_any_call("\n[bold blue]Expected snippet:[/bold blue]")
    # The following direct Panel assertion was failing due to different object instances:
    # mock_console.print.assert_any_call(Panel(original_snippet, title="Expected", border_style="blue", title_align="left"))
    mock_console.print.assert_any_call("\n[bold blue]Actual file content:[/bold blue]")
    # Assuming a similar assertion for the "Actual" panel was intended or would also fail:
    # mock_console.print.assert_any_call(Panel(original_content, title="Actual", border_style="yellow", title_align="left"))

    found_expected_panel = False
    found_actual_panel = False
    for call_args in mock_console.print.call_args_list:
        if call_args[0] and isinstance(call_args[0][0], Panel): # Check if first arg exists and is Panel
            panel_arg = call_args[0][0]
            if panel_arg.title == "Expected" and panel_arg.renderable == original_snippet:
                found_expected_panel = True
            elif panel_arg.title == "Actual" and panel_arg.renderable == original_content:
                found_actual_panel = True
    assert found_expected_panel, "Panel with expected snippet not printed or attributes mismatch."
    assert found_actual_panel, "Panel with actual file content not printed or attributes mismatch."

def test_apply_diff_edit_snippet_multiple_matches(mock_file_io_for_diff, mock_console):
    """Test applying a diff edit when the original snippet has multiple matches."""
    mock_read, mock_create, mock_normalize = mock_file_io_for_diff
    file_path = "/fake/path/to/file.txt"
    original_content = "Snippet\nAnother line\nSnippet\nLine 4"
    original_snippet = "Snippet" # Appears twice
    new_snippet = "Replacement"

    mock_read.return_value = original_content

    with pytest.raises(ValueError, match="Failed to apply diff to .*Ambiguous edit: 2 matches"):
        apply_diff_edit(file_path, original_snippet, new_snippet, mock_console, MOCK_MAX_FILE_SIZE_BYTES)

    mock_normalize.assert_called_once_with(file_path)
    mock_read.assert_called_once_with(file_path)
    mock_create.assert_not_called() # File should not be created/modified if multiple matches
    mock_console.print.assert_any_call(f"[bold yellow]⚠ Multiple matches (2) found - requiring line numbers for safety. Snippet found in '{file_path}'[/bold yellow]")
    mock_console.print.assert_any_call("[dim]Use format:\n--- original.py (lines X-Y)\n+++ modified.py[/dim]")
    # Check for the panels being printed
    mock_console.print.assert_any_call("\n[bold blue]Expected snippet:[/bold blue]")
    # The following direct Panel assertion was failing due to different object instances:
    # mock_console.print.assert_any_call(Panel(original_snippet, title="Expected", border_style="blue", title_align="left"))
    mock_console.print.assert_any_call("\n[bold blue]Actual file content:[/bold blue]")
    # Assuming a similar assertion for the "Actual" panel was intended or would also fail:
    # mock_console.print.assert_any_call(Panel(original_content, title="Actual", border_style="yellow", title_align="left"))

    found_expected_panel = False
    found_actual_panel = False
    for call_args in mock_console.print.call_args_list:
        if call_args[0] and isinstance(call_args[0][0], Panel): # Check if first arg exists and is Panel
            panel_arg = call_args[0][0]
            if panel_arg.title == "Expected" and panel_arg.renderable == original_snippet:
                found_expected_panel = True
            elif panel_arg.title == "Actual" and panel_arg.renderable == original_content:
                found_actual_panel = True
    assert found_expected_panel, "Panel with expected snippet not printed or attributes mismatch."
    assert found_actual_panel, "Panel with actual file content not printed or attributes mismatch."

def test_apply_diff_edit_file_not_found(mock_file_io_for_diff, mock_console):
    """Test applying a diff edit to a non-existent file."""
    mock_read, mock_create, mock_normalize = mock_file_io_for_diff
    file_path = "/fake/path/to/non_existent_file.txt"
    original_snippet = "Snippet"
    new_snippet = "Replacement"

    # Configure mock_read to raise FileNotFoundError
    mock_read.side_effect = FileNotFoundError("No such file")

    with pytest.raises(FileNotFoundError, match="File not found for diff editing:"):
        apply_diff_edit(file_path, original_snippet, new_snippet, mock_console, MOCK_MAX_FILE_SIZE_BYTES)

    mock_normalize.assert_called_once_with(file_path)
    mock_read.assert_called_once_with(file_path)
    mock_create.assert_not_called()
    mock_console.print.assert_called_with(f"[bold red]✗[/bold red] File not found for diff editing: '{file_path}'")


def test_apply_diff_edit_read_permission_error(mock_file_io_for_diff, mock_console):
    """Test applying a diff edit when reading the file fails due to permissions."""
    mock_read, mock_create, mock_normalize = mock_file_io_for_diff
    file_path = "/fake/path/to/permission_denied_read.txt"
    original_snippet = "Snippet"
    new_snippet = "Replacement"

    # Configure mock_read to raise OSError
    mock_read.side_effect = OSError("Permission denied on read")

    with pytest.raises(OSError, match="OS error during diff edit for .*Permission denied on read"):
        apply_diff_edit(file_path, original_snippet, new_snippet, mock_console, MOCK_MAX_FILE_SIZE_BYTES)

    mock_normalize.assert_called_once_with(file_path)
    mock_read.assert_called_once_with(file_path)
    mock_create.assert_not_called()
    mock_console.print.assert_called_with(f"[bold red]✗[/bold red] OS error during diff edit for '{file_path}': Permission denied on read")


def test_apply_diff_edit_write_permission_error(mock_file_io_for_diff, mock_console):
    """Test applying a diff edit when writing the file fails due to permissions."""
    mock_read, mock_create, mock_normalize = mock_file_io_for_diff
    file_path = "/fake/path/to/permission_denied_write.txt"
    original_content = "Snippet to replace"
    original_snippet = "Snippet to replace"
    new_snippet = "Replacement"

    mock_read.return_value = original_content
    # Configure mock_create to raise OSError
    mock_create.side_effect = OSError("Permission denied on write")

    with pytest.raises(OSError, match="OS error during diff edit for .*Permission denied on write"):
        apply_diff_edit(file_path, original_snippet, new_snippet, mock_console, MOCK_MAX_FILE_SIZE_BYTES)

    mock_normalize.assert_called_once_with(file_path)
    mock_read.assert_called_once_with(file_path)
    mock_create.assert_called_once() # create_file should have been called
    mock_console.print.assert_called_with(f"[bold red]✗[/bold red] OS error during diff edit for '{file_path}': Permission denied on write")


def test_apply_diff_edit_result_exceeds_size_limit(mock_file_io_for_diff, mock_console):
    """Test applying a diff edit where the resulting content exceeds the size limit."""
    mock_read, mock_create, mock_normalize = mock_file_io_for_diff
    file_path = "/fake/path/to/size_limit_edit.txt"
    original_content = "Short snippet"
    original_snippet = "Short snippet"
    new_snippet = "A" * (MOCK_MAX_FILE_SIZE_BYTES + 1) # Replacement makes content too large

    mock_read.return_value = original_content

    # Configure mock_create to raise ValueError for size limit
    mock_create.side_effect = ValueError(f"File content exceeds {MOCK_MAX_FILE_SIZE_BYTES // (1024*1024)}MB size limit")

    with pytest.raises(ValueError, match=f"Failed to apply diff to .*: File content exceeds {MOCK_MAX_FILE_SIZE_BYTES // (1024*1024)}MB size limit"):
        apply_diff_edit(file_path, original_snippet, new_snippet, mock_console, MOCK_MAX_FILE_SIZE_BYTES)

    mock_normalize.assert_called_once_with(file_path)
    mock_read.assert_called_once_with(file_path)
    mock_create.assert_called_once() # create_file should have been called

    # The mocked create_file only raises an error; it doesn't print its internal red message.
    # Assert for the warning message printed by apply_diff_edit
    mock_console.print.assert_any_call(f"[bold yellow]⚠[/bold yellow] File content exceeds {MOCK_MAX_FILE_SIZE_BYTES // (1024*1024)}MB size limit in '[bright_cyan]{file_path}[/bright_cyan]'. No changes made.")
    # Panels are not expected in this case based on the corrected logic
