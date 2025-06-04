# tests/test_config_utils.py
import pytest
import os
import toml # For creating and parsing TOML content (tomllib for load, toml for dump)
from unittest.mock import patch, MagicMock

# Import the functions and globals from src.config_utils
from src import config_utils

# Define a mock console object fixture
@pytest.fixture
def mock_console():
    """Fixture for a mock Rich console object."""
    return MagicMock()

@pytest.fixture(autouse=True)
def reset_config_globals_and_env(monkeypatch):
    """Reset global config dicts and relevant env vars before each test."""
    config_utils.CONFIG_FROM_TOML = {}
    config_utils.RUNTIME_OVERRIDES = {}
    
    # Clear any environment variables that might interfere with tests
    for param_name, p_config in config_utils.SUPPORTED_SET_PARAMS.items():
        if p_config["env_var"] in os.environ:
            monkeypatch.delenv(p_config["env_var"])
    if "AI_API_KEY" in os.environ: # Example of another common env var
        monkeypatch.delenv("AI_API_KEY")
    yield

@pytest.fixture
def temp_config_file(tmp_path):
    """Creates a temporary config.toml file and returns its path."""
    config_content = {
        "litellm": {
            "model": "toml_model",
            "api_base": "http://toml.api.base/v1",
            "max_tokens": 2048,
            "temperature": 0.5,
            "reasoning_effort": "high"
        },
        "ui": {
            "reasoning_style": "compact",
            "reply_effort": "low"
        },
        "defaults": { # Add a defaults section for testing get_config_value fallbacks
            "default_model": "default_toml_model",
            "default_api_base": "http://default.toml.api.base/v1",
            "default_max_tokens": 1024,
            "default_temperature": 0.8,
            "default_reasoning_style": "full",
            "default_reasoning_effort": "medium",
            "default_reply_effort": "medium"
        }
    }
    config_file = tmp_path / "config.toml"
    with open(config_file, "w", encoding="utf-8") as f: # toml.dump requires text mode
        toml.dump(config_content, f)
    return config_file

class TestLoadConfiguration:

    @patch('src.config_utils.load_dotenv')
    def test_load_configuration_success(self, mock_load_dotenv, temp_config_file, mock_console, monkeypatch):
        # Change CWD to where temp_config_file is, so "config.toml" is found
        monkeypatch.chdir(temp_config_file.parent)
        
        config_utils.load_configuration(mock_console)
        
        mock_load_dotenv.assert_called_once()
        assert config_utils.CONFIG_FROM_TOML["litellm"]["model"] == "toml_model"
        assert config_utils.CONFIG_FROM_TOML["ui"]["reasoning_style"] == "compact"
        mock_console.print.assert_not_called() # No warnings

    @patch('src.config_utils.load_dotenv')
    def test_load_configuration_file_not_found(self, mock_load_dotenv, mock_console, tmp_path, monkeypatch):
        # Ensure no config.toml exists in tmp_path
        monkeypatch.chdir(tmp_path)
        
        config_utils.load_configuration(mock_console)
        
        mock_load_dotenv.assert_called_once()
        assert config_utils.CONFIG_FROM_TOML == {} # Should be empty
        mock_console.print.assert_not_called() # No warning for missing file

    @patch('src.config_utils.load_dotenv')
    def test_load_configuration_toml_decode_error(self, mock_load_dotenv, tmp_path, mock_console, monkeypatch):
        config_file = tmp_path / "config.toml"
        config_file.write_text("this is not valid toml content {")
        monkeypatch.chdir(tmp_path)

        config_utils.load_configuration(mock_console)

        mock_load_dotenv.assert_called_once()
        assert config_utils.CONFIG_FROM_TOML == {}

        expected_message = "[yellow]Warning: Could not parse config.toml: Expected '=' after a key in a key/value pair (at line 1, column 6). Using defaults and environment variables.[/yellow]"

        # Check if any call to print had the expected message string
        found_expected_call = False
        for call_args in mock_console.print.call_args_list:
            # Check positional arguments (the message is the first one)
            if call_args[0] and str(call_args[0][0]) == expected_message:
                found_expected_call = True
                break

        assert found_expected_call, f"Expected console print call not found: '{expected_message}'\nActual calls: {mock_console.print.call_args_list}"

    @patch('src.config_utils.load_dotenv')
    @patch('builtins.open', side_effect=OSError("Test permission error"))
    def test_load_configuration_generic_error(self, mock_open, mock_load_dotenv, tmp_path, mock_console, monkeypatch):
        # Create a dummy config.toml so Path("config.toml").exists() is true
        config_file = tmp_path / "config.toml"
        config_file.touch()
        monkeypatch.chdir(tmp_path)

        config_utils.load_configuration(mock_console)

        mock_load_dotenv.assert_called_once()
        assert config_utils.CONFIG_FROM_TOML == {}
        mock_console.print.assert_any_call(
            "[yellow]Warning: Error loading config.toml: Test permission error. Using defaults and environment variables.[/yellow]"
        )


class TestGetConfigValue:

    # Test data: (param_name, env_value, toml_value_in_config, runtime_value, default_value, expected_type)
    # Note: toml_value_in_config will be set up by temp_config_file fixture
    test_params_data = [
        ("model", "env_model", "toml_model", "runtime_model", "default_model_code", str),
        ("api_base", "http://env.api/v1", "http://toml.api.base/v1", "http://runtime.api/v1", "http://default.api/v1", str),
        ("reasoning_style", "full", "compact", "silent", "medium", str), # medium is not in allowed, so default_reasoning_style from TOML/code
        ("max_tokens", "3000", 2048, "4000", 8192, int),
        ("reasoning_effort", "low", "high", "medium", "low", str),
        ("reply_effort", "high", "low", "medium", "high", str),
        ("temperature", "0.3", 0.5, "0.9", 0.7, float),
    ]

    @pytest.mark.parametrize("param_name, env_value, _, runtime_value, default_value, expected_type", test_params_data)
    def test_get_config_value_runtime_override(self, param_name, env_value, _, runtime_value, default_value, expected_type, monkeypatch, temp_config_file, mock_console):
        monkeypatch.setenv(config_utils.SUPPORTED_SET_PARAMS[param_name]["env_var"], str(env_value))
        # Load TOML so it's present but lower precedence
        monkeypatch.chdir(temp_config_file.parent)
        config_utils.load_configuration(mock_console)
        
        config_utils.RUNTIME_OVERRIDES[param_name] = runtime_value
        
        val = config_utils.get_config_value(param_name, default_value, mock_console)
        
        expected_val = runtime_value
        if expected_type is int and isinstance(runtime_value, str): expected_val = int(runtime_value)
        if expected_type is float and isinstance(runtime_value, str): expected_val = float(runtime_value)
        
        assert val == expected_val
        assert isinstance(val, expected_type)

    @pytest.mark.parametrize("param_name, env_value, _, runtime_value, default_value, expected_type", test_params_data)
    def test_get_config_value_env_variable(self, param_name, env_value, _, runtime_value, default_value, expected_type, monkeypatch, temp_config_file, mock_console):
        monkeypatch.setenv(config_utils.SUPPORTED_SET_PARAMS[param_name]["env_var"], str(env_value))
        # Load TOML so it's present but lower precedence
        monkeypatch.chdir(temp_config_file.parent)
        config_utils.load_configuration(mock_console)

        # Ensure no runtime override for this test
        config_utils.RUNTIME_OVERRIDES = {}

        val = config_utils.get_config_value(param_name, default_value, mock_console)
        
        expected_val = env_value
        if param_name in ["reasoning_style", "reasoning_effort", "reply_effort"]:
            expected_val = env_value.lower() # These are lowercased if from env
        elif expected_type is int: expected_val = int(env_value)
        elif expected_type is float: expected_val = float(env_value)

        assert val == expected_val
        assert isinstance(val, expected_type)

    @pytest.mark.parametrize("param_name, _, toml_value_in_config, runtime_value, default_value, expected_type", test_params_data)
    def test_get_config_value_toml_config(self, param_name, _, toml_value_in_config, runtime_value, default_value, expected_type, monkeypatch, temp_config_file, mock_console):
        # Env var for this param should NOT be set for this test
        # Runtime override should NOT be set
        config_utils.RUNTIME_OVERRIDES = {}
        
        monkeypatch.chdir(temp_config_file.parent)
        config_utils.load_configuration(mock_console) # This loads the TOML

        val = config_utils.get_config_value(param_name, default_value, mock_console)
        
        # Expected value comes from the temp_config_file fixture's content
        p_config = config_utils.SUPPORTED_SET_PARAMS[param_name]
        expected_toml_val = config_utils.CONFIG_FROM_TOML[p_config["toml_section"]][p_config["toml_key"]]

        assert val == expected_toml_val
        assert isinstance(val, expected_type)


    @pytest.mark.parametrize("param_name, _, __, runtime_value, default_value, expected_type", test_params_data)
    def test_get_config_value_default_value(self, param_name, _, __, runtime_value, default_value, expected_type, monkeypatch, tmp_path, mock_console):
        # Ensure no env var, no TOML, no runtime override
        config_utils.RUNTIME_OVERRIDES = {}
        # Change to a dir without config.toml
        monkeypatch.chdir(tmp_path)
        config_utils.load_configuration(mock_console) # CONFIG_FROM_TOML will be {}

        val = config_utils.get_config_value(param_name, default_value, mock_console)
        assert val == default_value
        assert isinstance(val, expected_type)

    def test_get_config_value_max_tokens_invalid_env(self, monkeypatch, mock_console):
        monkeypatch.setenv("LITELLM_MAX_TOKENS", "not_an_int")
        val = config_utils.get_config_value("max_tokens", 4096, mock_console)
        assert val == 4096 # Should fall back to default

    def test_get_config_value_temperature_invalid_env(self, monkeypatch, mock_console):
        monkeypatch.setenv("LITELLM_TEMPERATURE", "not_a_float")
        val = config_utils.get_config_value("temperature", 0.7, mock_console)
        assert val == 0.7 # Should fall back to default

    @pytest.mark.parametrize("param_name, allowed_values_key", [
        ("reasoning_style", "allowed_values"),
        ("reasoning_effort", "allowed_values"),
        ("reply_effort", "allowed_values"),
    ])
    def test_get_config_value_env_invalid_allowed_value(self, param_name, allowed_values_key, monkeypatch, temp_config_file, mock_console):
        p_config = config_utils.SUPPORTED_SET_PARAMS[param_name]
        monkeypatch.setenv(p_config["env_var"], "invalid_value_for_env")
        
        # Load TOML so it's present as a fallback
        monkeypatch.chdir(temp_config_file.parent)
        config_utils.load_configuration(mock_console)

        # The default value passed to get_config_value is the ultimate fallback
        code_default = "some_code_default" 
        
        val = config_utils.get_config_value(param_name, code_default, mock_console)
        
        # Expected value should be from TOML because env was invalid
        expected_toml_val = config_utils.CONFIG_FROM_TOML[p_config["toml_section"]][p_config["toml_key"]]
        assert val == expected_toml_val

    def test_get_config_value_env_valid_allowed_value_mixed_case(self, monkeypatch, mock_console):
        param_name = "reasoning_style"
        p_config = config_utils.SUPPORTED_SET_PARAMS[param_name]
        monkeypatch.setenv(p_config["env_var"], "FuLl") # Mixed case
        
        val = config_utils.get_config_value(param_name, "default_style", mock_console)
        assert val == "full" # Should be lowercased and match

    def test_get_config_value_no_param_in_supported_set_params(self, mock_console):
        with pytest.raises(KeyError):
            config_utils.get_config_value("non_existent_param", "default", mock_console)
