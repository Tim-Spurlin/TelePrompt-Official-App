# utils/file_utils.py
import yaml

def load_config(filepath):
    """Loads the configuration from a YAML file."""
    try:
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found at {filepath}, creating with defaults.")
        return {}  # Return an empty dict if file not found
    except yaml.YAMLError as e:
        print(f"Error parsing YAML in {filepath}: {e}")
        return {}

def save_config(config_data, filepath):
    """Saves the configuration to a YAML file."""
    try:
        with open(filepath, 'w') as f:
            yaml.safe_dump(config_data, f)
    except Exception as e:
        print(f"Error saving config to {filepath}: {e}")