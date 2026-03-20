import json
from src.setup import SETTINGS_FILE


def load_settings():
    """Reads the current user settings from the JSON file."""
    if not SETTINGS_FILE.exists():
        return {}
    with open(SETTINGS_FILE, "r") as f:
        return json.load(f)


def update_allocation(conservative: int, moderate: int, aggressive: int):
    """Updates the risk allocation in the settings file."""
    settings = load_settings()

    settings["risk_allocation"] = {
        "conservative": conservative,
        "moderate": moderate,
        "aggressive": aggressive,
    }

    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)
