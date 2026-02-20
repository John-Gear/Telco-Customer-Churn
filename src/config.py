from pathlib import Path
import yaml

def load_config(config_path: str | Path = 'config.yaml') -> dict:
    config_path = Path(config_path)
    with con