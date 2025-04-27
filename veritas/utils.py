"""
Utility functions for the Veritas package.
"""
from pathlib import Path
from veritas.config import BASE_DIR, DATA_DIR, MODELS_DIR, LOGS_DIR, SCRIPTS_DIR

def get_project_root() -> Path:
    """Get the absolute path to the project root directory."""
    return BASE_DIR

def get_model_path(model_name: str) -> Path:
    """Get the path to a model file."""
    return MODELS_DIR / model_name

def get_data_path(file_name: str) -> Path:
    """Get the path to a data file."""
    return DATA_DIR / file_name

def get_log_path(file_name: str) -> Path:
    """Get the path to a log file."""
    return LOGS_DIR / file_name

def get_script_path(script_name: str) -> Path:
    """Get the path to a script file."""
    return SCRIPTS_DIR / script_name

def ensure_directories():
    """Ensure all required directories exist."""
    for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, SCRIPTS_DIR]:
        directory.mkdir(exist_ok=True)

def resolve_path(path: str | Path, base_dir: Path = None) -> Path:
    """
    Resolve a path relative to a base directory.
    If the path is absolute, it will be returned as is.
    If base_dir is None, the project root will be used.
    """
    path = Path(path)
    if path.is_absolute():
        return path
    base_dir = base_dir or BASE_DIR
    return base_dir / path

def ensure_parent_dirs(path: Path):
    """Ensure the parent directories of a path exist."""
    path.parent.mkdir(parents=True, exist_ok=True) 