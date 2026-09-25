"""PyECLOUD: version metadata comes from pyproject.toml at installation."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("PyECLOUD")
except PackageNotFoundError:
    # Legacy in-place builds can be imported without an installed distribution.
    __version__ = "unknown (not installed)"
