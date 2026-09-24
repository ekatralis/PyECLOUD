"""Version and optional source-checkout provenance; Git is never required."""
from pathlib import Path
import subprocess
from ._version import __version__


def git_provenance():
    root = Path(__file__).resolve().parent
    if not (root / ".git").exists():
        return "git hash: unavailable (installed package)", "git branch: unavailable (installed package)"
    def query(*args):
        try:
            return subprocess.check_output(["git", "-C", str(root), *args],
                stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            return "unavailable"
    return "git hash: " + query("rev-parse", "HEAD"), "git branch: " + query("rev-parse", "--abbrev-ref", "HEAD")
