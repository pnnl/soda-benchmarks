"""Registry management for sb-cli experiments.

Manages experiments/registry.py — a plain Python file containing
EXPERIMENTS: dict[str, str] mapping logical names to directory paths.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REGISTRY_FILENAME = "experiments/registry.py"
_REGISTRY_TEMPLATE = """\
# sb-cli experiment registry
# Maps experiment names to their directory paths (relative to benches/).
# Comment out entries to exclude from 'sb-cli collect'.
EXPERIMENTS: dict[str, str] = {
}
"""


class Registry:
    """Load, append to, and resolve entries in experiments/registry.py."""

    def __init__(self, base_dir: Path) -> None:
        self.path = base_dir / _REGISTRY_FILENAME

    def _ensure_exists(self) -> None:
        """Create registry file with empty EXPERIMENTS dict if absent."""
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(_REGISTRY_TEMPLATE, encoding="utf-8")

    def load(self) -> dict[str, str]:
        """Load and return the EXPERIMENTS dict from registry.py.

        Returns:
            Dict mapping experiment names to paths.
        """
        self._ensure_exists()
        spec = importlib.util.spec_from_file_location("_sb_cli_registry", self.path)
        if spec is None or spec.loader is None:
            return {}
        mod = importlib.util.module_from_spec(spec)
        # Use a unique module name to avoid caching stale registry state
        mod_name = f"_sb_cli_registry_{id(self)}"
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
        except SyntaxError as exc:
            print(f"[sb-cli] ERROR: experiments/registry.py has a syntax error: {exc}")
            print("[sb-cli] Please fix the file manually before proceeding.")
            raise SystemExit(1) from exc
        finally:
            sys.modules.pop(mod_name, None)
        return getattr(mod, "EXPERIMENTS", {})

    def append(self, name: str, rel_path: str) -> None:
        """Append a new entry to the EXPERIMENTS dict in registry.py.

        Inserts before the closing '}' of the dict.

        Args:
            name: Logical experiment name.
            rel_path: Relative path string (relative to base_dir).
        """
        self._ensure_exists()
        text = self.path.read_text(encoding="utf-8")
        entry_line = f'    "{name}": "{rel_path}",\n'

        # Find the last closing brace and insert before it
        close_idx = text.rfind("}")
        if close_idx == -1:
            # Fallback: append entry_line before end
            text = text.rstrip() + "\n" + entry_line + "}\n"
        else:
            text = text[:close_idx] + entry_line + text[close_idx:]

        self.path.write_text(text, encoding="utf-8")

    def resolve(self, name_or_path: str, base_dir: Path) -> Path:
        """Resolve a name or path to an absolute experiment directory.

        First looks up name in the registry, then falls back to treating
        name_or_path as a filesystem path.

        Args:
            name_or_path: Registered experiment name or filesystem path.
            base_dir: Base directory (benches/) for relative path resolution.

        Returns:
            Absolute Path to the experiment directory.

        Raises:
            SystemExit: If name not found in registry and path doesn't exist.
        """
        experiments = self.load()
        if name_or_path in experiments:
            candidate = Path(experiments[name_or_path])
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            return candidate.resolve()

        # Fallback: treat as filesystem path
        candidate = Path(name_or_path)
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        candidate = candidate.resolve()
        if not candidate.is_dir():
            print(
                f"[sb-cli] ERROR: '{name_or_path}' not found in registry "
                "and is not a valid directory path."
            )
            raise SystemExit(1)
        return candidate
