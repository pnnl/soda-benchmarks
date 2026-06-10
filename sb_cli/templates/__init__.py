"""Template rendering utilities for sb-cli file generation.

Uses Python's string.Template for $variable substitution in .tmpl files.
"""

from __future__ import annotations

from pathlib import Path
from string import Template

_TEMPLATE_DIR = Path(__file__).parent


def render(template_name: str, context: dict[str, object]) -> str:
    """Render a template file with the given context variables.

    Args:
        template_name: Filename of the template (e.g. 'flow.py.tmpl').
        context: Dict of variable names to values for substitution.

    Returns:
        Rendered string with all $variable placeholders replaced.

    Raises:
        FileNotFoundError: If the template file does not exist.
    """
    tmpl_path = _TEMPLATE_DIR / template_name
    template_text = tmpl_path.read_text(encoding="utf-8")
    return Template(template_text).safe_substitute(context)
