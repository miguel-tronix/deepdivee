from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=False)


def render(template_name: str, **kwargs) -> str:
    template = _env.get_template(template_name)
    return template.render(**kwargs)
