"""Prompt text lives in ai/prompts/*.md so it can be read and edited apart from the code."""
from functools import cache
from importlib.resources import files


@cache
def load_prompt(name: str) -> str:
    return files("marketplace_agent.ai.prompts").joinpath(f"{name}.md").read_text(encoding="utf-8")
