"""Prompt templates used by the manuscript-aligned GoC-Mem pipeline."""
from pathlib import Path


PROMPT_DIR = Path(__file__).resolve().parent / "prompt_templates"


def load_prompt(name, **values):
    text = (PROMPT_DIR / name).read_text(encoding="utf-8")
    return text.format(**values)


def extraction_prompt(history, question):
    return load_prompt("claim_extraction.txt", history=history, question=question)


def verification_prompt(claims):
    return load_prompt("image_verification.txt", claims=claims)


def answering_prompt(question, memory):
    def lines(values):
        return "\n".join("- " + value for value in values) or "- None"
    return load_prompt(
        "memory_guided_answering.txt",
        question=question,
        supported=lines(memory["supported"]),
        uncertain=lines(memory["uncertain"]),
        corrections=lines(memory["corrections"]),
    )
