"""Image preprocessing and prompt generation for VLM profiling."""

from PIL import Image


def resize_image(image: Image.Image, resolution: int) -> Image.Image:
    """Resize image to resolution x resolution, preserving RGB mode."""
    return image.resize((resolution, resolution), Image.LANCZOS)


def generate_prompt(base_question: str, target_token_count: int) -> str:
    """Generate a prompt of approximately target_token_count tokens.

    Pads the base question with context filler to reach the desired length.
    A rough heuristic: 1 token ~ 4 characters for English text.
    """
    if target_token_count <= 0:
        return base_question

    base_char_count = len(base_question)
    target_chars = target_token_count * 4

    if base_char_count >= target_chars:
        return base_question

    padding_phrases = [
        "Please provide a detailed answer.",
        "Consider all visible elements in the image.",
        "Think step by step before answering.",
        "Take into account the context and background.",
        "Be precise and specific in your response.",
        "Look carefully at every detail shown.",
        "Explain your reasoning thoroughly.",
        "Include relevant observations about the scene.",
    ]

    result = base_question
    phrase_idx = 0
    while len(result) < target_chars:
        result += " " + padding_phrases[phrase_idx % len(padding_phrases)]
        phrase_idx += 1

    return result
