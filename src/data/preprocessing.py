"""Image preprocessing and prompt generation for VLM profiling."""

from PIL import Image

# Model-specific prompt templates. {question} is replaced with the actual question.
# OPT models need "Question: {} Answer:" format (BLIP-2 paper, Section 4.1)
# Vicuna/LLaMA models need chat-style templates
# FlanT5 models work with plain questions (encoder-decoder, no prompt echo)
PROMPT_TEMPLATES: dict[str, str] = {
    # BLIP-2 OPT (decoder-only, needs explicit QA format)
    "Salesforce/blip2-opt-2.7b": "Question: {question} Answer:",
    # InstructBLIP Vicuna (decoder-only, plain question — processor handles formatting)
    # Note: adding "Answer:" suffix causes garbage generation with this model
    "Salesforce/instructblip-vicuna-7b": "{question}",
    # LLaVA (decoder-only, needs <image> token)
    "llava-hf/llava-1.5-7b-hf": "USER: <image>\n{question}\nASSISTANT:",
    "llava-hf/llava-1.5-13b-hf": "USER: <image>\n{question}\nASSISTANT:",
    # Idefics2 (decoder-only, needs <image> token)
    "HuggingFaceM4/idefics2-8b": "User:<image>{question}<end_of_utterance>\nAssistant:",
    # Fuyu (decoder-only, plain question works)
    "adept/fuyu-8b": "{question}\n",
    # FlanT5 models: no template needed (encoder-decoder)
}


def resize_image(image: Image.Image, resolution: int) -> Image.Image:
    """Resize image to resolution x resolution, preserving RGB mode."""
    return image.resize((resolution, resolution), Image.LANCZOS)


def format_prompt_for_model(prompt: str, model_name: str) -> str:
    """Wrap prompt in model-specific template if needed."""
    template = PROMPT_TEMPLATES.get(model_name)
    if template:
        return template.format(question=prompt)
    return prompt


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
