"""Dataset loading for VLM profiling: ScienceQA, TextVQA, COCO Captions."""

import logging
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from PIL import Image

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data_cache"


@dataclass(frozen=True)
class VLMSample:
    image: Image.Image
    question: str
    answer: str
    dataset_name: str
    sample_id: int


def load_scienceqa(num_samples: int = 300) -> list[VLMSample]:
    """Load ScienceQA validation samples that have images."""
    logger.info("Loading ScienceQA (filtering for image-based questions)...")
    ds = load_dataset(
        "derek-thomas/ScienceQA", split="validation", cache_dir=str(CACHE_DIR)
    )

    samples = []
    for idx, row in enumerate(ds):
        if len(samples) >= num_samples:
            break
        if row.get("image") is None:
            continue
        img = row["image"]
        if not isinstance(img, Image.Image):
            continue
        samples.append(
            VLMSample(
                image=img.convert("RGB"),
                question=row["question"],
                answer=str(row.get("answer", "")),
                dataset_name="scienceqa",
                sample_id=idx,
            )
        )

    logger.info("ScienceQA: loaded %d samples", len(samples))
    return samples


def load_textvqa(num_samples: int = 300) -> list[VLMSample]:
    """Load TextVQA validation samples."""
    logger.info("Loading TextVQA...")
    ds = load_dataset("lmms-lab/textvqa", split="validation", cache_dir=str(CACHE_DIR))

    samples = []
    for idx, row in enumerate(ds):
        if len(samples) >= num_samples:
            break
        img = row.get("image")
        if img is None or not isinstance(img, Image.Image):
            continue
        answers = row.get("answers", [])
        answer = answers[0] if answers else ""
        samples.append(
            VLMSample(
                image=img.convert("RGB"),
                question=row["question"],
                answer=str(answer),
                dataset_name="textvqa",
                sample_id=idx,
            )
        )

    logger.info("TextVQA: loaded %d samples", len(samples))
    return samples


def load_coco_captions(num_samples: int = 300) -> list[VLMSample]:
    """Load COCO Caption validation samples."""
    logger.info("Loading COCO Captions...")
    ds = load_dataset(
        "lmms-lab/COCO-Caption", split="val", cache_dir=str(CACHE_DIR)
    )

    samples = []
    for idx, row in enumerate(ds):
        if len(samples) >= num_samples:
            break
        img = row.get("image")
        if img is None or not isinstance(img, Image.Image):
            continue
        raw_caption = row.get("answer", row.get("caption", ""))
        if isinstance(raw_caption, list):
            caption = raw_caption[0] if raw_caption else ""
        else:
            caption = str(raw_caption)
        samples.append(
            VLMSample(
                image=img.convert("RGB"),
                question="Describe this image.",
                answer=str(caption),
                dataset_name="coco_caption",
                sample_id=idx,
            )
        )

    logger.info("COCO Captions: loaded %d samples", len(samples))
    return samples


DATASET_LOADERS = {
    "scienceqa": load_scienceqa,
    "textvqa": load_textvqa,
    "coco_caption": load_coco_captions,
}


def load_dataset_by_name(name: str, num_samples: int = 300) -> list[VLMSample]:
    """Load a dataset by name."""
    loader = DATASET_LOADERS.get(name)
    if loader is None:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_LOADERS)}")
    return loader(num_samples=num_samples)
