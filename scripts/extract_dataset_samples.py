"""One-shot helper: pull one sample image + Q/A from each dataset for slides."""

from __future__ import annotations

import logging
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_coco_captions, load_scienceqa, load_textvqa  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")

OUT_DIR = REPO_ROOT / "slides" / "Images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_PX = 480


def save_sample(name: str, sample, idx: int = 0) -> None:
    img = sample.image.copy()
    img.thumbnail((MAX_PX, MAX_PX))
    img.save(OUT_DIR / f"sample_{name}.jpg", quality=88)

    caption = textwrap.shorten(sample.answer.strip(), width=140, placeholder="...")
    question = textwrap.shorten(sample.question.strip(), width=140, placeholder="...")
    (OUT_DIR / f"sample_{name}.txt").write_text(
        f"Q: {question}\nA: {caption}\n", encoding="utf-8"
    )
    print(f"[{name}] image={img.size} Q={question!r} A={caption!r}")


def main() -> None:
    save_sample("scienceqa", load_scienceqa(num_samples=1)[0])
    # Sample 5 is a cleaner illustrative example than 0 ("nous les gosses").
    save_sample("textvqa", load_textvqa(num_samples=10)[5])
    save_sample("coco_caption", load_coco_captions(num_samples=1)[0])


if __name__ == "__main__":
    main()
