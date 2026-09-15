"""Hosted Qwen3.5-4B sampling; labels never enter the prediction interface."""

from importlib.metadata import version
from pathlib import Path
import subprocess
import sys
from threading import Lock
from time import perf_counter

from .inference import PROMPT

COOKBOOK_REVISION = "485726f55d3b2b5abe5fcb4a0d2f3e18e4599dfe"
PROCESSOR_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SEED = 20260913
TEACHER_MODEL = "Qwen/Qwen3.6-35B-A3B"
TEACHER_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
PROCESSOR_REVISIONS = {
    "Qwen/Qwen3.5-4B": PROCESSOR_REVISION,
    TEACHER_MODEL: TEACHER_REVISION,
}


def verify_cookbook(directory):
    directory = Path(directory).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip()
    if revision != COOKBOOK_REVISION or dirty:
        raise ValueError(
            "Tinker cookbook must be a clean checkout of the pinned revision"
        )
    return directory


def load_renderer(model, revision, cookbook_dir):
    """Load the shared training/inference template without contacting Tinker."""
    source = verify_cookbook(cookbook_dir)
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(c not in "0123456789abcdef" for c in revision.lower())
    ):
        raise ValueError(
            "Tinker processor revision must be a full 40-character commit SHA"
        )
    if model not in PROCESSOR_REVISIONS or revision != PROCESSOR_REVISIONS[model]:
        raise ValueError(
            "Use an explicitly supported model and pinned processor revision"
        )
    sys.path.insert(0, str(source))
    import tinker_cookbook.renderers as renderers
    from transformers import AutoImageProcessor, AutoTokenizer

    if not Path(renderers.__file__).resolve().is_relative_to(source):
        raise ValueError("A different Tinker cookbook was already imported")
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    processor = AutoImageProcessor.from_pretrained(model, revision=revision)
    return tokenizer, renderers.get_renderer(
        "qwen3_5_disable_thinking", tokenizer, image_processor=processor
    )


def image_message(image_path, max_pixels):
    from PIL import Image

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    if image.width * image.height > max_pixels:
        scale = (max_pixels / (image.width * image.height)) ** 0.5
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        )
    return {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }


class TinkerPredictor:
    def __init__(self, model, revision, max_new_tokens, max_pixels, cookbook_dir):
        self.tokenizer, self.renderer = load_renderer(model, revision, cookbook_dir)
        import tinker

        self.processor_revision = revision
        self.revision = (
            None  # The sampling API does not expose hosted weight revisions.
        )
        self.sdk_version = version("tinker")
        self.sdk = tinker
        self.client = tinker.ServiceClient().create_sampling_client(base_model=model)
        self.max_new_tokens = max_new_tokens
        self.max_pixels = max_pixels
        self.stop = self.renderer.get_stop_sequences()
        self.prepare_lock = Lock()

    def __call__(self, image_path):
        start = perf_counter()
        # Tokenizer/renderer preparation is serialized; hosted requests overlap.
        with self.prepare_lock:
            prompt = self.renderer.build_generation_prompt(
                [image_message(image_path, self.max_pixels)]
            )
        result = self.client.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=self.sdk.SamplingParams(
                max_tokens=self.max_new_tokens, temperature=0, seed=SEED, stop=self.stop
            ),
        ).result(timeout=600)
        sequence = result.sequences[0]
        with self.prepare_lock:
            html = self.tokenizer.decode(sequence.tokens, skip_special_tokens=True)
        return {
            "html": html,
            "input_tokens": prompt.length,
            "output_tokens": len(sequence.tokens),
            "cached_input_tokens": result.prompt_cache_hit_tokens,
            "stop_reason": sequence.stop_reason,
            "latency_seconds": perf_counter() - start,
        }
