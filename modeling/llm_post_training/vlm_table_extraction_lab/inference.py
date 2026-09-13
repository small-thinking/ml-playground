"""Optional in-process Transformers inference; no training or hosted API."""

from time import perf_counter

PROMPT = (
    "Extract the table from this image as HTML. Return exactly one <table> with "
    "all rows and cells, preserving rowspan and colspan. Preserve text, signs, "
    "numbers and units. Do not add explanations or Markdown fences."
)


class TransformersPredictor:
    def __init__(self, model_path, revision, device, max_new_tokens, max_pixels):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.max_pixels = max_pixels
        self.processor = AutoProcessor.from_pretrained(model_path, revision=revision)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                model_path, revision=revision, torch_dtype="auto"
            )
            .to(device)
            .eval()
        )
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def __call__(self, image_path):
        from PIL import Image

        with Image.open(image_path) as source:
            im = source.convert("RGB")
        if im.width * im.height > self.max_pixels:
            scale = (self.max_pixels / (im.width * im.height)) ** 0.5
            im = im.resize(
                (max(1, int(im.width * scale)), max(1, int(im.height * scale)))
            )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": im},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        start = perf_counter()
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        tokens = output[0, inputs["input_ids"].shape[-1] :]
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        reason = "stop" if len(tokens) and int(tokens[-1]) in eos else "length"
        return {
            "html": self.processor.decode(tokens, skip_special_tokens=True),
            "input_tokens": int(inputs["input_ids"].shape[-1]),
            "output_tokens": len(tokens),
            "stop_reason": reason,
            "latency_seconds": perf_counter() - start,
        }


class MLXPredictor:
    """Apple Metal inference using the same original, unquantized HF snapshot."""

    def __init__(self, model_path, revision, device, max_new_tokens, max_pixels):
        from pathlib import Path

        if device not in {"auto", "mps"}:
            raise ValueError("MLX backend requires Apple Metal (--device mps or auto)")
        import mlx.core as mx
        from mlx_vlm import load

        if not Path(model_path).is_dir():
            raise ValueError(
                "MLX requires an explicitly downloaded local model directory"
            )
        self.mx = mx
        mx.set_default_device(mx.gpu)
        self.model, self.processor = load(model_path)
        self.max_new_tokens = max_new_tokens
        self.max_pixels = max_pixels
        self.revision = revision

    def __call__(self, image_path):
        from PIL import Image
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        with Image.open(image_path) as source:
            image = source.convert("RGB")
        if image.width * image.height > self.max_pixels:
            scale = (self.max_pixels / (image.width * image.height)) ** 0.5
            image = image.resize(
                (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
            )
        prompt = apply_chat_template(
            self.processor,
            self.model.config,
            PROMPT,
            num_images=1,
            enable_thinking=False,
        )
        start = perf_counter()
        result = generate(
            self.model,
            self.processor,
            prompt,
            [image],
            max_tokens=self.max_new_tokens,
            temperature=0.0,
            verbose=False,
        )
        self.mx.synchronize()
        elapsed = perf_counter() - start
        self.mx.clear_cache()
        return {
            "html": result.text,
            "input_tokens": result.prompt_tokens,
            "output_tokens": result.generation_tokens,
            "stop_reason": result.finish_reason,
            "latency_seconds": elapsed,
        }
