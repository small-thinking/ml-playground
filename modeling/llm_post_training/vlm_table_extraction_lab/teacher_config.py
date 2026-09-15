"""Pinned visual teachers and uncached 64K Tinker rates (USD / million tokens)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TeacherSpec:
    model: str
    revision: str
    forward_rate: float
    sample_rate: float


DEFAULT_TEACHER_MODEL = "Qwen/Qwen3.6-35B-A3B"
LARGE_TEACHER_MODEL = "Qwen/Qwen3.5-397B-A17B"
TEACHERS = {
    DEFAULT_TEACHER_MODEL: TeacherSpec(
        DEFAULT_TEACHER_MODEL, "995ad96eacd98c81ed38be0c5b274b04031597b0", 0.54, 1.335
    ),
    LARGE_TEACHER_MODEL: TeacherSpec(
        LARGE_TEACHER_MODEL, "8472618112abcbd45acbcdc58436aff4233c23f7", 3.0, 7.5
    ),
}


def teacher_spec(model=DEFAULT_TEACHER_MODEL):
    """Reject unpriced/unpinned models rather than borrowing another model's rate."""
    if model not in TEACHERS:
        raise ValueError(f"Unsupported teacher model: {model}")
    return TEACHERS[model]
