"""Dataset loader registry.

Use :func:`get_loader` so the sampling script stays dataset-agnostic::

    from data.loaders import get_loader
    records = get_loader("musique").load("data/raw/musique/musique_ans_v1.0_dev.jsonl")
"""
from __future__ import annotations

from .base import (
    BaseLoader,
    GoldEvidence,
    GoldStep,
    LoaderError,
    Paragraph,
    QuestionRecord,
    download_file,
    paragraph_id,
    read_json_records,
    read_question_records,
    write_jsonl,
)
from .hotpotqa import HotpotQALoader
from .musique import MusiqueLoader
from .twiki import TwikiLoader

__all__ = [
    "BaseLoader",
    "GoldEvidence",
    "GoldStep",
    "LoaderError",
    "Paragraph",
    "QuestionRecord",
    "HotpotQALoader",
    "MusiqueLoader",
    "TwikiLoader",
    "LOADERS",
    "get_loader",
    "download_file",
    "paragraph_id",
    "read_json_records",
    "read_question_records",
    "write_jsonl",
]

LOADERS = {cls.name: cls() for cls in (MusiqueLoader, TwikiLoader, HotpotQALoader)}


def get_loader(name: str) -> BaseLoader:
    """Return the loader registered for ``name``."""
    try:
        return LOADERS[name]
    except KeyError:
        raise KeyError(
            f"unknown dataset {name!r}; choose from {sorted(LOADERS)}"
        ) from None
