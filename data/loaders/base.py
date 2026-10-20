"""Shared types and IO helpers for the dataset loaders (Role 1).

Every loader normalizes its raw dataset into :class:`QuestionRecord`, the single
schema consumed by graph construction (Roles 2 and 3) and evaluation (Role 8):

    {
      "question_id": str,
      "dataset": str,
      "hop_level": int,
      "question": str,
      "answer": str,
      "paragraphs": [{"id": str, "title": str, "text": str}],
      "gold_evidence": {
        "paragraph_ids": [str],
        "sentence_facts": [{"paragraph_id": str, "sentence_index": int}]
      },
      "gold_chain": [
        {"step": int, "head": str|null, "relation": str,
         "tail": str|null, "support_paragraph_ids": [str]}
      ],
      "metadata": dict
    }

Raw dataset files must never be committed: ``.gitignore`` keeps ``data/*`` out
of git except for the loader code itself.

Paragraph ids are ``"<dataset>::<title>"`` so the same source paragraph has the
same id across questions; that makes them usable as edge provenance.
"""
from __future__ import annotations

import gzip
import json
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


class LoaderError(Exception):
    """Raised when a dataset file is missing, malformed, or cannot be normalized."""


# --------------------------------------------------------------------------- #
# Record schema
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Paragraph:
    """One source paragraph of the corpus."""

    id: str
    title: str
    text: str

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "text": self.text}


@dataclass(frozen=True)
class GoldStep:
    """One step of a gold reasoning chain.

    ``relation`` is the natural-language step (a MuSiQue sub-question or a
    2Wiki ``evidences`` relation). ``head``/``tail`` are optional because some
    datasets only annotate the sub-question and its answer.
    """

    step: int
    relation: str
    tail: Optional[str] = None
    head: Optional[str] = None
    support_paragraph_ids: tuple = ()

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
            "support_paragraph_ids": list(self.support_paragraph_ids),
        }


@dataclass(frozen=True)
class GoldEvidence:
    """Gold evidence used for evidence recall/precision and path recall."""

    paragraph_ids: tuple = ()
    sentence_facts: tuple = ()

    def to_dict(self) -> dict:
        return {
            "paragraph_ids": list(self.paragraph_ids),
            "sentence_facts": [dict(f) for f in self.sentence_facts],
        }


@dataclass
class QuestionRecord:
    """Normalized record shared by every dataset."""

    question_id: str
    dataset: str
    hop_level: int
    question: str
    answer: str
    paragraphs: list
    gold_evidence: GoldEvidence
    gold_chain: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "dataset": self.dataset,
            "hop_level": self.hop_level,
            "question": self.question,
            "answer": self.answer,
            "paragraphs": [p.to_dict() for p in self.paragraphs],
            "gold_evidence": self.gold_evidence.to_dict(),
            "gold_chain": [s.to_dict() for s in self.gold_chain],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "QuestionRecord":
        gold = raw.get("gold_evidence") or {}
        steps = []
        for s in raw.get("gold_chain") or []:
            steps.append(
                GoldStep(
                    step=int(s.get("step", len(steps) + 1)),
                    relation=str(s.get("relation", "")),
                    tail=s.get("tail"),
                    head=s.get("head"),
                    support_paragraph_ids=tuple(s.get("support_paragraph_ids") or ()),
                )
            )
        return cls(
            question_id=str(raw["question_id"]),
            dataset=str(raw["dataset"]),
            hop_level=int(raw["hop_level"]),
            question=str(raw.get("question", "")),
            answer=str(raw.get("answer", "")),
            paragraphs=[Paragraph(**p) for p in raw.get("paragraphs") or []],
            gold_evidence=GoldEvidence(
                paragraph_ids=tuple(gold.get("paragraph_ids") or ()),
                sentence_facts=tuple(gold.get("sentence_facts") or ()),
            ),
            gold_chain=steps,
            metadata=dict(raw.get("metadata") or {}),
        )


# --------------------------------------------------------------------------- #
# Normalization helpers
# --------------------------------------------------------------------------- #
def paragraph_id(dataset: str, title: str) -> str:
    """Stable paragraph id shared across questions: ``<dataset>::<title>``."""
    return f"{dataset}::{str(title).strip()}"


def infer_hop(raw: dict, default: int = 2) -> int:
    """Read an explicit hop count if the dataset provides one."""
    for key in ("hop", "num_hops", "hop_level", "num_hop"):
        value = raw.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return default


def record_id(raw: dict, dataset: str, index: int) -> str:
    """Use the dataset's own id when present, otherwise a stable fallback."""
    for key in ("id", "_id", "question_id", "qid"):
        value = raw.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return f"{dataset}_{index}"


def dedupe(items: Iterable[Any]) -> list:
    """Order-preserving de-duplication for hashable items."""
    return list(dict.fromkeys(items))


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def read_json_records(path: str | Path) -> list:
    """Read a ``.json``/``.jsonl`` file (optionally gzipped) as a list of dicts."""
    p = Path(path)
    if not p.exists():
        raise LoaderError(
            f"dataset file not found: {p}\n"
            "Download the raw dataset and point `path` at it in configs/data.yaml."
        )

    name = p.name.lower()
    is_gzip = name.endswith(".gz")
    inner = name[:-3] if is_gzip else name
    opener = gzip.open if is_gzip else open

    if inner.endswith(".jsonl"):
        with opener(p, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    if not inner.endswith(".json"):
        raise LoaderError(f"unsupported dataset extension: {p.name} (expected .json or .jsonl)")

    with opener(p, "rt", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "questions", "records", "examples"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    raise LoaderError(f"could not find a list of records in {p}")


def write_jsonl(records: Iterable, path: str | Path) -> Path:
    """Write records (dataclasses or dicts) as JSON Lines."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for record in records:
            payload = record.to_dict() if isinstance(record, QuestionRecord) else record
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return p


def read_question_records(path: str | Path) -> list:
    """Read a JSON/JSONL file into :class:`QuestionRecord` objects."""
    return [QuestionRecord.from_dict(raw) for raw in read_json_records(path)]


def download_file(url: str, dest: str | Path, *, timeout: int = 60) -> Path:
    """Download ``url`` to ``dest`` atomically (writes a ``.part`` file first)."""
    p = Path(dest)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as response, open(tmp, "wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(p)
    return p


# --------------------------------------------------------------------------- #
# Loader base class
# --------------------------------------------------------------------------- #
class BaseLoader:
    """Base class: read a file, then normalize each raw record."""

    name: str = ""
    default_filename: str = ""
    default_url: Optional[str] = None

    def load(self, path: str | Path) -> list:
        return self.parse_records(read_json_records(path))

    def parse_records(self, raws: Iterable[dict]) -> list:
        records = []
        for index, raw in enumerate(raws):
            try:
                records.append(self.parse_record(raw, index))
            except LoaderError:
                raise
            except Exception as exc:  # noqa: BLE001 - add record context
                raise LoaderError(
                    f"{self.name}: failed to parse record {index}: {exc}"
                ) from exc
        return records

    def parse_record(self, raw: dict, index: int) -> QuestionRecord:
        raise NotImplementedError

    # -- small helpers shared by the concrete loaders --------------------- #
    def _paragraphs_from_context(self, context: Iterable) -> list:
        """Convert HotpotQA-style ``[[title, [sentences]], ...]`` context."""
        paragraphs = []
        for entry in context or []:
            title = str(entry[0]).strip()
            sentences = entry[1] if len(entry) > 1 else []
            if isinstance(sentences, (list, tuple)):
                text = " ".join(str(s) for s in sentences)
            else:
                text = str(sentences)
            paragraphs.append(
                Paragraph(id=paragraph_id(self.name, title), title=title, text=text)
            )
        return paragraphs
