"""Gold-evidence helpers and record validation (Role 1).

The loaders build :class:`~data.loaders.base.GoldEvidence` and
:class:`~data.loaders.base.GoldStep` objects from each dataset's annotations.
This module adds the checks that catch the mistakes which silently poison later
metrics: dangling paragraph ids, duplicate ids, and missing gold evidence.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

from .loaders.base import GoldEvidence, GoldStep, QuestionRecord


class EvidenceError(Exception):
    """Raised when sampled records are internally inconsistent."""


def make_gold_evidence(paragraph_ids: Iterable[str], sentence_facts: Iterable[dict] = ()) -> GoldEvidence:
    """Build gold evidence, removing duplicate paragraph/fact entries."""
    pids = list(dict.fromkeys(str(p) for p in paragraph_ids))
    facts = []
    seen = set()
    for fact in sentence_facts:
        pid = str(fact.get("paragraph_id", ""))
        index = fact.get("sentence_index")
        key = (pid, index)
        if not pid or key in seen:
            continue
        seen.add(key)
        facts.append({"paragraph_id": pid, "sentence_index": index})
    return GoldEvidence(paragraph_ids=tuple(pids), sentence_facts=tuple(facts))


def make_gold_step(
    step: int,
    relation: str,
    *,
    head: str | None = None,
    tail: str | None = None,
    support_paragraph_ids: Iterable[str] = (),
) -> GoldStep:
    """Build one gold-chain step with de-duplicated support ids."""
    return GoldStep(
        step=int(step),
        relation=str(relation),
        head=head,
        tail=tail,
        support_paragraph_ids=tuple(dict.fromkeys(support_paragraph_ids)),
    )


def validate_record(record: QuestionRecord, *, require_gold: bool = True) -> list:
    """Return a list of human-readable problems with ``record`` (empty if valid)."""
    errors = []
    prefix = f"[{record.dataset}:{record.question_id}]"

    if not record.question.strip():
        errors.append(f"{prefix} empty question")
    if record.hop_level < 1:
        errors.append(f"{prefix} invalid hop_level {record.hop_level!r}")
    if not record.paragraphs:
        errors.append(f"{prefix} no paragraphs")

    paragraph_ids = [p.id for p in record.paragraphs]
    duplicates = [pid for pid, n in Counter(paragraph_ids).items() if n > 1]
    if duplicates:
        errors.append(f"{prefix} duplicate paragraph ids: {duplicates[:5]}")

    known = set(paragraph_ids)
    gold = record.gold_evidence.paragraph_ids
    if require_gold and not gold:
        errors.append(f"{prefix} no gold evidence paragraphs")
    for pid in gold:
        if pid not in known:
            errors.append(f"{prefix} gold paragraph id not in paragraphs: {pid!r}")

    for fact in record.gold_evidence.sentence_facts:
        pid = fact.get("paragraph_id")
        if pid not in known:
            errors.append(f"{prefix} sentence fact references unknown paragraph: {pid!r}")

    for step in record.gold_chain:
        for pid in step.support_paragraph_ids:
            if pid not in known:
                errors.append(
                    f"{prefix} gold step {step.step} references unknown paragraph: {pid!r}"
                )

    return errors


def validate_records(records: Iterable[QuestionRecord], *, strict: bool = True) -> list:
    """Validate every record; raise :class:`EvidenceError` in strict mode."""
    all_errors = []
    for record in records:
        all_errors.extend(validate_record(record))
    if strict and all_errors:
        shown = "\n".join(f"  - {e}" for e in all_errors[:20])
        extra = "" if len(all_errors) <= 20 else f"\n  ... and {len(all_errors) - 20} more"
        raise EvidenceError(f"invalid records:\n{shown}{extra}")
    return all_errors


def summarise(records: Iterable[QuestionRecord]) -> dict:
    """Counts useful in the sampling manifest."""
    records = list(records)
    by_dataset = Counter(r.dataset for r in records)
    by_hop = Counter(r.hop_level for r in records)
    return {
        "total": len(records),
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_hop": {str(k): v for k, v in sorted(by_hop.items())},
        "with_gold_chain": sum(1 for r in records if r.gold_chain),
        "empty_answer": sum(1 for r in records if not r.answer.strip()),
    }
