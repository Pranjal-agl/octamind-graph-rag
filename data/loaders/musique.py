"""MuSiQue-Ans loader.

Each MuSiQue record already carries a ``question_decomposition``: one sub-question
per hop, each pointing at the paragraph that supports it. We use that as the gold
chain and the ``is_supporting`` flag for gold evidence.

Official release: https://github.com/StonyBrookNLP/musique
(the data itself is distributed via Google Drive, not in the repo).
"""
from __future__ import annotations

from .base import (
    BaseLoader,
    GoldEvidence,
    GoldStep,
    LoaderError,
    Paragraph,
    QuestionRecord,
    dedupe,
    infer_hop,
    paragraph_id,
    record_id,
)


class MusiqueLoader(BaseLoader):
    name = "musique"
    default_filename = "musique_ans_v1.0_dev.jsonl"
    # Distributed via Google Drive (gdown id 1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h).
    default_url = None

    def parse_record(self, raw: dict, index: int) -> QuestionRecord:
        raw_paragraphs = raw.get("paragraphs") or []
        if not raw_paragraphs:
            raise LoaderError("MuSiQue record has no 'paragraphs'")

        paragraphs = []
        gold_paragraphs = []
        idx_to_pid = {}
        for position, item in enumerate(raw_paragraphs):
            title = str(item.get("title", "")).strip()
            pid = paragraph_id(self.name, title)
            paragraphs.append(
                Paragraph(
                    id=pid,
                    title=title,
                    text=str(item.get("paragraph_text", item.get("text", ""))),
                )
            )
            raw_idx = item.get("idx", position)
            try:
                idx_to_pid[int(raw_idx)] = pid
            except (TypeError, ValueError):
                idx_to_pid[position] = pid
            if item.get("is_supporting"):
                gold_paragraphs.append(pid)

        decomposition = (
            raw.get("question_decomposition")
            or raw.get("decomposition")
            or raw.get("question_decomposition_list")
            or []
        )

        chain = []
        for step_no, step in enumerate(decomposition, start=1):
            support_idx = step.get("paragraph_support_idx")
            support_pids = []
            if support_idx is not None:
                try:
                    pid = idx_to_pid.get(int(support_idx))
                except (TypeError, ValueError):
                    pid = None
                if pid is not None:
                    support_pids.append(pid)
                    gold_paragraphs.append(pid)

            chain.append(
                GoldStep(
                    step=step_no,
                    relation=str(step.get("question", "")),
                    tail=(str(step["answer"]) if step.get("answer") is not None else None),
                    support_paragraph_ids=tuple(support_pids),
                )
            )

        hop_level = len(decomposition) if decomposition else infer_hop(raw, default=2)

        return QuestionRecord(
            question_id=record_id(raw, self.name, index),
            dataset=self.name,
            hop_level=hop_level,
            question=str(raw.get("question", "")),
            answer=str(raw.get("answer", "")),
            paragraphs=paragraphs,
            gold_evidence=GoldEvidence(paragraph_ids=tuple(dedupe(gold_paragraphs))),
            gold_chain=chain,
            metadata={
                "answer_aliases": raw.get("answer_aliases") or [],
                "answerable": raw.get("answerable"),
                "num_paragraphs": len(paragraphs),
            },
        )
