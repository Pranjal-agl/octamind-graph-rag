"""HotpotQA loader.

HotpotQA uses the same ``context`` / ``supporting_facts`` shapes as 2Wiki, so
the normalization is nearly identical. HotpotQA comes with a direct download
URL for the distractor dev set; all questions are 2-hop.

Official release: https://hotpotqa.github.io/
"""
from __future__ import annotations

from .base import (
    BaseLoader,
    GoldEvidence,
    QuestionRecord,
    dedupe,
    infer_hop,
    record_id,
)


class HotpotQALoader(BaseLoader):
    name = "hotpotqa"
    default_filename = "hotpot_dev_distractor_v1.json"
    default_url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"

    def parse_record(self, raw: dict, index: int) -> QuestionRecord:
        paragraphs = self._paragraphs_from_context(raw.get("context") or [])
        pid_by_title = {p.title: p.id for p in paragraphs}

        gold_paragraphs = []
        sentence_facts = []
        seen_facts = set()
        for fact in raw.get("supporting_facts") or []:
            if not isinstance(fact, (list, tuple)) or len(fact) < 2:
                continue
            title, sentence_index = str(fact[0]).strip(), fact[1]
            pid = pid_by_title.get(title)
            if pid is None:
                continue
            gold_paragraphs.append(pid)
            try:
                sentence_index = int(sentence_index)
            except (TypeError, ValueError):
                continue
            key = (pid, sentence_index)
            if key not in seen_facts:
                seen_facts.add(key)
                sentence_facts.append(
                    {"paragraph_id": pid, "sentence_index": sentence_index}
                )

        hop_level = infer_hop(raw, default=2)

        return QuestionRecord(
            question_id=record_id(raw, self.name, index),
            dataset=self.name,
            hop_level=hop_level,
            question=str(raw.get("question", "")),
            answer=str(raw.get("answer", "")),
            paragraphs=paragraphs,
            gold_evidence=GoldEvidence(
                paragraph_ids=tuple(dedupe(gold_paragraphs)),
                sentence_facts=tuple(sentence_facts),
            ),
            # HotpotQA has no relation triples in the released annotations.
            gold_chain=[],
            metadata={
                "type": raw.get("type"),
                "level": raw.get("level"),
                "num_paragraphs": len(paragraphs),
            },
        )
