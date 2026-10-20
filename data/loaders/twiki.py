"""2WikiMultiHopQA loader.

The dataset follows the HotpotQA format: ``context`` is ``[[title, [sentences]],
...]`` and ``supporting_facts`` is ``[[title, sentence_index], ...]``. It also
ships ``evidences`` as ``[subject, relation, object]`` triples, which we keep as
the gold chain when present.

Official release: https://github.com/Alab-NII/2wikimultihop
(data is distributed as a Dropbox zip, not in the repo).
"""
from __future__ import annotations

from .base import (
    BaseLoader,
    GoldEvidence,
    GoldStep,
    QuestionRecord,
    dedupe,
    infer_hop,
    record_id,
)


class TwikiLoader(BaseLoader):
    name = "twiki"
    default_filename = "dev.json"
    # Distributed via Dropbox: https://www.dropbox.com/s/npidmtadreo6df2/data.zip
    default_url = None

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

        chain = []
        for step_no, evidence in enumerate(raw.get("evidences") or [], start=1):
            if isinstance(evidence, (list, tuple)) and len(evidence) == 3:
                chain.append(
                    GoldStep(
                        step=step_no,
                        head=str(evidence[0]),
                        relation=str(evidence[1]),
                        tail=str(evidence[2]),
                    )
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
            gold_chain=chain,
            metadata={
                "type": raw.get("type"),
                "entity_ids": raw.get("entity_ids"),
                "num_paragraphs": len(paragraphs),
            },
        )
