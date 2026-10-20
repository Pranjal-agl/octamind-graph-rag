"""Tests for Role 1 data ingestion: loaders, validation, and sampling.

Everything here runs on synthetic fixtures, so the tests need no network and no
downloaded datasets.
"""
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from data.evidence import (
    EvidenceError,
    make_gold_evidence,
    summarise,
    validate_record,
    validate_records,
)
from data.loaders import LOADERS, get_loader
from data.loaders.base import (
    GoldEvidence,
    LoaderError,
    Paragraph,
    QuestionRecord,
    read_json_records,
    write_jsonl,
)
from data.sampling import load_config, run, sample_records

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def musique_raw(hop=3):
    titles = ["Alpha", "Beta", "Gamma", "Delta"][:hop]
    paragraphs = [
        {
            "idx": i,
            "title": title,
            "paragraph_text": f"{title} text.",
            "is_supporting": True,
        }
        for i, title in enumerate(titles)
    ]
    paragraphs.append(
        {
            "idx": len(titles),
            "title": "Distractor",
            "paragraph_text": "Noise.",
            "is_supporting": False,
        }
    )
    decomposition = [
        {
            "id": i + 1,
            "question": f"Question {i}?",
            "answer": f"A{i + 1}",
            "paragraph_support_idx": i,
        }
        for i in range(hop)
    ]
    return {
        "id": "musique-1",
        "question": "Multi hop question?",
        "answer": "Final answer",
        "paragraphs": paragraphs,
        "question_decomposition": decomposition,
        "answer_aliases": ["Final answer"],
        "answerable": True,
    }


def twiki_raw():
    return {
        "_id": "twiki-1",
        "type": "bridge",
        "question": "Which entity links A and B?",
        "answer": "X",
        "context": [
            ["Title A", ["s0.", "s1."]],
            ["Title B", ["t0."]],
        ],
        "supporting_facts": [["Title A", 1], ["Title B", 0]],
        "evidences": [["Entity A", "relation", "Entity B"]],
        "entity_ids": "Q1_Q2",
    }


def hotpot_raw():
    return {
        "_id": "hotpot-1",
        "question": "Which page links A and B?",
        "answer": "Y",
        "type": "bridge",
        "level": "medium",
        "context": [
            ["Page A", ["a0.", "a1."]],
            ["Page B", ["b0."]],
        ],
        "supporting_facts": [["Page A", 0], ["Page B", 0]],
    }


def make_records(dataset="musique", per_hop=300, hops=(2, 3, 4)):
    records = []
    for hop in hops:
        for i in range(per_hop):
            pid = f"{dataset}::p{hop}_{i}"
            records.append(
                QuestionRecord(
                    question_id=f"{dataset}_{hop}_{i}",
                    dataset=dataset,
                    hop_level=hop,
                    question=f"q {hop} {i}",
                    answer="a",
                    paragraphs=[Paragraph(id=pid, title=f"p{hop}_{i}", text="t")],
                    gold_evidence=GoldEvidence(paragraph_ids=(pid,)),
                )
            )
    return records


# --------------------------------------------------------------------------- #
# Loader registry
# --------------------------------------------------------------------------- #
def test_loader_registry():
    assert set(LOADERS) == {"musique", "twiki", "hotpotqa"}
    assert get_loader("musique").name == "musique"
    with pytest.raises(KeyError):
        get_loader("nope")


# --------------------------------------------------------------------------- #
# MuSiQue
# --------------------------------------------------------------------------- #
def test_musique_normalizes_schema():
    record = get_loader("musique").parse_records([musique_raw(hop=3)])[0]

    assert record.dataset == "musique"
    assert record.question_id == "musique-1"
    assert record.hop_level == 3
    assert [p.id for p in record.paragraphs] == [
        "musique::Alpha",
        "musique::Beta",
        "musique::Gamma",
        "musique::Distractor",
    ]
    assert record.gold_evidence.paragraph_ids == (
        "musique::Alpha",
        "musique::Beta",
        "musique::Gamma",
    )
    assert len(record.gold_chain) == 3
    assert record.gold_chain[0].relation == "Question 0?"
    assert record.gold_chain[2].tail == "A3"
    assert record.gold_chain[0].support_paragraph_ids == ("musique::Alpha",)
    assert validate_record(record) == []
    assert record.to_dict()["metadata"]["answerable"] is True


def test_musique_hop_follows_decomposition_length():
    for hop in (2, 3):
        record = get_loader("musique").parse_records([musique_raw(hop=hop)])[0]
        assert record.hop_level == hop


# --------------------------------------------------------------------------- #
# 2Wiki / HotpotQA
# --------------------------------------------------------------------------- #
def test_twiki_normalizes_schema():
    record = get_loader("twiki").parse_records([twiki_raw()])[0]

    assert record.dataset == "twiki"
    assert record.hop_level == 2
    assert [p.id for p in record.paragraphs] == ["twiki::Title A", "twiki::Title B"]
    assert record.gold_evidence.paragraph_ids == ("twiki::Title A", "twiki::Title B")
    assert record.gold_evidence.sentence_facts == (
        {"paragraph_id": "twiki::Title A", "sentence_index": 1},
        {"paragraph_id": "twiki::Title B", "sentence_index": 0},
    )
    assert len(record.gold_chain) == 1
    step = record.gold_chain[0]
    assert (step.head, step.relation, step.tail) == ("Entity A", "relation", "Entity B")
    assert validate_record(record) == []


def test_hotpotqa_normalizes_schema():
    record = get_loader("hotpotqa").parse_records([hotpot_raw()])[0]

    assert record.dataset == "hotpotqa"
    assert record.hop_level == 2
    assert [p.id for p in record.paragraphs] == ["hotpotqa::Page A", "hotpotqa::Page B"]
    assert record.gold_evidence.paragraph_ids == ("hotpotqa::Page A", "hotpotqa::Page B")
    assert record.gold_chain == []  # HotpotQA has no relation triples
    assert record.metadata["level"] == "medium"
    assert validate_record(record) == []


# --------------------------------------------------------------------------- #
# Evidence helpers
# --------------------------------------------------------------------------- #
def test_make_gold_evidence_dedupes():
    evidence = make_gold_evidence(
        ["a", "a", "b"],
        [
            {"paragraph_id": "a", "sentence_index": 1},
            {"paragraph_id": "a", "sentence_index": 1},
            {"paragraph_id": "b", "sentence_index": 0},
        ],
    )
    assert evidence.paragraph_ids == ("a", "b")
    assert len(evidence.sentence_facts) == 2


def test_validation_catches_dangling_gold_paragraph():
    record = make_records(per_hop=1, hops=(2,))[0]
    bad = QuestionRecord(
        question_id=record.question_id,
        dataset=record.dataset,
        hop_level=record.hop_level,
        question=record.question,
        answer=record.answer,
        paragraphs=record.paragraphs,
        gold_evidence=GoldEvidence(paragraph_ids=("missing::pid",)),
    )
    errors = validate_record(bad)
    assert any("gold paragraph id not in paragraphs" in e for e in errors)
    with pytest.raises(EvidenceError):
        validate_records([bad])


def test_validation_catches_missing_gold_evidence():
    record = make_records(per_hop=1, hops=(2,))[0]
    bad = QuestionRecord(
        question_id=record.question_id,
        dataset=record.dataset,
        hop_level=record.hop_level,
        question=record.question,
        answer=record.answer,
        paragraphs=record.paragraphs,
        gold_evidence=GoldEvidence(),
    )
    assert any("no gold evidence" in e for e in validate_record(bad))


def test_validation_rejects_duplicate_paragraph_ids():
    record = make_records(per_hop=1, hops=(2,))[0]
    duplicate = record.paragraphs[0]
    bad = QuestionRecord(
        question_id=record.question_id,
        dataset=record.dataset,
        hop_level=record.hop_level,
        question=record.question,
        answer=record.answer,
        paragraphs=[duplicate, duplicate],
        gold_evidence=GoldEvidence(paragraph_ids=(duplicate.id,)),
    )
    assert any("duplicate paragraph ids" in e for e in validate_record(bad))


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def test_sampling_is_deterministic_and_capped():
    records = make_records(per_hop=300, hops=(2, 3, 4))

    first = sample_records(records, per_hop=200, seed=0, hop_levels=(2, 3, 4))
    second = sample_records(records, per_hop=200, seed=0, hop_levels=(2, 3, 4))

    assert len(first) == 600
    assert [r.question_id for r in first] == [r.question_id for r in second]
    assert Counter(r.hop_level for r in first) == {2: 200, 3: 200, 4: 200}


def test_sampling_changes_with_seed():
    records = make_records(per_hop=100, hops=(2,))
    first = sample_records(records, per_hop=20, seed=0, hop_levels=(2,))
    second = sample_records(records, per_hop=20, seed=1, hop_levels=(2,))
    assert [r.question_id for r in first] != [r.question_id for r in second]


def test_sampling_skips_missing_hops():
    records = make_records(dataset="hotpotqa", per_hop=10, hops=(2,))
    sampled = sample_records(records, per_hop=200, seed=0, hop_levels=(2, 3, 4))
    assert len(sampled) == 10
    assert {r.hop_level for r in sampled} == {2}


def test_sampling_caps_at_available_pool():
    records = make_records(per_hop=5, hops=(2,))
    assert len(sample_records(records, per_hop=200, seed=0, hop_levels=(2,))) == 5


def test_summarise_counts():
    summary = summarise(make_records(per_hop=2, hops=(2, 3)))
    assert summary["total"] == 4
    assert summary["by_hop"] == {"2": 2, "3": 2}
    assert summary["with_gold_chain"] == 0


# --------------------------------------------------------------------------- #
# IO and config
# --------------------------------------------------------------------------- #
def test_jsonl_roundtrip(tmp_path):
    records = make_records(per_hop=2, hops=(2,))
    path = write_jsonl(records, tmp_path / "out.jsonl")

    raws = read_json_records(path)
    assert len(raws) == 2
    assert QuestionRecord.from_dict(raws[0]).to_dict() == records[0].to_dict()


def test_read_json_records_wrapped_json(tmp_path):
    path = tmp_path / "wrapped.json"
    path.write_text(json.dumps({"data": [{"id": 1}, {"id": 2}]}), encoding="utf-8")
    assert read_json_records(path) == [{"id": 1}, {"id": 2}]


def test_missing_file_raises(tmp_path):
    with pytest.raises(LoaderError):
        get_loader("musique").load(tmp_path / "nope.jsonl")


def test_repo_config_loads():
    config = load_config(REPO_ROOT / "configs" / "data.yaml")
    assert config.seed == 0
    assert config.dev_per_hop == 200
    assert config.hop_levels == (2, 3, 4)
    assert set(config.datasets) == {"musique", "twiki", "hotpotqa"}
    assert config.datasets["hotpotqa"].url
    assert config.datasets["musique"].path.endswith("musique_ans_v1.0_dev.jsonl")


def test_run_end_to_end(tmp_path):
    raw_path = tmp_path / "musique.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for i in range(3):
            for hop in (2, 3, 4):
                raw = musique_raw(hop=hop)
                raw["id"] = f"m{hop}_{i}"
                f.write(json.dumps(raw) + "\n")

    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 0,
                "dev_per_hop": 2,
                "hop_levels": [2, 3, 4],
                "raw_dir": str(tmp_path / "raw"),
                "out_dir": str(tmp_path / "out"),
                "datasets": {
                    "musique": {"enabled": True, "path": str(raw_path), "url": None}
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = run(config_path, datasets=["musique"])

    assert manifest["seed"] == 0
    assert manifest["datasets"]["musique"]["loaded"] == 9
    assert manifest["datasets"]["musique"]["sampled"] == 6  # 2 per hop
    assert manifest["datasets"]["musique"]["by_hop"] == {"2": 2, "3": 2, "4": 2}

    out = tmp_path / "out" / "musique_dev.jsonl"
    assert out.exists()
    assert len(read_json_records(out)) == 6
    assert (tmp_path / "out" / "manifest.json").exists()
