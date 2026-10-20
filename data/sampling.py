"""Fixed-seed dev sampling for the three project datasets (Role 1).

Usage (from the repository root)::

    python -m data.sampling --config configs/data.yaml
    python -m data.sampling --dataset musique --download
    python -m data.sampling --dataset hotpotqa --per-hop 50 --out-dir data/debug

The script writes one ``<dataset>_dev.jsonl`` per dataset plus a
``manifest.json`` recording the seed and counts. Sampling is deterministic:
each (seed, dataset, hop) pair gets its own SHA-256-derived RNG, and the pool is
sorted by question id before sampling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import yaml

from .evidence import summarise, validate_records
from .loaders import LOADERS, get_loader
from .loaders.base import LoaderError, QuestionRecord, download_file, write_jsonl

DEFAULT_CONFIG = "configs/data.yaml"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DatasetConfig:
    path: str
    enabled: bool = True
    url: Optional[str] = None


@dataclass(frozen=True)
class DataConfig:
    seed: int = 0
    dev_per_hop: int = 200
    hop_levels: tuple = (2, 3, 4)
    raw_dir: str = "data/raw"
    out_dir: str = "data/processed"
    datasets: dict = field(default_factory=dict)


def load_config(path: str | Path = DEFAULT_CONFIG) -> DataConfig:
    """Load ``configs/data.yaml`` into a typed config."""
    p = Path(path)
    if not p.exists():
        raise LoaderError(f"config file not found: {p}")
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    raw_dir = str(raw.get("raw_dir", "data/raw"))
    datasets = {}
    for name, cfg in (raw.get("datasets") or {}).items():
        cfg = cfg or {}
        loader = get_loader(name)  # raises a clear error for unknown names
        default_path = str(Path(raw_dir) / name / loader.default_filename)
        datasets[name] = DatasetConfig(
            path=str(cfg.get("path") or default_path),
            enabled=bool(cfg.get("enabled", True)),
            url=cfg.get("url"),
        )

    return DataConfig(
        seed=int(raw.get("seed", 0)),
        dev_per_hop=int(raw.get("dev_per_hop", 200)),
        hop_levels=tuple(int(h) for h in (raw.get("hop_levels") or (2, 3, 4))),
        raw_dir=raw_dir,
        out_dir=str(raw.get("out_dir", "data/processed")),
        datasets=datasets,
    )


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def _rng(seed: int, dataset: str, hop: int) -> random.Random:
    """Deterministic RNG independent of Python's hash randomisation."""
    digest = hashlib.sha256(f"{seed}:{dataset}:{hop}".encode("utf-8")).hexdigest()
    return random.Random(int(digest, 16))


def group_by_hop(records: Iterable[QuestionRecord]) -> dict:
    groups = {}
    for record in records:
        groups.setdefault(record.hop_level, []).append(record)
    return groups


def sample_records(
    records: Iterable[QuestionRecord],
    *,
    per_hop: int,
    seed: int,
    hop_levels: Iterable[int] = (2, 3, 4),
    dataset: str | None = None,
) -> list:
    """Sample up to ``per_hop`` records per hop level, deterministically.

    Hops absent from the dataset are skipped (e.g. HotpotQA only has 2-hop
    questions). The result is sorted by ``question_id`` for stable diffs.
    """
    records = list(records)
    dataset = dataset or (records[0].dataset if records else "unknown")
    groups = group_by_hop(records)
    sampled = []

    for hop in hop_levels:
        pool = sorted(groups.get(hop, []), key=lambda r: r.question_id)
        if not pool:
            continue
        k = min(int(per_hop), len(pool))
        chosen = _rng(seed, dataset, hop).sample(pool, k)
        sampled.extend(sorted(chosen, key=lambda r: r.question_id))

    return sampled


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def _count_by_hop(records: Iterable[QuestionRecord]) -> dict:
    counts = {}
    for record in records:
        counts[str(record.hop_level)] = counts.get(str(record.hop_level), 0) + 1
    return dict(sorted(counts.items()))


def run(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    datasets: Optional[Iterable[str]] = None,
    download: bool = False,
    skip_missing: bool = False,
    out_dir: Optional[str] = None,
    per_hop: Optional[int] = None,
    seed: Optional[int] = None,
    hop_levels: Optional[Iterable[int]] = None,
) -> dict:
    """Load, validate, sample and write dev sets. Returns the manifest."""
    config = load_config(config_path)
    selected = list(datasets) if datasets else [
        name for name, cfg in config.datasets.items() if cfg.enabled
    ]
    unknown = [name for name in selected if name not in config.datasets]
    if unknown:
        raise KeyError(f"datasets not in {config_path}: {unknown}")

    seed = config.seed if seed is None else int(seed)
    per_hop = config.dev_per_hop if per_hop is None else int(per_hop)
    hop_levels = tuple(hop_levels) if hop_levels is not None else config.hop_levels
    out_dir = Path(out_dir or config.out_dir)

    manifest = {
        "seed": seed,
        "dev_per_hop": per_hop,
        "hop_levels": list(hop_levels),
        "datasets": {},
    }

    for name in selected:
        ds_cfg = config.datasets[name]
        path = Path(ds_cfg.path)
        loader = get_loader(name)

        if not path.exists() and download:
            if not ds_cfg.url:
                raise LoaderError(
                    f"{name}: no direct download URL configured; "
                    f"download the dataset manually and put it at {path}"
                )
            print(f"[{name}] downloading {ds_cfg.url} -> {path}")
            download_file(ds_cfg.url, path)
        if not path.exists():
            if skip_missing:
                print(f"[{name}] skipped: missing {path}")
                continue
            raise LoaderError(
                f"{name}: dataset not found at {path}. "
                "Download it first or pass --skip-missing."
            )

        print(f"[{name}] loading {path}")
        records = loader.load(path)
        sampled = sample_records(
            records,
            per_hop=per_hop,
            seed=seed,
            hop_levels=hop_levels,
            dataset=name,
        )
        validate_records(sampled)  # strict: only clean records are shipped
        out_path = write_jsonl(sampled, out_dir / f"{name}_dev.jsonl")

        stats = summarise(sampled)
        manifest["datasets"][name] = {
            "source": str(path),
            "output": str(out_path),
            "loaded": len(records),
            "sampled": len(sampled),
            "by_hop": _count_by_hop(sampled),
            "with_gold_chain": stats["with_gold_chain"],
        }
        print(
            f"[{name}] sampled {len(sampled)}/{len(records)} "
            f"by hop {manifest['datasets'][name]['by_hop']} -> {out_path}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")
    return manifest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(LOADERS),
        help="dataset to process (repeatable); default: all enabled in config",
    )
    parser.add_argument("--download", action="store_true", help="download when the file is missing and a URL is configured")
    parser.add_argument("--skip-missing", action="store_true", help="skip datasets whose raw file is absent")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--per-hop", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--hop",
        action="append",
        type=int,
        default=None,
        help="hop level to sample (repeatable); default: config hop_levels",
    )
    args = parser.parse_args(argv)

    run(
        args.config,
        datasets=args.dataset,
        download=args.download,
        skip_missing=args.skip_missing,
        out_dir=args.out_dir,
        per_hop=args.per_hop,
        seed=args.seed,
        hop_levels=args.hop,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
