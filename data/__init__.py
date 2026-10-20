"""Dataset ingestion and gold-evidence extraction (Role 1).

This package owns the offline corpus: it loads the raw QA datasets, normalizes
them to the shared `QuestionRecord` schema, and samples fixed-seed dev sets.
Raw dataset files live under ``data/raw`` and are never committed.
"""
