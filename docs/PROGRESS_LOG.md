# Progress log

Add one entry per work session: date, who, what changed, decisions, what is blocked.

| Date | Who | What | Decisions / blockers |
|---|---|---|---|
| 2026-10-02 | Pranjal | Repo skeleton, traversal scaffold (types, frontier, stubs, toy-graph tests) | Edge format proposed in docs/INTERFACE.md, to agree with Role 3 |
| 2026-10-20 | Chirantan S (Role 1) | Dataset loaders for MuSiQue/2Wiki/HotpotQA, normalized `QuestionRecord` schema, fixed-seed dev sampling, gold-evidence validation, `configs/data.yaml`, 18 tests | Draft schema pending review with Roles 2 & 3. Paragraph ids are `<dataset>::<title>` and feed edge provenance. Raw data stays git-ignored. Traversal stubs still fail as expected (Role 5). |
| 2026-10-21 | Chirantan S (Role 1) | MuSiQue + 2Wiki raw data downloaded; end-to-end sampling verified (MuSiQue 600 = 200/hop 2-4, 2Wiki 200). Fixed duplicate paragraph titles in MuSiQue with content-hash ids. HotpotQA temporarily disabled in `configs/data.yaml`. 20 data tests pass. | BLOCKER: every HotpotQA download route is dead (CMU 502, hotpotqa.github.io links dead, HF mirror is parquet-only; `datasets` needs Windows symlinks). Loader is ready; we will revisit getting it online later. Details in `docs/DATA_NOTES.md`. |
