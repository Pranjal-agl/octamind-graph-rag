# AI-tool usage log

The course requires us to state which tools we used, for what, and how we verified the output.
Add a line whenever AI help is used. We remain responsible for correctness, and every member must
be able to explain what they submitted in the viva.

| Date | Who | Tool | Used for | How verified |
|---|---|---|---|---|
| 2026-10-02 | Pranjal | Claude | Drafting the traversal scaffold (types, frontier, test stubs, interface doc) | Ran the tests; the scoring, search and selection code is written by the member |
| 2026-10-20 | Chirantan S | Claude (OpenCode) | Drafting Role 1 dataset loaders, sampling, evidence validation, config and tests | Reviewed the code and ran `pytest tests/test_data.py` (18 passed); every file is owned and explainable by the member |
| 2026-10-21 | Chirantan S | Claude (OpenCode) | Investigating HotpotQA download sources; adding the temporary-skip notes; fixing duplicate MuSiQue paragraph titles | Ran `python -m data.sampling` on the real MuSiQue/2Wiki files (600 + 200 records) and `pytest tests/test_data.py` (20 passed); reviewed the duplicate-id fix |
