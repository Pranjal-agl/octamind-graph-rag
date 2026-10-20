# Data ingestion notes (Role 1)

Status of the raw datasets as of **2026-10-21**.

## Quick status

| Dataset | Format | Status | Expected path |
|---|---|---|---|
| MuSiQue-Ans | JSONL | Downloaded; sampled (600 = 200 per hop) | `data/raw/musique/musique_ans_v1.0_dev.jsonl` |
| 2WikiMultiHopQA | JSON | Downloaded; sampled (200) | `data/raw/twiki/dev.json` |
| HotpotQA | JSON | **Temporarily disabled** | `data/raw/hotpotqa/hotpot_dev_distractor_v1.json` |

## HotpotQA: temporary blocker

HotpotQA is **temporarily disabled** in `configs/data.yaml` (`enabled: false`).

All public download routes we tried are currently dead or unusable:

- The official URLs used by `hotpotqa/download.sh`
  (`http://curtis.ml.cmu.edu/datasets/hotpot/...`) return **502**.
- The download links on <https://hotpotqa.github.io/> are also dead.
- The HuggingFace mirror (`thinkall/HotpotQA`) currently exposes **parquet
  only**, and the direct parquet URLs return 404.
- The `datasets` library route fails on Windows without symlink support
  (Developer Mode or admin required), which we are not enabling.

This is **temporary**. We will revisit getting HotpotQA online later. The
loader is already implemented and tested; only the raw file is missing.

### To re-enable later

1. Obtain a valid `hotpot_dev_distractor_v1.json` (HotpotQA native JSON).
2. Put it at `data/raw/hotpotqa/hotpot_dev_distractor_v1.json`.
3. Set `datasets.hotpotqa.enabled: true` in `configs/data.yaml`.
4. Run `python -m data.sampling --dataset hotpotqa`.

## Working sources for the other two

- **MuSiQue-Ans** (`musique_ans_v1.0_dev.jsonl`): HuggingFace
  `dgslibisey/MuSiQue`, or the official Google Drive release via `gdown`
  (`1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h`).
- **2WikiMultiHopQA** (`dev.json`): HuggingFace `voidful/2WikiMultihopQA`,
  or the official Dropbox `data.zip`.

## Running the sampler

```powershell
python -m data.sampling              # MuSiQue + 2Wiki (HotpotQA disabled)
python -m data.sampling --skip-missing
python -m pytest tests/test_data.py -q
```
