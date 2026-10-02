# Octamind: Query-Aware Semantic Graph Retrieval for Multi-Hop QA

> **Status: proposal stage.** This repository supports the CS F407 / U407 Artificial Intelligence project (Team 18, **Research** track) at BITS Pilani, Hyderabad Campus. Sections marked *planned* describe the design in our proposal and will be updated as code lands.

## What this project is about

Standard retrieval-augmented generation (RAG) retrieves text chunks by embedding similarity. For multi-hop questions, where the answer needs a chain of facts across entities and documents, no single chunk holds the chain, so similarity search either misses a link or pads the prompt with text that is similar but not needed.

We build a **Semantic Knowledge Network** (entities and typed relationships, each with a confidence and a pointer to its source text) and answer each question by **query-aware traversal**: score every relationship by its relevance to the question, search for the chain the question needs, and give the LLM only that evidence.

We do **not** claim novelty in knowledge graphs or GraphRAG in general. The question we test is narrow:

> Can query-specific traversal of an entity-relation graph assemble a much smaller and more relevant context than similarity-based retrieval, while keeping answer accuracy?

A partial or negative result, with analysis of where and why, is an acceptable outcome.

## Research questions

| | Question | Hypothesis |
|---|---|---|
| RQ1 (headline) | Does query-aware traversal reach the accuracy of tuned vector RAG with substantially fewer context tokens? | H1: at the best F1 reached by tuned vector RAG, our method needs fewer context tokens; at equal token budget its F1 is at least as high. |
| RQ2 | Does the advantage grow with hop depth and on distractor-heavy questions? | H2: the gain over vector RAG and naive k-hop expansion is larger at 4 hops than at 2 hops. |
| RQ3 | Which scoring signals matter? | H3: query relevance contributes most. |
| RQ4 | How sensitive is the method to graph-construction errors? | H4: performance degrades with graph errors; the oracle-graph gap separates construction errors from traversal errors. |

## Approach (planned)

**Offline, built once:** corpus, then entity/concept/fact extraction, then relationship discovery, then the Semantic Knowledge Network (triples + confidence + provenance).

**Per query:** query analysis and seed entities, then relationship selection with beam traversal, then minimal relevant subgraph under a token budget, then context construction, then LLM answer.

Traversal is a search problem (CS F407 Modules 2 and 3):

- Edge cost: `cost(e | q) = α(1 − rel(e,q)) + β(1 − c(e)) + η(1 − s(e)) + γ·depth(e)`
- Evaluation function (weighted A*): `f(n) = g(n) + W·h(n)`, with a beam of width `k` or margin `m`
- Context selection: add candidate paths in decreasing utility-per-token order until budget `B` is reached or marginal utility falls below `ε`

The heuristic `h` is not admissible, so we use it for ordering and pruning, not to claim optimal paths. `W`, `k` and `m` are config values, so the search-strategy ablations (uniform-cost, greedy best-first, A*, weighted A*) need no extra code.

## Systems compared

| ID | System |
|---|---|
| S1 | Full context |
| S2 | Vector RAG (chunk size and k swept) |
| S3 | Naive k-hop expansion (depth-limited BFS, no query scoring) |
| S4 | HippoRAG (LightRAG as fallback) |
| S5 | IRCoT (optional) |
| S6 | Proposed method on the automatically built graph |
| S7 | Proposed method on an oracle graph |

## Data and evaluation

- **Datasets:** MuSiQue-Ans (2-, 3- and 4-hop), 2WikiMultiHopQA, HotpotQA. 200 sampled dev questions per hop level, fixed seed.
- **Metrics:** EM and token-level F1; context tokens sent to the LLM; evidence recall and precision (mapped to source paragraphs); path recall where a verified gold chain exists; context reduction at matched accuracy; nodes expanded, peak frontier size and latency.
- **Statistics:** paired bootstrap 95% confidence intervals.
- **Fixed settings:** one LLM at temperature 0 and one embedding model for every system *(models: TODO, fill in once chosen)*.

## Repository layout (planned)

```
octamind-graph-rag/
├── README.md
├── configs/             # all run settings (models, W, k, m, budgets, seeds)
├── data/                # dataset loaders and sampling (raw data is not committed)
├── src/
│   ├── graph/           # extraction, entity linking, network construction, oracle graph
│   ├── query/           # query analysis and seed-entity identification
│   ├── traversal/       # edge cost, weighted A*, beam search
│   ├── context/         # subgraph serialization, token budget, prompts
│   ├── baselines/       # full context, vector RAG, k-hop, HippoRAG
│   └── evaluation/      # metrics, token counting, bootstrap, plots
├── experiments/         # scripts for each experiment
├── tests/               # unit tests, including a hand-built toy graph
├── results/             # tables and figures
└── docs/
    ├── PROGRESS_LOG.md  # running log of work and decisions
    └── AI_USAGE.md      # AI-tool disclosure log
```

## Getting started

*TODO: fill in as the code is written.* Planned: create an environment, install `requirements.txt`, set the model names and API keys in `configs/`, then run the experiment scripts from `experiments/`. Every result in the report should be reproducible from a clean clone using a config file and a seed.

## Working agreements

- **Commit steadily.** The course checks the commit history across the whole semester. Small, frequent commits with clear messages; no single last-minute upload.
- **Work on branches and open pull requests** so each member's contribution is visible and reviewed.
- **Seeds and configs, not magic numbers.** Anything tunable lives in `configs/`.
- **Keep `docs/PROGRESS_LOG.md` up to date** (date, who, what, decisions).
- **Do not commit** API keys, large datasets or model weights.

## Team (Team 18)

| Member | Role in proposal | GitHub |
|---|---|---|
| Pranjal Agrawal | Role 5: Relationship Ranking and Graph Traversal | @Pranjal-agl |
| Chirantan S | TBD | TBD |
| Shantanu Pandey | TBD | TBD |
| Utkarsh Gopal Bhartariya | TBD | TBD |
| Harshini Reddy Goli | TBD | TBD |
| Suchendra Kumar Pothukuchi | TBD | TBD |
| Prashant Pandurang Kadam | TBD | TBD |

## AI-tool disclosure

The course allows AI tools but requires disclosure and that every member can explain what they submit. We record which tools were used, for what, and how the output was verified, in `docs/AI_USAGE.md`, and summarize it in the final report.

## References

1. Lewis et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020. <https://arxiv.org/abs/2005.11401>
2. Edge et al. From Local to Global: A Graph RAG Approach to Query-Focused Summarization. 2024. <https://arxiv.org/abs/2404.16130>
3. Gutierrez et al. HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs. NeurIPS 2024. <https://arxiv.org/abs/2405.14831>
4. Guo et al. LightRAG: Simple and Fast Retrieval-Augmented Generation. 2024. <https://arxiv.org/abs/2410.05779>
5. Sun et al. Think-on-Graph: Deep and Responsible Reasoning of LLM on Knowledge Graph. ICLR 2024. <https://arxiv.org/abs/2307.07697>
6. He et al. G-Retriever: RAG for Textual Graph Understanding and QA. NeurIPS 2024. <https://arxiv.org/abs/2402.07630>
7. Trivedi et al. Interleaving Retrieval with Chain-of-Thought Reasoning (IRCoT). ACL 2023. <https://arxiv.org/abs/2212.10509>
8. Trivedi et al. MuSiQue: Multihop Questions via Single-hop Question Composition. TACL 2022. <https://arxiv.org/abs/2108.00573>
9. Ho et al. Constructing a Multi-hop QA Dataset (2WikiMultiHopQA). COLING 2020. <https://arxiv.org/abs/2011.01060>
10. Yang et al. HotpotQA: A Dataset for Diverse, Explainable Multi-hop QA. EMNLP 2018. <https://arxiv.org/abs/1809.09600>

## License

MIT

