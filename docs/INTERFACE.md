# Interface between graph construction (Role 3) and traversal (Role 5)

Agree this first; everything else depends on it. If you change a field, change it here, in
`src/traversal/types.py`, and tell the other role.

## Edge (one relationship)

| Field | Type | Meaning |
|---|---|---|
| `head`, `tail` | str | canonical entity ids (after normalization and linking) |
| `relation` | str | relation label, e.g. `works_in` |
| `confidence` | float in [0,1] | extractor's confidence, c(e) |
| `support` | int >= 1 | number of source sentences that support this edge |
| `provenance` | tuple of str | source paragraph ids (needed for evidence recall/precision) |
| `embedding` | tuple of float | phi(e): embedding of the text "head relation tail", from the shared embedding model |

## Graph

- A list of edges plus `entity_embeddings: dict[str, tuple]` (one embedding per entity, from the same
  model), used for the semantic-distance heuristic h(n).
- Persisted as JSON Lines (one edge per line) plus one JSON file for entity embeddings, so every
  member can load it without the extraction code. Exact file names: TBD with Role 3.

## Shared services (one implementation, used by everyone)

- `embed(text) -> tuple[float]`: the fixed embedding model (Gemini Embedding 2).
- `count_tokens(text) -> int`: one tokenizer for all systems, so context sizes are comparable.

## Query side (Role 4 to Role 5)

- `seeds: list[str]` entity ids linked from the question.
- `q_emb: tuple[float]` question embedding.

## Output of traversal (Role 5 to Role 6)

- `H`: the selected list of `Edge` (the selected subgraph), plus counters: nodes expanded, peak
  frontier size. Role 6 serializes H (triple plus source sentence) into the prompt.
