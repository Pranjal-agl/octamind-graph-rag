"""Context selection under a token budget (Section 6, 'dynamic context budget')."""
from .types import Config


def select_context(paths: list, q_emb, token_counter, cfg: Config) -> list:
    """Return the selected edges H.

    Add candidate paths in decreasing utility-per-token order until the budget B is reached,
    or until the marginal utility of the next path falls below epsilon. The utility of a path is
    the sum of (1 - cost(e | q)) over its edges NOT already selected, and a path only costs the
    tokens of those new edges.

    `token_counter(text) -> int` is the single shared tokenizer. TODO (Role 5): implement.
    """
    raise NotImplementedError
