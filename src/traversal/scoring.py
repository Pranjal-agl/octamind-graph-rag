"""Scoring signals (Section 6 of the proposal). YOU write the marked functions.

Notation: q = query embedding, phi(e) = edge embedding, phi(n) = entity embedding.
    rel(e, q)   = cos(phi(e), phi(q))
    cost(e | q) = alpha*(1 - rel) + beta*(1 - c(e)) + eta*(1 - s(e)) + gamma*depth(e)
    h(n)        = 1 - cos(phi(end(n)), phi(q))       (semantic distance, the heuristic)
    rho(P)      = cos(phi(P), phi(q))                 (multi-hop path relevance)
"""
import math

from .types import Config, Edge


def cosine(a, b) -> float:
    """Cosine similarity of two equal-length vectors; 0.0 if either has zero norm."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def strength(edge: Edge, max_support: int) -> float:
    """s(e) in [0,1]: the support count normalized by the largest support in the graph."""
    return min(1.0, edge.support / max_support) if max_support > 0 else 0.0


def edge_cost(edge: Edge, q_emb, depth: int, max_support: int, cfg: Config) -> float:
    """cost(e | q). TODO (Role 5): implement from the formula in the module docstring.
    Must be >= 0 for every edge (Module 2: step costs must be positive for UCS/A* to behave)."""
    raise NotImplementedError


def semantic_distance(entity_emb, q_emb) -> float:
    """h(n) = 1 - cos(phi(end(n)), phi(q)). TODO (Role 5)."""
    raise NotImplementedError


def path_relevance(path_edges: list, q_emb, embed_fn) -> float:
    """rho(P) = cos(phi(P), phi(q)), where phi(P) embeds the concatenated text of the path.
    `embed_fn(text) -> vector` comes from the shared embedding model. TODO (Role 5)."""
    raise NotImplementedError
