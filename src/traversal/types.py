"""Shared data types for traversal (Role 5). The edge format is the contract with Role 3
(graph construction); see docs/INTERFACE.md. Do not change field names without telling them."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

Vector = tuple  # a tuple of floats; embeddings from the fixed embedding model


@dataclass(frozen=True)
class Edge:
    """One relationship e = (head, relation, tail) in the Semantic Knowledge Network."""
    head: str                     # canonical entity id
    relation: str                 # relation label, e.g. "works_in"
    tail: str                     # canonical entity id
    confidence: float             # c(e) in [0, 1], extractor confidence
    support: int                  # number of source sentences supporting e (>= 1)
    provenance: tuple = ()        # source paragraph ids, used for evidence recall/precision
    embedding: Vector = ()        # phi(e): embedding of the text "head relation tail"

    @property
    def text(self) -> str:
        return f"{self.head} {self.relation} {self.tail}"


class Graph:
    """Directed multigraph with adjacency lists and one embedding per entity."""

    def __init__(self, edges: list, entity_embeddings: Optional[dict] = None):
        self.edges = list(edges)
        self.entity_embeddings = entity_embeddings or {}
        self._out: dict = {}
        for e in self.edges:
            self._out.setdefault(e.head, []).append(e)

    def out_edges(self, entity: str) -> list:
        return self._out.get(entity, [])

    def entities(self) -> set:
        return {e.head for e in self.edges} | {e.tail for e in self.edges}


@dataclass
class SearchNode:
    """Node data structure from Module 2: STATE, PARENT, ACTION, PATH-COST.
    The state is the entity the partial path currently ends at; the path itself is
    recovered by following parent pointers."""
    state: str
    parent: Optional["SearchNode"] = None
    action: Optional[Edge] = None     # the edge that led here (None for a seed node)
    path_cost: float = 0.0            # g(n)
    depth: int = 0

    def path(self) -> list:
        """Edges from the seed to this node, in order."""
        edges, n = [], self
        while n is not None and n.action is not None:
            edges.append(n.action)
            n = n.parent
        return list(reversed(edges))


@dataclass
class Config:
    alpha: float = 1.0
    beta: float = 0.5
    eta: float = 0.25
    gamma: float = 0.1
    W: float = 1.5
    use_g: bool = True
    beam_k: Optional[int] = 5
    beam_margin: Optional[float] = None
    max_depth: int = 4
    tau: float = 0.0
    budget_B: int = 1000
    epsilon: float = 0.0
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})
