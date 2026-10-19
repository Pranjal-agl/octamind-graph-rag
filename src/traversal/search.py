"""Query-aware traversal: weighted A* with a beam (Modules 2 and 3 in the proposal).

    f(n) = g(n) + W * h(n)
    W = 0 -> uniform-cost search; use_g=False -> greedy best-first; W = 1 -> A*; W > 1 -> weighted A*.

Search problem (Module 2):
    initial state : the seed entities of the question
    actions(s)    : outgoing edges of the entity the path ends at
    result(s, a)  : the path extended by that edge
    cost(s, a, s'): cost(e | q)
    goal          : a path ending at an answer-bearing entity, or no frontier left / max_depth reached
"""
from .frontier import Frontier
from .types import Config, Graph, SearchNode


def make_seed_nodes(seeds: list) -> list:
    return [SearchNode(state=s) for s in seeds]


def traverse(graph: Graph, seeds: list, q_emb, cfg: Config, embed_fn=None) -> list:
    """Return candidate paths (each a list of Edge, best first) for the question.

    TODO (Role 5): implement. Steps to think through before coding:
      1. push seed nodes on the Frontier with f = (g if use_g) + W*h
      2. pop the best node; expand its out_edges; skip edges that would revisit an entity on the path
      3. child.path_cost = parent.path_cost + edge_cost(...); child.depth = parent.depth + 1
      4. stop expanding at cfg.max_depth; apply frontier.keep_best(cfg.beam_k, cfg.beam_margin)
      5. record every path reached, prune those with rho(P) < cfg.tau, rank the rest
    Keep counters for nodes expanded and peak frontier size (reported in the experiments).
    """
    raise NotImplementedError
