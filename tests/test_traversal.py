"""Tests for the traversal module on the Figure 1 toy graph.
Tests that depend on functions you have not written yet fail with NotImplementedError;
that is expected at the start (write the code until they pass)."""
import math

from src.traversal.frontier import Frontier
from src.traversal.scoring import cosine, edge_cost
from src.traversal.search import traverse
from src.traversal.select import select_context
from src.traversal.toy_graph import GOLD_CHAIN, Q_EMB, build_toy_graph
from src.traversal.types import Config, SearchNode


# ---- already working (scaffolding) -------------------------------------------------
def test_cosine():
    assert math.isclose(cosine((1, 0), (1, 0)), 1.0)
    assert math.isclose(cosine((1, 0), (0, 1)), 0.0, abs_tol=1e-12)
    assert cosine((0, 0), (1, 0)) == 0.0


def test_out_edges_and_entities():
    g = build_toy_graph()
    assert {e.relation for e in g.out_edges("Priya")} == {"works_in", "joined_on", "sits_in"}
    assert "PyTorch" in g.entities()


def test_search_node_path_follows_parents():
    g = build_toy_graph()
    e1, e2 = g.out_edges("Priya")[0], g.out_edges("DeptA")[0]
    n0 = SearchNode("Priya")
    n1 = SearchNode(e1.tail, n0, e1, 0.2, 1)
    n2 = SearchNode(e2.tail, n1, e2, 0.5, 2)
    assert n2.path() == [e1, e2]


def test_frontier_orders_by_f_and_beam_prunes():
    fr = Frontier()
    for i, f in enumerate([3.0, 1.0, 2.0, 5.0]):
        fr.push(SearchNode(str(i)), f)
    fr.keep_best(k=2)
    assert len(fr) == 2
    assert fr.pop()[1] == 1.0 and fr.pop()[1] == 2.0


def test_frontier_margin():
    fr = Frontier()
    for i, f in enumerate([1.0, 1.2, 4.0]):
        fr.push(SearchNode(str(i)), f)
    fr.keep_best(margin=0.5)
    assert len(fr) == 2


# ---- you implement these (expected to fail until then) -----------------------------
def test_relevant_edges_cost_less_than_distractors():
    g = build_toy_graph()
    cfg = Config()
    by_rel = {e.relation: edge_cost(e, Q_EMB, 1, 3, cfg) for e in g.edges}
    assert all(c >= 0 for c in by_rel.values())
    for good in GOLD_CHAIN:
        for bad in ("joined_on", "sits_in", "near", "funded_by"):
            assert by_rel[good] < by_rel[bad]


def test_traverse_finds_gold_chain_first():
    g = build_toy_graph()
    paths = traverse(g, ["Priya"], Q_EMB, Config(beam_k=5))
    best = paths[0]
    assert tuple(e.relation for e in best) == GOLD_CHAIN
    assert best[-1].tail == "PyTorch"


def test_w0_uniform_cost_returns_same_best_path():
    g = build_toy_graph()
    uniform = traverse(g, ["Priya"], Q_EMB, Config(W=0.0, beam_k=None))
    assert tuple(e.relation for e in uniform[0]) == GOLD_CHAIN  # W changes search effort, not the best path here


def test_select_context_keeps_gold_and_drops_distractors_under_budget():
    g = build_toy_graph()
    cfg = Config(budget_B=12)
    paths = traverse(g, ["Priya"], Q_EMB, cfg)
    H = select_context(paths, Q_EMB, lambda t: len(t.split()), cfg)
    rels = {e.relation for e in H}
    assert set(GOLD_CHAIN) <= rels
    assert not rels & {"joined_on", "sits_in", "near"}
    assert sum(len(e.text.split()) for e in H) <= cfg.budget_B
