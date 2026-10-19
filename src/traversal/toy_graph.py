"""The Figure 1 example as a deterministic toy graph, with hand-made 2-D embeddings so
tests need no API. The query is 'What technology does the department where Priya works rely on?'
and its embedding points along (1, 0); relevant edges point close to it, distractors away."""
from .types import Edge, Graph

Q_EMB = (1.0, 0.0)


def build_toy_graph() -> Graph:
    E = Edge
    edges = [
        # required three-hop chain
        E("Priya", "works_in", "DeptA", 0.95, 3, ("p1",), (0.90, 0.10)),
        E("DeptA", "develops", "ProjectX", 0.90, 2, ("p2",), (0.80, 0.20)),
        E("ProjectX", "uses", "PyTorch", 0.92, 2, ("p3",), (0.95, 0.05)),
        # true but irrelevant facts (Figure 1, grey)
        E("Priya", "joined_on", "Date2021", 0.99, 1, ("p4",), (0.00, 1.00)),
        E("Priya", "sits_in", "Office3", 0.97, 1, ("p5",), (0.10, 0.90)),
        E("DeptA", "near", "Cafeteria", 0.85, 1, ("p6",), (0.05, 0.95)),
        E("ProjectX", "funded_by", "Budget7", 0.88, 1, ("p7",), (0.20, 0.80)),
    ]
    ent = {
        "Priya": (0.5, 0.5), "DeptA": (0.7, 0.3), "ProjectX": (0.8, 0.2), "PyTorch": (1.0, 0.0),
        "Date2021": (0.0, 1.0), "Office3": (0.1, 0.9), "Cafeteria": (0.05, 0.95), "Budget7": (0.2, 0.8),
    }
    return Graph(edges, ent)


GOLD_CHAIN = ("works_in", "develops", "uses")
