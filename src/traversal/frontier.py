"""Priority-queue frontier for best-first search (Module 2: frontier + reached)."""
import heapq
import itertools


class Frontier:
    """Min-heap ordered by f-score. A counter breaks ties so SearchNodes are never compared."""

    def __init__(self):
        self._heap = []
        self._counter = itertools.count()

    def push(self, node, f: float) -> None:
        heapq.heappush(self._heap, (f, next(self._counter), node))

    def pop(self):
        f, _, node = heapq.heappop(self._heap)
        return node, f

    def peek_f(self) -> float:
        return self._heap[0][0]

    def __len__(self) -> int:
        return len(self._heap)

    def keep_best(self, k=None, margin=None) -> None:
        """Beam pruning (Module 2, memory-bounded search): keep the k best f-scores,
        and/or only nodes within `margin` of the best f. None means that rule is off."""
        if not self._heap:
            return
        items = sorted(self._heap)
        if margin is not None:
            best = items[0][0]
            items = [it for it in items if it[0] <= best + margin]
        if k is not None:
            items = items[:k]
        self._heap = items
        heapq.heapify(self._heap)
