"""CUDA graph of ``greedy_decode``. Position/valid_len are GPU buffers, not capture scalars."""

from __future__ import annotations

import gc

import torch

from .linear import DeviceWeights
from .state import DecodeState
from .step import greedy_decode

__all__ = ["GreedyGraph", "capture_greedy"]


class GreedyGraph:
    def __init__(self, graph: torch.cuda.CUDAGraph, state: DecodeState, weights: DeviceWeights) -> None:
        self.graph = graph
        self.state = state
        self.weights = weights

    def replay(self) -> None:
        self.graph.replay()

    def __call__(self, state: DecodeState, weights: DeviceWeights) -> None:
        if state is not self.state or weights is not self.weights:
            raise RuntimeError("replay is bound to the captured state/weights")
        self.replay()


def capture_greedy(
    state: DecodeState,
    weights: DeviceWeights,
    *,
    warmup: int = 3,
) -> GreedyGraph:
    """Capture one greedy step. Caller must be on CUDA, N=1, static arena."""
    if state.device.type != "cuda":
        raise RuntimeError("greedy graph is CUDA-only")
    stream = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    side.wait_stream(stream)
    with torch.cuda.stream(side):
        for _ in range(max(1, warmup)):
            greedy_decode(state, weights)
        side.synchronize()
        graph = torch.cuda.CUDAGraph()
        graph.capture_begin()
        try:
            greedy_decode(state, weights)
        finally:
            graph.capture_end()
    stream.wait_stream(side)
    gc.collect()
    return GreedyGraph(graph, state, weights)
