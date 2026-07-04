"""Pairwise evaluation helpers for future arena-style comparisons."""

from nyssa_bench.arena.arena_report import save_arena_report, save_pairwise_results, save_preference_table
from nyssa_bench.arena.pairwise_runner import PairwiseOutcome, PairwiseSummary, compare_episode_pairs
from nyssa_bench.arena.preference_schema import PreferenceRecord

__all__ = [
    "PairwiseOutcome",
    "PairwiseSummary",
    "PreferenceRecord",
    "compare_episode_pairs",
    "save_arena_report",
    "save_pairwise_results",
    "save_preference_table",
]
