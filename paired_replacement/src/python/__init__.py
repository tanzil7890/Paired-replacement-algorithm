"""
Python implementation of the Paired Replacement Algorithm.

This module contains the NumPy reference implementation of the cache replacement algorithm.
"""

from .paired_cache import PairedCache, baseline_full_rebuild

__all__ = ['PairedCache', 'baseline_full_rebuild']