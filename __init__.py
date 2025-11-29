"""
Game Theory Module
==================

This module provides tools for analyzing zero-sum games and Nash equilibria
in sports contexts, particularly penalty kick scenarios.

Components:
    - penalty_kick_nash: Visualization of mixed-strategy Nash equilibrium
    - payoff_matrix: 2×2 payoff matrix calculations and analysis

Reference:
    Park, C. & Fath, B.D. (2025). Bounded Rationality Produces Nash Equilibria 
    in Sports Networks. Physica A.
"""

from .penalty_kick_nash import plot_penalty_kick_equilibrium
from .payoff_matrix import calculate_mixed_strategy_equilibrium

__all__ = ['plot_penalty_kick_equilibrium', 'calculate_mixed_strategy_equilibrium']
