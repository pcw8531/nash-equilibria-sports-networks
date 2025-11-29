"""
Payoff Matrix Analysis for Zero-Sum Games
==========================================

This module provides functions for analyzing 2×2 payoff matrices and calculating
Nash equilibria in zero-sum game contexts.

Mathematical Foundation:
    For a two-player zero-sum game, the payoff structure satisfies:
    U^A(s^A, s^B) + U^B(s^A, s^B) = 0

    A Nash equilibrium (s^A*, s^B*) satisfies:
    U^A(s^A*, s^B*) >= U^A(x, s^B*) for all strategies x of player A
    U^B(s^A*, s^B*) >= U^B(s^A*, y) for all strategies y of player B

Reference:
    Park, C. & Fath, B.D. (2025). Bounded Rationality Produces Nash Equilibria 
    in Sports Networks. Physica A. (Section 2.4, Equations 9-10)

Author: Chulwook Park
License: MIT
"""

import numpy as np
from typing import Tuple, Dict, Optional


def create_payoff_matrix(payoffs: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Create payoff matrices for both players in a zero-sum game.
    
    Parameters
    ----------
    payoffs : np.ndarray
        2×2 array of payoffs for Player A (row player).
        Player B's payoffs are automatically computed as negative of Player A's.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'player_A': Payoff matrix for Player A
        - 'player_B': Payoff matrix for Player B (= -player_A for zero-sum)
        - 'is_zero_sum': Boolean confirming zero-sum property
    
    Example
    -------
    >>> payoffs_A = np.array([[0, 1], [-1, 0]])  # Matching pennies
    >>> matrices = create_payoff_matrix(payoffs_A)
    >>> print(matrices['player_B'])
    [[ 0 -1]
     [ 1  0]]
    """
    player_A = np.array(payoffs)
    player_B = -player_A  # Zero-sum property
    
    return {
        'player_A': player_A,
        'player_B': player_B,
        'is_zero_sum': True
    }


def calculate_mixed_strategy_equilibrium(payoff_A: np.ndarray) -> Dict[str, float]:
    """
    Calculate the mixed-strategy Nash equilibrium for a 2×2 zero-sum game.
    
    For a general 2×2 zero-sum game with payoff matrix:
        [[a, b],
         [c, d]]
    
    The optimal mixed strategy for the row player (probability of choosing row 1):
        p* = (d - c) / (a - b - c + d)
    
    The optimal mixed strategy for the column player (probability of choosing col 1):
        q* = (d - b) / (a - b - c + d)
    
    Parameters
    ----------
    payoff_A : np.ndarray
        2×2 payoff matrix for Player A (row player).
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'p_star': Optimal probability for Player A choosing strategy 1
        - 'q_star': Optimal probability for Player B choosing strategy 1
        - 'expected_value': Expected payoff at equilibrium
        - 'equilibrium_type': 'pure' or 'mixed'
    
    Example
    -------
    >>> # Penalty kick game (matching pennies variant)
    >>> payoff = np.array([[-1, 1], [1, -1]])
    >>> result = calculate_mixed_strategy_equilibrium(payoff)
    >>> print(f"p* = {result['p_star']:.2f}, q* = {result['q_star']:.2f}")
    p* = 0.50, q* = 0.50
    
    Notes
    -----
    This implements Equation (10) from the manuscript:
    x = (x_1, x_2, ..., x_n), where x_i >= 0 and sum(x_i) = 1
    """
    a, b = payoff_A[0, 0], payoff_A[0, 1]
    c, d = payoff_A[1, 0], payoff_A[1, 1]
    
    denominator = a - b - c + d
    
    # Check for pure strategy equilibrium
    if abs(denominator) < 1e-10:
        # Degenerate case - check for dominant strategies
        return {
            'p_star': 0.5,
            'q_star': 0.5,
            'expected_value': 0.0,
            'equilibrium_type': 'degenerate'
        }
    
    p_star = (d - c) / denominator
    q_star = (d - b) / denominator
    
    # Clamp to [0, 1] for numerical stability
    p_star = np.clip(p_star, 0, 1)
    q_star = np.clip(q_star, 0, 1)
    
    # Calculate expected value at equilibrium
    # E[U^A] = p*q*a + p*(1-q)*b + (1-p)*q*c + (1-p)*(1-q)*d
    expected_value = (p_star * q_star * a + 
                      p_star * (1 - q_star) * b + 
                      (1 - p_star) * q_star * c + 
                      (1 - p_star) * (1 - q_star) * d)
    
    # Determine equilibrium type
    if p_star in [0, 1] and q_star in [0, 1]:
        eq_type = 'pure'
    else:
        eq_type = 'mixed'
    
    return {
        'p_star': p_star,
        'q_star': q_star,
        'expected_value': expected_value,
        'equilibrium_type': eq_type
    }


def expected_payoff(p: float, q: float, payoff_A: np.ndarray) -> Tuple[float, float]:
    """
    Calculate expected payoffs for both players given mixed strategies.
    
    Implements Equation (16) from the manuscript:
    U^R(p, q) = pq·U^R(P,P) + p(1-q)·U^R(P,Q) + (1-p)q·U^R(Q,P) + (1-p)(1-q)·U^R(Q,Q)
    
    Parameters
    ----------
    p : float
        Probability that Player A (row) chooses strategy 1 (0 <= p <= 1).
    q : float
        Probability that Player B (column) chooses strategy 1 (0 <= q <= 1).
    payoff_A : np.ndarray
        2×2 payoff matrix for Player A.
    
    Returns
    -------
    tuple
        (expected_payoff_A, expected_payoff_B)
    
    Example
    -------
    >>> payoff = np.array([[-1, 1], [1, -1]])
    >>> u_A, u_B = expected_payoff(0.5, 0.5, payoff)
    >>> print(f"E[U^A] = {u_A:.2f}, E[U^B] = {u_B:.2f}")
    E[U^A] = 0.00, E[U^B] = 0.00
    """
    a, b = payoff_A[0, 0], payoff_A[0, 1]
    c, d = payoff_A[1, 0], payoff_A[1, 1]
    
    # Expected payoff for Player A (row player)
    u_A = p * q * a + p * (1 - q) * b + (1 - p) * q * c + (1 - p) * (1 - q) * d
    
    # Zero-sum: Player B's payoff is negative of Player A's
    u_B = -u_A
    
    return u_A, u_B


def verify_nash_equilibrium(p: float, q: float, payoff_A: np.ndarray, 
                            tolerance: float = 1e-6) -> Dict[str, bool]:
    """
    Verify whether a strategy profile (p, q) constitutes a Nash equilibrium.
    
    A strategy profile is a Nash equilibrium if neither player can improve
    their expected payoff by unilaterally deviating.
    
    Parameters
    ----------
    p : float
        Probability that Player A chooses strategy 1.
    q : float
        Probability that Player B chooses strategy 1.
    payoff_A : np.ndarray
        2×2 payoff matrix for Player A.
    tolerance : float
        Numerical tolerance for equilibrium verification.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'is_equilibrium': Boolean indicating if (p, q) is Nash equilibrium
        - 'player_A_stable': Boolean for Player A's best response condition
        - 'player_B_stable': Boolean for Player B's best response condition
        - 'max_deviation_gain': Maximum possible gain from deviation
    """
    current_u_A, current_u_B = expected_payoff(p, q, payoff_A)
    
    # Check if Player A can improve by deviating
    u_A_if_p0, _ = expected_payoff(0, q, payoff_A)
    u_A_if_p1, _ = expected_payoff(1, q, payoff_A)
    max_u_A = max(u_A_if_p0, u_A_if_p1)
    player_A_stable = current_u_A >= max_u_A - tolerance
    
    # Check if Player B can improve by deviating
    _, u_B_if_q0 = expected_payoff(p, 0, payoff_A)
    _, u_B_if_q1 = expected_payoff(p, 1, payoff_A)
    max_u_B = max(u_B_if_q0, u_B_if_q1)
    player_B_stable = current_u_B >= max_u_B - tolerance
    
    max_deviation = max(max_u_A - current_u_A, max_u_B - current_u_B)
    
    return {
        'is_equilibrium': player_A_stable and player_B_stable,
        'player_A_stable': player_A_stable,
        'player_B_stable': player_B_stable,
        'max_deviation_gain': max_deviation
    }


def penalty_kick_payoff_matrix() -> np.ndarray:
    """
    Return the standard penalty kick payoff matrix.
    
    This represents the zero-sum game between kicker and goalkeeper
    as described in Figure 1 and Equation (9) of the manuscript.
    
    Payoff structure:
        - Kicker scores (+1 for kicker, -1 for goalkeeper) if they choose
          opposite directions
        - Goalkeeper saves (-1 for kicker, +1 for goalkeeper) if they choose
          the same direction
    
    Returns
    -------
    np.ndarray
        2×2 payoff matrix for the kicker (row player).
        Rows: Kick Left, Kick Right
        Columns: Goalkeeper Left, Goalkeeper Right
    """
    return np.array([
        [-1, +1],  # Kick Left: (same direction = save, opposite = goal)
        [+1, -1]   # Kick Right: (opposite = goal, same direction = save)
    ])


if __name__ == "__main__":
    # Demonstration of payoff matrix analysis
    print("=" * 60)
    print("PAYOFF MATRIX ANALYSIS - PENALTY KICK GAME")
    print("=" * 60)
    
    # Create penalty kick payoff matrix
    payoff = penalty_kick_payoff_matrix()
    print("\nPayoff Matrix (Kicker's perspective):")
    print("            GK Left   GK Right")
    print(f"Kick Left    {payoff[0,0]:+d}        {payoff[0,1]:+d}")
    print(f"Kick Right   {payoff[1,0]:+d}        {payoff[1,1]:+d}")
    
    # Calculate Nash equilibrium
    result = calculate_mixed_strategy_equilibrium(payoff)
    print(f"\nNash Equilibrium:")
    print(f"  Kicker probability (Left): p* = {result['p_star']:.2f}")
    print(f"  Goalkeeper probability (Left): q* = {result['q_star']:.2f}")
    print(f"  Expected payoff: {result['expected_value']:.2f}")
    print(f"  Equilibrium type: {result['equilibrium_type']}")
    
    # Verify equilibrium
    verification = verify_nash_equilibrium(result['p_star'], result['q_star'], payoff)
    print(f"\nEquilibrium Verification:")
    print(f"  Is Nash equilibrium: {verification['is_equilibrium']}")
    print(f"  Max deviation gain: {verification['max_deviation_gain']:.6f}")
