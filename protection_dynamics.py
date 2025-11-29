"""
Protection Dynamics Simulation
==============================

This module implements the agent-based simulation of protection strategies
in sports networks, demonstrating how Nash equilibria emerge from bounded
rationality through social learning and strategic exploration.

Mathematical Foundation (from manuscript):
    
    Capital update (Equation 5):
        c_i(t+1) = 1 + (1 - f_m - f_p) · c_i(t)
    
    Protection probability (Equation 7):
        p_p = p_{p,max} / (1 + c_{p,1/2} / (f_p · c_i))
    
    Protection level (Equation 6):
        f_p = strategy_0 + strategy_1 × C_i
    
    Imitation probability (Equation 8a):
        π = 1 / (1 + exp(-s × (capital_j - capital_i)))

Four Equilibrium Scenarios (Section 3.1):
    A: p_{p,max}=1.0, c_p=1.0 → Coexistence (p_p ≈ 0.47)
    B: p_{p,max}=0.1, c_p=1.0 → System failure (p_p ≈ 0.05)
    C: p_{p,max}=0.1, c_p=0.1 → Partial coexistence
    D: p_{p,max}=1.0, c_p=0.1 → Non-failure (p_p ≈ 0.50)

Reference:
    Park, C. & Fath, B.D. (2025). Bounded Rationality Produces Nash Equilibria 
    in Sports Networks. Physica A. (Sections 2.2-2.3, 3.1)

Author: Chulwook Park
License: MIT
"""

import networkx as nx
import numpy as np


def create_scale_free_graph(n, m):
    """
    Create scale-free network using Barabási-Albert preferential attachment.
    
    Parameters
    ----------
    n : int
        Total number of nodes.
    m : int
        Number of edges each new node creates.
    
    Returns
    -------
    nx.Graph
        Scale-free network.
    
    Notes
    -----
    Implements the network construction described in Section 2.1 and Appendix 2.
    Connection probability is proportional to node degree: p(i) = k_i / Σ_j k_j
    """
    G = nx.complete_graph(m)
    target_nodes = list(G.nodes())
    for i in range(m, n):
        targets = np.random.choice(target_nodes, m, replace=True)
        G.add_node(i)
        for t in targets:
            G.add_edge(i, t)
        target_nodes.extend([i] * m)
        target_nodes.extend(targets)
    return G


def run_simulation(n=100, m=5, time_period=100, num_realizations=10,
                   fm=0.1, pr=0.99, pe=0.99, sigma=0.1, s=1.0,
                   pn=0.1, pl=0.01, pmax=1.0, cp=1.0,
                   seed=None, verbose=True):
    """
    Run protection dynamics simulation.
    
    Parameters
    ----------
    n : int
        Number of nodes (default: 100).
    m : int
        Scale-free network parameter (default: 5).
    time_period : int
        Number of time steps (default: 100).
    num_realizations : int
        Number of independent runs to average (default: 10).
    fm : float
        Maintenance cost, f_m in Equation 5 (default: 0.1).
    pr : float
        Imitation probability, p_r in Equation 8 (default: 0.99).
    pe : float
        Exploration probability, p_e (default: 0.99).
    sigma : float
        Exploration noise standard deviation (default: 0.1).
    s : float
        Selection intensity in imitation (default: 1.0).
    pn : float
        Node failure probability, p_n (default: 0.1).
    pl : float
        Link-based failure probability, p_l (default: 0.01).
    pmax : float
        Maximum protection, p_{p,max} in Equation 7 (default: 1.0).
    cp : float
        Protection scaling, c_{p,1/2} in Equation 7 (default: 1.0).
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool
        Print progress and results (default: True).
    
    Returns
    -------
    dict
        Results containing:
        - 'capital_mean': Average capital per node
        - 'failure_mean': Average failure rate per node
        - 'protection_mean': Average protection probability per node
        - 'system_failure_rate': Overall system failure rate
        - 'system_capital': Overall system capital
        - 'system_protection': Overall system protection
        - 'parameters': Input parameters dictionary
    
    Example
    -------
    >>> # Scenario A: Sufficient protection
    >>> results = run_simulation(pmax=1.0, cp=1.0, verbose=True)
    
    >>> # Scenario B: Low protection  
    >>> results = run_simulation(pmax=0.1, cp=1.0, verbose=True)
    
    >>> # Scenario D: Robust protection
    >>> results = run_simulation(pmax=1.0, cp=0.1, verbose=True)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if verbose:
        print("=" * 60)
        print("PROTECTION DYNAMICS SIMULATION")
        print("=" * 60)
        print(f"\nNetwork: n={n}, m={m}")
        print(f"Time period: {time_period}, Realizations: {num_realizations}")
        print(f"\nKey parameters:")
        print(f"  p_{{p,max}} = {pmax}")
        print(f"  c_{{p,1/2}} = {cp}")
        print(f"  p_r (imitation) = {pr}")
        print(f"  p_e (exploration) = {pe}")
        print(f"  p_n (node failure) = {pn}")
        print(f"  p_l (link failure) = {pl}")
    
    # Create network (same topology for all realizations)
    G = create_scale_free_graph(n, m)
    
    # Eigenvector centrality
    C_dict = nx.eigenvector_centrality(G)
    Centrality = np.array(list(C_dict.values()))
    
    if verbose:
        print(f"\nNetwork created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"Eigenvector centrality range: [{Centrality.min():.4f}, {Centrality.max():.4f}]")
    
    # Storage arrays
    Capital_final = np.zeros((n, num_realizations))
    Failure_final = np.zeros((n, num_realizations))
    Protect_final = np.zeros((n, num_realizations))
    
    # Run multiple realizations
    for realization in range(num_realizations):
        
        # Initialize
        Capital = np.ones(n)
        Strategy_0 = np.zeros(n)
        Strategy_1 = np.zeros(n)
        Failure = np.zeros(n)
        failure_potential = np.zeros(n)
        
        # Time evolution
        for t in range(time_period):
            
            # --- Imitation of Strategy_0 (Equation 8a) ---
            for i in range(n):
                if np.random.random() <= pr:
                    focal = i
                    rr = np.random.choice([j for j in range(n) if j != focal])
                    # Imitation probability: π = 1/(1 + exp(-s(c_j - c_i)))
                    pi = 1.0 / (1.0 + np.exp(-s * (Capital[rr] - Capital[focal])))
                    if np.random.random() <= pi:
                        Strategy_0[focal] = Strategy_0[rr]
            
            # --- Imitation of Strategy_1 ---
            for i in range(n):
                if np.random.random() <= pr:
                    focal = i
                    rr = np.random.choice([j for j in range(n) if j != focal])
                    pi = 1.0 / (1.0 + np.exp(-s * (Capital[rr] - Capital[focal])))
                    if np.random.random() <= pi:
                        Strategy_1[focal] = Strategy_1[rr]
            
            # --- Exploration (mutation) ---
            for i in range(n):
                if np.random.random() <= pe:
                    Strategy_0[i] += np.random.normal(0, sigma)
                if np.random.random() <= pe:
                    Strategy_1[i] += np.random.normal(0, sigma)
            
            # --- Protection level (Equation 6) ---
            # f_p = strategy_0 + strategy_1 × C_i
            fp = Strategy_0 + Strategy_1 * Centrality
            fp = np.clip(fp, 0, 1 - fm)
            
            # --- Capital update (Equation 5) ---
            # c_i(t+1) = 1 + (1 - f_m - f_p) · c_i(t)
            Capital = 1.0 + (1.0 - fm - fp) * Capital
            
            # --- Failure dynamics ---
            # New failure potential (probability p_n)
            newly_potential = np.random.random(n) <= pn
            failure_potential[newly_potential] = 1
            
            # Propagation from failed neighbors (probability p_l)
            for i in range(n):
                if Failure[i] > 0:
                    for j in G.neighbors(i):
                        if np.random.random() <= pl:
                            failure_potential[j] = 1
            
            # --- Protection probability (Equation 7) ---
            # p_p = p_{p,max} / (1 + c_{p,1/2} / (f_p · c_i))
            protection_probability = np.zeros(n)
            idx_potential = failure_potential > 0
            with np.errstate(divide='ignore', invalid='ignore'):
                mask = (fp > 0) & (Capital > 0) & idx_potential
                protection_probability[mask] = pmax / (1.0 + cp / (fp[mask] * Capital[mask]))
            
            # --- Determine failures ---
            # Fail if random > protection_probability
            fail_roll = np.random.random(n)
            new_failures = (fail_roll > protection_probability) & idx_potential
            Failure[new_failures] = 1
            Capital[new_failures] = 0.0
            
            # Reset failure potential
            failure_potential[:] = 0
        
        # Store final state
        Capital_final[:, realization] = Capital
        Failure_final[:, realization] = Failure
        Protect_final[:, realization] = protection_probability
    
    # Compute averages
    Capital_mean = np.mean(Capital_final, axis=1)
    Failure_mean = np.mean(Failure_final, axis=1)
    Protect_mean = np.mean(Protect_final, axis=1)
    
    # System-level metrics
    system_failure_rate = np.mean(Failure_mean)
    system_capital = np.mean(Capital_mean)
    system_protection = np.mean(Protect_mean)
    
    if verbose:
        print("\n" + "=" * 60)
        print("SIMULATION RESULTS")
        print("=" * 60)
        print(f"\nSystem-level outcomes (averaged over {num_realizations} realizations):")
        print(f"  Failure rate: {system_failure_rate:.2%}")
        print(f"  Average capital: {system_capital:.4f}")
        print(f"  Average protection: {system_protection:.4f}")
        
        # Theoretical protection probability for reference
        # Using average f_p·c ≈ 0.9 (typical value after convergence)
        fp_c_typical = 0.9
        theoretical_pp = pmax / (1.0 + cp / fp_c_typical)
        print(f"\nTheoretical protection probability (Equation 7):")
        print(f"  p_p = {pmax} / (1 + {cp} / 0.9) = {theoretical_pp:.4f}")
        print(f"  Failure potential = 1 - p_p = {1 - theoretical_pp:.4f}")
        
        # Interpretation
        print("\n" + "-" * 60)
        if system_failure_rate < 0.1:
            print("Interpretation: SCENARIO D (Robust protection → Non-failure)")
        elif system_failure_rate < 0.3:
            print("Interpretation: SCENARIO A (Sufficient protection → Coexistence)")
        elif system_failure_rate < 0.7:
            print("Interpretation: SCENARIO C (Limited protection → Partial coexistence)")
        else:
            print("Interpretation: SCENARIO B (Low protection → System failure)")
        print("-" * 60)
    
    return {
        'capital_mean': Capital_mean,
        'failure_mean': Failure_mean,
        'protection_mean': Protect_mean,
        'system_failure_rate': system_failure_rate,
        'system_capital': system_capital,
        'system_protection': system_protection,
        'network': G,
        'centrality': Centrality,
        'parameters': {
            'n': n, 'm': m, 'time_period': time_period,
            'num_realizations': num_realizations,
            'fm': fm, 'pr': pr, 'pe': pe, 'sigma': sigma, 's': s,
            'pn': pn, 'pl': pl, 'pmax': pmax, 'cp': cp
        }
    }


def calculate_protection_probability(fp, capital, pmax, cp):
    """
    Calculate protection probability using Equation 7.
    
    p_p = p_{p,max} / (1 + c_{p,1/2} / (f_p · c))
    
    Parameters
    ----------
    fp : float
        Protection investment level.
    capital : float
        Node's capital.
    pmax : float
        Maximum protection probability.
    cp : float
        Protection scaling parameter.
    
    Returns
    -------
    float
        Protection probability.
    
    Example
    -------
    >>> # Scenario A: p_{p,max}=1.0, c_p=1.0
    >>> pp = calculate_protection_probability(fp=0.9, capital=1.0, pmax=1.0, cp=1.0)
    >>> print(f"p_p = {pp:.4f}, failure potential = {1-pp:.4f}")
    p_p = 0.4737, failure potential = 0.5263
    """
    if fp <= 0 or capital <= 0:
        return 0.0
    return pmax / (1.0 + cp / (fp * capital))


if __name__ == "__main__":
    
    # Run with default parameters (similar to Scenario A)
    print("\n" + "=" * 70)
    print("EXAMPLE: Default parameters (Scenario A-like)")
    print("=" * 70)
    results = run_simulation(seed=42)
    
    print("\n\n")
    
    # Demonstrate Equation 7 calculations for each scenario
    print("=" * 70)
    print("THEORETICAL PROTECTION PROBABILITY (Equation 7)")
    print("p_p = p_{p,max} / (1 + c_{p,1/2} / (f_p × c))")
    print("=" * 70)
    print("\nAssuming f_p × c ≈ 0.9 (typical converged value):\n")
    
    scenarios = {
        'A': {'pmax': 1.0, 'cp': 1.0, 'desc': 'Sufficient protection → Coexistence'},
        'B': {'pmax': 0.1, 'cp': 1.0, 'desc': 'Low protection → System failure'},
        'C': {'pmax': 0.1, 'cp': 0.1, 'desc': 'Limited protection → Partial coexistence'},
        'D': {'pmax': 1.0, 'cp': 0.1, 'desc': 'Robust protection → Non-failure'},
    }
    
    for scenario, params in scenarios.items():
        pp = calculate_protection_probability(
            fp=0.9, capital=1.0, 
            pmax=params['pmax'], cp=params['cp']
        )
        print(f"Scenario {scenario}: {params['desc']}")
        print(f"  p_{{p,max}} = {params['pmax']}, c_{{p,1/2}} = {params['cp']}")
        print(f"  p_p = {pp:.4f}, failure potential = {1-pp:.4f}")
        print()