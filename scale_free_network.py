"""
Scale-Free Network Construction
================================

This module implements the Barabási-Albert preferential attachment model
for generating scale-free networks that represent sports systems.

Mathematical Foundation:
    In a scale-free network, the probability of a new node connecting to
    an existing node i is proportional to node i's degree:
    
    p(i) = k_i / Σ_j k_j
    
    This results in a power-law degree distribution:
    P(k) ~ k^(-γ), where γ typically ranges from 2 to 3

    The adjacency matrix A = (a_ij) where:
    a_ij = 1 if edge exists between nodes i and j
    a_ij = 0 otherwise

Reference:
    Park, C. & Fath, B.D. (2025). Bounded Rationality Produces Nash Equilibria 
    in Sports Networks. Physica A. (Section 2.1, Equation 4)
    
    Barabási, A.L. & Albert, R. (1999). Emergence of scaling in random networks.
    Science, 286, 509-512.

Author: Chulwook Park
License: MIT
"""

import numpy as np
import networkx as nx
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt


def create_scale_free_graph(n: int, m: int, seed: Optional[int] = None) -> nx.Graph:
    """
    Create a scale-free network using the Barabási-Albert preferential attachment model.
    
    The algorithm:
    1. Start with a complete graph of m nodes
    2. Add new nodes one at a time
    3. Each new node connects to m existing nodes
    4. Connection probability is proportional to existing node degrees
    
    Parameters
    ----------
    n : int
        Total number of nodes in the final network.
    m : int
        Number of edges each new node creates (minimum degree).
        Must satisfy m < n.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    nx.Graph
        The generated scale-free network.
    
    Example
    -------
    >>> G = create_scale_free_graph(n=100, m=5, seed=42)
    >>> print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    Nodes: 100, Edges: 475
    
    Notes
    -----
    This implements the network construction described in Section 2.1 and
    Appendix 2 of the manuscript. The resulting network captures essential
    features of sports systems where strategic information and risk
    propagate through interconnected agents.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize with complete graph of m nodes
    G = nx.complete_graph(m)
    
    # Track nodes for preferential attachment
    # Each node appears in this list proportional to its degree
    target_nodes = list(G.nodes())
    
    # Add remaining nodes with preferential attachment
    for new_node in range(m, n):
        # Select m targets with probability proportional to degree
        targets = np.random.choice(target_nodes, size=m, replace=False)
        
        # Add new node
        G.add_node(new_node)
        
        # Connect to targets
        for target in targets:
            G.add_edge(new_node, target)
        
        # Update target list for preferential attachment
        # New node appears m times (its degree)
        target_nodes.extend([new_node] * m)
        # Each target gains one connection
        target_nodes.extend(targets)
    
    return G


def create_scale_free_graph_directed(n: int, m: int, seed: Optional[int] = None) -> nx.DiGraph:
    """
    Create a directed scale-free network for asymmetric sports interactions.
    
    Parameters
    ----------
    n : int
        Total number of nodes.
    m : int
        Number of outgoing edges per new node.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    nx.DiGraph
        Directed scale-free network.
    
    Notes
    -----
    Useful for modeling directed influence in sports networks,
    such as tactical dependencies between positions.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Start with directed complete graph
    G = nx.complete_graph(m, create_using=nx.DiGraph())
    
    in_degree_list = list(G.nodes())
    
    for new_node in range(m, n):
        # Preferential attachment based on in-degree
        targets = np.random.choice(in_degree_list, size=m, replace=False)
        
        G.add_node(new_node)
        
        for target in targets:
            G.add_edge(new_node, target)
        
        in_degree_list.extend(targets)
        in_degree_list.append(new_node)
    
    return G


def get_network_properties(G: nx.Graph) -> Dict:
    """
    Calculate key network properties relevant to sports network analysis.
    
    Parameters
    ----------
    G : nx.Graph
        Input network (can be directed or undirected).
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'n_nodes': Number of nodes
        - 'n_edges': Number of edges
        - 'avg_degree': Average degree
        - 'degree_distribution': Degree frequency distribution
        - 'clustering': Average clustering coefficient
        - 'density': Network density
        - 'adjacency_matrix': Numpy adjacency matrix
        - 'eigenvalues': Eigenvalues of adjacency matrix
        - 'dominant_eigenvalue': Largest eigenvalue (λ₁)
        - 'eigenvector_centrality': Dict of eigenvector centrality values
    
    Example
    -------
    >>> G = create_scale_free_graph(100, 5, seed=42)
    >>> props = get_network_properties(G)
    >>> print(f"Dominant eigenvalue λ₁ = {props['dominant_eigenvalue']:.3f}")
    """
    # Basic properties
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    # Degree statistics
    degrees = [d for n, d in G.degree()]
    avg_degree = np.mean(degrees)
    
    # Degree distribution
    degree_counts = {}
    for d in degrees:
        degree_counts[d] = degree_counts.get(d, 0) + 1
    
    # Clustering coefficient
    clustering = nx.average_clustering(G)
    
    # Density
    density = nx.density(G)
    
    # Adjacency matrix and eigenvalues
    A = nx.to_numpy_array(G)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Sort by magnitude
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    dominant_eigenvalue = np.abs(eigenvalues[0])
    
    # Eigenvector centrality
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigenvector_centrality = nx.eigenvector_centrality_numpy(G)
    
    return {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'avg_degree': avg_degree,
        'degree_distribution': degree_counts,
        'clustering': clustering,
        'density': density,
        'adjacency_matrix': A,
        'eigenvalues': eigenvalues,
        'dominant_eigenvalue': dominant_eigenvalue,
        'dominant_eigenvector': eigenvectors[:, 0],
        'eigenvector_centrality': eigenvector_centrality
    }


def verify_scale_free_property(G: nx.Graph, plot: bool = False) -> Dict:
    """
    Verify that the network exhibits scale-free properties.
    
    Scale-free networks have degree distributions following a power law:
    P(k) ~ k^(-γ)
    
    Parameters
    ----------
    G : nx.Graph
        Network to analyze.
    plot : bool
        Whether to generate a log-log plot of degree distribution.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'gamma': Estimated power-law exponent
        - 'r_squared': Goodness of fit
        - 'is_scale_free': Boolean assessment
    """
    degrees = [d for n, d in G.degree()]
    
    # Count degree frequencies
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    
    # Filter out zero degrees for log transformation
    mask = unique_degrees > 0
    log_k = np.log10(unique_degrees[mask])
    log_p = np.log10(counts[mask] / len(degrees))
    
    # Linear fit in log-log space
    if len(log_k) > 2:
        coeffs = np.polyfit(log_k, log_p, 1)
        gamma = -coeffs[0]  # Power-law exponent
        
        # Calculate R-squared
        p_pred = np.polyval(coeffs, log_k)
        ss_res = np.sum((log_p - p_pred) ** 2)
        ss_tot = np.sum((log_p - np.mean(log_p)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    else:
        gamma = np.nan
        r_squared = np.nan
    
    # Scale-free networks typically have γ between 2 and 3
    is_scale_free = 2 <= gamma <= 3.5 if not np.isnan(gamma) else False
    
    if plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.loglog(unique_degrees[mask], counts[mask] / len(degrees), 'bo', 
                  markersize=8, label='Data')
        
        if not np.isnan(gamma):
            k_fit = np.logspace(np.log10(unique_degrees[mask].min()),
                               np.log10(unique_degrees[mask].max()), 50)
            p_fit = 10**(coeffs[1]) * k_fit**(-gamma)
            ax.loglog(k_fit, p_fit, 'r-', linewidth=2, 
                      label=f'Power law fit (γ = {gamma:.2f})')
        
        ax.set_xlabel('Degree k', fontsize=14)
        ax.set_ylabel('P(k)', fontsize=14)
        ax.set_title('Degree Distribution (Log-Log Scale)', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
    
    return {
        'gamma': gamma,
        'r_squared': r_squared,
        'is_scale_free': is_scale_free,
        'degrees': degrees,
        'degree_counts': dict(zip(unique_degrees, counts))
    }


def visualize_network(G: nx.Graph, 
                      node_colors: Optional[List] = None,
                      node_sizes: Optional[List] = None,
                      title: str = "Scale-Free Network",
                      figsize: Tuple[int, int] = (10, 10),
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize the network with customizable styling.
    
    Parameters
    ----------
    G : nx.Graph
        Network to visualize.
    node_colors : list, optional
        Colors for each node. Defaults to eigenvector centrality.
    node_sizes : list, optional
        Sizes for each node. Defaults to degree-based sizing.
    title : str
        Figure title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    
    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Layout
    pos = nx.spring_layout(G, seed=42)
    
    # Default node sizes based on degree
    if node_sizes is None:
        degrees = dict(G.degree())
        node_sizes = [300 + 100 * degrees[n] for n in G.nodes()]
    
    # Default colors based on eigenvector centrality
    if node_colors is None:
        try:
            centrality = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            centrality = {n: 0.5 for n in G.nodes()}
        node_colors = [centrality[n] for n in G.nodes()]
    
    # Draw network
    nodes = nx.draw_networkx_nodes(G, pos, 
                                   node_color=node_colors,
                                   node_size=node_sizes,
                                   cmap=plt.cm.viridis,
                                   edgecolors='black',
                                   linewidths=1,
                                   ax=ax)
    
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    
    # Colorbar
    plt.colorbar(nodes, ax=ax, label='Eigenvector Centrality')
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    print("=" * 60)
    print("SCALE-FREE NETWORK CONSTRUCTION")
    print("=" * 60)
    
    # Create network
    G = create_scale_free_graph(n=100, m=5, seed=42)
    print(f"\nNetwork created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Get properties
    props = get_network_properties(G)
    print(f"\nNetwork Properties:")
    print(f"  Average degree: {props['avg_degree']:.2f}")
    print(f"  Clustering coefficient: {props['clustering']:.4f}")
    print(f"  Density: {props['density']:.4f}")
    print(f"  Dominant eigenvalue (λ₁): {props['dominant_eigenvalue']:.3f}")
    
    # Verify scale-free property
    sf_props = verify_scale_free_property(G)
    print(f"\nScale-Free Verification:")
    print(f"  Power-law exponent (γ): {sf_props['gamma']:.2f}")
    print(f"  R-squared: {sf_props['r_squared']:.3f}")
    print(f"  Is scale-free: {sf_props['is_scale_free']}")
    
    # Visualize
    fig = visualize_network(G, title="Barabási-Albert Scale-Free Network (n=100, m=5)")
    plt.show()
