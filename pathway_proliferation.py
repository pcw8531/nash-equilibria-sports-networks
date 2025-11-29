"""
Pathway Proliferation Analysis
==============================

This module analyzes pathway proliferation in networks, a key mechanism
for understanding how strategic influences propagate in sports systems.

Mathematical Foundation:
    Pathway proliferation describes the geometric increase in the number
    of pathways as path length increases. For adjacency matrix A:
    
    - A^m counts all pathways of length m
    - The dominant eigenvalue λ₁ characterizes the proliferation rate:
      A^(m+1) / A^m → λ₁ as m → ∞
    
    This property explains how strategic adaptations spread through
    strongly connected components in sports networks.

Reference:
    Park, C. & Fath, B.D. (2025). Bounded Rationality Produces Nash Equilibria 
    in Sports Networks. Physica A. (Section 2.1, Appendix 2)
    
    Borrett, S.R., Fath, B.D. & Patten, B.C. (2007). Functional integration 
    of ecological networks through pathway proliferation. J. Theor. Biol.

Author: Chulwook Park
License: MIT
"""

import numpy as np
import networkx as nx
from numpy.linalg import matrix_power, eig
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def calculate_pathway_proliferation(G: nx.Graph, 
                                    max_length: int = 20) -> Dict:
    """
    Calculate pathway proliferation metrics for a network.
    
    Pathway proliferation quantifies how the number of indirect pathways
    between nodes grows with increasing path length. This is characterized
    by the dominant eigenvalue of the adjacency matrix.
    
    Parameters
    ----------
    G : nx.Graph
        Input network (directed or undirected).
    max_length : int
        Maximum pathway length to compute.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'adjacency_matrix': Network adjacency matrix
        - 'eigenvalues': All eigenvalues (sorted by magnitude)
        - 'dominant_eigenvalue': λ₁ (largest eigenvalue)
        - 'dominant_eigenvector': Corresponding eigenvector
        - 'path_lengths': Array of path lengths [1, max_length]
        - 'total_paths': Total pathways at each length
        - 'asymptotic_growth': Theoretical growth based on λ₁
        - 'proliferation_rate': Observed A^(m+1)/A^m ratios
    
    Example
    -------
    >>> G = nx.barabasi_albert_graph(50, 3, seed=42)
    >>> results = calculate_pathway_proliferation(G)
    >>> print(f"Dominant eigenvalue λ₁ = {results['dominant_eigenvalue']:.3f}")
    
    Notes
    -----
    This implements the pathway proliferation analysis described in
    Section 2.1 and Appendix 2 of the manuscript. The dominant eigenvalue
    captures how rapidly strategic influences propagate through the network.
    """
    # Get adjacency matrix
    A = nx.to_numpy_array(G)
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eig(A)
    
    # Sort by magnitude (descending)
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    dominant_eigenvalue = np.abs(eigenvalues[0])
    dominant_eigenvector = np.abs(eigenvectors[:, 0])
    
    # Normalize eigenvector
    dominant_eigenvector = dominant_eigenvector / np.max(dominant_eigenvector)
    
    # Calculate total pathways for each length
    path_lengths = np.arange(1, max_length + 1)
    total_paths = []
    
    for m in path_lengths:
        Am = matrix_power(A, m)
        total_paths.append(np.sum(Am))
    
    total_paths = np.array(total_paths)
    
    # Theoretical asymptotic growth
    # Use path length 5 as reference point
    ref_idx = min(4, len(total_paths) - 1)
    asymptotic_growth = np.array([
        total_paths[ref_idx] * (dominant_eigenvalue ** (m - path_lengths[ref_idx]))
        for m in path_lengths
    ])
    
    # Observed proliferation rate (ratio of consecutive path counts)
    proliferation_rate = np.zeros(len(path_lengths) - 1)
    for i in range(len(total_paths) - 1):
        if total_paths[i] > 0:
            proliferation_rate[i] = total_paths[i + 1] / total_paths[i]
    
    return {
        'adjacency_matrix': A,
        'eigenvalues': eigenvalues,
        'dominant_eigenvalue': dominant_eigenvalue,
        'dominant_eigenvector': dominant_eigenvector,
        'path_lengths': path_lengths,
        'total_paths': total_paths,
        'asymptotic_growth': asymptotic_growth,
        'proliferation_rate': proliferation_rate
    }


def calculate_pairwise_pathways(G: nx.Graph, 
                                source: int, 
                                target: int,
                                max_length: int = 20) -> np.ndarray:
    """
    Calculate the number of pathways between two specific nodes.
    
    Parameters
    ----------
    G : nx.Graph
        Input network.
    source : int
        Source node index.
    target : int
        Target node index.
    max_length : int
        Maximum pathway length.
    
    Returns
    -------
    np.ndarray
        Array of pathway counts for each length [1, max_length].
    
    Notes
    -----
    The (i,j) entry of A^m gives the number of pathways of length m
    from node j to node i.
    """
    A = nx.to_numpy_array(G)
    
    pathways = []
    for m in range(1, max_length + 1):
        Am = matrix_power(A, m)
        pathways.append(Am[target, source])
    
    return np.array(pathways)


def identify_strongly_connected_components(G: nx.Graph) -> Dict:
    """
    Identify strongly connected components and their pathway proliferation rates.
    
    In sports networks, strongly connected components represent functional
    modules (e.g., defensive units, offensive formations) that may develop
    internal strategic equilibria at different rates.
    
    Parameters
    ----------
    G : nx.Graph
        Input network (will be converted to directed if undirected).
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'components': List of node sets for each component
        - 'component_sizes': Size of each component
        - 'component_eigenvalues': Dominant eigenvalue for each component
        - 'largest_component': Nodes in the largest component
    """
    # Convert to directed if needed
    if not G.is_directed():
        G_dir = G.to_directed()
    else:
        G_dir = G
    
    # Find strongly connected components
    components = list(nx.strongly_connected_components(G_dir))
    
    # Sort by size (descending)
    components = sorted(components, key=len, reverse=True)
    
    component_sizes = [len(c) for c in components]
    component_eigenvalues = []
    
    for component in components:
        if len(component) > 1:
            # Create subgraph
            subgraph = G_dir.subgraph(component)
            A_sub = nx.to_numpy_array(subgraph)
            
            # Calculate dominant eigenvalue
            eigenvalues = np.linalg.eigvals(A_sub)
            dominant = np.max(np.abs(eigenvalues))
            component_eigenvalues.append(dominant)
        else:
            component_eigenvalues.append(0)
    
    return {
        'components': components,
        'component_sizes': component_sizes,
        'component_eigenvalues': component_eigenvalues,
        'largest_component': components[0] if components else set()
    }


def plot_proliferation(results: Dict,
                       figsize: Tuple[int, int] = (10, 6),
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize pathway proliferation analysis.
    
    Parameters
    ----------
    results : dict
        Output from calculate_pathway_proliferation().
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    
    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Define colors
    DATA_COLOR = '#E63946'
    THEORY_COLOR = '#FCBF49'
    EDGE_COLOR = '#4D4D4D'
    
    # Plot 1: Pathway counts (log scale)
    ax1 = axes[0]
    
    path_lengths = results['path_lengths']
    total_paths = results['total_paths']
    asymptotic_growth = results['asymptotic_growth']
    lambda1 = results['dominant_eigenvalue']
    
    ax1.semilogy(path_lengths, total_paths, 'o-', color=DATA_COLOR,
                 markersize=8, linewidth=2, label='Observed pathways')
    ax1.semilogy(path_lengths[4:], asymptotic_growth[4:], '--', 
                 color=THEORY_COLOR, linewidth=2.5,
                 label=f'λ₁ asymptote ({lambda1:.2f})')
    
    ax1.set_xlabel('Pathway Length (m)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Pathways (log scale)', fontsize=12, fontweight='bold')
    ax1.set_title('Pathway Proliferation', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Proliferation rate convergence
    ax2 = axes[1]
    
    rate = results['proliferation_rate']
    ax2.plot(path_lengths[:-1], rate, 'o-', color=DATA_COLOR,
             markersize=8, linewidth=2, label='Observed rate')
    ax2.axhline(y=lambda1, color=THEORY_COLOR, linestyle='--', 
                linewidth=2.5, label=f'λ₁ = {lambda1:.2f}')
    
    ax2.set_xlabel('Pathway Length (m)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Proliferation Rate (A^(m+1)/A^m)', fontsize=12, fontweight='bold')
    ax2.set_title('Convergence to λ₁', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_figure_2_network_panel(G: nx.Graph,
                                  positions: Optional[Dict] = None,
                                  figsize: Tuple[int, int] = (12, 4),
                                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Create the three-panel network visualization from Figure 2.
    
    Panels:
        A. Adjacency matrix with strongly connected component highlighted
        B. Network topology with eigenvector centrality
        C. Pathway proliferation with eigenvalue asymptote
    
    Parameters
    ----------
    G : nx.Graph
        Input network.
    positions : dict, optional
        Node positions for network layout.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    
    Returns
    -------
    plt.Figure
        The complete three-panel figure.
    """
    # Define colors
    COMPONENT_COLOR = '#2A9D8F'
    PLAYER_A_COLOR = '#E63946'
    PLAYER_B_COLOR = '#1D3557'
    NASH_COLOR = '#FCBF49'
    EDGE_COLOR = '#4D4D4D'
    
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=150)
    
    # Get network properties
    A = nx.to_numpy_array(G)
    eigenvalues, eigenvectors = eig(A)
    dominant_idx = np.argmax(np.abs(eigenvalues))
    lambda1 = np.abs(eigenvalues[dominant_idx])
    
    # Calculate pathway proliferation
    prolif_results = calculate_pathway_proliferation(G)
    
    # Panel A: Adjacency matrix
    ax = axes[0]
    im = ax.imshow(A, cmap='Blues', interpolation='none')
    ax.set_xlabel('To Node', fontweight='bold')
    ax.set_ylabel('From Node', fontweight='bold')
    ax.set_title('A. Adjacency Matrix', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Connection')
    
    # Panel B: Network topology
    ax = axes[1]
    
    if positions is None:
        positions = nx.spring_layout(G, seed=42)
    
    # Node sizes based on eigenvector centrality
    try:
        centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        centrality = {n: 0.5 for n in G.nodes()}
    
    node_sizes = [500 + 1000 * centrality[n] for n in G.nodes()]
    node_colors = [centrality[n] for n in G.nodes()]
    
    nodes = nx.draw_networkx_nodes(G, positions, 
                                   node_color=node_colors,
                                   node_size=node_sizes,
                                   cmap=plt.cm.viridis,
                                   edgecolors='black',
                                   linewidths=1,
                                   ax=ax)
    
    nx.draw_networkx_edges(G, positions, alpha=0.4, 
                           edge_color=EDGE_COLOR, ax=ax)
    
    ax.set_title('B. Network Topology', fontsize=12, fontweight='bold')
    ax.text(0.5, -0.1, f'λ₁ = {lambda1:.2f}', transform=ax.transAxes,
            fontsize=11, ha='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor=EDGE_COLOR))
    ax.axis('off')
    
    # Panel C: Pathway proliferation
    ax = axes[2]
    
    path_lengths = prolif_results['path_lengths']
    total_paths = prolif_results['total_paths']
    asymptotic = prolif_results['asymptotic_growth']
    
    ax.semilogy(path_lengths, total_paths, 'o-', color=PLAYER_A_COLOR,
                markersize=6, linewidth=2, label='Total pathways')
    ax.semilogy(path_lengths[5:], asymptotic[5:], '-', color=NASH_COLOR,
                linewidth=2.5, label=f'λ₁ asymptote')
    
    ax.set_xlabel('Pathway Length (m)', fontweight='bold')
    ax.set_ylabel('Number of Pathways', fontweight='bold')
    ax.set_title('C. Pathway Proliferation', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add convergence annotation
    ax.axvline(x=12, color=NASH_COLOR, linestyle='-.', alpha=0.5)
    ax.text(12.5, total_paths[5], 'Convergence\nto λ₁', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    print("=" * 60)
    print("PATHWAY PROLIFERATION ANALYSIS")
    print("=" * 60)
    
    # Create a sample network
    G = nx.barabasi_albert_graph(50, 3, seed=42)
    print(f"\nNetwork: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Calculate proliferation
    results = calculate_pathway_proliferation(G)
    
    print(f"\nDominant eigenvalue (λ₁): {results['dominant_eigenvalue']:.4f}")
    print(f"Proliferation rate convergence:")
    for i in [5, 10, 15]:
        if i < len(results['proliferation_rate']):
            print(f"  m={i}: rate = {results['proliferation_rate'][i-1]:.4f}")
    
    # Identify strongly connected components
    scc = identify_strongly_connected_components(G)
    print(f"\nStrongly connected components: {len(scc['components'])}")
    print(f"Largest component size: {scc['component_sizes'][0]}")
    
    # Visualize
    fig = plot_proliferation(results)
    plt.show()
