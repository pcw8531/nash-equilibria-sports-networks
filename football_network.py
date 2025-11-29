"""
Football Network Analysis - FIFA World Cup Data
================================================

This module provides empirical network analysis using 2014 FIFA World Cup
team interaction data, demonstrating pathway proliferation in real sports.

Data Source:
    FIFA World Cup 2014 (www.fifa.com/worldcup/archive/brazil2014)
    Table 3.3.7 from manuscript Appendix 2

The analysis demonstrates:
    - Team formation as network structure
    - Eigenvector centrality identifying key positions
    - Pathway proliferation through tactical connections

Reference:
    Park, C. & Fath, B.D. (2025). Bounded Rationality Produces Nash Equilibria 
    in Sports Networks. Physica A. (Appendix 2, Figure 2)

Author: Chulwook Park
License: MIT
"""

import numpy as np
import pandas as pd
import networkx as nx
from numpy.linalg import matrix_power, eig
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Define colors consistent with manuscript
PLAYER_A_COLOR = '#E63946'  # Attack (red)
PLAYER_B_COLOR = '#1D3557'  # Defense (blue)
MIDFIELD_COLOR = '#8338EC'  # Midfield (purple)
NASH_COLOR = '#FCBF49'      # Highlight (yellow)
EDGE_COLOR = '#4D4D4D'      # Edges (gray)
COMPONENT_COLOR = '#2A9D8F' # Connected component (teal)


def get_fifa_2014_data() -> pd.DataFrame:
    """
    Return FIFA 2014 World Cup team network data.
    
    This data represents player position interactions and centrality
    measures from actual match analysis.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - position: Player position abbreviation
        - full_name: Full position name
        - in_degree: Normalized in-degree centrality
        - out_degree: Normalized out-degree centrality
        - centrality: Overall centrality measure
        - eigen_cent: Eigenvector centrality
    
    Notes
    -----
    Position abbreviations:
        GK = Goalkeeper, LCD/RCD = Central Defenders,
        LD/RD = Full Backs, DM = Defensive Midfielder,
        LM/RM = Wide Midfielders, OM = Offensive Midfielder,
        LF/RF = Forwards
    """
    positions = {
        'DM': 'Defensive Midfielder',
        'GK': 'Goalkeeper',
        'LCD': 'Left Central Defender',
        'LD': 'Left Defender',
        'LF': 'Left Forward',
        'LM': 'Left Midfielder',
        'OM': 'Offensive Midfielder',
        'RCD': 'Right Central Defender',
        'RD': 'Right Defender',
        'RF': 'Right Forward',
        'RM': 'Right Midfielder'
    }
    
    data = {
        'position': list(positions.keys()),
        'full_name': list(positions.values()),
        'in_degree': [0.12, 0.03, 0.09, 0.08, 0.09, 0.10, 0.11, 0.09, 0.10, 0.09, 0.11],
        'out_degree': [0.13, 0.06, 0.10, 0.10, 0.06, 0.08, 0.10, 0.11, 0.12, 0.06, 0.09],
        'centrality': [0.11, 0.06, 0.09, 0.09, 0.08, 0.09, 0.10, 0.10, 0.10, 0.08, 0.10],
        'eigen_cent': [0.49, 0.13, 0.26, 0.26, 0.21, 0.32, 0.37, 0.27, 0.26, 0.21, 0.32]
    }
    
    return pd.DataFrame(data)


def create_football_network(df: Optional[pd.DataFrame] = None,
                            weight_threshold: float = 0.5) -> nx.DiGraph:
    """
    Create a directed network from football position data.
    
    Edge weights are determined by the product of eigenvector centralities,
    representing the strength of tactical interaction between positions.
    
    Parameters
    ----------
    df : pd.DataFrame, optional
        Football data. If None, uses FIFA 2014 data.
    weight_threshold : float
        Minimum edge weight to include (filters weak connections).
    
    Returns
    -------
    nx.DiGraph
        Directed network of player positions.
    
    Example
    -------
    >>> G = create_football_network()
    >>> print(f"Network has {G.number_of_edges()} tactical connections")
    """
    if df is None:
        df = get_fifa_2014_data()
    
    G = nx.DiGraph()
    
    # Add nodes
    for position in df['position']:
        G.add_node(position)
    
    # Add weighted edges based on centrality products
    for i, pos1 in enumerate(df['position']):
        for j, pos2 in enumerate(df['position']):
            if i != j:
                weight = df['eigen_cent'].iloc[i] * df['eigen_cent'].iloc[j] * 10
                if weight > weight_threshold:
                    G.add_edge(pos1, pos2, weight=weight)
    
    return G


def get_formation_positions() -> Dict[str, Tuple[float, float]]:
    """
    Return standard football formation positions for visualization.
    
    Returns
    -------
    dict
        Position name to (x, y) coordinate mapping.
        Represents a 4-3-3 formation layout.
    """
    return {
        'GK': (-0.8, 0),
        'LCD': (-0.5, 0.5),
        'RCD': (-0.5, -0.5),
        'LD': (-0.2, 0.8),
        'RD': (-0.2, -0.8),
        'DM': (0, 0),
        'LM': (0.3, 0.6),
        'RM': (0.3, -0.6),
        'OM': (0.5, 0),
        'LF': (0.8, 0.4),
        'RF': (0.8, -0.4)
    }


def analyze_football_centrality(G: Optional[nx.DiGraph] = None,
                                df: Optional[pd.DataFrame] = None) -> Dict:
    """
    Analyze centrality and pathway proliferation in football network.
    
    Parameters
    ----------
    G : nx.DiGraph, optional
        Football network. Created from data if not provided.
    df : pd.DataFrame, optional
        Football data. Uses FIFA 2014 if not provided.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'network': The football network
        - 'adjacency_matrix': Weighted adjacency matrix
        - 'eigenvalues': Eigenvalues of adjacency matrix
        - 'dominant_eigenvalue': λ₁
        - 'eigenvector_centrality': Dict of position centralities
        - 'key_position': Position with highest centrality
        - 'pathway_proliferation': Pathway counts by length
    """
    if df is None:
        df = get_fifa_2014_data()
    
    if G is None:
        G = create_football_network(df)
    
    # Adjacency matrix
    A = nx.to_numpy_array(G, nodelist=df['position'])
    
    # Eigenvalue analysis
    eigenvalues, eigenvectors = eig(A)
    dominant_idx = np.argmax(np.abs(eigenvalues))
    lambda1 = np.abs(eigenvalues[dominant_idx])
    
    # Eigenvector centrality from data
    eigen_cent = dict(zip(df['position'], df['eigen_cent']))
    
    # Key position
    key_position = df.loc[df['eigen_cent'].idxmax(), 'position']
    
    # Pathway proliferation
    max_length = 20
    path_lengths = np.arange(1, max_length + 1)
    total_paths = []
    
    for m in path_lengths:
        Am = matrix_power(A, m)
        total_paths.append(np.sum(Am))
    
    return {
        'network': G,
        'data': df,
        'adjacency_matrix': A,
        'eigenvalues': eigenvalues,
        'dominant_eigenvalue': lambda1,
        'eigenvector_centrality': eigen_cent,
        'key_position': key_position,
        'key_position_centrality': df['eigen_cent'].max(),
        'pathway_proliferation': {
            'lengths': path_lengths,
            'total_paths': np.array(total_paths)
        }
    }


def plot_football_network(results: Optional[Dict] = None,
                          figsize: Tuple[int, int] = (15, 5),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Create the three-panel football network visualization.
    
    This generates the empirical validation figure from Appendix 2,
    showing:
        A. Team interaction matrix
        B. Network topology with formation layout
        C. Eigenvector centrality and pathway growth
    
    Parameters
    ----------
    results : dict, optional
        Output from analyze_football_centrality(). Computed if not provided.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    
    Returns
    -------
    plt.Figure
        The three-panel figure.
    """
    if results is None:
        results = analyze_football_centrality()
    
    df = results['data']
    G = results['network']
    A = results['adjacency_matrix']
    lambda1 = results['dominant_eigenvalue']
    
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=150)
    
    # Panel A: Interaction Matrix
    ax = axes[0]
    im = ax.imshow(A, cmap='Blues', interpolation='none')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Interaction Strength', fontsize=10)
    
    # Labels
    ax.set_xticks(np.arange(len(df['position'])))
    ax.set_yticks(np.arange(len(df['position'])))
    ax.set_xticklabels(df['position'], rotation=90, fontsize=8)
    ax.set_yticklabels(df['position'], fontsize=8)
    
    # Highlight DM (highest centrality)
    dm_idx = list(df['position']).index('DM')
    rect = plt.Rectangle((dm_idx - 0.5, -0.5), 1, len(df), 
                          linewidth=2, edgecolor=MIDFIELD_COLOR, facecolor='none')
    ax.add_patch(rect)
    
    ax.set_title('A. Team Interaction Matrix', fontsize=12, fontweight='bold')
    ax.set_xlabel('To Position', fontweight='bold')
    ax.set_ylabel('From Position', fontweight='bold')
    
    # Panel B: Network Topology
    ax = axes[1]
    
    pos = get_formation_positions()
    
    # Node colors by role
    node_colors = []
    for p in df['position']:
        if p in ['GK', 'LCD', 'RCD', 'LD', 'RD']:
            node_colors.append(PLAYER_B_COLOR)
        elif p in ['DM', 'LM', 'RM', 'OM']:
            node_colors.append(MIDFIELD_COLOR)
        else:
            node_colors.append(PLAYER_A_COLOR)
    
    # Node sizes by centrality
    node_sizes = [v * 1500 for v in df['eigen_cent']]
    
    # Draw network
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors,
                           node_size=node_sizes,
                           edgecolors='black',
                           linewidths=1.5,
                           ax=ax)
    
    # Edge widths
    edge_widths = [G[u][v]['weight'] / 2 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos,
                           width=edge_widths,
                           edge_color='#666666',
                           alpha=0.6,
                           connectionstyle='arc3,rad=0.1',
                           ax=ax)
    
    nx.draw_networkx_labels(G, pos,
                            font_color='white',
                            font_weight='bold',
                            font_size=9,
                            ax=ax)
    
    ax.axis('off')
    
    # Legend
    defense_patch = mpatches.Patch(color=PLAYER_B_COLOR, label='Defense')
    midfield_patch = mpatches.Patch(color=MIDFIELD_COLOR, label='Midfield')
    attack_patch = mpatches.Patch(color=PLAYER_A_COLOR, label='Attack')
    ax.legend(handles=[defense_patch, midfield_patch, attack_patch],
              loc='upper right', fontsize=8)
    
    ax.set_title('B. Football Network Topology', fontsize=12, fontweight='bold')
    ax.text(0.5, -0.1, f'λ₁ = {lambda1:.2f}', transform=ax.transAxes,
            fontsize=10, ha='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor=EDGE_COLOR))
    
    # Panel C: Centrality Bar Chart
    ax = axes[2]
    
    sorted_df = df.sort_values('eigen_cent', ascending=False)
    
    # Bar colors by role
    bar_colors = []
    for p in sorted_df['position']:
        if p in ['GK', 'LCD', 'RCD', 'LD', 'RD']:
            bar_colors.append(PLAYER_B_COLOR)
        elif p in ['DM', 'LM', 'RM', 'OM']:
            bar_colors.append(MIDFIELD_COLOR)
        else:
            bar_colors.append(PLAYER_A_COLOR)
    
    bars = ax.bar(np.arange(len(sorted_df)), sorted_df['eigen_cent'],
                  color=bar_colors, alpha=0.7, edgecolor='black')
    
    # Add values
    for i, v in enumerate(sorted_df['eigen_cent']):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
    
    ax.set_xticks(np.arange(len(sorted_df)))
    ax.set_xticklabels(sorted_df['position'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Eigenvector Centrality', fontweight='bold')
    ax.set_title('C. Position Centrality Ranking', fontsize=12, fontweight='bold')
    
    # Highlight DM
    dm_bar_idx = list(sorted_df['position']).index('DM')
    ax.annotate('DM: Highest\ncentrality',
                xy=(dm_bar_idx, sorted_df['eigen_cent'].iloc[dm_bar_idx]),
                xytext=(dm_bar_idx + 2, 0.35),
                arrowprops=dict(arrowstyle='->', color=NASH_COLOR),
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8))
    
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add equation
    ax.text(0.65, 0.85, r'$\lambda_1$ characterizes' + '\npathway proliferation',
            transform=ax.transAxes, fontsize=9, ha='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor=EDGE_COLOR))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def export_data_to_csv(filepath: str = 'fifa_2014_network.csv') -> None:
    """
    Export FIFA 2014 data to CSV file.
    
    Parameters
    ----------
    filepath : str
        Output file path.
    """
    df = get_fifa_2014_data()
    df.to_csv(filepath, index=False)
    print(f"Data exported to: {filepath}")


if __name__ == "__main__":
    print("=" * 60)
    print("FOOTBALL NETWORK ANALYSIS - FIFA 2014 WORLD CUP")
    print("=" * 60)
    
    # Load data
    df = get_fifa_2014_data()
    print("\nPosition Data:")
    print(df.to_string(index=False))
    
    # Analyze network
    results = analyze_football_centrality()
    
    print(f"\nNetwork Analysis:")
    print(f"  Nodes: {results['network'].number_of_nodes()}")
    print(f"  Edges: {results['network'].number_of_edges()}")
    print(f"  Dominant eigenvalue (λ₁): {results['dominant_eigenvalue']:.4f}")
    print(f"  Key position: {results['key_position']} "
          f"(centrality = {results['key_position_centrality']:.2f})")
    
    # Visualize
    fig = plot_football_network(results)
    plt.show()
