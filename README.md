# Nash Equilibria in Sports Networks

**Code repository for:** "Bounded Rationality Produces Nash Equilibria in Sports Networks: Protection, Learning, and Strategic Adaptation"

## Overview

This repository demonstrates how Nash equilibria emerge naturally in sports networks through bounded rationality—without requiring perfect information or exhaustive utility calculations. Using scale-free networks where nodes represent sports agents, the model simulates evolutionary game dynamics under varying protection capacity, social learning, and strategic adaptation parameters.

The key finding is that four distinct equilibrium patterns emerge based on the balance between protection capacity and risk propagation.

## Repository Structure
```
nash-equilibria-sports-networks/
├── requirements.txt
│
├── game_theory/
│   ├── payoff_matrix.py          # Nash equilibrium calculations
│   └── penalty_kick_nash.py      # Zero-sum game visualization
│
├── network_analysis/
│   ├── scale_free_network.py     # Barabási-Albert network construction
│   ├── pathway_proliferation.py  # Eigenvalue analysis (λ₁ convergence)
│   └── football_network.py       # FIFA 2014 World Cup network data
│
├── simulation/
│   └── protection_dynamics.py    # Agent-based protection simulation
│
└── empirical_validation/
    └── sports_equilibria.py      # NBA strategy and European football data
```

## Installation
```bash
git clone https://github.com/pcw8531/nash-equilibria-sports-networks.git
cd nash-equilibria-sports-networks
pip install -r requirements.txt
```

## Quick Start
```python
from simulation.protection_dynamics import run_simulation, calculate_protection_probability

# Run simulation
results = run_simulation(pmax=1.0, cp=1.0, seed=42)

# Calculate theoretical protection probability
pp = calculate_protection_probability(fp=0.9, capital=1.0, pmax=1.0, cp=1.0)
print(f"p_p = {pp:.4f}")  # Output: 0.4737
```

## License

MIT License

## Contact

- **Chulwook Park** - Seoul National University / OIST - pcw8531@snu.ac.kr
