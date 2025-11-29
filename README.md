# Nash Equilibria in Sports Networks

**Code repository for:** "Bounded Rationality Produces Nash Equilibria in Sports Networks: Protection, Learning, and Strategic Adaptation"

## Overview

This repository demonstrates how Nash equilibria emerge naturally in sports networks through bounded rationality—without requiring perfect information or exhaustive utility calculations. Using scale-free networks where nodes represent sports agents, the model simulates evolutionary game dynamics under varying protection capacity, social learning, and strategic adaptation parameters.

The key finding is that four distinct equilibrium patterns emerge based on the balance between protection capacity and risk propagation:

| Scenario | Parameters | Outcome | Protection Probability |
|----------|------------|---------|------------------------|
| A | p_{p,max}=1.0, c_p=1.0 | Coexistence | p_p ≈ 0.47 |
| B | p_{p,max}=0.1, c_p=1.0 | System failure | p_p ≈ 0.05 |
| C | p_{p,max}=0.1, c_p=0.1 | Partial coexistence | p_p ≈ 0.05-0.10 |
| D | p_{p,max}=1.0, c_p=0.1 | Non-failure | p_p ≈ 0.50 |

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

## Core Equations

**Capital Update:**
```
c_i(t+1) = 1 + (1 - f_m - f_p) · c_i(t)
```

**Protection Probability:**
```
p_p = p_{p,max} / (1 + c_{p,1/2} / (f_p · c_i))
```

**Imitation Probability:**
```
π = 1 / (1 + exp(-s × (capital_j - capital_i)))
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
