# Nash Equilibria in Sports Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code repository for: **"Bounded Rationality Produces Nash Equilibria in Sports Networks: Protection, Learning, and Strategic Adaptation"**



## Overview

This repository provides analysis code demonstrating how Nash equilibria emerge from bounded rationality in sports networks through social learning and strategic adaptation.

## Key Concepts

### Nash Equilibrium
A stable strategy profile where no player can improve their payoff by unilaterally changing strategy

### Protection Probability
Agent's likelihood of resisting failure by protection potential

### Scale-Free Network
Generated via Barabási-Albert preferential attachment where connection probability is proportional to node degree

## Repository Structure

| Directory | Description |
|-----------|-------------|
| `game_theory/` | Zero-sum game and payoff matrix analysis |
| `network_analysis/` | Scale-free network construction and pathway proliferation |
| `simulation/` | Protection dynamics and four equilibria scenarios |
| `empirical_validation/` | NBA and European football validation |

## Installation
```bash
git clone https://github.com/pcw8531/nash-equilibria-sports-networks.git
cd nash-equilibria-sports-networks
pip install -r requirements.txt
```

## Requirements

- Python ≥ 3.8
- NumPy
- Matplotlib
- NetworkX
- SciPy
- Pandas
- Seaborn

## Usage
```python
# Example: Generate scale-free network
from network_analysis.scale_free_network import create_scale_free_graph

G = create_scale_free_graph(n=100, m=5)
```

## Citation
```bibtex
@article{park2025nash,
  title={Bounded Rationality Produces Nash Equilibria in Sports Networks: Protection, Learning, and Strategic Adaptation},
  author={Park, Chulwook},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Chulwook Park** - Seoul National University / OIST - pcw8531@snu.ac.kr
