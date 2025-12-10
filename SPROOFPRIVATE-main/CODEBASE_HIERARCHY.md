
# Codebase Hierarchy

This document provides a high-level overview of the project's structure.

- **Core Logic (`src/`)**: This directory likely contains the main source code for the trading system, including:
    - Data providers and collectors
    - Machine learning models and algorithms
    - Trading bots and execution logic
    - Risk management and position management modules

- **Tests (`tests/`)**: This directory probably contains unit tests, integration tests, and other tests for the code in `src/`.

- **Scripts (`scripts/`)**: This directory likely contains helper scripts for various tasks, such as:
    - Data processing and analysis
    - Model training and evaluation
    - System deployment and maintenance

- **Documentation (`docs/`)**: This directory is for documentation, such as this file.

- **Data (`data/`, `*_cache/`, `*.db`)**: Various directories and files are used for storing data, including:
    - Market data
    - Backtest results
    - Application-specific databases

- **Models (`models/`, `ml_models/`)**: These directories probably store trained machine learning models.

- **Configuration (`.env`, `*.json`, `*.toml`)**: A variety of files are used for configuration, including:
    - API keys and secrets
    - Application settings
    - Project dependencies

- **Demos and Runners (`run_*.py`, `demo_*.py`)**: A large number of Python scripts in the root directory are used for running demos, tests, and various parts of the system.
